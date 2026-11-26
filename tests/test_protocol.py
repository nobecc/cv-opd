import asyncio
import copy
import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path

from finance_opd.protocol import (aligned_teacher_logps, assert_compatible_tokenizers,
                                  freeze_teacher_view, sampled_output, score_payload)


class ProtocolTests(unittest.TestCase):
    def test_teacher_cannot_shift_or_truncate_sampled_output(self):
        body = score_payload([1, 2, 3], [8, 9])
        self.assertEqual(body["sampling_params"]["max_new_tokens"], 0)
        response = {"meta_info": {"input_token_logprobs": [[None, 1], [-1, 2], [-1, 3], [-2, 8], [-3, 9]]}}
        self.assertEqual(aligned_teacher_logps(response, [1, 2, 3], [8, 9]), [-2, -3])
        for bad in [[[-2, 8]], [[-1, 1], [-1, 2], [-1, 3], [-2, 9], [-3, 8]]]:
            with self.assertRaises(ValueError):
                aligned_teacher_logps({"meta_info": {"input_token_logprobs": bad}}, [1, 2, 3], [8, 9])
        response["meta_info"]["input_token_logprobs"][-1][0] = float("nan")
        with self.assertRaises(ValueError):
            aligned_teacher_logps(response, [1, 2, 3], [8, 9])

    def test_engine_probabilities_are_authoritative(self):
        value = {"text": "answer", "output_ids": [3], "meta_info": {
            "output_token_logprobs": [[-0.2, 3]], "finish_reason": {"type": "stop"}, "weight_version": "v7"}}
        self.assertEqual(sampled_output(value)[0], [3])
        value["output_ids"] = [4]
        with self.assertRaises(ValueError):
            sampled_output(value)

    def test_teacher_must_confirm_the_conditioning_prefix_too(self):
        response = {"meta_info": {"input_token_logprobs": [[None, 100], [-1, 2], [-2, 8]]}}
        with self.assertRaisesRegex(ValueError, "aligned"):
            aligned_teacher_logps(response, [1, 2], [8])

    def test_hook_gets_no_future_output_and_is_copied(self):
        messages = [{"role": "user", "content": "task"}]
        holder = {}
        module = types.ModuleType("opd_test_hook")
        def hook(request):
            self.assertEqual(set(request), {"messages", "task", "turn"})
            holder["result"] = {"messages": request["messages"], "weight": 0.5}
            return holder["result"]
        module.make = hook
        sys.modules[module.__name__] = module
        try:
            view = freeze_teacher_view("opd_test_hook.make", messages, {"task": "task"}, 0)
            holder["result"]["messages"][0]["content"] = "mutated after sampling"
            self.assertEqual(view["messages"], messages)
            self.assertEqual(view["weight"], 0.5)
        finally:
            del sys.modules[module.__name__]

    def test_tokenizer_extra_ids_allowed_but_remapping_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            a, b = Path(root) / "a", Path(root) / "b"
            a.mkdir(); b.mkdir()
            value = {"model": {"vocab": {"x": 0}}, "added_tokens": [{"id": 1, "content": "<tool>"}]}
            (a / "tokenizer.json").write_text(json.dumps(value))
            extra = copy.deepcopy(value)
            extra["added_tokens"].append({"id": 2, "content": "<audio>"})
            (b / "tokenizer.json").write_text(json.dumps(extra))
            assert_compatible_tokenizers(a, b)
            extra["added_tokens"][0]["content"] = "wrong"
            (b / "tokenizer.json").write_text(json.dumps(extra))
            with self.assertRaises(ValueError):
                assert_compatible_tokenizers(a, b)


    def test_merge_serialization_preserves_order_and_pretokenizer_checks(self):
        with tempfile.TemporaryDirectory() as root:
            a, b = Path(root) / "a", Path(root) / "b"
            a.mkdir()
            b.mkdir()
            first = {"model": {"type": "BPE", "vocab": {"a": 0, "b": 1},
                               "merges": [["a", "b"], ["b", "a"]]},
                     "pre_tokenizer": {"type": "ByteLevel", "use_regex": False}}
            second = copy.deepcopy(first)
            second["model"]["merges"] = ["a b", "b a"]
            (a / "tokenizer.json").write_text(json.dumps(first))
            (b / "tokenizer.json").write_text(json.dumps(second))
            assert_compatible_tokenizers(a, b)
            second["model"]["merges"].reverse()
            (b / "tokenizer.json").write_text(json.dumps(second))
            with self.assertRaises(ValueError):
                assert_compatible_tokenizers(a, b)
            second["model"]["merges"].reverse()
            second["pre_tokenizer"]["use_regex"] = True
            (b / "tokenizer.json").write_text(json.dumps(second))
            with self.assertRaises(ValueError):
                assert_compatible_tokenizers(a, b)


if __name__ == "__main__":
    unittest.main()
