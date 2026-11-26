"""vLLM teacher-transport contract tests (no network, no models)."""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finance_opd.teacher_transport import (  # noqa: E402
    parse_vllm_completion, parse_vllm_models, parse_vllm_prompt_logprobs,
    transport_name, validate_teacher_config, vllm_generate_payload, vllm_score_payload)


def sglang_config():
    return {"teacher": {"url": "http://127.0.0.1:31051",
                        "model": "/models/qwen-teacher"}}


def vllm_config():
    return {"teacher": {"url": "http://10.0.0.1:8000/", "transport": "vllm",
                        "model": "qwen-teacher-snapshot",
                        "tokenizer_path": "/models/qwen-teacher"}}


class TransportConfig(unittest.TestCase):
    def test_default_is_sglang(self):
        self.assertEqual(transport_name(sglang_config()), "sglang")

    def test_unknown_transport_rejected(self):
        config = vllm_config()
        config["teacher"]["transport"] = "triton"
        with self.assertRaises(ValueError):
            transport_name(config)

    def test_vllm_requires_served_name_and_tokenizer(self):
        validate_teacher_config(vllm_config())
        for missing in ("model", "tokenizer_path"):
            config = vllm_config()
            config["teacher"][missing] = ""
            with self.assertRaises(ValueError):
                validate_teacher_config(config)

    def test_sglang_requires_local_directory(self):
        config = sglang_config()
        config["teacher"]["model"] = "/definitely/not/a/model"
        with self.assertRaises(ValueError):
            validate_teacher_config(config)


class ScoringPayload(unittest.TestCase):
    def test_payload_shape(self):
        payload = vllm_score_payload([5, 6], [7], "served")
        self.assertEqual(payload["prompt"], [5, 6, 7])
        self.assertEqual(payload["model"], "served")
        self.assertEqual(payload["prompt_logprobs"], 0)
        self.assertEqual(payload["max_tokens"], 1)

    def test_empty_sides_rejected(self):
        with self.assertRaises(ValueError):
            vllm_score_payload([], [7], "served")
        with self.assertRaises(ValueError):
            vllm_score_payload([5], [], "served")


def score_response(entries, prompt_tokens=None):
    choice = {"prompt_logprobs": entries}
    response = {"choices": [choice]}
    if prompt_tokens is not None:
        response["usage"] = {"prompt_tokens": prompt_tokens}
    return response


class ScoringParse(unittest.TestCase):
    def test_happy_path_returns_output_logprobs(self):
        entries = [None, {6: -0.25}, {7: -1.5}]
        logps = parse_vllm_prompt_logprobs(score_response(entries, prompt_tokens=3), [5, 6], [7])
        self.assertEqual(logps, [-1.5])

    def test_string_token_keys_accepted(self):
        entries = [None, {"6": -0.25}, {"7": -1.5}]
        logps = parse_vllm_prompt_logprobs(score_response(entries), [5, 6], [7])
        self.assertEqual(logps, [-1.5])

    def test_extra_top_tokens_do_not_break_alignment(self):
        entries = [None, {6: -0.25, 99: -0.1}, {7: -1.5, 98: -0.2}]
        logps = parse_vllm_prompt_logprobs(score_response(entries), [5, 6], [7])
        self.assertEqual(logps, [-1.5])

    def test_missing_first_none_rejected(self):
        entries = [{5: -0.1}, {6: -0.25}, {7: -1.5}]
        with self.assertRaises(ValueError):
            parse_vllm_prompt_logprobs(score_response(entries), [5, 6], [7])

    def test_wrong_length_rejected(self):
        with self.assertRaises(ValueError):
            parse_vllm_prompt_logprobs(score_response([None, {6: -0.25}]), [5, 6], [7])

    def test_shifted_prompt_rejected(self):
        # A one-token shift must not pass: position keys no longer match.
        entries = [None, {5: -0.1}, {6: -0.25}]
        with self.assertRaises(ValueError):
            parse_vllm_prompt_logprobs(score_response(entries), [5, 6], [7])

    def test_usage_mismatch_rejected(self):
        entries = [None, {6: -0.25}, {7: -1.5}]
        with self.assertRaises(ValueError):
            parse_vllm_prompt_logprobs(score_response(entries, prompt_tokens=2), [5, 6], [7])

    def test_nonfinite_rejected(self):
        entries = [None, {6: -0.25}, {7: math.inf}]
        with self.assertRaises(ValueError):
            parse_vllm_prompt_logprobs(score_response(entries), [5, 6], [7])

    def test_multiple_choices_rejected(self):
        response = {"choices": [{}, {}]}
        with self.assertRaises(ValueError):
            parse_vllm_prompt_logprobs(response, [5], [7])


def completion(text="ok", finish="stop", prompt_tokens=2, completion_tokens=2, logprobs=None):
    choice = {"text": text, "finish_reason": finish}
    if logprobs is not None:
        choice["logprobs"] = logprobs
    response = {"choices": [choice]}
    if prompt_tokens is not None:
        response["usage"] = {"prompt_tokens": prompt_tokens,
                             "completion_tokens": completion_tokens}
    return response


class GeneratePayload(unittest.TestCase):
    def test_sampling_params_map_to_vllm_fields(self):
        params = {"temperature": 0.0, "top_p": 1.0, "top_k": -1,
                  "max_new_tokens": 16, "skip_special_tokens": True, "sampling_seed": 42}
        payload = vllm_generate_payload([1, 2], params, "served", sampled=True)
        self.assertEqual(payload["prompt"], [1, 2])
        self.assertEqual(payload["max_tokens"], 16)
        self.assertEqual(payload["seed"], 42)
        self.assertEqual(payload["top_k"], -1)
        self.assertEqual(payload["logprobs"], 1)
        self.assertTrue(payload["return_tokens_as_token_ids"])
        self.assertFalse(payload["skip_special_tokens"])
        self.assertNotIn("sampling_seed", payload)
        self.assertNotIn("no_stop_trim", payload)

    def test_plain_generation_omits_logprobs(self):
        payload = vllm_generate_payload([1], {"max_new_tokens": 4}, "served", sampled=False)
        self.assertNotIn("logprobs", payload)


class CompletionParse(unittest.TestCase):
    def test_original_sampled_ids_are_required_without_retokenization(self):
        from finance_opd.teacher_transport import reconstruct_sampled_ids
        choice = {"text": "arbitrary decoded text", "logprobs": {
            "tokens": ["token_id:12", "token_id:45"], "token_logprobs": [-0.1, -0.2]}}
        self.assertEqual(reconstruct_sampled_ids({}, choice, [99]), ([12, 45], choice["text"]))
        for tokens in [None, ["a", "b"], [12, 45], ["token_id:12"], ["token_id:-1", "token_id:45"]]:
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                reconstruct_sampled_ids({}, {**choice, "logprobs": {**choice["logprobs"], "tokens": tokens}}, [99])
        with self.assertRaises(ValueError):
            reconstruct_sampled_ids({}, {**choice, "token_ids": [12, 46]}, [99])

    def test_plain_completion_fields(self):
        parsed = parse_vllm_completion(completion(), prefix_len=2, max_new_tokens=8)
        self.assertEqual(parsed["text"], "ok")
        self.assertEqual(parsed["finish"], "stop")
        self.assertEqual(parsed["prompt_tokens"], 2)

    def test_abnormal_finish_rejected(self):
        with self.assertRaises(ValueError):
            parse_vllm_completion(completion(finish="content_filter"),
                                  prefix_len=2, max_new_tokens=8)

    def test_prompt_truncation_rejected(self):
        with self.assertRaises(ValueError):
            parse_vllm_completion(completion(prompt_tokens=1),
                                  prefix_len=2, max_new_tokens=8)

    def test_completion_overrun_rejected(self):
        with self.assertRaises(ValueError):
            parse_vllm_completion(completion(completion_tokens=9),
                                  prefix_len=2, max_new_tokens=8)

    def test_sampled_probabilities_accept_openai_layout(self):
        from finance_opd.teacher_transport import parse_vllm_sampled_logprobs
        self.assertEqual(parse_vllm_sampled_logprobs({"logprobs": {"token_logprobs": [-0.1, -0.2]}}), [-0.1, -0.2])

    def test_sampled_nonfinite_or_missing_probabilities_rejected(self):
        from finance_opd.teacher_transport import parse_vllm_sampled_logprobs
        for value in [None, {}, [], {"token_logprobs": []}, {"token_logprobs": [float("nan")]},
                      {"token_logprobs": [True]}]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_vllm_sampled_logprobs({"logprobs": value})


class ModelsParse(unittest.TestCase):
    def test_single_model(self):
        self.assertEqual(parse_vllm_models({"data": [{"id": "served"}]}), "served")

    def test_multiple_models_rejected(self):
        with self.assertRaises(ValueError):
            parse_vllm_models({"data": [{"id": "a"}, {"id": "b"}]})

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            parse_vllm_models({"data": []})


if __name__ == "__main__":
    unittest.main()
