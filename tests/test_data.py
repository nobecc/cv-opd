import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from finance_opd import data


class DataTests(unittest.TestCase):
    def test_duplicate_questions_use_native_identity_and_share_split(self):
        candidates = [{"task_id": "CN/case-a", "input_files": ["a.pdf"]},
                      {"task_id": "CN/case-b", "input_files": ["b.pdf"]}]
        matched = data.match_indexed_task({"native_case_id": "case-b", "task": "same question"},
                                         {"same question": candidates})
        self.assertEqual(matched["input_files"], ["b.pdf"])
        rows = [{"metadata": {}} for _ in range(3)]
        split = data.assign_splits(rows, [["question-a", "doc-a"],
                                         ["question-a", "doc-b"],
                                         ["question-c", "doc-b"]], 0.5, 42)
        self.assertEqual(len({label for label, _ in split}), 1)
        self.assertEqual(len({row["metadata"]["split_group"] for _, row in split}), 1)

    def test_gold_is_separate_and_unicode_line_separator_stays_in_row(self):
        with tempfile.TemporaryDirectory() as root:
            workspace = Path(root)
            index = workspace / "eval_finance/data/finfirst"
            datasets = workspace / "SAfactory/env/pi_finance/datasets"
            index.mkdir(parents=True); datasets.mkdir(parents=True)
            objective = "question\u2028still the same task"
            row = {"id": "finfirst:1", "benchmark": "finfirst", "native_case_id": "1",
                   "task": objective, "source": {}, "evaluation": {"row": {"answer": "SECRET_GOLD"}}}
            (datasets / "finfirst.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n")
            (index / "tasks.jsonl").write_text(json.dumps({"task_id": "1", "objective": objective}) + "\n")
            out = workspace / "standalone/output"
            with patch.object(data, "ROOT", workspace / "standalone"):
                result = data.prepare(workspace, out, ["finfirst"], validation_fraction=0)
            self.assertEqual(result["counts"], {"finfirst": 1})
            self.assertNotIn("SECRET_GOLD", (out / "train.jsonl").read_text())
            self.assertNotIn("evaluation", (out / "train.jsonl").read_text())
            self.assertIn("SECRET_GOLD", (out / "private_references.jsonl").read_text())
            self.assertEqual(list(data.read_jsonl(out / "train.jsonl"))[0]["task"], objective)
            self.assertEqual((out / "private_references.jsonl").stat().st_mode & 0o777, 0o600)
            with patch.object(data, "ROOT", workspace / "standalone"), self.assertRaises(FileExistsError):
                data.prepare(workspace, out, ["finfirst"])

    def test_dataclaw_only_exposes_database(self):
        with tempfile.TemporaryDirectory() as root:
            index, out = Path(root) / "index", Path(root) / "out"
            database = index / "dataclawbench/assets/database"
            database.mkdir(parents=True)
            (database / "facts.csv").write_text("a,b\n1,2\n")
            gold = database.parent / "qa_gold"
            gold.mkdir(); (gold / "answers.json").write_text("secret")
            source = data.stage_source({"id": "task", "benchmark": "dataclawbench", "source": {"data_path": "/workspace/data/files"}}, index, None, out)
            visible = Path(source["data_path"])
            self.assertEqual([p.name for p in visible.iterdir()], ["database"])
            self.assertFalse((visible / "qa_gold").exists())


if __name__ == "__main__":
    unittest.main()
