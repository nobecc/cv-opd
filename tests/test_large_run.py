import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from finance_opd.config import ROOT
from finance_opd.checkpoints import prune_checkpoints
from finance_opd.cluster_worker import remaining_seconds
from finance_opd.sqlite_data import prepare


class LargeRunTests(unittest.TestCase):
    def test_deadline_replaces_three_hour_limit_and_rejects_expired(self):
        with patch("finance_opd.cluster_worker.time.time", return_value=1000):
            self.assertEqual(remaining_seconds({}), 10800)
            self.assertEqual(remaining_seconds({"job": {"deadline_epoch": 1000 + 7 * 86400}}), 7 * 86400)
            with self.assertRaises(TimeoutError):
                remaining_seconds({"job": {"deadline_epoch": 999}})

    def test_retention_requires_complete_marker_and_keeps_in_progress_save(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as tmp:
            output = Path(tmp) / "training"
            root = output / "checkpoints"
            for iteration in range(5):
                for rank in range(2):
                    path = root / f"iter_{iteration:07d}/mp_rank_{rank:02d}/model_optim_rng.pt"
                    path.parent.mkdir(parents=True)
                    path.write_bytes(b"weights")
            marker = root / "latest_checkpointed_iteration.txt"
            marker.write_text("3")
            config = {"output": str(output), "training": {
                "checkpoint_format": "torch", "tensor_parallel_size": 2, "keep_checkpoints": 2}}
            missing = root / "iter_0000003/mp_rank_01/model_optim_rng.pt"
            missing.unlink()
            self.assertEqual(prune_checkpoints(config), [])
            missing.write_bytes(b"weights")
            removed = prune_checkpoints(config)
            self.assertEqual(len(removed), 2)
            self.assertTrue((root / "iter_0000002").is_dir())
            self.assertTrue((root / "iter_0000003").is_dir())
            self.assertTrue((root / "iter_0000004").is_dir())
            self.assertEqual(prune_checkpoints(config), [])
            marker.write_text("")
            self.assertEqual(prune_checkpoints(config), [])

    def test_retention_rejects_external_path(self):
        with self.assertRaises(ValueError):
            prune_checkpoints({"output": "/tmp/unowned", "training": {
                "keep_checkpoints": 2, "checkpoint_format": "torch"}})

    def test_single_checkpoint_retention_preserves_latest_complete_save(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as tmp:
            output = Path(tmp) / "training"
            root = output / "checkpoints"
            for iteration in range(3):
                p = root / f"iter_{iteration:07d}/mp_rank_00/model_optim_rng.pt"
                p.parent.mkdir(parents=True)
                p.write_bytes(b"weights")
            (root / "latest_checkpointed_iteration.txt").write_text("1")
            config = {"output": str(output), "training": {
                "checkpoint_format": "torch", "tensor_parallel_size": 1, "keep_checkpoints": 1}}
            self.assertEqual(len(prune_checkpoints(config)), 1)
            self.assertTrue((root / "iter_0000001").is_dir())
            self.assertTrue((root / "iter_0000002").is_dir())
            config["training"]["keep_checkpoints"] = 0
            with self.assertRaises(ValueError):
                prune_checkpoints(config)

    def test_company_split_readonly_and_eval_overlap(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as tmp:
            base = Path(tmp)
            db = base / "source.db"
            with sqlite3.connect(db) as connection:
                connection.executescript("CREATE TABLE companies(cik TEXT, name TEXT);"
                                         "CREATE TABLE filings(cik TEXT); CREATE TABLE financial_facts(cik TEXT);")
                for cik in range(10):
                    connection.execute("INSERT INTO companies VALUES(?,?)", (str(cik), str(cik)))
                    connection.execute("INSERT INTO filings VALUES(?)", (str(cik),))
                    connection.execute("INSERT INTO financial_facts VALUES(?)", (str(cik),))
            before = db.read_bytes()
            manifest = prepare(db, base / "data", validation_count=2)
            self.assertEqual(manifest["counts"], {"train": 8, "validation": 2})
            sets = []
            for split in ("train", "validation"):
                rows = [json.loads(line) for line in (base / f"data/{split}.jsonl").read_text().splitlines()]
                sets.append({r["metadata"]["entity_id"] for r in rows})
                self.assertTrue(all(r["metadata"]["source"]["db_path"] == str(db) for r in rows))
            self.assertFalse(sets[0] & sets[1])
            self.assertEqual(db.read_bytes(), before)
            with self.assertRaisesRegex(ValueError, "overlap"):
                prepare(db, base / "overlap", validation_count=2, eval_database=db)
