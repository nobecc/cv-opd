import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("probe_logprobs", Path(__file__).parents[1] / "scripts/probe_logprobs.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class ProbeGeometryTests(unittest.TestCase):
    def test_full_256k_is_covered_once_per_tp_rank_with_cp4(self):
        for tensor_rank in (0, 1):
            covered = []
            for rank in range(tensor_rank, 8, 2):
                ranges = probe.sequence_ranges(262144, 2, 4, 8, rank)
                self.assertEqual(sum(end - start for start, end in ranges), 65536)
                covered.extend(i for start, end in ranges for i in range(start, end))
            self.assertEqual(sorted(covered), list(range(262144)))
        self.assertEqual(probe.sequence_ranges(262144, 2, 4, 8, 0),
                         [(0, 32768), (229376, 262144)])

    def test_cp1_and_data_parallel_replicas(self):
        self.assertEqual(probe.sequence_ranges(17, 2, 1, 4, 3), [(0, 17)])
        self.assertEqual(probe.sequence_ranges(32, 2, 2, 8, 0),
                         probe.sequence_ranges(32, 2, 2, 8, 4))
        for values in [(31, 2, 2, 8, 0), (32, 2, 3, 8, 0), (32, 2, 2, 8, 8)]:
            with self.assertRaises(ValueError):
                probe.sequence_ranges(*values)

    def test_older_worker_reads_owning_run_and_rejects_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"training": {"gpus": 8, "context_parallel_size": 4,
                                                    "tensor_parallel_size": 2},
                                        "runtime": {"context_length": 262144}}))
            kwargs = dict(config_path=str(path), sequence_length=262144, tensor_parallel_size=2)
            self.assertEqual(probe.context_parallel_size(None, **kwargs), 4)
            with self.assertRaises(ValueError):
                probe.context_parallel_size(1, **kwargs)
            with self.assertRaises(ValueError):
                probe.context_parallel_size(None, **{**kwargs, "sequence_length": 65536})
        self.assertEqual(probe.context_parallel_size(None, config_path=None,
                                                     sequence_length=17, tensor_parallel_size=2), 1)
