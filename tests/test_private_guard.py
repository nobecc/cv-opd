"""Exercise the actual child-interpreter leakage guard, not a mocked open()."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from finance_opd.config import ROOT
from finance_opd.episode import native_node_wrapper, native_settings
from finance_opd.privacy import guard_environment


class PrivateGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.private = self.root / "private"
        self.private.mkdir()
        (self.private / "gold.json").write_text("PRIVATE_GOLD_SENTINEL")
        self.public = self.root / "report.md"
        self.public.write_text("public revenue 120")
        self.alias = self.root / "alias"
        self.alias.symlink_to(self.private, target_is_directory=True)
        self.env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONPATH": str(ROOT / "runtime_guards"),
                    "FINANCE_OPD_PRIVATE_PATHS": json.dumps([str(self.private)])}

    def python(self, code, env=None):
        return subprocess.run([sys.executable, "-B", "-c", code], env=self.env if env is None else env,
                              capture_output=True, text=True, timeout=10)

    def test_isolated_web_config_is_available_to_runtime_but_private_to_python_tools(self):
        from finance_opd.agent_state import prepare_agent_state
        source = self.private / "web.json"
        source.write_text('{"BraveApiKey":"WEB_CONFIG_SENTINEL"}')
        config = {"pi_finance": str(ROOT.parent / "pi-finance"),
                  "runtime": {"web_config_path": str(source)}, "continuation": {}}
        episode = self.root / "episode"
        copied = prepare_agent_state(config, episode) / "web-search.json"
        self.assertEqual(copied.read_bytes(), source.read_bytes())
        self.assertEqual(copied.stat().st_mode & 0o777, 0o600)
        environment = {**os.environ, **guard_environment(config, run_dir=episode)}
        for path in (source, copied):
            result = self.python(f"print(open({str(path)!r}).read())", env=environment)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("WEB_CONFIG_SENTINEL", result.stdout)

    def test_ordinary_private_reads_symlink_aliases_and_directory_listing_are_denied(self):
        for code in [f"print(open({str(self.private / 'gold.json')!r}).read())",
                     f"print(open({str(self.alias / 'gold.json')!r}).read())",
                     f"import os; print(os.listdir({str(self.private)!r}))",
                     f"import os; print(list(os.scandir({str(self.alias)!r})))",
                     f"open({str(self.private / 'gold.json')!r}, 'w').write('changed')"]:
            with self.subTest(code=code):
                result = self.python(code)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("private supervision file access denied", result.stderr)
                self.assertNotIn("PRIVATE_GOLD_SENTINEL", result.stdout)
        self.assertEqual((self.private / "gold.json").read_text(), "PRIVATE_GOLD_SENTINEL")

    def test_sample_journal_with_teacher_prefix_is_private_to_native_tools(self):
        episode = self.root / "episode"
        journal = episode / "sampled.private"
        journal.mkdir(parents=True)
        sample = journal / "sample.json"
        sample.write_text('{"teacher_prompt_ids": [123456]}')
        config = {"pi_finance": str(ROOT.parent / "pi-finance"), "runtime": {}, "continuation": {}}
        environment = {**os.environ, **guard_environment(config, run_dir=episode)}
        for code in [f"print(open({str(sample)!r}).read())",
                     f"import os; print(os.listdir({str(journal)!r}))"]:
            result = self.python(code, environment)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("private supervision file access denied", result.stderr)
            self.assertNotIn("123456", result.stdout)

    def test_public_files_remain_readable_and_no_environment_means_no_guard(self):
        result = self.python(f"print(open({str(self.public)!r}).read())")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("public revenue 120", result.stdout)
        environment = dict(self.env)
        environment.pop("FINANCE_OPD_PRIVATE_PATHS")
        result = self.python(f"print(open({str(self.private / 'gold.json')!r}).read())", environment)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PRIVATE_GOLD_SENTINEL", result.stdout)

    def test_invalid_guard_configuration_cannot_silently_start_unguarded(self):
        for raw in ["invalid json", '"/private"', '["relative/path"]', '["/"]']:
            result = self.python("print('unguarded')", {**self.env, "FINANCE_OPD_PRIVATE_PATHS": raw})
            self.assertEqual(result.returncode, 78)
            self.assertNotIn("unguarded", result.stdout)

    def test_native_launch_wrapper_propagates_guard_without_guarding_evaluator(self):
        config = {"safactory": str(ROOT.parent / "SAfactory"), "pi_finance": str(ROOT.parent / "pi-finance"),
                  "runtime": {}, "continuation": {"enabled": True, "references_path": str(self.private / "gold.json")}}
        settings = native_settings(config, "ddr_10k", base_url="http://127.0.0.1:1", run_dir=self.root / "episode")
        self.assertNotIn("FINANCE_OPD_PRIVATE_PATHS", settings["env"])
        environment = guard_environment(config, run_dir=self.root / "episode")
        paths = json.loads(environment["FINANCE_OPD_PRIVATE_PATHS"])
        self.assertIn(str(self.private / "gold.json"), paths)
        self.assertIn(str(self.root / "episode/continuation"), paths)
        self.assertIn(str(self.root / "episode/turns.jsonl.gz"), paths)
        self.assertNotIn(str(self.root / "episode/native"), paths)
        teacher = guard_environment(config)
        self.assertEqual(json.loads(teacher["FINANCE_OPD_PRIVATE_PATHS"]), [str(self.private / "gold.json")])
        wrapper = self.root / "node-wrapper.sh"
        wrapper.write_text(native_node_wrapper(config, self.root / "episode", sys.executable))
        wrapper.chmod(0o700)
        # Use the actual launch wrapper, with a Python stand-in for Node, to
        # verify its env survives the same process exec boundary.
        result = subprocess.run([str(wrapper), "-B", "-c", f"print(open({str(self.private / 'gold.json')!r}).read())"],
                                env={key: value for key, value in os.environ.items()
                                     if key != "FINANCE_OPD_PRIVATE_PATHS"},
                                capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("private supervision file access denied", result.stderr)
        config["continuation"]["enabled"] = False
        plain = native_settings(config, "ddr_10k", base_url="http://127.0.0.1:1", run_dir=self.root)
        self.assertNotIn("FINANCE_OPD_PRIVATE_PATHS", plain["env"])
        self.assertNotIn("FINANCE_OPD_PRIVATE_PATHS", native_node_wrapper(config, self.root, sys.executable))


if __name__ == "__main__":
    unittest.main()
