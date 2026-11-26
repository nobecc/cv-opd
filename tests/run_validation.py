"""Reproducible local validation; real CPU tensors/tools, fixture model servers."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import unittest
from collections import Counter
from datetime import datetime
from pathlib import Path

from finance_opd.config import BENCHMARKS, ROOT, read_config
from finance_opd.data import read_jsonl
from finance_opd.episode import write_json


def audit_dataset(config):
    directory = Path(config["dataset"]).parent
    workspace = Path(config["workspace"])
    manifest = json.loads((directory / "manifest.json").read_text())
    originals = {}
    for benchmark in BENCHMARKS:
        path = workspace / "SAfactory/env/pi_finance/datasets" / f"{benchmark}.jsonl"
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["input_sha256"][str(path)]:
            raise AssertionError(f"source dataset changed since export: {benchmark}")
        originals.update((row["id"], row) for row in read_jsonl(path))
    seen, questions, documents, groups = {}, {}, {}, {}
    counts, benchmarks, selected = Counter(), Counter(), {}
    material_files = 0
    for split in ["train", "validation"]:
        for row in read_jsonl(directory / f"{split}.jsonl"):
            assert set(row) == {"task", "metadata"}
            meta = row["metadata"]
            assert set(meta) <= {"id", "benchmark", "native_case_id", "entity_id", "source", "split_group"}
            identity = meta["id"]
            assert identity not in seen
            source = originals[identity]
            assert row["task"] == source["task"]
            assert meta["native_case_id"] == source["native_case_id"]
            benchmark = meta["benchmark"]
            assert benchmark == source["benchmark"]
            seen[identity] = split
            assert groups.setdefault(meta["split_group"], split) == split
            assert questions.setdefault((benchmark, " ".join(row["task"].split())), split) == split
            files = []
            for key, name in meta["source"].items():
                path = Path(name)
                assert path.exists(), name
                if key == "db_path":
                    assert path.is_file()
                elif benchmark == "finance_complex_qa":
                    for material in path.iterdir():
                        assert material.is_file() and material.is_symlink()
                        assert "Reference_documents" in material.resolve().parts
                        files.append(str(material.resolve()))
                        material_files += 1
                elif benchmark == "dataclawbench":
                    assert {item.name for item in path.iterdir()} == {"database"}
                    assert (path / "database").resolve() == (workspace / "eval_finance/data/dataclawbench/assets/database").resolve()
                elif benchmark in {"finance_agent_bench_v1_1", "finance_agent_bench_v2"}:
                    assert path.resolve() == (workspace / "eval_finance/data" / benchmark / "materials" / meta["native_case_id"]).resolve()
            if files:
                assert documents.setdefault((benchmark, tuple(sorted(files))), split) == split
            counts[split] += 1
            benchmarks[benchmark] += 1
            selected.setdefault(benchmark, {**meta, "task": row["task"]})
    assert set(seen) == set(originals)
    references = list(read_jsonl(directory / "private_references.jsonl"))
    assert len(references) == len(seen)
    assert len({r["id"] for r in references}) == len(seen)
    for row in references:
        assert row["split"] == seen[row["id"]]
        assert row["evaluation"] == originals[row["id"]]["evaluation"]
    assert (directory / "private_references.jsonl").stat().st_mode & 0o777 == 0o600
    assert dict(counts) == manifest["splits"]
    assert dict(benchmarks) == manifest["counts"]
    return {"passed": True, "total": len(seen), "splits": dict(counts), "benchmarks": dict(benchmarks),
            "checked_material_links": material_files, "question_or_document_split_overlap": False,
            "task_only_schema": True, "private_references_match_originals": True}, selected


async def profiles(output, config, selected):
    from native_smoke import run, text_builder
    tokenizer, builder = await asyncio.to_thread(text_builder, config)
    backend = (tokenizer, type(builder))
    semaphore = asyncio.Semaphore(3)
    async def one(benchmark):
        async with semaphore:
            print(f"native profile start: {benchmark}", flush=True)
            try:
                report = await run(output / "profiles" / benchmark, config,
                                   profile_task=selected[benchmark], shared_text_backend=backend)
                result = {"passed": True, "turns": report["generated_turns"],
                          "tool_calls": report["native_result"]["tool_calls"],
                          "source": selected[benchmark]["source"], "task_id": selected[benchmark]["id"]}
            except Exception as error:
                result = {"passed": False, "error": f"{type(error).__name__}: {error}"}
            print(f"native profile done: {benchmark}: {result['passed']}", flush=True)
            return benchmark, result
    return dict(await asyncio.gather(*(one(benchmark) for benchmark in BENCHMARKS)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "validation" / datetime.now().strftime("expanded-%Y%m%d-%H%M%S"))
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("validation output must remain in cv-opd")
    output.mkdir(parents=True, exist_ok=False)
    print(f"Validation artifacts: {output}", flush=True)
    config = read_config()
    import torch
    torch.set_num_threads(1)
    result = {"torch_version": torch.__version__, "cuda_available": torch.cuda.is_available(),
              "model_services": "deterministic HTTP fixtures", "full_gpu_training_executed": False,
              "virtual_cp_test_sizes": [1, 2, 4, 8], "distributed_collectives_tested": False}
    try:
        suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*.py")
        with (output / "tests.log").open("w") as stream:
            tested = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        result["tests"] = {"run": tested.testsRun, "failures": len(tested.failures), "errors": len(tested.errors),
                           "skipped": [(str(test), reason) for test, reason in tested.skipped],
                           "passed": tested.wasSuccessful() and not tested.skipped}
        print(f"Tests: {result['tests']}", flush=True)
        if not tested.wasSuccessful():
            raise AssertionError("test suite failed; see tests.log")
        result["dataset"], selected = audit_dataset(config)
        write_json(output / "dataset.json", result["dataset"])
        print(f"Dataset audited: {result['dataset']['total']} tasks", flush=True)
        result["profiles"] = asyncio.run(profiles(output, config, selected))
        result["passed"] = result["tests"]["passed"] and all(r["passed"] for r in result["profiles"].values())
    except Exception as error:
        result.update(passed=False, error=f"{type(error).__name__}: {error}")
    finally:
        result["source_sha256"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for folder in ["finance_opd", "tests"] for path in (ROOT / folder).glob("*.py")}
        write_json(output / "RESULTS.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"source_sha256", "profiles"}}, indent=2), flush=True)
    raise SystemExit(0 if result.get("passed") else 1)


if __name__ == "__main__":
    main()
