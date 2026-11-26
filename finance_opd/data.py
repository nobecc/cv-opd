"""Prepare task-only Slime input and separate private evaluator references."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from .config import BENCHMARKS, ROOT


def read_jsonl(path: Path):
    with path.open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected an object")
            yield row


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def match_indexed_task(row: dict, by_objective: dict) -> dict | None:
    candidates = by_objective.get(row["task"], [])
    identity = str(row["native_case_id"])
    exact = [item for item in candidates
             if str(item["task_id"]) == identity or str(item["task_id"]).rsplit("/", 1)[-1] == identity]
    if len(exact) == 1:
        return exact[0]
    return candidates[0] if len(candidates) == 1 else None


def assign_splits(prepared: list[dict], group_keys: list[list[str]], fraction: float, seed: int):
    # Connected components prevent a duplicate question with different documents
    # from crossing the split through a second task sharing either document set.
    parents = list(range(len(prepared)))
    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    owners = {}
    for i, keys in enumerate(group_keys):
        for key in keys:
            if key in owners:
                parents[root(i)] = root(owners[key])
            else:
                owners[key] = i
    components = {}
    for i, keys in enumerate(group_keys):
        components.setdefault(root(i), set()).update(keys)
    identifiers = {r: digest(sorted(keys)) for r, keys in components.items()}
    result = []
    for i, row in enumerate(prepared):
        group = identifiers[root(i)]
        row["metadata"]["split_group"] = group
        split = "validation" if int(digest([seed, group])[:16], 16) / 2**64 < fraction else "train"
        result.append((split, row))
    return result


def checked_material(index: Path, relative: str) -> Path:
    path = (index / relative).resolve()
    if not path.is_relative_to(index.resolve()) or not path.is_file():
        raise ValueError(f"missing/unsafe indexed material: {relative}")
    with path.open("rb") as stream:
        if stream.read(50).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError(f"unresolved Git LFS material: {relative}")
    return path


def stage_source(row: dict, index: Path, indexed: dict | None, output: Path) -> dict:
    """Link only tool-visible materials, never the benchmark/reference directory."""
    benchmark = row["benchmark"]
    source = row.get("source") or {}
    if not source:
        return {}
    stage = output / "materials" / benchmark / digest(row["id"])[:16]
    if benchmark == "ddr_10k":
        return {"db_path": str(checked_material(index, "ddr_10k/raw/10k_financial_data.db"))}
    if benchmark == "dataclawbench":
        database = index / "dataclawbench/assets/database"
        if not database.is_dir() or not any(database.iterdir()):
            raise ValueError(f"missing database materials: {database}")
        stage.mkdir(parents=True)
        (stage / "database").symlink_to(database.resolve(), target_is_directory=True)
    elif benchmark in {"finance_agent_bench_v1_1", "finance_agent_bench_v2"}:
        # Native task-specific bundles have no rubric/checklist files.
        material = index / benchmark / "materials" / row["native_case_id"]
        if not material.is_dir() or not any(material.iterdir()):
            raise ValueError(f"missing task material bundle: {material}")
        return {"data_path": str(material.resolve())}
    elif benchmark == "finance_complex_qa":
        if indexed is None:
            raise ValueError(f"no matching indexed task for {row['id']}")
        paths = [checked_material(index, p) for p in indexed.get("input_files", [])]
        if not any(p.suffix.lower() == ".pdf" for p in paths):
            raise ValueError(f"no reference document PDF for {row['id']}")
        stage.mkdir(parents=True)
        for material in paths:
            target = stage / material.name
            if target.exists() or target.is_symlink():
                if target.resolve() != material:
                    raise ValueError(f"material basename collision: {material.name}")
            else:
                target.symlink_to(material)
    else:
        raise ValueError(f"unhandled native source layout: {benchmark}")
    return {"data_path": str(stage.resolve())}


def prepare(workspace: Path, output: Path, benchmarks=BENCHMARKS, *, limit=None,
            validation_fraction=0.1, seed=42) -> dict:
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0,1)")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    workspace, output = workspace.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"choose a new output directory: {output}")
    # All generated files, including material links, belong to this standalone project.
    if not output.is_relative_to(ROOT):
        raise ValueError(f"output must be under {ROOT}")
    output.mkdir(parents=True)
    index = workspace / "eval_finance/data"
    datasets = workspace / "SAfactory/env/pi_finance/datasets"
    prepared, references, seen, group_keys = [], [], set(), []
    input_hashes = {}
    try:
        for benchmark in benchmarks:
            if benchmark not in BENCHMARKS:
                raise ValueError(f"unknown benchmark: {benchmark}")
            dataset = datasets / f"{benchmark}.jsonl"
            input_hashes[str(dataset)] = hashlib.sha256(dataset.read_bytes()).hexdigest()
            by_objective = {}
            for item in read_jsonl(index / benchmark / "tasks.jsonl"):
                by_objective.setdefault(item["objective"], []).append(item)
            rows = read_jsonl(dataset)
            for position, row in enumerate(rows):
                if limit is not None and position >= limit:
                    break
                if row["benchmark"] != benchmark or row["id"] in seen:
                    raise ValueError(f"duplicate/mismatched task identity: {row.get('id')}")
                seen.add(row["id"])
                indexed = match_indexed_task(row, by_objective)
                source = stage_source(row, index, indexed, output)
                # Explicit allowlist. evaluation, answers, rubrics and reward metadata never
                # enter the model dataset or the context hook.
                metadata = {k: row[k] for k in ["id", "benchmark", "native_case_id", "entity_id"] if k in row}
                metadata["source"] = source
                keys = [digest([benchmark, "question", " ".join(row["task"].split())])]
                if indexed and indexed.get("input_files") and benchmark != "dataclawbench":
                    keys.append(digest([benchmark, "documents", sorted(indexed["input_files"])]))
                group_keys.append(keys)
                prepared.append({"task": row["task"], "metadata": metadata})
                references.append({"id": row["id"], "benchmark": benchmark,
                                   "native_case_id": row["native_case_id"],
                                   "evaluation": row["evaluation"]})
        prepared = assign_splits(prepared, group_keys, validation_fraction, seed)
        for reference, (split, _) in zip(references, prepared):
            reference["split"] = split
        for split in ["train", "validation"]:
            with (output / f"{split}.jsonl").open("x") as stream:
                for label, row in prepared:
                    if label == split:
                        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        private = output / "private_references.jsonl"
        with private.open("x") as stream:
            private.chmod(0o600)
            for row in references:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        manifest = {"schema_version": "finance_opd.dataset.v1", "input_sha256": input_hashes,
                    "counts": dict(Counter(row["metadata"]["benchmark"] for _, row in prepared)),
                    "splits": dict(Counter(label for label, _ in prepared)),
                    "validation_fraction": validation_fraction, "seed": seed,
                    "split_scope": "identical question or identical indexed document set; no company/time generalization claim",
                    "scope": "adaptation of existing evaluation tasks; preserve untouched external tests for generalization",
                    "references_visible_to_model": False}
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        return manifest
    except Exception as error:
        (output / "FAILED.json").write_text(json.dumps({"error": str(error), "complete": False}, ensure_ascii=False) + "\n")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=ROOT.parent)
    parser.add_argument("--output", type=Path, default=ROOT / "data/current")
    parser.add_argument("--benchmarks", nargs="+", choices=BENCHMARKS, default=list(BENCHMARKS))
    parser.add_argument("--limit-per-benchmark", type=int)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(prepare(args.workspace, args.output, args.benchmarks,
                             limit=args.limit_per_benchmark, validation_fraction=args.validation_fraction,
                             seed=args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
