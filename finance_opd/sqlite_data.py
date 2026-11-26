"""Prepare company-disjoint OPD tasks directly from a read-only DDR database."""
import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from .config import ROOT


def companies(path):
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.execute("PRAGMA query_only=ON")
        rows = db.execute("SELECT cik, name FROM companies ORDER BY cik").fetchall()
        filings = {str(int(row[0])) for row in db.execute("SELECT DISTINCT cik FROM filings")}
        facts = {str(int(row[0])) for row in db.execute("SELECT DISTINCT cik FROM financial_facts")}
    result = {str(int(cik)): name for cik, name in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate CIK after numeric normalization")
    return result, filings, facts


def prepare(database, output, validation_count=100, seed=42, eval_database=None):
    database, output = database.resolve(), output.resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        raise ValueError("use a new output directory inside cv-opd")
    entities, filings, facts = companies(database)
    eligible = sorted(entities.keys() & filings & facts)
    if not 0 < validation_count < len(eligible):
        raise ValueError("validation count must leave nonempty training and validation sets")
    overlap = []
    if eval_database is not None:
        eval_entities, _, _ = companies(eval_database)
        overlap = sorted(set(eligible) & eval_entities.keys())
        if overlap:
            raise ValueError(f"{len(overlap)} companies overlap the external evaluation database")
    order = sorted(eligible, key=lambda cik: hashlib.sha256(f"{seed}:{cik}".encode()).hexdigest())
    validation = set(order[:validation_count])
    output.mkdir(parents=True)
    counts = {}
    for split in ["train", "validation"]:
        rows = []
        for cik in eligible:
            if (cik in validation) != (split == "validation"):
                continue
            rows.append({"task": f"Analyze company with CIK {cik}", "metadata": {
                "id": f"ddr_10k:{cik}__entity", "benchmark": "ddr_10k",
                "native_case_id": f"{cik}__entity", "entity_id": cik,
                "source": {"db_path": str(database)}, "split_group": f"company:{cik}"}})
        (output / f"{split}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
        counts[split] = len(rows)
    stat = database.stat()
    manifest = {"database": str(database), "database_size": stat.st_size, "database_mtime_ns": stat.st_mtime_ns,
                "companies_total": len(entities), "eligible_companies": len(eligible), "counts": counts,
                "seed": seed, "split_unit": "company_cik", "train_validation_company_overlap": 0,
                "external_eval_database": str(eval_database.resolve()) if eval_database else None,
                "external_eval_company_overlap": len(overlap) if eval_database else None,
                "database_access": "read-only; original database path used by tools",
                "scope": "task-company split; shared tool database still contains held-out companies; no access isolation claim",
                "input_kind": "task-only; no answers or reference trajectories",
                "files_sha256": {split: hashlib.sha256((output / f"{split}.jsonl").read_bytes()).hexdigest() for split in counts}}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-database", type=Path)
    parser.add_argument("--validation-count", type=int, default=100)
    args = parser.parse_args()
    print(json.dumps(prepare(args.database, args.output, args.validation_count, eval_database=args.eval_database), indent=2))


if __name__ == "__main__":
    main()
