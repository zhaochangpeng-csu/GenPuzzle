from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from task_registry import TASKS, dataset_root, load_raw_rows, parse_tasks


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate all benchmark datasets and assets locally.")
    p.add_argument("--suite-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--tasks", default="all")
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def check_asset(root: Path, rel: str) -> bool:
    p = (root / rel).resolve()
    try:
        p.relative_to(root.resolve())
    except ValueError:
        return False
    return p.is_file() and p.stat().st_size > 0


def validate_task(suite_root: Path, task_name: str) -> dict[str, Any]:
    spec = TASKS[task_name]
    root = dataset_root(suite_root, spec)
    rows = load_raw_rows(suite_root, spec)
    ids = [str(r[spec.id_field]) for r in rows]
    missing_inputs: list[str] = []
    missing_refs: list[str] = []

    for row in rows:
        item_id = str(row[spec.id_field])
        if spec.loader_kind in {"civil_service", "maze", "sudoku", "nonogram", "tangram"}:
            if not check_asset(root, str(row["image"])):
                missing_inputs.append(item_id)
            if not check_asset(root, str(row["answer"])):
                missing_refs.append(item_id)
        else:
            for im in row.get("input_images") or []:
                if not check_asset(root, str(im["path"])):
                    missing_inputs.append(item_id)
            for im in row.get("reference_images") or []:
                if not check_asset(root, str(im["path"])):
                    missing_refs.append(item_id)

    result = {
        "task": task_name,
        "expected_count": spec.expected_count,
        "actual_count": len(rows),
        "unique_ids": len(set(ids)),
        "duplicate_ids": sorted({x for x in ids if ids.count(x) > 1}),
        "missing_input_assets": sorted(set(missing_inputs)),
        "missing_reference_assets": sorted(set(missing_refs)),
    }
    result["ok"] = (
        result["actual_count"] == result["expected_count"]
        and result["unique_ids"] == result["actual_count"]
        and not result["missing_input_assets"]
        and not result["missing_reference_assets"]
    )
    return result


def main() -> None:
    args = parse_args()
    suite_root = args.suite_root.resolve()
    task_names = parse_tasks(args.tasks)
    results = [validate_task(suite_root, name) for name in task_names]
    summary = {
        "suite_root": str(suite_root),
        "tasks": results,
        "total_expected": sum(x["expected_count"] for x in results),
        "total_actual": sum(x["actual_count"] for x in results),
        "ok": all(x["ok"] for x in results),
    }
    for row in results:
        print(f"{'OK' if row['ok'] else 'FAIL':4} {row['task']:28} {row['actual_count']:4}/{row['expected_count']}")
        if row["missing_input_assets"]:
            print(f"  missing inputs: {row['missing_input_assets'][:10]}")
        if row["missing_reference_assets"]:
            print(f"  missing refs: {row['missing_reference_assets'][:10]}")
    print(f"TOTAL {summary['total_actual']}/{summary['total_expected']}  ok={summary['ok']}")
    out = args.output or (suite_root / "validation_summary.json")
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not summary["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
