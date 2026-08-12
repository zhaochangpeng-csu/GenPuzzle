from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from task_registry import TASKS, parse_tasks


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate leaderboard and radar chart from one or more benchmark runs.")
    p.add_argument("--suite-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--runs", required=True, help="Comma-separated run names.")
    p.add_argument("--runs-dir", type=Path, default=None)
    p.add_argument("--tasks", default="all")
    p.add_argument(
        "--judge-model",
        default="gpt-5.5",
        help="Judge model name, or comma-separated judge model names aligned with --runs.",
    )
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--count-errors-as-zero", action="store_true", default=True)
    return p.parse_args()


def safe_model_variants(judge_model: str) -> list[str]:
    safe = judge_model.replace("/", "_")
    variants = [safe]
    if not safe.startswith("openai_"):
        variants.append(f"openai_{safe}")
    if safe.startswith("openai_"):
        variants.append(safe.removeprefix("openai_"))
    out: list[str] = []
    for value in variants:
        if value not in out:
            out.append(value)
    return out


def result_path_for_safe(root_run: Path, task_name: str, safe: str) -> Path:
    task_run = root_run / task_name
    evaluator = TASKS[task_name].evaluator
    if evaluator == "civil_service":
        return task_run / "evaluation" / safe / "results.jsonl"
    if evaluator == "maze":
        return task_run / "evaluation" / f"maze_{safe}" / "results.jsonl"
    if evaluator == "sudoku":
        return task_run / "evaluation" / f"sudoku_{safe}" / "results.jsonl"
    if evaluator == "nonogram":
        return task_run / "evaluation" / f"nonogram_{safe}" / "results.jsonl"
    if evaluator == "tangram":
        return task_run / "evaluation" / f"tangram_{safe}" / "results.jsonl"
    if evaluator == "board_game":
        return task_run / "evaluation" / f"board_game_{safe}.jsonl"
    if evaluator == "matchsticks":
        return task_run / "evaluation" / f"matchsticks_{safe}.jsonl"
    if evaluator == "orthographic":
        return task_run / "evaluation" / f"orthographic_{safe}.jsonl"
    if evaluator == "mathematical_proof":
        return task_run / "evaluation" / f"mathematical_proof_{safe}.jsonl"
    raise ValueError(evaluator)


def result_path(root_run: Path, task_name: str, judge_model: str) -> Path:
    candidates = [result_path_for_safe(root_run, task_name, safe) for safe in safe_model_variants(judge_model)]
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


def infer_model_label(root_run: Path) -> str:
    manifest = root_run / "run_manifest.json"
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            return str(data.get("model") or root_run.name)
        except Exception:
            pass
    return root_run.name


def main() -> None:
    args = parse_args()
    suite_root = args.suite_root.resolve()
    runs_dir = (args.runs_dir or (suite_root / "runs")).resolve()
    run_names = [x.strip() for x in args.runs.split(",") if x.strip()]
    judge_models = [x.strip() for x in args.judge_model.split(",") if x.strip()]
    if len(judge_models) == 1:
        judge_models = judge_models * len(run_names)
    elif len(judge_models) != len(run_names):
        raise ValueError(
            "--judge-model must be a single model name or a comma-separated list "
            "with the same length as --runs"
        )
    task_names = parse_tasks(args.tasks)
    inputs: list[str] = []

    for run_name, judge_model in zip(run_names, judge_models):
        root_run = runs_dir / run_name
        model_label = infer_model_label(root_run)
        for task_name in task_names:
            path = result_path(root_run, task_name, judge_model)
            if path.is_file():
                inputs.append(f"{model_label}:{TASKS[task_name].category}:{path}")
            else:
                print(f"WARN missing result: {path}")

    if not inputs:
        raise RuntimeError("No evaluation result files found")
    out_dir = args.out_dir or (runs_dir / ("report_" + "_vs_".join(run_names)))
    script = Path(__file__).resolve().parent / "report/report_generator.py"
    cmd = [sys.executable, str(script), "--inputs", *inputs, "--out-dir", str(out_dir)]
    if args.count_errors_as_zero:
        cmd.append("--count-errors-as-zero")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Report written to: {out_dir}")


if __name__ == "__main__":
    main()
