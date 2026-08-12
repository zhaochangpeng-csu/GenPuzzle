from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from task_registry import TASKS, parse_tasks


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch-evaluate generated images across benchmark tracks.")
    p.add_argument("--suite-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--tasks", default="all")
    p.add_argument("--run-name", required=True)
    p.add_argument("--runs-dir", type=Path, default=None)
    p.add_argument("--judge-model", default="gpt-5.5")
    p.add_argument("--reasoning-effort", choices=["none", "low", "medium", "high", "xhigh"], default="high")
    p.add_argument("--passes", type=int, default=1)
    p.add_argument("--limit-per-task", type=int, default=None)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--retry-delay", type=float, default=2.0)
    p.add_argument("--request-timeout", type=float, default=180.0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--stop-on-error", action="store_true")
    return p.parse_args()


def safe_model(name: str) -> str:
    return name.replace("/", "_")


def main() -> None:
    args = parse_args()
    if not os.getenv("OPENAI_API_KEY") and not args.dry_run:
        raise RuntimeError("OPENAI_API_KEY is not set")

    suite_root = args.suite_root.resolve()
    code_root = Path(__file__).resolve().parent
    runs_dir = (args.runs_dir or (suite_root / "runs")).resolve()
    root_run = runs_dir / args.run_name
    task_names = parse_tasks(args.tasks)
    judge_safe = safe_model(args.judge_model)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(code_root) + os.pathsep + env.get("PYTHONPATH", "")

    failures: list[str] = []
    for i, task_name in enumerate(task_names, 1):
        spec = TASKS[task_name]
        task_run = root_run / task_name
        if not task_run.exists():
            print(f"SKIP {task_name}: missing run directory {task_run}")
            continue
        dataset_root = (suite_root / spec.dataset_dir).resolve()
        limit_args = ["--limit", str(args.limit_per_task)] if args.limit_per_task is not None else []

        if spec.evaluator == "civil_service":
            script = code_root / "evaluators/civil_service.py"
            cmd = [sys.executable, str(script), "--dataset", str(dataset_root), "--run", str(task_run),
                   "--judge-model", args.judge_model, "--reasoning-effort", args.reasoning_effort,
                   "--passes", str(args.passes), "--workers", str(args.workers),
                   "--max-retries", str(args.max_retries), "--retry-delay", str(args.retry_delay),
                   "--request-timeout", str(args.request_timeout)] + limit_args
            if args.overwrite:
                cmd.append("--overwrite")
        elif spec.evaluator == "maze":
            script = code_root / "evaluators/maze.py"
            cmd = [sys.executable, str(script), "--dataset", str(dataset_root), "--run", str(task_run),
                   "--judge-model", args.judge_model, "--reasoning-effort", args.reasoning_effort,
                   "--passes", str(args.passes), "--workers", str(args.workers),
                   "--max-retries", str(args.max_retries), "--retry-delay", str(args.retry_delay),
                   "--request-timeout", str(args.request_timeout)] + limit_args
            if args.overwrite:
                cmd.append("--overwrite")
        elif spec.evaluator == "sudoku":
            script = code_root / "evaluators/sudoku.py"
            cmd = [sys.executable, str(script), "--dataset", str(dataset_root), "--run", str(task_run),
                   "--reader-model", args.judge_model, "--reasoning-effort", args.reasoning_effort,
                   "--passes", str(args.passes), "--workers", str(args.workers),
                   "--max-retries", str(args.max_retries), "--retry-delay", str(args.retry_delay),
                   "--request-timeout", str(args.request_timeout)] + limit_args
            if args.overwrite:
                cmd.append("--overwrite")
        elif spec.evaluator == "nonogram":
            script = code_root / "evaluators/nonogram.py"
            cmd = [sys.executable, str(script), "--dataset", str(dataset_root), "--run", str(task_run),
                   "--reader-model", args.judge_model, "--reasoning-effort", args.reasoning_effort,
                   "--passes", str(args.passes), "--workers", str(args.workers),
                   "--max-retries", str(args.max_retries), "--retry-delay", str(args.retry_delay),
                   "--request-timeout", str(args.request_timeout)] + limit_args
            if args.overwrite:
                cmd.append("--overwrite")
        elif spec.evaluator == "tangram":
            script = code_root / "evaluators/tangram.py"
            cmd = [sys.executable, str(script), "--dataset", str(dataset_root), "--run", str(task_run),
                   "--judge-model", args.judge_model, "--reasoning-effort", args.reasoning_effort,
                   "--passes", str(args.passes), "--workers", str(args.workers),
                   "--max-retries", str(args.max_retries), "--retry-delay", str(args.retry_delay),
                   "--request-timeout", str(args.request_timeout)] + limit_args
            if args.overwrite:
                cmd.append("--overwrite")
        elif spec.evaluator == "board_game":
            script = code_root / "evaluators/board_game.py"
            result = task_run / "evaluation" / f"board_game_{judge_safe}.jsonl"
            if args.overwrite and result.exists():
                result.unlink()
            result.parent.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, str(script), "--dataset", str(dataset_root / spec.data_file),
                   "--dataset-root", str(dataset_root), "--outputs-dir", str(task_run / "images"),
                   "--result-file", str(result), "--judge-model", args.judge_model,
                   "--max-retries", str(args.max_retries), "--retry-delay", str(args.retry_delay),
                   "--request-timeout", str(args.request_timeout)] + limit_args
            if args.overwrite:
                cmd.append("--overwrite")
        elif spec.evaluator == "matchsticks":
            script = code_root / "evaluators/matchsticks.py"
            result = task_run / "evaluation" / f"matchsticks_{judge_safe}.jsonl"
            if result.exists() and not args.overwrite:
                print(f"SKIP {task_name}: result exists {result}")
                continue
            if args.overwrite and result.exists():
                result.unlink()
            result.parent.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, str(script), "--dataset", str(dataset_root / spec.data_file),
                   "--root", str(dataset_root), "--generated-dir", str(task_run / "images"),
                   "--output", str(result), "--judge-model", args.judge_model,
                   "--reasoning-effort", args.reasoning_effort] + limit_args
        elif spec.evaluator == "orthographic":
            script = code_root / "evaluators/orthographic.py"
            result = task_run / "evaluation" / f"orthographic_{judge_safe}.jsonl"
            if result.exists() and not args.overwrite:
                print(f"SKIP {task_name}: result exists {result}")
                continue
            if args.overwrite and result.exists():
                result.unlink()
            result.parent.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, str(script), "--dataset", str(dataset_root / spec.data_file),
                   "--dataset-root", str(dataset_root), "--outputs-dir", str(task_run / "images"),
                   "--result-file", str(result), "--judge-model", args.judge_model,
                   "--reasoning-effort", args.reasoning_effort] + limit_args
        elif spec.evaluator == "mathematical_proof":
            script = code_root / "evaluators/mathematical_proof.py"
            result = task_run / "evaluation" / f"mathematical_proof_{judge_safe}.jsonl"
            if args.overwrite and result.exists():
                result.unlink()
            result.parent.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, str(script), "--dataset", str(dataset_root / spec.data_file),
                   "--dataset-root", str(dataset_root), "--generated-dir", str(task_run / "images"),
                   "--output", str(result), "--judge-model", args.judge_model,
                   "--reasoning-effort", args.reasoning_effort,
                   "--max-retries", str(args.max_retries), "--retry-delay", str(args.retry_delay),
                   "--request-timeout", str(args.request_timeout)] + limit_args
            if args.overwrite:
                cmd.append("--overwrite")
        else:
            raise ValueError(f"Unknown evaluator: {spec.evaluator}")

        print(f"\n=== [{i}/{len(task_names)}] Evaluate {task_name} ===")
        print(" ".join(cmd))
        if args.dry_run:
            continue
        proc = subprocess.run(cmd, cwd=str(code_root), env=env)
        if proc.returncode != 0:
            failures.append(task_name)
            print(f"ERROR {task_name}: exit code {proc.returncode}")
            if args.stop_on_error:
                raise SystemExit(proc.returncode)

    if failures:
        print("\nEvaluation completed with failures: " + ", ".join(failures))
        raise SystemExit(1)
    print("\nEvaluation finished.")


if __name__ == "__main__":
    main()
