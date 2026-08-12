from __future__ import annotations

import argparse
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import append_jsonl, call_with_retry, successful_generation_ids, write_json
from prompts import EXPLANATION_SUFFIX
from providers import create_generator
from task_registry import TASKS, load_items, parse_tasks


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch-generate answer images across all benchmark tracks.")
    p.add_argument("--suite-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--tasks", default="all", help="Comma-separated task names or all.")
    p.add_argument("--provider", choices=["openai", "openai-generation", "google", "ark"], required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--run-name", required=True)
    p.add_argument("--runs-dir", type=Path, default=None)
    p.add_argument("--explanation-mode", choices=["optional", "none"], default="optional")
    p.add_argument("--limit-per-task", type=int, default=None)
    p.add_argument("--ids", default=None, help="Optional comma-separated IDs; applied within every selected task.")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--retry-delay", type=float, default=2.0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Validate selection and print prompts without API calls.")

    p.add_argument("--openai-size", default="1024x1024")
    p.add_argument("--openai-quality", choices=["auto", "low", "medium", "high"], default="high")
    p.add_argument("--openai-input-fidelity", choices=["low", "high"], default="high")
    p.add_argument("--gemini-aspect-ratio", default="1:1")
    p.add_argument("--gemini-image-size", default="1K")
    p.add_argument("--ark-size", default="2K")
    p.add_argument("--ark-base-url", default=None)
    p.add_argument("--ark-watermark", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    suite_root = args.suite_root.resolve()
    task_names = parse_tasks(args.tasks)
    runs_dir = (args.runs_dir or (suite_root / "runs")).resolve()
    root_run = runs_dir / args.run_name
    root_run.mkdir(parents=True, exist_ok=True)
    requested_ids = {x.strip() for x in args.ids.split(",") if x.strip()} if args.ids else None

    selected_counts: dict[str, int] = {}
    for task_name in task_names:
        spec = TASKS[task_name]
        items = load_items(suite_root, spec)
        if requested_ids is not None:
            items = [x for x in items if x["id"] in requested_ids]
        if args.limit_per_task is not None:
            items = items[: args.limit_per_task]
        selected_counts[task_name] = len(items)

    if args.dry_run:
        print(f"Suite: {suite_root}")
        for task_name in task_names:
            spec = TASKS[task_name]
            items = load_items(suite_root, spec)
            if requested_ids is not None:
                items = [x for x in items if x["id"] in requested_ids]
            if args.limit_per_task is not None:
                items = items[: args.limit_per_task]
            print(f"{task_name}: {len(items)} item(s)")
            if items:
                print(f"  first id: {items[0]['id']}")
                print(f"  inputs: {items[0]['input_images']}")
                print(f"  prompt: {items[0]['prompt'][:240]}")
        return

    generator = create_generator(args)
    write_json(root_run / "run_manifest.json", {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "suite_root": str(suite_root),
        "provider": generator.provider,
        "model": generator.model,
        "run_name": args.run_name,
        "tasks": task_names,
        "selected_counts": selected_counts,
        "explanation_mode": args.explanation_mode,
    })

    for task_index, task_name in enumerate(task_names, 1):
        spec = TASKS[task_name]
        items = load_items(suite_root, spec)
        if requested_ids is not None:
            items = [x for x in items if x["id"] in requested_ids]
        if args.limit_per_task is not None:
            items = items[: args.limit_per_task]

        task_run = root_run / task_name
        image_dir = task_run / "images"
        records_path = task_run / "records.jsonl"
        image_dir.mkdir(parents=True, exist_ok=True)
        if args.overwrite and records_path.exists():
            records_path.unlink()

        done = set() if args.overwrite else successful_generation_ids(records_path, image_dir)
        pending = [x for x in items if x["id"] not in done]
        write_json(task_run / "config.json", {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "task": task_name,
            "display_name": spec.display_name,
            "dataset": str((suite_root / spec.dataset_dir).resolve()),
            "provider": generator.provider,
            "model": generator.model,
            "selected_count": len(items),
            "pending_count_at_start": len(pending),
            "workers": args.workers,
        })

        print(f"\n=== [{task_index}/{len(task_names)}] {task_name}: {len(pending)} pending / {len(items)} selected ===")
        if not pending:
            continue
        lock = threading.Lock()

        def run_one(item: dict[str, Any]) -> dict[str, Any]:
            started = time.perf_counter()
            item_id = item["id"]
            input_paths = [(item["dataset_root"] / p).resolve() for p in item["input_images"]]
            for path in input_paths:
                if not path.is_file():
                    raise FileNotFoundError(path)
            prompt = item["prompt"]
            if args.explanation_mode == "optional":
                prompt = prompt.rstrip() + EXPLANATION_SUFFIX
            try:
                image_bytes, explanation, provider_meta = call_with_retry(
                    lambda: generator.generate(input_paths, prompt),
                    max_retries=args.max_retries,
                    base_delay=args.retry_delay,
                )
                (image_dir / f"{item_id}.png").write_bytes(image_bytes)
                status, error = "success", None
            except Exception as exc:
                explanation, provider_meta = None, {}
                status, error = "error", f"{type(exc).__name__}: {exc}"
            return {
                "id": item_id,
                "task": task_name,
                "status": status,
                "provider": generator.provider,
                "model": generator.model,
                "prompt": prompt,
                "input_images": item["input_images"],
                "output_image": f"images/{item_id}.png" if status == "success" else None,
                "explanation": explanation,
                "provider_meta": provider_meta,
                "error": error,
                "latency_seconds": round(time.perf_counter() - started, 3),
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_one, item): item for item in pending}
            completed = 0
            for future in as_completed(futures):
                try:
                    row = future.result()
                except Exception as exc:
                    item = futures[future]
                    row = {
                        "id": item["id"], "task": task_name, "status": "error",
                        "provider": generator.provider, "model": generator.model,
                        "prompt": item["prompt"], "input_images": item["input_images"],
                        "output_image": None, "explanation": None, "provider_meta": {},
                        "error": f"{type(exc).__name__}: {exc}", "latency_seconds": None,
                        "finished_at": datetime.now(timezone.utc).isoformat(),
                    }
                with lock:
                    append_jsonl(records_path, row)
                completed += 1
                print(f"[{completed}/{len(pending)}] {task_name}/{row['id']} {row['status']}")

    print(f"\nGeneration finished: {root_run}")


if __name__ == "__main__":
    main()
