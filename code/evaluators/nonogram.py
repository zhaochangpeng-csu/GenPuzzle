from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath
sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

import argparse
import os
import statistics
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from common import (
    append_jsonl,
    call_with_retry,
    image_data_url,
    latest_records,
    load_jsonl,
    openai_pydantic_call,
    select_items,
    write_json,
)


class NonogramTranscription(BaseModel):
    grid: list[list[int | None]]
    confidence: float = Field(ge=0.0, le=1.0)
    unreadable_cells: list[str] = Field(default_factory=list)
    note: str = ""


TRANSCRIBE_SYSTEM_PROMPT = """
You are an exact transcriber of Nonogram candidate answer images, not a solver.

Your only task is to read the actual state of the grid in the candidate image and output a two-dimensional matrix:
- 1: the cell is clearly filled black;
- 0: the cell is clearly white/blank;
- null: the cell is occluded, blurry, misaligned at the boundary, or cannot be reliably recognized.

Hard rules:
- Do not solve the puzzle based on the numerical clues, do not guess, and do not automatically correct errors.
- Do not refer to what you think the “correct answer should be.”
- Ignore any text, titles, decorations, or explanations outside the grid.
- The original problem image may only be used to help locate the grid; the final transcription must reflect the actually visible cell states in the candidate image.
- The number of rows and columns in grid must match the user-specified size.
- Return only the structured transcription result. Do not output a long chain of thought.
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Transcribe and deterministically evaluate Nonogram generations.")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--reader-model", default="gpt-5.5")
    p.add_argument("--reasoning-effort", choices=["none", "low", "medium", "high", "xhigh"], default="medium")
    p.add_argument("--passes", type=int, default=1)
    p.add_argument("--reader-mode", choices=["auto", "cv", "mlm"], default="auto")
    p.add_argument("--confidence-threshold", type=float, default=0.75)
    p.add_argument("--start-id", default=None)
    p.add_argument("--end-id", default=None)
    p.add_argument("--ids", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--retry-delay", type=float, default=2.0)
    p.add_argument("--request-timeout", type=float, default=180.0)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def normalize_grid(raw: list[list[int | None]], size: int) -> tuple[list[list[int | None]], bool]:
    malformed = len(raw) != size or any(len(row) != size for row in raw)
    out: list[list[int | None]] = [[None] * size for _ in range(size)]
    for r in range(min(size, len(raw))):
        row = raw[r]
        for c in range(min(size, len(row))):
            value = row[c]
            out[r][c] = value if value in (0, 1) else None
    return out, malformed


def consensus_grid(transcriptions: list[NonogramTranscription], size: int) -> tuple[list[list[int | None]], int]:
    grids = [normalize_grid(t.grid, size)[0] for t in transcriptions]
    threshold = len(grids) // 2 + 1
    out: list[list[int | None]] = [[None] * size for _ in range(size)]
    disagreements = 0
    for r in range(size):
        for c in range(size):
            values = [grid[r][c] for grid in grids]
            counts = Counter(v for v in values if v is not None)
            if counts:
                value, count = counts.most_common(1)[0]
                if count >= threshold:
                    out[r][c] = value
            if len(set(values)) > 1:
                disagreements += 1
    return out, disagreements


def line_clues(values: list[int]) -> list[int]:
    clues: list[int] = []
    run = 0
    for value in values:
        if value == 1:
            run += 1
        elif run:
            clues.append(run)
            run = 0
    if run:
        clues.append(run)
    return clues


def score_candidate(
    candidate: list[list[int | None]],
    solution: list[list[int]],
    row_clues: list[list[int]],
    col_clues: list[list[int]],
) -> dict[str, Any]:
    size = len(solution)
    total = size * size
    readable = sum(candidate[r][c] in (0, 1) for r in range(size) for c in range(size))
    correct = sum(candidate[r][c] == solution[r][c] for r in range(size) for c in range(size))
    completion_rate = readable / total
    cell_accuracy = correct / total

    valid_rows = 0
    for r in range(size):
        if all(candidate[r][c] in (0, 1) for c in range(size)):
            vals = [int(candidate[r][c]) for c in range(size)]
            if line_clues(vals) == row_clues[r]:
                valid_rows += 1

    valid_cols = 0
    for c in range(size):
        col = [candidate[r][c] for r in range(size)]
        if all(v in (0, 1) for v in col):
            if line_clues([int(v) for v in col]) == col_clues[c]:
                valid_cols += 1

    is_valid = readable == total and valid_rows == size and valid_cols == size
    exact_match = readable == total and all(candidate[r][c] == solution[r][c] for r in range(size) for c in range(size))

    # The dataset is unique-solution by construction, so a valid complete grid should equal the reference solution.
    if is_valid:
        score = 3
    elif completion_rate >= 0.90 and cell_accuracy >= 0.90:
        score = 2
    elif completion_rate >= 0.50 and cell_accuracy >= 0.50:
        score = 1
    else:
        score = 0

    return {
        "score": score,
        "normalized_score": round(score / 3 * 100, 2),
        "is_valid_solution": is_valid,
        "exact_match": exact_match,
        "cell_accuracy": round(cell_accuracy, 6),
        "completion_rate": round(completion_rate, 6),
        "valid_rows": valid_rows,
        "valid_columns": valid_cols,
        "row_constraint_accuracy": round(valid_rows / size, 6),
        "column_constraint_accuracy": round(valid_cols / size, 6),
    }


def cv_transcribe(candidate_path: Path, meta_row: dict[str, Any]) -> tuple[list[list[int | None]], float, int]:
    size = int(meta_row["size"])
    render = meta_row["render"]
    canvas_size = int(render["canvas_size"])
    x0, y0, x1, y1 = [int(v) for v in render["grid_bbox"]]

    image = Image.open(candidate_path).convert("RGB").resize((canvas_size, canvas_size), Image.Resampling.BILINEAR)
    arr = np.asarray(image).astype(np.float32)
    # Perceptual luminance; red paths/text are dark enough to be treated as non-white.
    gray = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
    cell_w = (x1 - x0) / size
    cell_h = (y1 - y0) / size
    grid: list[list[int | None]] = [[None] * size for _ in range(size)]
    uncertain = 0
    margins = 0.22

    for r in range(size):
        for c in range(size):
            xa = int(round(x0 + (c + margins) * cell_w))
            xb = int(round(x0 + (c + 1 - margins) * cell_w))
            ya = int(round(y0 + (r + margins) * cell_h))
            yb = int(round(y0 + (r + 1 - margins) * cell_h))
            patch = gray[max(0, ya):min(canvas_size, yb), max(0, xa):min(canvas_size, xb)]
            if patch.size == 0:
                uncertain += 1
                continue
            dark_ratio = float(np.mean(patch < 105))
            medium_ratio = float(np.mean(patch < 190))
            if dark_ratio >= 0.42 or medium_ratio >= 0.62:
                grid[r][c] = 1
            elif dark_ratio <= 0.07 and medium_ratio <= 0.16:
                grid[r][c] = 0
            else:
                uncertain += 1

    confidence = max(0.0, 1.0 - uncertain / (size * size))
    return grid, confidence, uncertain


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if r.get("status") == "success"]
    failed = [r for r in rows if r.get("status") != "success"]
    if not ok:
        return {"evaluated_successfully": 0, "evaluation_failures": len(failed)}
    scores = [int(r["score"]) for r in ok]
    dist = Counter(scores)
    by_difficulty: dict[str, dict[str, Any]] = {}
    for difficulty in ("easy", "medium", "hard"):
        subset = [r for r in ok if r.get("difficulty") == difficulty]
        if subset:
            by_difficulty[difficulty] = {
                "count": len(subset),
                "exact_solve_rate": round(statistics.mean(bool(r["is_valid_solution"]) for r in subset), 6),
                "normalized_score": round(statistics.mean(int(r["score"]) for r in subset) / 3 * 100, 4),
                "mean_cell_accuracy": round(statistics.mean(float(r["cell_accuracy"]) for r in subset), 6),
            }
    return {
        "evaluated_successfully": len(ok),
        "evaluation_failures": len(failed),
        "exact_solve_rate": round(statistics.mean(bool(r["is_valid_solution"]) for r in ok), 6),
        "mean_cell_accuracy": round(statistics.mean(float(r["cell_accuracy"]) for r in ok), 6),
        "mean_completion_rate": round(statistics.mean(float(r["completion_rate"]) for r in ok), 6),
        "mean_row_constraint_accuracy": round(statistics.mean(float(r["row_constraint_accuracy"]) for r in ok), 6),
        "mean_column_constraint_accuracy": round(statistics.mean(float(r["column_constraint_accuracy"]) for r in ok), 6),
        "mean_tier": round(statistics.mean(scores), 4),
        "normalized_score": round(statistics.mean(scores) / 3 * 100, 4),
        "score_distribution": {str(k): dist.get(k, 0) for k in [0, 1, 2, 3]},
        "human_review_count": sum(bool(r.get("needs_human_review")) for r in ok),
        "by_difficulty": by_difficulty,
    }


def main() -> None:
    args = parse_args()
    if args.passes < 1 or args.workers < 1:
        raise ValueError("--passes and --workers must be >= 1")
    if args.reader_mode in {"auto", "mlm"} and not os.getenv("OPENAI_API_KEY"):
        if args.reader_mode == "mlm":
            raise RuntimeError("OPENAI_API_KEY is not set")
        print("WARN OPENAI_API_KEY not set: auto mode will use CV only")

    client = None
    if args.reader_mode in {"auto", "mlm"} and os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install dependencies: pip install -r requirements.txt") from exc
        client = OpenAI(
            **({"base_url": os.environ["OPENAI_BASE_URL"]} if os.getenv("OPENAI_BASE_URL") else {}),
            timeout=args.request_timeout,
        )

    dataset_root = args.dataset.resolve()
    run_dir = args.run.resolve()
    items = load_jsonl(dataset_root / "data.jsonl")
    meta = {str(row["id"]): row for row in load_jsonl(dataset_root / "eval_meta.jsonl")}
    requested_ids = set(args.ids.split(",")) if args.ids else None
    items = select_items(items, args.start_id, args.end_id, requested_ids, args.limit)
    generations = latest_records(run_dir / "records.jsonl")

    eval_dir = run_dir / "evaluation" / f"nonogram_{args.reader_model.replace('/', '_')}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    results_path = eval_dir / "results.jsonl"
    review_path = eval_dir / "human_review.jsonl"
    if args.overwrite:
        for path in (results_path, review_path):
            if path.exists():
                path.unlink()
    done = set() if args.overwrite else {
        str(r["id"]) for r in load_jsonl(results_path) if r.get("status") == "success"
    } if results_path.exists() else set()
    pending = [item for item in items if str(item["id"]) not in done]

    write_json(eval_dir / "config.json", {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_root),
        "run": str(run_dir),
        "reader_model": args.reader_model,
        "reader_mode": args.reader_mode,
        "reasoning_effort": args.reasoning_effort,
        "passes": args.passes,
        "confidence_threshold": args.confidence_threshold,
        "selected_count": len(items),
    })

    write_lock = threading.Lock()

    def transcribe_once(question_path: Path, candidate_path: Path, size: int) -> NonogramTranscription:
        assert client is not None
        return openai_pydantic_call(
            client,
            model=args.reader_model,
            reasoning_effort=args.reasoning_effort,
            input_messages=[
                {"role": "system", "content": TRANSCRIBE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": f"This is a {size}x{size} Nonogram. The first image is the original problem and is used only to locate the grid; the second image is the candidate answer. Please transcribe the actual black/white state of every cell in the second image."},
                        {"type": "input_image", "image_url": image_data_url(question_path), "detail": "original"},
                        {"type": "input_image", "image_url": image_data_url(candidate_path), "detail": "original"},
                    ],
                },
            ],
            output_model=NonogramTranscription,
            schema_name="nonogram_transcription",
        )

    def evaluate_one(item: dict[str, Any]) -> dict[str, Any]:
        item_id = str(item["id"])
        started = time.perf_counter()
        generation = generations.get(item_id, {})
        candidate_path = run_dir / "images" / f"{item_id}.png"
        question_path = dataset_root / str(item["image"])
        if generation.get("status") != "success" or not candidate_path.is_file():
            return {
                "id": item_id,
                "status": "error",
                "error": "missing successful generation record or candidate image",
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
        if item_id not in meta:
            return {"id": item_id, "status": "error", "error": "missing eval_meta row"}

        try:
            meta_row = meta[item_id]
            size = int(meta_row["size"])
            reader_source = "cv"
            disagreements = 0
            malformed = False
            mlm_passes: list[NonogramTranscription] = []

            if args.reader_mode in {"auto", "cv"}:
                grid, confidence, uncertain = cv_transcribe(candidate_path, meta_row)
            else:
                grid, confidence, uncertain = [[None] * size for _ in range(size)], 0.0, size * size

            use_mlm = args.reader_mode == "mlm" or (
                args.reader_mode == "auto"
                and client is not None
                and (confidence < args.confidence_threshold or uncertain > max(2, int(size * size * 0.04)))
            )
            if use_mlm:
                reader_source = "mlm"
                mlm_passes = [
                    call_with_retry(
                        lambda: transcribe_once(question_path, candidate_path, size),
                        max_retries=args.max_retries,
                        base_delay=args.retry_delay,
                    )
                    for _ in range(args.passes)
                ]
                if args.passes == 1:
                    grid, malformed = normalize_grid(mlm_passes[0].grid, size)
                    disagreements = 0
                else:
                    grid, disagreements = consensus_grid(mlm_passes, size)
                    malformed = any(normalize_grid(t.grid, size)[1] for t in mlm_passes)
                confidence = statistics.mean(t.confidence for t in mlm_passes)
                uncertain = sum(cell is None for row in grid for cell in row)

            metrics = score_candidate(
                grid,
                [[int(v) for v in row] for row in meta_row["solution"]],
                [[int(v) for v in row] for row in meta_row["row_clues"]],
                [[int(v) for v in row] for row in meta_row["column_clues"]],
            )
            needs_review = malformed or confidence < args.confidence_threshold or disagreements > 0 or uncertain > 0
            reasons: list[str] = []
            if malformed:
                reasons.append("reader returned malformed grid shape")
            if confidence < args.confidence_threshold:
                reasons.append(f"reader confidence {confidence:.2f} below threshold")
            if disagreements > 0:
                reasons.append(f"reader passes disagreed on {disagreements} cells")
            if uncertain > 0:
                reasons.append(f"{uncertain} cells remain unreadable")

            return {
                "id": item_id,
                "status": "success",
                "difficulty": meta_row["difficulty"],
                "size": size,
                "reader_source": reader_source,
                "reader_model": args.reader_model if reader_source == "mlm" else None,
                "transcribed_grid": grid,
                "transcription_confidence": round(float(confidence), 4),
                "unreadable_cells": uncertain,
                "transcription_disagreements": disagreements,
                **metrics,
                "needs_human_review": needs_review,
                "review_reason": "; ".join(reasons) or None,
                "reader_passes": [t.model_dump() for t in mlm_passes],
                "latency_seconds": round(time.perf_counter() - started, 3),
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as exc:
            return {
                "id": item_id,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "latency_seconds": round(time.perf_counter() - started, 3),
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }

    total = len(pending)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {executor.submit(evaluate_one, item): item for item in pending}
        completed = 0
        for future in as_completed(future_map):
            row = future.result()
            with write_lock:
                append_jsonl(results_path, row)
                if row.get("needs_human_review"):
                    append_jsonl(review_path, row)
            completed += 1
            print(f"[{completed}/{total}] {row['id']} {row['status']} score={row.get('score')} reader={row.get('reader_source')}")

    all_rows = load_jsonl(results_path) if results_path.exists() else []
    write_json(eval_dir / "summary.json", summarize(all_rows))
    print(f"Evaluation complete: {eval_dir}")


if __name__ == "__main__":
    main()
