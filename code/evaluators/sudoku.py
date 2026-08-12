from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath
sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

import argparse
import itertools
import os
import statistics
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


class SudokuTranscription(BaseModel):
    grid: list[list[int | None]]
    confidence: float = Field(ge=0.0, le=1.0)
    unreadable_cells: list[str] = Field(default_factory=list)
    note: str = ""


TRANSCRIBE_SYSTEM_PROMPT = """You are an exact transcriber for 4x4 Sudoku answer images, not a puzzle solver.

Your only task is to read the actual visible contents of the 4x4 grid in the candidate image and transcribe them as four rows and four columns.

Hard rules:
- Each cell may contain only 1, 2, 3, 4, or null.
- If a cell is blank, occluded, blurry, outside the grid, or not reliably recognizable, output null.
- Do not infer, complete, or correct numbers based on Sudoku rules.
- Do not change the transcription because you know what a valid Sudoku should be.
- Ignore digit color; read both black and red digits as their actual values.
- Ignore text, explanations, decorations, paper backgrounds, or photo backgrounds outside the grid.
- If the candidate image redraws the grid or changes the style, still transcribe only the actual visible numbers inside the candidate 4x4 grid; do not repair it.
- The grid should be four rows whenever possible, with four elements in each row.

Return only the structured transcription result. Do not output a long chain of thought."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe and deterministically evaluate 4x4 Sudoku generations.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reader-model", default="gpt-5.5")
    parser.add_argument("--reasoning-effort", choices=["none", "low", "medium", "high", "xhigh"], default="medium")
    parser.add_argument("--passes", type=int, default=1, help="Use 1 for normal runs; 3 gives majority-cell consensus.")
    parser.add_argument("--confidence-threshold", type=float, default=0.75)
    parser.add_argument("--start-id", default=None)
    parser.add_argument("--end-id", default=None)
    parser.add_argument("--ids", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def enumerate_solutions() -> list[tuple[tuple[int, ...], ...]]:
    perms = list(itertools.permutations([1, 2, 3, 4]))
    solutions: list[tuple[tuple[int, ...], ...]] = []
    for rows in itertools.product(perms, repeat=4):
        if any(len({rows[r][c] for r in range(4)}) != 4 for c in range(4)):
            continue
        ok = all(
            len({rows[r][c] for r in range(br, br + 2) for c in range(bc, bc + 2)}) == 4
            for br in (0, 2)
            for bc in (0, 2)
        )
        if ok:
            solutions.append(rows)
    return solutions


VALID_SOLUTIONS = enumerate_solutions()


def parse_grid_string(value: str) -> list[list[int]]:
    if len(value) != 16 or any(ch not in "01234" for ch in value):
        raise ValueError(f"Expected 16 digits 0-4, got: {value!r}")
    nums = [int(ch) for ch in value]
    return [nums[i:i + 4] for i in range(0, 16, 4)]


def normalize_grid(raw: list[list[int | None]]) -> tuple[list[list[int | None]], bool]:
    malformed = len(raw) != 4 or any(len(row) != 4 for row in raw)
    out: list[list[int | None]] = [[None] * 4 for _ in range(4)]
    for r in range(min(4, len(raw))):
        row = raw[r]
        for c in range(min(4, len(row))):
            value = row[c]
            out[r][c] = value if value in (1, 2, 3, 4) else None
    return out, malformed


def consensus_grid(transcriptions: list[SudokuTranscription]) -> tuple[list[list[int | None]], int]:
    grids = [normalize_grid(t.grid)[0] for t in transcriptions]
    n = len(grids)
    threshold = n // 2 + 1
    consensus: list[list[int | None]] = [[None] * 4 for _ in range(4)]
    disagreements = 0
    for r in range(4):
        for c in range(4):
            counts = Counter(grid[r][c] for grid in grids if grid[r][c] is not None)
            if not counts:
                continue
            value, count = counts.most_common(1)[0]
            if count >= threshold:
                consensus[r][c] = value
            if len({grid[r][c] for grid in grids}) > 1:
                disagreements += 1
    return consensus, disagreements


def compatible_solutions(givens: list[list[int]]) -> list[tuple[tuple[int, ...], ...]]:
    return [
        sol for sol in VALID_SOLUTIONS
        if all(givens[r][c] == 0 or givens[r][c] == sol[r][c] for r in range(4) for c in range(4))
    ]


def score_candidate(
    candidate: list[list[int | None]],
    givens: list[list[int]],
    reference: list[list[int]],
) -> dict[str, Any]:
    compatible = compatible_solutions(givens)
    if not compatible:
        raise RuntimeError("No valid Sudoku solution is compatible with stored givens")

    given_positions = [(r, c) for r in range(4) for c in range(4) if givens[r][c] != 0]
    blank_positions = [(r, c) for r in range(4) for c in range(4) if givens[r][c] == 0]
    given_correct = sum(candidate[r][c] == givens[r][c] for r, c in given_positions)
    given_rate = given_correct / len(given_positions) if given_positions else 1.0
    filled_blank = sum(candidate[r][c] is not None for r, c in blank_positions)
    completion_rate = filled_blank / len(blank_positions) if blank_positions else 1.0

    best_blank_matches = max(
        sum(candidate[r][c] == sol[r][c] for r, c in blank_positions)
        for sol in compatible
    )
    blank_accuracy = best_blank_matches / len(blank_positions) if blank_positions else 1.0

    complete = all(candidate[r][c] in (1, 2, 3, 4) for r in range(4) for c in range(4))
    valid_rows = sum(complete and set(candidate[r]) == {1, 2, 3, 4} for r in range(4))
    valid_cols = sum(complete and {candidate[r][c] for r in range(4)} == {1, 2, 3, 4} for c in range(4))
    valid_boxes = 0
    if complete:
        for br in (0, 2):
            for bc in (0, 2):
                vals = {candidate[r][c] for r in range(br, br + 2) for c in range(bc, bc + 2)}
                valid_boxes += vals == {1, 2, 3, 4}

    given_preserved = given_rate == 1.0
    is_valid = complete and given_preserved and valid_rows == 4 and valid_cols == 4 and valid_boxes == 4
    alternative = is_valid and candidate != reference

    if is_valid:
        score = 3
    elif given_rate == 1.0 and blank_accuracy >= 0.85 and completion_rate >= 0.85:
        score = 2
    elif given_rate >= 0.75 and blank_accuracy >= 0.50 and completion_rate >= 0.50:
        score = 1
    else:
        score = 0

    return {
        "score": score,
        "normalized_score": round(score / 3 * 100, 2),
        "is_valid_solution": is_valid,
        "alternative_valid_solution": alternative,
        "complete": complete,
        "given_preservation_rate": round(given_rate, 6),
        "blank_cell_accuracy": round(blank_accuracy, 6),
        "blank_completion_rate": round(completion_rate, 6),
        "valid_rows": valid_rows,
        "valid_columns": valid_cols,
        "valid_boxes": valid_boxes,
        "compatible_solution_count": len(compatible),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if r.get("status") == "success"]
    failed = [r for r in rows if r.get("status") != "success"]
    if not ok:
        return {"evaluated_successfully": 0, "evaluation_failures": len(failed)}
    scores = [int(r["score"]) for r in ok]
    dist = Counter(scores)
    return {
        "evaluated_successfully": len(ok),
        "evaluation_failures": len(failed),
        "exact_solve_rate": round(statistics.mean(bool(r["is_valid_solution"]) for r in ok), 6),
        "mean_blank_cell_accuracy": round(statistics.mean(float(r["blank_cell_accuracy"]) for r in ok), 6),
        "mean_blank_completion_rate": round(statistics.mean(float(r["blank_completion_rate"]) for r in ok), 6),
        "mean_given_preservation_rate": round(statistics.mean(float(r["given_preservation_rate"]) for r in ok), 6),
        "mean_tier": round(statistics.mean(scores), 4),
        "normalized_score": round(statistics.mean(scores) / 3 * 100, 4),
        "score_distribution": {str(k): dist.get(k, 0) for k in [0, 1, 2, 3]},
        "alternative_valid_solution_count": sum(bool(r.get("alternative_valid_solution")) for r in ok),
        "human_review_count": sum(bool(r.get("needs_human_review")) for r in ok),
    }


def main() -> None:
    args = parse_args()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")
    if args.passes < 1 or args.workers < 1:
        raise ValueError("--passes and --workers must be >= 1")

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

    eval_dir = run_dir / "evaluation" / f"sudoku_{args.reader_model.replace('/', '_')}"
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
        "reasoning_effort": args.reasoning_effort,
        "passes": args.passes,
        "confidence_threshold": args.confidence_threshold,
        "selected_count": len(items),
    })

    write_lock = threading.Lock()

    def transcribe_once(candidate_path: Path) -> SudokuTranscription:
        return openai_pydantic_call(
            client,
            model=args.reader_model,
            reasoning_effort=args.reasoning_effort,
            input_messages=[
                {"role": "system", "content": TRANSCRIBE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Transcribe only the 16 cells that are actually visible in the following candidate 4x4 Sudoku answer image."},
                        {"type": "input_image", "image_url": image_data_url(candidate_path), "detail": "original"},
                    ],
                },
            ],
            output_model=SudokuTranscription,
            schema_name="sudoku_transcription",
        )

    def evaluate_one(item: dict[str, Any]) -> dict[str, Any]:
        item_id = str(item["id"])
        started = time.perf_counter()
        generation = generations.get(item_id, {})
        candidate_path = run_dir / "images" / f"{item_id}.png"
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
            transcriptions = [
                call_with_retry(
                    lambda: transcribe_once(candidate_path),
                    max_retries=args.max_retries,
                    base_delay=args.retry_delay,
                )
                for _ in range(args.passes)
            ]
            if args.passes == 1:
                grid, malformed = normalize_grid(transcriptions[0].grid)
                disagreements = 0
            else:
                grid, disagreements = consensus_grid(transcriptions)
                malformed = any(normalize_grid(t.grid)[1] for t in transcriptions)

            meta_row = meta[item_id]
            givens = parse_grid_string(str(meta_row["puzzle"]))
            reference = parse_grid_string(str(meta_row["reference_solution"]))
            metrics = score_candidate(grid, givens, reference)
            mean_confidence = statistics.mean(t.confidence for t in transcriptions)
            needs_review = (
                malformed
                or mean_confidence < args.confidence_threshold
                or disagreements > 0
            )
            reasons: list[str] = []
            if malformed:
                reasons.append("reader returned malformed grid shape")
            if mean_confidence < args.confidence_threshold:
                reasons.append(f"mean transcription confidence {mean_confidence:.2f} below threshold")
            if disagreements > 0:
                reasons.append(f"reader passes disagreed on {disagreements} cells")

            return {
                "id": item_id,
                "status": "success",
                "reader_model": args.reader_model,
                "transcribed_grid": grid,
                "transcription_confidence": round(mean_confidence, 4),
                "transcription_disagreements": disagreements,
                **metrics,
                "needs_human_review": needs_review,
                "review_reason": "; ".join(reasons) or None,
                "reader_passes": [t.model_dump() for t in transcriptions],
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
            print(f"[{completed}/{total}] {row['id']} {row['status']} score={row.get('score')}")

    all_rows = load_jsonl(results_path) if results_path.exists() else []
    write_json(eval_dir / "summary.json", summarize(all_rows))
    print(f"Evaluation complete: {eval_dir}")


if __name__ == "__main__":
    main()
