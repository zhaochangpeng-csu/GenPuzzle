import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from common import call_with_retry, openai_json_call
from utils import (
    append_jsonl,
    get_first_image_path,
    image_to_data_url,
    infer_dataset_root,
    read_jsonl,
    safe_filename,
    select_records,
)


RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "task_id": {"type": "string"},
        "score_total": {"type": "integer", "minimum": 0, "maximum": 100},
        "m1_layout_result": {"type": "integer", "minimum": 0, "maximum": 15},
        "m2_logic_correctness": {"type": "integer", "minimum": 0, "maximum": 45},
        "m3_input_preservation": {"type": "integer", "minimum": 0, "maximum": 20},
        "m4_text_and_labels": {"type": "integer", "minimum": 0, "maximum": 10},
        "m5_task_completion": {"type": "integer", "minimum": 0, "maximum": 10},
        "pass": {"type": "boolean"},
        "error_flags": {"type": "array", "items": {"type": "string"}},
        "short_reason": {"type": "string"},
        "detailed_reason": {"type": "string"},
        "suggested_human_review": {"type": "boolean"},
    },
    "required": [
        "task_id",
        "score_total",
        "m1_layout_result",
        "m2_logic_correctness",
        "m3_input_preservation",
        "m4_text_and_labels",
        "m5_task_completion",
        "pass",
        "error_flags",
        "short_reason",
        "detailed_reason",
        "suggested_human_review",
    ],
}


RETRYABLE_ERROR_FLAGS = {
    "evaluation_script_error",
    "judge_output_not_json",
}


JUDGE_SYSTEM_PROMPT = """You are a strict VLM judge for visual reasoning answer images.

You will receive:
1. The original input image.
2. The GT reference image.
3. The model-generated answer image.
4. The structured dataset record, including the hidden ground truth.

Judge whether the generated image solves the task. Do not require identical styling to the GT image. Focus on visual correctness, rule correctness, input preservation, and legibility.

Scoring:
- m1_layout_result: 0-15. Clear, complete, readable answer diagram with the expected board/grid/result structure.
- m2_logic_correctness: 0-45. Correct game/puzzle logic, legal move/fill/placement, correct final result, correct line/capture/target when applicable.
- m3_input_preservation: 0-20. Preserves all relevant input pieces, digits, givens, grid geometry, labels, and coordinates except the intended answer change.
- m4_text_and_labels: 0-10. Text, labels, coordinates, arrows, highlights, and annotations are accurate and not misleading.
- m5_task_completion: 0-10. Fully completes the requested task as an answer image.

Policy:
- If solution_policy is single_solution, the answer must match the GT logic.
- If solution_policy is open_solution and allow_novel_valid_solution is true, a different valid solution may receive full credit.
- Penalize illegal moves, wrong side to move, changed unrelated pieces/digits, wrong board size, floating Connect Four discs, invalid Sudoku/N-Queens/checkers/chess logic, or an answer that is only text with no visual solution.
- Penalize cross-game contamination: generated images must not introduce piece icons, notation, labels, symbols, or visual conventions from another board game. For example, simplified Shogi letter boards must not turn into western chess piece icons or chess-style CHECK/CHECKMATE diagrams.
- Set pass to true only when score_total >= 80 and no serious rule error is present.
- Output strict JSON only. No Markdown, prose wrapper, or code block."""


def build_judge_prompt(record: dict[str, Any]) -> str:
    compact_record = {
        "task_id": record.get("task_id"),
        "title": record.get("title"),
        "category": record.get("category"),
        "sub_category": record.get("sub_category"),
        "user_prompt": record.get("user_prompt"),
        "solution_policy": record.get("solution_policy"),
        "allow_novel_valid_solution": record.get("allow_novel_valid_solution"),
        "gt": record.get("gt"),
    }
    return (
        "Evaluate the generated answer image for this visual reasoning task.\n\n"
        "Structured task and ground truth:\n"
        f"{json.dumps(compact_record, ensure_ascii=False, indent=2)}\n\n"
        "Return exactly one JSON object matching the required schema."
    )


def extract_output_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return str(text)

    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            value = getattr(content, "text", None)
            if value:
                chunks.append(str(value))
    return "\n".join(chunks)


def extract_chat_output_text(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", "") if message is not None else ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(parts)
    return str(content)


def call_judge(
    client: OpenAI,
    judge_model: str,
    record: dict[str, Any],
    input_image: Path,
    reference_image: Path,
    generated_image: Path,
) -> str:
    result = openai_json_call(
        client,
        model=judge_model,
        schema=RESULT_SCHEMA,
        schema_name="visual_reasoning_eval_result",
        input_messages=[
            {"role": "system", "content": [{"type": "input_text", "text": JUDGE_SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "input_text", "text": build_judge_prompt(record)},
                {"type": "input_text", "text": "Original input image:"},
                {"type": "input_image", "image_url": image_to_data_url(input_image)},
                {"type": "input_text", "text": "GT reference image:"},
                {"type": "input_image", "image_url": image_to_data_url(reference_image)},
                {"type": "input_text", "text": "Generated answer image to evaluate:"},
                {"type": "input_image", "image_url": image_to_data_url(generated_image)},
            ]},
        ],
    )
    return json.dumps(result, ensure_ascii=False)


def error_result(task_id: str, flag: str, short_reason: str, detailed_reason: str = "") -> dict[str, Any]:
    return {
        "task_id": task_id,
        "score_total": 0,
        "m1_layout_result": 0,
        "m2_logic_correctness": 0,
        "m3_input_preservation": 0,
        "m4_text_and_labels": 0,
        "m5_task_completion": 0,
        "pass": False,
        "error_flags": [flag],
        "short_reason": short_reason,
        "detailed_reason": detailed_reason,
        "suggested_human_review": True,
    }


def is_retryable_result(row: dict[str, Any]) -> bool:
    flags = row.get("error_flags")
    if isinstance(flags, list) and any(str(flag) in RETRYABLE_ERROR_FLAGS for flag in flags):
        return True
    detail = str(row.get("detailed_reason") or row.get("error") or "")
    return "APITimeoutError" in detail or "timed out" in detail.lower()


def completed_eval_ids(result_file: Path) -> set[str]:
    if not result_file.exists():
        return set()

    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(result_file):
        task_id = str(row.get("task_id") or "")
        if task_id:
            latest[task_id] = row
    return {task_id for task_id, row in latest.items() if not is_retryable_result(row)}


def normalize_result(result: dict[str, Any], task_id: str) -> dict[str, Any]:
    normalized = error_result(task_id, "judge_result_missing_fields", "Judge result was incomplete.")
    normalized.update(result)
    normalized["task_id"] = task_id

    component_keys = [
        "m1_layout_result",
        "m2_logic_correctness",
        "m3_input_preservation",
        "m4_text_and_labels",
        "m5_task_completion",
    ]
    max_scores = {
        "m1_layout_result": 15,
        "m2_logic_correctness": 45,
        "m3_input_preservation": 20,
        "m4_text_and_labels": 10,
        "m5_task_completion": 10,
    }

    component_sum = 0
    for key in component_keys:
        try:
            value = int(normalized.get(key, 0))
        except Exception:
            value = 0
        value = max(0, min(value, max_scores[key]))
        normalized[key] = value
        component_sum += value

    # Never trust an inconsistent free-form score_total. The five dimensions define the score.
    normalized["score_total"] = component_sum

    if not isinstance(normalized.get("error_flags"), list):
        normalized["error_flags"] = [str(normalized.get("error_flags"))]

    serious_flags = {
        "illegal_move", "wrong_move", "wrong_final_board", "wrong_board_size",
        "ignored_input_image", "empty_or_irrelevant_output", "single_solution_mismatch",
    }
    has_serious_error = any(flag in serious_flags for flag in normalized["error_flags"])
    normalized["pass"] = bool(component_sum >= 80 and not has_serious_error)
    normalized["suggested_human_review"] = bool(normalized.get("suggested_human_review"))

    return {key: normalized[key] for key in RESULT_SCHEMA["required"]}


def evaluate_one(
    client: OpenAI,
    record: dict[str, Any],
    dataset_root: Path,
    outputs_dir: Path,
    judge_model: str,
) -> dict[str, Any]:
    task_id = str(record["task_id"])
    input_image = get_first_image_path(record, "input_images", dataset_root)
    reference_image = get_first_image_path(record, "reference_images", dataset_root)
    generated_image = outputs_dir / f"{safe_filename(task_id)}.png"
    if not generated_image.exists():
        return error_result(
            task_id,
            "missing_generated_image",
            f"Generated image not found: {generated_image}",
        )

    raw_text = call_judge(
        client=client,
        judge_model=judge_model,
        record=record,
        input_image=input_image,
        reference_image=reference_image,
        generated_image=generated_image,
    )
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        return error_result(
            task_id,
            "judge_output_not_json",
            "Judge did not return valid JSON.",
            raw_text[:4000],
        )

    return normalize_result(parsed, task_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-evaluate generated visual reasoning images.")
    parser.add_argument("--dataset", default="data/dataset_board_game.jsonl")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--outputs-dir", default="outputs/gpt-image-2")
    parser.add_argument("--result-file", default="results/eval_gpt-image-2.jsonl")
    parser.add_argument("--judge-model", default="gpt-5.5")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--sample-every",
        type=int,
        default=None,
        help="Evaluate one record every N dataset rows, matching generate.py sampling.",
    )
    parser.add_argument(
        "--sample-offset",
        type=int,
        default=0,
        help="Zero-based offset used with --sample-every. Default 0 selects the first row in each stride.",
    )
    parser.add_argument("--base-url", default=None)
    return parser.parse_args()


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    args = parse_args()

    dataset_path = Path(args.dataset).resolve()
    dataset_root = infer_dataset_root(dataset_path, args.dataset_root)
    outputs_dir = Path(args.outputs_dir).resolve()
    result_file = Path(args.result_file).resolve()
    if args.overwrite and result_file.exists():
        result_file.unlink()

    all_records = read_jsonl(dataset_path)
    selected_records = select_records(
        all_records,
        limit=args.limit,
        sample_every=args.sample_every,
        sample_offset=args.sample_offset,
    )
    done = completed_eval_ids(result_file)
    records = [record for record in selected_records if str(record.get("task_id")) not in done]

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    client = OpenAI(
        api_key=api_key,
        base_url=args.base_url or os.environ.get("OPENAI_BASE_URL"),
        timeout=args.request_timeout,
    )

    print(f"Dataset: {dataset_path}")
    print(f"Dataset root: {dataset_root}")
    print(f"Outputs dir: {outputs_dir}")
    print(f"Result file: {result_file}")
    print(f"Judge model: {args.judge_model}")
    print(f"Records: {len(records)} pending / {len(selected_records)} selected from {len(all_records)}")
    if done:
        print(f"Resume: skipping {len(done)} completed records")
    if args.sample_every is not None:
        print(f"Sampling: every {args.sample_every} records, offset {args.sample_offset}")

    for index, record in enumerate(records, start=1):
        task_id = str(record.get("task_id", f"record_{index}"))
        started = time.time()
        try:
            result = call_with_retry(
                lambda: evaluate_one(
                    client=client,
                    record=record,
                    dataset_root=dataset_root,
                    outputs_dir=outputs_dir,
                    judge_model=args.judge_model,
                ),
                max_retries=args.max_retries,
                base_delay=args.retry_delay,
            )
        except Exception as exc:
            result = error_result(
                task_id,
                "evaluation_script_error",
                "Evaluation script failed before receiving a judge result.",
                repr(exc),
            )

        append_jsonl(result_file, result)
        print(
            f"[{index}/{len(records)}] {task_id} "
            f"score={result['score_total']} pass={result['pass']} "
            f"elapsed={time.time() - started:.1f}s"
        )


if __name__ == "__main__":
    main()
