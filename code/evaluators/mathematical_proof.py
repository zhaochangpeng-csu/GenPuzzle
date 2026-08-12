from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath
sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

import argparse
import base64
import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from common import call_with_retry, image_payload_for_eval, openai_json_call


DIMENSIONS = {
    "m1_instruction_following": 20,
    "m2_visual_structure": 25,
    "m3_reasoning_correctness": 30,
    "m4_text_symbol_accuracy": 15,
    "m5_clarity": 10,
}
FATAL_CAPS = {
    "wrong_final_answer": 40,
    "mathematical_contradiction": 40,
    "missing_core_visual_structure": 50,
    "unreadable_key_text": 60,
    "ignored_input_image": 30,
    "empty_or_irrelevant_output": 20,
}

RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "task_id": {"type": "string"},
        "score_total": {"type": "number", "minimum": 0, "maximum": 100},
        "pass": {"type": "boolean"},
        "grades": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                name: {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "grade": {"type": "string", "enum": ["A", "B", "C", "D", "F"]},
                        "score": {"type": "number", "minimum": 0, "maximum": max_score},
                        "max_score": {"type": "number"},
                        "reason": {"type": "string"},
                    },
                    "required": ["grade", "score", "max_score", "reason"],
                }
                for name, max_score in DIMENSIONS.items()
            },
            "required": list(DIMENSIONS),
        },
        "fatal_error_flags": {"type": "array", "items": {"type": "string"}},
        "score_cap_applied": {"type": ["number", "null"]},
        "missing_required_elements": {"type": "array", "items": {"type": "string"}},
        "observed_errors": {"type": "array", "items": {"type": "string"}},
        "matches_reference_solution": {"type": "boolean"},
        "novel_valid_solution": {"type": "boolean"},
        "short_judgment": {"type": "string"},
        "suggested_human_review": {"type": "boolean"},
    },
    "required": [
        "task_id", "score_total", "pass", "grades", "fatal_error_flags",
        "score_cap_applied", "missing_required_elements", "observed_errors",
        "matches_reference_solution", "novel_valid_solution", "short_judgment",
        "suggested_human_review",
    ],
}

SYSTEM_PROMPT = """You are a strict evaluator for visual mathematical reasoning.
You will see the task, structured scoring requirements, a reference answer image, and the test model's generated image; some tasks also provide an input problem image.

Core principles:
- The reference image is one known valid visual proof; pixel-level or layout-level identity is not required.
- If the task specifies a particular visual construction, focus on whether the same proof logic is preserved.
- For visual_proof_generation tasks, focus on whether the generated image proves the target statement using the visual construction required by the task.
- For image_text_to_image_reasoning tasks, first understand the input problem image, then check whether the generated image correctly answers the same problem.
- A correct formula with only textual or purely algebraic derivation should not receive a high score.
- Mathematical relations, symbols, angles, area correspondences, dot-array counting, and auxiliary lines must be correct.
- Do not reward logical errors because the image looks polished.
- Only when allow_novel_valid_solution=true may a different proof path be treated as a full-credit candidate.
- Output only an object that conforms to the JSON Schema."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def image_to_data_url(path: Path) -> str:
    payload, mime = image_payload_for_eval(path)
    mime = mime or mimetypes.guess_type(str(path))[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(payload).decode('ascii')}"


def resolve(dataset_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else dataset_root / path


def build_judge_text(item: dict[str, Any]) -> str:
    gt = item.get("gt") or {}
    compact = {
        "task_id": item.get("task_id"),
        "title": item.get("title"),
        "task_mode": item.get("task_mode"),
        "user_prompt": item.get("user_prompt"),
        "has_input_problem_image": bool(item.get("input_images")),
        "solution_policy": item.get("solution_policy"),
        "allow_novel_valid_solution": item.get("allow_novel_valid_solution"),
        "gt": {
            "problem_summary": gt.get("problem_summary"),
            "target_answer": gt.get("target_answer"),
            "target_formula": gt.get("target_formula"),
            "answer_summary": gt.get("answer_summary"),
            "required_elements": gt.get("required_elements", []),
            "acceptable_variations": gt.get("acceptable_variations", []),
            "forbidden_errors": gt.get("forbidden_errors", []),
        },
    }
    return "Evaluate the test image. Structured task information follows:\n" + json.dumps(compact, ensure_ascii=False, indent=2)


def normalize_result(result: dict[str, Any], task_id: str) -> dict[str, Any]:
    result["task_id"] = task_id
    grades = result.get("grades") if isinstance(result.get("grades"), dict) else {}
    missing_grade_scores = []
    component_sum = 0.0
    for name, max_score in DIMENSIONS.items():
        part = grades.get(name) if isinstance(grades.get(name), dict) else {}
        if "score" not in part:
            missing_grade_scores.append(name)
        try:
            score = float(part.get("score", 0))
        except Exception:
            score = 0.0
        score = min(max(score, 0.0), float(max_score))
        part["score"] = score
        part["max_score"] = max_score
        grades[name] = part
        component_sum += score
    if missing_grade_scores:
        raise ValueError(
            "Judge result did not match mathematical proof schema; missing grade scores: "
            + ", ".join(missing_grade_scores)
        )
    result["grades"] = grades

    flags = result.get("fatal_error_flags") if isinstance(result.get("fatal_error_flags"), list) else []
    caps = [FATAL_CAPS[f] for f in flags if f in FATAL_CAPS]
    cap = min(caps) if caps else None
    score_total = component_sum if cap is None else min(component_sum, cap)
    result["score_total"] = round(score_total, 2)
    result["score_cap_applied"] = cap
    result["pass"] = bool(score_total >= 80 and not caps)
    result["suggested_human_review"] = bool(
        result.get("suggested_human_review")
        or result.get("novel_valid_solution")
        or (50 <= score_total < 80)
    )
    return result


def select(rows: list[dict[str, Any]], limit: int | None, start_id: str | None, end_id: str | None) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        task_id = str(row.get("task_id", ""))
        if start_id and task_id < start_id:
            continue
        if end_id and task_id > end_id:
            continue
        out.append(row)
        if limit is not None and len(out) >= limit:
            break
    return out


def completed_eval_ids(output: Path) -> set[str]:
    if not output.exists():
        return set()
    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(output):
        task_id = str(row.get("task_id") or "")
        if task_id:
            latest[task_id] = row
    return {
        task_id
        for task_id, row in latest.items()
        if "error" not in row and "score_total" in row and isinstance(row.get("grades"), dict)
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate visual mathematical proof generations.")
    p.add_argument("--dataset", default="dataset.jsonl")
    p.add_argument("--dataset-root", default=None)
    p.add_argument("--generated-dir", default="outputs/image2")
    p.add_argument("--output", default="results/eval_image2.jsonl")
    p.add_argument("--judge-model", default="gpt-5.5")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--start-id", default=None)
    p.add_argument("--end-id", default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument("--reasoning-effort", default="medium", choices=["none", "low", "medium", "high", "xhigh"])
    p.add_argument("--sleep", type=float, default=0.5)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--retry-delay", type=float, default=2.0)
    p.add_argument("--request-timeout", type=float, default=180.0)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset = Path(args.dataset).resolve()
    dataset_root = Path(args.dataset_root).resolve() if args.dataset_root else dataset.parent
    generated_dir = Path(args.generated_dir).resolve()
    output = Path(args.output).resolve()
    if args.overwrite and output.exists():
        output.unlink()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    client = OpenAI(
        api_key=api_key,
        base_url=args.base_url or os.environ.get("OPENAI_BASE_URL"),
        timeout=args.request_timeout,
    )

    selected_rows = select(read_jsonl(dataset), args.limit, args.start_id, args.end_id)
    done = set() if args.overwrite else completed_eval_ids(output)
    rows = [row for row in selected_rows if str(row.get("task_id")) not in done]
    if done:
        print(f"Resume: skipping {len(done)} completed records")
    print(f"Records: {len(rows)} pending / {len(selected_rows)} selected")
    for i, item in enumerate(rows, 1):
        task_id = str(item["task_id"])
        ref = resolve(dataset_root, item["reference_images"][0]["path"])
        gen = generated_dir / f"{task_id}.png"
        if not gen.exists():
            append_jsonl(output, {"task_id": task_id, "error": f"missing generated image: {gen}"})
            continue
        content: list[dict[str, Any]] = [
            {"type": "input_text", "text": build_judge_text(item)},
        ]
        for image in item.get("input_images") or []:
            p = resolve(dataset_root, image["path"])
            content.extend([
                {"type": "input_text", "text": "Input problem image:"},
                {"type": "input_image", "image_url": image_to_data_url(p)},
            ])
        content.extend([
            {"type": "input_text", "text": "Reference answer image:"},
            {"type": "input_image", "image_url": image_to_data_url(ref)},
            {"type": "input_text", "text": "Test model generated image:"},
            {"type": "input_image", "image_url": image_to_data_url(gen)},
        ])
        try:
            result = normalize_result(call_with_retry(
                lambda: openai_json_call(
                    client,
                    model=args.judge_model,
                    reasoning_effort=args.reasoning_effort,
                    schema=RESULT_SCHEMA,
                    schema_name="visual_math_eval",
                    input_messages=[
                        {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
                        {"role": "user", "content": content},
                    ],
                ),
                max_retries=args.max_retries,
                base_delay=args.retry_delay,
            ), task_id)
            result["generated_image"] = str(gen)
            result["gt_image"] = str(ref)
            result["judge_model"] = args.judge_model
            append_jsonl(output, result)
            print(f"[{i}/{len(rows)}] {task_id} score={result['score_total']}")
        except Exception as exc:
            append_jsonl(output, {"task_id": task_id, "error": repr(exc), "suggested_human_review": True})
            print(f"[{i}/{len(rows)}] ERROR {task_id}: {exc}")
        time.sleep(args.sleep)


if __name__ == "__main__":
    main()
