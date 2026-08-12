from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from common import image_payload_for_eval, openai_json_call


RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "task_id": {"type": "string"},
        "score": {"type": "integer", "enum": [0, 100]},
        "grade": {"type": "string", "enum": ["correct", "wrong"]},
        "matches_gt": {"type": "boolean"},
        "novel_valid_solution": {"type": "boolean"},
        "is_correct": {"type": "boolean"},
        "fatal_error": {"type": "boolean"},
        "error_type": {"type": ["string", "null"]},
        "extracted_generated_equation": {"type": ["string", "null"]},
        "reason": {"type": "string"},
        "suggested_human_review": {"type": "boolean"},
    },
    "required": ["task_id", "score", "grade", "matches_gt", "novel_valid_solution", "is_correct", "fatal_error", "error_type", "extracted_generated_equation", "reason", "suggested_human_review"],
}

SYSTEM_PROMPT = """You are a strict evaluator for matchstick-equation visual reasoning.
You will see the original problem image, one known correct reference answer image, and the test model's generated image.

Judging principles:
- The reference answer is only one known valid solution; if the data allow a novel valid solution, a different but legal solution should also receive full credit.
- Do not require pixel-level or layout-level identity between the test image and the reference image.
- Strictly check the original equation, the required number of moved matchsticks, conservation of the total number of matchsticks, legibility of seven-segment digits, and correctness of the final equation.
- The test image should be a clean final corrected equation. If it copies segment labels such as A0/B1/C2, dashed candidate sticks, or auxiliary marks from the input image into the final answer, treat this as an obvious error.
- If the final equation is true, the digits are legible, and the result can be obtained from the original by moving exactly the required number of sticks, mark it correct and set novel_valid_solution=true even if it differs from known_target_answer.
- Do not award credit for adding/removing matchsticks from nowhere, changing more than the required number of sticks, or producing an incorrect equation.
- Output only an object that conforms to the JSON Schema."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def resolve(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def image_url(path: Path) -> str:
    payload, mime = image_payload_for_eval(path)
    mime = mime or mimetypes.guess_type(str(path))[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(payload).decode('ascii')}"


def move_constraint(item: dict[str, Any]) -> str:
    mapping = {"one_stick_move": 1, "two_stick_move": 2, "three_stick_move": 3, "four_stick_move": 4}
    n = mapping.get(item.get("sub_category"))
    return f"Must move exactly {n} matchstick(s)" if n else "Must strictly follow the required number of moved matchsticks"


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate MathSticks benchmark outputs.")
    p.add_argument("--dataset", default="data/dataset_mathsticks.jsonl")
    p.add_argument("--root", default=".")
    p.add_argument("--generated-dir", default="outputs/gpt-image-2")
    p.add_argument("--output", default="results/eval_gpt-image-2.jsonl")
    p.add_argument("--judge-model", default="gpt-5.5")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--start-id", default=None)
    p.add_argument("--end-id", default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument("--reasoning-effort", default="high", choices=["none", "low", "medium", "high", "xhigh"])
    p.add_argument("--sleep", type=float, default=0.5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset = Path(args.dataset).resolve()
    root = Path(args.root).resolve()
    generated_dir = Path(args.generated_dir).resolve()
    output = Path(args.output).resolve()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    client = OpenAI(api_key=api_key, base_url=args.base_url or os.environ.get("OPENAI_BASE_URL"))

    rows = []
    for item in read_jsonl(dataset):
        task_id = str(item["task_id"])
        if args.start_id and task_id < args.start_id:
            continue
        if args.end_id and task_id > args.end_id:
            continue
        rows.append(item)
        if args.limit is not None and len(rows) >= args.limit:
            break

    for i, item in enumerate(rows, 1):
        task_id = str(item["task_id"])
        input_path = resolve(root, item["input_images"][0]["path"])
        gt_path = resolve(root, item["reference_images"][0]["path"])
        gen_path = generated_dir / f"{task_id}.png"
        if not gen_path.exists():
            append_jsonl(output, {"task_id": task_id, "error": f"missing generated image: {gen_path}"})
            continue
        gt = item.get("gt") or {}
        prompt = {
            "task_id": task_id,
            "user_prompt": item.get("user_prompt"),
            "move_constraint": move_constraint(item),
            "solution_policy": item.get("solution_policy"),
            "allow_novel_valid_solution": item.get("allow_novel_valid_solution"),
            "known_target_answer_one_valid_solution": gt.get("target_answer"),
            "known_target_answer_is_not_exclusive": bool(item.get("allow_novel_valid_solution")),
            "known_problem_summary": gt.get("problem_summary"),
            "required_elements": gt.get("required_elements", []),
            "acceptable_variations": gt.get("acceptable_variations", []),
            "forbidden_errors": gt.get("forbidden_errors", []),
        }
        try:
            result = openai_json_call(
                client,
                model=args.judge_model,
                reasoning_effort=args.reasoning_effort,
                schema=RESULT_SCHEMA,
                schema_name="mathsticks_eval",
                input_messages=[
                    {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
                    {"role": "user", "content": [
                        {"type": "input_text", "text": json.dumps(prompt, ensure_ascii=False, indent=2)},
                        {"type": "input_text", "text": "Original problem image:"},
                        {"type": "input_image", "image_url": image_url(input_path)},
                        {"type": "input_text", "text": "Known correct reference image:"},
                        {"type": "input_image", "image_url": image_url(gt_path)},
                        {"type": "input_text", "text": "Test model generated image:"},
                        {"type": "input_image", "image_url": image_url(gen_path)},
                    ]},
                ],
            )
            result["task_id"] = task_id
            result["score"] = 100 if result.get("is_correct") else 0
            result["grade"] = "correct" if result["score"] == 100 else "wrong"
            result["suggested_human_review"] = bool(result.get("suggested_human_review") or result.get("novel_valid_solution"))
            result["_meta"] = {
                "task_id": task_id,
                "judge_model": args.judge_model,
                "sub_category": item.get("sub_category"),
                "move_constraint": move_constraint(item),
            }
            append_jsonl(output, result)
            print(f"[{i}/{len(rows)}] {task_id} score={result['score']}")
        except Exception as exc:
            append_jsonl(output, {"task_id": task_id, "error": repr(exc)})
            print(f"[{i}/{len(rows)}] ERROR {task_id}: {exc}")
        time.sleep(args.sleep)


if __name__ == "__main__":
    main()
