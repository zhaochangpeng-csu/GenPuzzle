from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from common import openai_json_call
from utils import append_jsonl, get_first_image_path, image_to_data_url, read_jsonl, safe_filename, select_records


DIMENSIONS = {
    "m1_instruction_following": 20,
    "m2_spatial_correctness": 35,
    "m3_visual_structure": 20,
    "m4_text_label_accuracy": 15,
    "m5_task_completion": 10,
}
FATAL_CAPS = {
    "wrong_task_type": 40,
    "swapped_views": 60,
    "inconsistent_projection": 50,
    "missing_core_output": 45,
    "unreadable_grid": 60,
    "ignored_input_image": 30,
    "empty_or_irrelevant_output": 20,
}
RESULT_SCHEMA: dict[str, Any] = {
    "type":"object","additionalProperties":False,
    "properties":{
        "task_id":{"type":"string"},
        "score_total":{"type":"number","minimum":0,"maximum":100},
        "pass":{"type":"boolean"},
        "grades":{"type":"object","additionalProperties":False,
            "properties":{name:{"type":"object","additionalProperties":False,"properties":{
                "grade":{"type":"string","enum":["A","B","C","D","F"]},
                "score":{"type":"number","minimum":0,"maximum":mx},
                "max_score":{"type":"number"},"reason":{"type":"string"}},
                "required":["grade","score","max_score","reason"]} for name,mx in DIMENSIONS.items()},
            "required":list(DIMENSIONS)},
        "fatal_error_flags":{"type":"array","items":{"type":"string"}},
        "score_cap_applied":{"type":["number","null"]},
        "missing_required_elements":{"type":"array","items":{"type":"string"}},
        "observed_errors":{"type":"array","items":{"type":"string"}},
        "short_judgment":{"type":"string"},
        "suggested_human_review":{"type":"boolean"},
    },
    "required":["task_id","score_total","pass","grades","fatal_error_flags","score_cap_applied","missing_required_elements","observed_errors","short_judgment","suggested_human_review"]
}
SYSTEM_PROMPT = """You are a strict evaluator for orthographic-view and cube-based spatial reasoning.

You will see:
1. The input problem image.
2. Structured ground truth containing the correct front_view, top_view, right_view, cube_count, projection convention, required elements, and forbidden errors.
3. The ground-truth reference answer image.
4. The test model's generated image.

Scoring principles:
- Pixel-level or drawing-style identity with the ground truth is not required.
- Focus on whether the projection is correct, grid positions are correct, Front/Top/Right directions are correct, the three views are mutually consistent, and the cube count and spatial relations are correct.
- If the task asks for three views from a solid, check whether the generated image contains the correct front, top, and right views.
- If the task asks for a solid from three views, check whether the generated image expresses a cube structure equivalent to the ground truth.
- If the task asks for the missing third view, check whether the completed view is uniquely correct and consistent with the given views.
- Do not award points because the image looks polished; spatial projection correctness is the priority.

You must strictly score with the following five grades fields and must not invent field names:
1. m1_instruction_following, max 20: whether the output follows the requested task type.
2. m2_spatial_correctness, max 35: whether the projection, grid positions, directions, cube count, and spatial consistency are correct.
3. m3_visual_structure, max 20: whether the grid/cube structure is clear, readable, and aligned.
4. m4_text_label_accuracy, max 15: whether Front/Top/Right labels, direction marks, and text are correct.
5. m5_task_completion, max 10: whether the task is fully completed.

Fatal error flags must be selected only from these values:
- wrong_task_type: wrong output type, e.g. only drawing a decorative 3D image when the task asks for orthographic views.
- swapped_views: Front/Top/Right views are swapped or labels cause view misassignment.
- inconsistent_projection: multiple views contradict each other and cannot correspond to the same cube structure.
- missing_core_output: a core output is missing, such as one required view.
- unreadable_grid: the grid or cubes are unreadable.
- ignored_input_image: the input image is clearly ignored.
- empty_or_irrelevant_output: the output is blank or irrelevant.

Output requirements:
- Output only one JSON object.
- It must contain task_id, score_total, pass, grades, fatal_error_flags, score_cap_applied, missing_required_elements, observed_errors, short_judgment, and suggested_human_review.
- The grades object must contain the five m1...m5 fields above, and each field must contain grade, score, max_score, and reason.
- Do not output invented fields such as overall_score, front_view_correct, top_view_correct, right_view_correct, or evaluation_metrics."""


def compact_record(record: dict) -> dict:
    gt=record.get("gt") or {}
    return {"task_id":record.get("task_id"),"title":record.get("title"),"sub_category":record.get("sub_category"),"difficulty":record.get("difficulty"),"user_prompt":record.get("user_prompt"),"solution_policy":record.get("solution_policy"),"allow_novel_valid_solution":record.get("allow_novel_valid_solution"),"gt":{
        "problem_summary":gt.get("problem_summary"),"target_answer":gt.get("target_answer"),"answer_summary":gt.get("answer_summary"),"voxel_grid_size":gt.get("voxel_grid_size"),"voxel_occupancy":gt.get("voxel_occupancy"),"front_view":gt.get("front_view"),"top_view":gt.get("top_view"),"right_view":gt.get("right_view"),"projection_conventions":gt.get("projection_conventions"),"cube_count":gt.get("cube_count"),"required_elements":gt.get("required_elements",[]),"acceptable_variations":gt.get("acceptable_variations",[]),"forbidden_errors":gt.get("forbidden_errors",[])}}


def normalize(result: dict[str, Any], task_id: str) -> dict[str, Any]:
    result["task_id"]=task_id
    grades=result.get("grades") if isinstance(result.get("grades"),dict) else {}
    missing_grade_scores = []
    total=0.0
    for name,mx in DIMENSIONS.items():
        part=grades.get(name) if isinstance(grades.get(name),dict) else {}
        if "score" not in part:
            missing_grade_scores.append(name)
        try: score=float(part.get("score",0))
        except Exception: score=0.0
        score=min(max(score,0.0),float(mx)); part["score"]=score; part["max_score"]=mx; grades[name]=part; total+=score
    if missing_grade_scores:
        raise ValueError(
            "Judge result did not match orthographic schema; missing grade scores: "
            + ", ".join(missing_grade_scores)
        )
    result["grades"]=grades
    flags=result.get("fatal_error_flags") if isinstance(result.get("fatal_error_flags"),list) else []
    caps=[FATAL_CAPS[f] for f in flags if f in FATAL_CAPS]; cap=min(caps) if caps else None
    result["score_total"]=round(min(total,cap) if cap is not None else total,2)
    result["score_cap_applied"]=cap
    result["pass"]=bool(result["score_total"]>=80 and not caps)
    result["suggested_human_review"]=bool(result.get("suggested_human_review") or 50<=result["score_total"]<80)
    return result


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[1]/".env")
    p=argparse.ArgumentParser(description="Evaluate orthographic benchmark answers.")
    p.add_argument("--dataset",default="data/dataset_orthographic.jsonl")
    p.add_argument("--dataset-root",default=None)
    p.add_argument("--outputs-dir",default="outputs/gpt-image-2")
    p.add_argument("--result-file",default="results/eval_orthographic_gpt-image-2.jsonl")
    p.add_argument("--judge-model",default="gpt-5.5")
    p.add_argument("--limit",type=int,default=None)
    p.add_argument("--sample-every",type=int,default=None)
    p.add_argument("--sample-offset",type=int,default=0)
    p.add_argument("--base-url",default=None)
    p.add_argument("--reasoning-effort",default="high",choices=["none","low","medium","high","xhigh"])
    p.add_argument("--sleep",type=float,default=0.5)
    args=p.parse_args()

    dataset=Path(args.dataset).resolve(); dataset_root=Path(args.dataset_root).resolve() if args.dataset_root else (dataset.parent.parent if dataset.parent.name.lower()=="data" else dataset.parent)
    outputs=Path(args.outputs_dir).resolve(); result_file=Path(args.result_file).resolve()
    api_key=os.environ.get("OPENAI_API_KEY")
    if not api_key: raise RuntimeError("OPENAI_API_KEY is not set")
    client=OpenAI(api_key=api_key,base_url=args.base_url or os.environ.get("OPENAI_BASE_URL"))
    rows=select_records(read_jsonl(dataset),limit=args.limit,sample_every=args.sample_every,sample_offset=args.sample_offset)

    for i,record in enumerate(rows,1):
        task_id=record["task_id"]
        try:
            input_image=get_first_image_path(record,"input_images",dataset_root)
            gt_image=get_first_image_path(record,"reference_images",dataset_root)
            generated=outputs/f"{safe_filename(task_id)}.png"
            if not generated.exists(): raise FileNotFoundError(f"Missing generated image: {generated}")
            result=normalize(openai_json_call(
                client,
                model=args.judge_model,
                reasoning_effort=args.reasoning_effort,
                schema=RESULT_SCHEMA,
                schema_name="orthographic_eval",
                input_messages=[
                    {"role":"system","content":[{"type":"input_text","text":SYSTEM_PROMPT}]},
                    {"role":"user","content":[
                        {"type":"input_text","text":json.dumps(compact_record(record),ensure_ascii=False,indent=2)},
                        {"type":"input_text","text":"Input problem image:"},{"type":"input_image","image_url":image_to_data_url(input_image)},
                        {"type":"input_text","text":"GT reference answer image:"},{"type":"input_image","image_url":image_to_data_url(gt_image)},
                        {"type":"input_text","text":"Test model generated image:"},{"type":"input_image","image_url":image_to_data_url(generated)},
                    ]},
                ],
            ),task_id)
            result.update({"judge_model":args.judge_model,"input_image":str(input_image),"gt_image":str(gt_image),"generated_image":str(generated)})
            append_jsonl(result_file,result)
            print(f"[{i}/{len(rows)}] {task_id} score={result['score_total']}")
        except Exception as exc:
            append_jsonl(result_file,{"task_id":task_id,"score_total":0,"pass":False,"error":repr(exc),"suggested_human_review":True})
            print(f"[{i}/{len(rows)}] ERROR {task_id}: {exc}")
        time.sleep(args.sleep)


if __name__=="__main__":
    main()
