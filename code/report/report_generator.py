from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Categories
# -----------------------------------------------------------------------------

CATEGORY_ALIASES = {
    # Public benchmark tracks
    "figure": "figure_completion",
    "figure_completion": "figure_completion",
    "missing_figure": "figure_completion",
    "graphic_completion": "figure_completion",

    "spatial": "spatial_generation",
    "spatial_generation": "spatial_generation",
    "spatial_reasoning": "spatial_generation",

    "maze_beginner": "maze_beginner",
    "beginner_maze": "maze_beginner",
    "maze_intermediate": "maze_intermediate",
    "intermediate_maze": "maze_intermediate",
    "maze_advanced": "maze_advanced",
    "advanced_maze": "maze_advanced",
    "maze": "maze",

    "sudoku": "sudoku_reasoning",
    "sudoku_reasoning": "sudoku_reasoning",

    "nonogram": "nonogram_reasoning",
    "picross": "nonogram_reasoning",
    "nonogram_reasoning": "nonogram_reasoning",

    "tangram": "tangram_reasoning",
    "tangram_reasoning": "tangram_reasoning",

    "board": "board_game_reasoning",
    "board_game": "board_game_reasoning",
    "boardgame": "board_game_reasoning",
    "board_game_reasoning": "board_game_reasoning",
    "chess": "board_game_reasoning",
    "chess_reasoning": "board_game_reasoning",

    "stick": "matchstick_reasoning",
    "matchstick": "matchstick_reasoning",
    "mathsticks": "matchstick_reasoning",
    "matchstick_reasoning": "matchstick_reasoning",

    "ortho": "orthographic_reasoning",
    "orthographic": "orthographic_reasoning",
    "orthographic_reasoning": "orthographic_reasoning",
    "three_view": "orthographic_reasoning",

    "math": "math_visual_reasoning",
    "visual_math": "math_visual_reasoning",
    "math_visual_reasoning": "math_visual_reasoning",
    "mathematical_proof": "math_visual_reasoning",
}

DEFAULT_CATEGORY_ORDER = [
    "figure_completion",
    "spatial_generation",
    "maze_beginner",
    "maze_intermediate",
    "maze_advanced",
    "sudoku_reasoning",
    "nonogram_reasoning",
    "tangram_reasoning",
    "board_game_reasoning",
    "matchstick_reasoning",
    "orthographic_reasoning",
    "math_visual_reasoning",
]

CATEGORY_DISPLAY_NAMES = {
    "figure_completion": "Figure Completion",
    "spatial_generation": "Spatial Generation",
    "maze_beginner": "Maze · Beginner",
    "maze_intermediate": "Maze · Intermediate",
    "maze_advanced": "Maze · Advanced",
    "sudoku_reasoning": "Sudoku",
    "nonogram_reasoning": "Nonogram",
    "tangram_reasoning": "Tangram",
    "board_game_reasoning": "Board Games",
    "matchstick_reasoning": "Matchsticks",
    "orthographic_reasoning": "Orthographic",
    "math_visual_reasoning": "Math Visual Proof",
    "unknown": "Unknown",
}

# Prefix order matters: more specific prefixes first.
TASK_ID_CATEGORY_PREFIXES: List[Tuple[str, str]] = [
    ("ORTHO_", "orthographic_reasoning"),
    ("STICK_", "matchstick_reasoning"),
    ("VRG_", "math_visual_reasoning"),
    ("MATH_", "math_visual_reasoning"),
    ("IMG_", "math_visual_reasoning"),
    ("VIS_", "math_visual_reasoning"),

    # Board-game benchmark prefixes.
    ("AMAZONS_", "board_game_reasoning"),
    ("AMAZON_", "board_game_reasoning"),
    ("BREAKTHROUGH_", "board_game_reasoning"),
    ("CHECKERS_", "board_game_reasoning"),
    ("CHESS_", "board_game_reasoning"),
    ("CONNECT_FOUR_", "board_game_reasoning"),
    ("CONNECT4_", "board_game_reasoning"),
    ("DOTSBOXES_", "board_game_reasoning"),
    ("DOTS_AND_BOXES_", "board_game_reasoning"),
    ("GO_", "board_game_reasoning"),
    ("GOMOKU_", "board_game_reasoning"),
    ("HEX_", "board_game_reasoning"),
    ("LIGHTSOUT_", "board_game_reasoning"),
    ("LIGHTS_OUT_", "board_game_reasoning"),
    ("LOA_", "board_game_reasoning"),
    ("OWARE_", "board_game_reasoning"),
    ("NMM_", "board_game_reasoning"),
    ("NQUEENS_", "board_game_reasoning"),
    ("OTHELLO_", "board_game_reasoning"),
    ("PEG_", "board_game_reasoning"),
    ("SHOGI_", "board_game_reasoning"),
    ("SUDOKU_", "board_game_reasoning"),
    ("TTT_", "board_game_reasoning"),
    ("XQ_", "board_game_reasoning"),
]

MAZE_OR_SUDOKU_TIER_CATEGORIES = {
    "maze_beginner",
    "maze_intermediate",
    "maze_advanced",
    "sudoku_reasoning",
    "nonogram_reasoning",
    "tangram_reasoning",
}


def normalize_category(category: Optional[str]) -> str:
    if not category:
        return "unknown"
    c = str(category).strip().lower().replace("-", "_").replace(" ", "_")
    return CATEGORY_ALIASES.get(c, c)


def infer_category_from_task_id(task_id: str) -> str:
    task_id = str(task_id or "").upper()
    for prefix, category in TASK_ID_CATEGORY_PREFIXES:
        if task_id.startswith(prefix):
            return category
    return "unknown"


def infer_category_from_filename(path: Path) -> str:
    name = path.stem.lower().replace("-", "_")

    if "maze" in name:
        if "beginner" in name or "basic" in name or "easy" in name:
            return "maze_beginner"
        if "intermediate" in name or "medium" in name:
            return "maze_intermediate"
        if "advanced" in name or "hard" in name:
            return "maze_advanced"
        return "maze"

    if "figure_completion" in name or "missing_figure" in name:
        return "figure_completion"
    if "spatial_generation" in name or "spatial_reasoning" in name:
        return "spatial_generation"
    if "orthographic" in name or "ortho" in name or "three_view" in name:
        return "orthographic_reasoning"
    if "mathstick" in name or "matchstick" in name or "stick" in name:
        return "matchstick_reasoning"
    if "board" in name or "chess" in name or "lichess" in name:
        return "board_game_reasoning"
    if "sudoku" in name:
        return "sudoku_reasoning"
    if "nonogram" in name or "picross" in name:
        return "nonogram_reasoning"
    if "tangram" in name:
        return "tangram_reasoning"
    if "proof" in name or "visual_math" in name or "math_visual" in name:
        return "math_visual_reasoning"

    return "unknown"


def infer_category_from_record(record: dict) -> str:
    for key in ("category", "task", "task_family", "group", "track"):
        value = record.get(key)
        if value:
            cat = normalize_category(str(value))
            if cat != "unknown":
                return cat

    meta = record.get("_meta")
    if isinstance(meta, dict):
        for key in ("category", "task", "task_family", "group", "track", "difficulty"):
            value = meta.get(key)
            if value:
                cat = normalize_category(str(value))
                if cat != "unknown":
                    return cat

    return "unknown"


def infer_model_from_filename(path: Path) -> str:
    """Best-effort model-name inference. Explicit model:category:path is safer."""
    stem = path.stem
    stem = re.sub(r"^(eval|evaluation|results?|scores?)_+", "", stem, flags=re.I)

    remove_tokens = {
        "gpt55", "gpt5", "judge", "pilot", "eval", "evaluation", "result", "results",
        "math", "proof", "visual", "reasoning", "mathsticks", "matchsticks", "stick", "sticks",
        "board", "game", "games", "chess", "lichess", "orthographic", "ortho", "maze",
        "beginner", "intermediate", "advanced", "easy", "medium", "hard", "sudoku",
        "nonogram", "picross", "tangram",
        "figure", "completion", "missing", "spatial", "generation",
    }
    parts = [p for p in re.split(r"[_\s]+", stem) if p]
    kept = [p for p in parts if p.lower() not in remove_tokens]
    return "_".join(kept) if kept else path.stem


# -----------------------------------------------------------------------------
# Input specs and score extraction
# -----------------------------------------------------------------------------


def parse_input_spec(spec: str) -> Tuple[Optional[str], Optional[str], Path]:
    """Accepts path or model:category:path.

    Model labels may contain colons (for example ``openai:gpt-image@2``), and
    Windows absolute paths contain a drive colon. Work from the right and use
    the longest suffix that resolves to an existing path when possible.
    """
    colon_positions = [i for i, char in enumerate(spec) if char == ":"]
    for index in colon_positions:
        candidate_path = Path(spec[index + 1 :])
        if not candidate_path.exists():
            continue
        prefix = spec[:index]
        split_at = prefix.rfind(":")
        if split_at >= 0:
            model = prefix[:split_at]
            category = prefix[split_at + 1 :]
            return model, normalize_category(category), candidate_path

    parts = spec.rsplit(":", 2)
    if len(parts) == 3:
        model, category, path = parts
        return model, normalize_category(category), Path(path)
    return None, None, Path(spec)


def _as_float(value) -> Optional[float]:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return x




def _parse_score_cap(value) -> Optional[float]:
    cap = _as_float(value)
    if cap is not None:
        return cap
    if isinstance(value, str):
        numbers = re.findall(r"(?:^|\s)(\d+(?:\.\d+)?)", value)
        if numbers:
            return float(numbers[-1])
    return None

def _sum_nested_grade_scores(record: dict) -> Optional[float]:
    grades = record.get("grades")
    if not isinstance(grades, dict) or not grades:
        return None

    values: List[float] = []
    for item in grades.values():
        if not isinstance(item, dict):
            return None
        score = _as_float(item.get("score"))
        if score is None:
            return None
        max_score = _as_float(item.get("max_score"))
        if max_score is not None:
            score = min(max(score, 0.0), max_score)
        values.append(score)
    return sum(values) if values else None


def _sum_flat_component_scores(record: dict) -> Optional[float]:
    """Board-game evaluators use m1_...m5_... flat numeric fields."""
    values = []
    for key, value in record.items():
        if not re.match(r"^m\d+_", str(key)):
            continue
        x = _as_float(value)
        if x is not None:
            values.append(x)
    return sum(values) if values else None


def extract_score(
    record: dict,
    category: str,
    count_error_as_zero: bool = True,
) -> Tuple[Optional[float], str, bool, str]:
    """
    Returns (score_0_to_100, source, repaired, note).

    Precedence:
      1. normalized_score
      2. recomputed nested grades
      3. recomputed flat m1_... components
      4. score_total
      5. score (0-3 tiers are normalized only for maze/sudoku tracks)
      6. error -> 0, if requested
    """
    normalized = _as_float(record.get("normalized_score"))
    if normalized is not None:
        return min(max(normalized, 0.0), 100.0), "normalized_score", False, ""

    reported_total = _as_float(record.get("score_total"))

    nested = _sum_nested_grade_scores(record)
    if nested is not None:
        cap = _parse_score_cap(record.get("score_cap_applied"))
        effective = min(nested, cap) if cap is not None else nested
        repaired = reported_total is not None and abs(reported_total - effective) > 1e-6
        if repaired:
            note = f"reported score_total={reported_total:g}, recomputed={effective:g}"
            if cap is not None:
                note += f" (component sum={nested:g}, cap={cap:g})"
        else:
            note = ""
        source = "grades_component_sum_with_cap" if cap is not None else "grades_component_sum"
        return min(max(effective, 0.0), 100.0), source, repaired, note

    flat = _sum_flat_component_scores(record)
    if flat is not None:
        repaired = reported_total is not None and abs(reported_total - flat) > 1e-6
        note = f"reported score_total={reported_total:g}, component sum={flat:g}" if repaired else ""
        return min(max(flat, 0.0), 100.0), "flat_component_sum", repaired, note

    if reported_total is not None:
        return min(max(reported_total, 0.0), 100.0), "score_total", False, ""

    raw_score = _as_float(record.get("score"))
    if raw_score is not None:
        if category in MAZE_OR_SUDOKU_TIER_CATEGORIES and 0.0 <= raw_score <= 3.0:
            return raw_score / 3.0 * 100.0, "score_tier_0_3", False, ""
        return min(max(raw_score, 0.0), 100.0), "score", False, ""

    if "error" in record and count_error_as_zero:
        return 0.0, "error_as_zero", False, str(record.get("error", ""))

    return None, "missing", False, ""


# -----------------------------------------------------------------------------
# Read evaluation JSONL
# -----------------------------------------------------------------------------


def read_eval_file(
    path: Path,
    model_override: Optional[str] = None,
    category_override: Optional[str] = None,
    count_error_as_zero: bool = True,
) -> List[dict]:
    rows = []
    model = model_override or infer_model_from_filename(path)
    file_category = category_override or infer_category_from_filename(path)

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                if count_error_as_zero:
                    rows.append({
                        "model": model,
                        "category": file_category,
                        "task_id": f"JSON_ERROR_LINE_{line_no}",
                        "score": 0.0,
                        "score_source": "json_error",
                        "score_repaired": False,
                        "score_note": str(exc),
                        "grade": "json_error",
                        "is_error": True,
                    })
                continue

            task_id = (
                record.get("task_id")
                or record.get("id")
                or (record.get("_meta") or {}).get("task_id", "")
            )

            if category_override is not None:
                category = category_override
            else:
                category = infer_category_from_task_id(str(task_id))
                if category == "unknown":
                    category = infer_category_from_record(record)
                if category == "unknown":
                    category = file_category
            category = normalize_category(category)

            score, source, repaired, note = extract_score(
                record,
                category=category,
                count_error_as_zero=count_error_as_zero,
            )
            if score is None:
                continue

            rows.append({
                "model": model,
                "category": category,
                "task_id": task_id,
                "score": score,
                "score_source": source,
                "score_repaired": repaired,
                "score_note": note,
                "grade": record.get("grade", ""),
                "is_correct": record.get("is_correct"),
                "matches_gt": record.get("matches_gt", record.get("matches_reference_solution")),
                "novel_valid_solution": record.get("novel_valid_solution", record.get("alternative_valid_solution")),
                "fatal_error": record.get("fatal_error", bool(record.get("fatal_error_flags"))),
                "error_type": record.get("error_type"),
                "reason": record.get("reason", record.get("short_reason", record.get("short_judgment", ""))),
            })

    return rows


def keep_latest_task_rows(rows: List[dict]) -> List[dict]:
    latest: Dict[Tuple[str, str, str], dict] = {}
    for row in rows:
        key = (
            str(row.get("model", "")),
            str(row.get("category", "")),
            str(row.get("task_id", "")),
        )
        latest[key] = row
    return list(latest.values())


# -----------------------------------------------------------------------------
# Aggregation
# -----------------------------------------------------------------------------


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def aggregate_scores(
    rows: List[dict],
    category_order: List[str],
    category_weights: Optional[Dict[str, float]] = None,
) -> Tuple[List[dict], Dict[str, Dict[str, float]]]:
    by_model_category = defaultdict(lambda: defaultdict(list))
    by_model_all = defaultdict(list)

    for row in rows:
        model = row["model"]
        category = row["category"]
        score = float(row["score"])
        by_model_category[model][category].append(score)
        by_model_all[model].append(score)
        if category not in category_order and category != "unknown":
            category_order.append(category)

    if category_weights is None:
        category_weights = {cat: 1.0 for cat in category_order}

    leaderboard = []
    per_model_category_scores: Dict[str, Dict[str, float]] = {}

    for model in sorted(by_model_all):
        cat_scores: Dict[str, float] = {}
        cat_counts: Dict[str, int] = {}

        for cat in category_order:
            scores = by_model_category[model].get(cat, [])
            cat_scores[cat] = mean(scores)
            cat_counts[cat] = len(scores)

        weighted_sum = 0.0
        weight_total = 0.0
        for cat in category_order:
            score = cat_scores[cat]
            if math.isnan(score):
                continue
            weight = category_weights.get(cat, 1.0)
            weighted_sum += score * weight
            weight_total += weight

        macro_overall = weighted_sum / weight_total if weight_total else float("nan")
        micro_overall = mean(by_model_all[model])

        row = {
            "model": model,
            "macro_overall": macro_overall,
            "micro_overall": micro_overall,
            "total_count": len(by_model_all[model]),
        }
        for cat in category_order:
            row[f"{cat}_score"] = cat_scores[cat]
            row[f"{cat}_count"] = cat_counts[cat]

        leaderboard.append(row)
        per_model_category_scores[model] = cat_scores

    leaderboard.sort(
        key=lambda row: float("-inf") if math.isnan(row["macro_overall"]) else row["macro_overall"],
        reverse=True,
    )
    return leaderboard, per_model_category_scores


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------


def fmt_score(value: float) -> str:
    return "-" if value is None or math.isnan(value) else f"{value:.2f}"


def display_category(category: str) -> str:
    return CATEGORY_DISPLAY_NAMES.get(category, category)


def write_leaderboard_md(leaderboard: List[dict], category_order: List[str], out_path: Path) -> None:
    labels = [display_category(cat) for cat in category_order]
    lines = [
        "# Visual Reasoning Generation Leaderboard",
        "",
        "Main score: **Macro Overall**, the equal-weight average across the benchmark categories available for each model.",
        "",
        "| Rank | Model | Macro Overall | Micro Overall | Total Count | " + " | ".join(labels) + " |",
        "|---:|---|---:|---:|---:|" + "|".join(["---:"] * len(category_order)) + "|",
    ]

    for rank, row in enumerate(leaderboard, start=1):
        cells = []
        for cat in category_order:
            score = row.get(f"{cat}_score", float("nan"))
            count = row.get(f"{cat}_count", 0)
            cells.append(f"{fmt_score(score)} ({count})")
        lines.append(
            f"| {rank} | {row['model']} | {fmt_score(row['macro_overall'])} | "
            f"{fmt_score(row['micro_overall'])} | {row['total_count']} | " + " | ".join(cells) + " |"
        )

    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_leaderboard_csv(leaderboard: List[dict], category_order: List[str], out_path: Path) -> None:
    fieldnames = ["model", "macro_overall", "micro_overall", "total_count"]
    for cat in category_order:
        fieldnames.extend([f"{cat}_score", f"{cat}_count"])

    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(leaderboard)


def write_raw_scores_csv(rows: List[dict], out_path: Path) -> None:
    fieldnames = [
        "model", "category", "task_id", "score", "score_source", "score_repaired", "score_note",
        "grade", "is_correct", "matches_gt", "novel_valid_solution", "fatal_error", "error_type", "reason",
    ]
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_audit_summary(rows: List[dict], out_path: Path) -> None:
    repaired = [r for r in rows if r.get("score_repaired")]
    unknown = [r for r in rows if r.get("category") == "unknown"]
    by_source = defaultdict(int)
    for r in rows:
        by_source[r.get("score_source", "unknown")] += 1

    payload = {
        "record_count": len(rows),
        "score_repaired_count": len(repaired),
        "unknown_category_count": len(unknown),
        "score_sources": dict(sorted(by_source.items())),
        "repaired_examples": [
            {
                "model": r["model"],
                "task_id": r["task_id"],
                "category": r["category"],
                "score": r["score"],
                "note": r.get("score_note", ""),
            }
            for r in repaired[:20]
        ],
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------


def plot_overall_bar(leaderboard: List[dict], out_path: Path) -> None:
    models = [row["model"] for row in leaderboard]
    scores = [row["macro_overall"] for row in leaderboard]
    plt.figure(figsize=(max(8, len(models) * 1.2), 5))
    plt.bar(models, scores)
    plt.ylabel("Macro Overall Score")
    plt.ylim(0, 100)
    plt.title("Overall Leaderboard")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_radar(
    per_model_category_scores: Dict[str, Dict[str, float]],
    category_order: List[str],
    out_path: Path,
) -> None:
    if len(category_order) < 3:
        return

    labels = [display_category(cat) for cat in category_order]
    n = len(labels)
    angles = [2 * math.pi * i / n for i in range(n)]
    closed_angles = angles + angles[:1]

    plt.figure(figsize=(9, 9))
    ax = plt.subplot(111, polar=True)
    for model, cat_scores in per_model_category_scores.items():
        values = []
        for cat in category_order:
            score = cat_scores.get(cat, float("nan"))
            values.append(0.0 if math.isnan(score) else score)
        values += values[:1]
        ax.plot(closed_angles, values, linewidth=2, label=model)
        ax.fill(closed_angles, values, alpha=0.06)

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_title("Capability Radar by Category")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15))
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_weights(weight_args: Optional[List[str]]) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    for item in weight_args or []:
        if "=" not in item:
            raise ValueError(f"Invalid weight format: {item}. Expected category=weight")
        category, weight = item.split("=", 1)
        weights[normalize_category(category)] = float(weight)
    return weights


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leaderboard, bar chart and radar chart from eval JSONL files.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help=(
            "Evaluation JSONL files. Use either PATH or MODEL:CATEGORY:PATH. "
            "Explicit MODEL:CATEGORY:PATH is recommended for numeric-ID tracks."
        ),
    )
    parser.add_argument("--out-dir", "--out_dir", dest="out_dir", default="results/report")
    parser.add_argument(
        "--weights",
        nargs="*",
        default=None,
        help="Optional category weights, e.g. figure_completion=1 maze_beginner=1.",
    )
    parser.add_argument(
        "--count-errors-as-zero", "--count_errors_as_zero",
        dest="count_errors_as_zero",
        action="store_true",
        help="Count malformed/error records as score 0 instead of skipping them.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    category_order = DEFAULT_CATEGORY_ORDER.copy()
    category_weights = parse_weights(args.weights)
    rows: List[dict] = []

    for spec in args.inputs:
        model, category, path = parse_input_spec(spec)
        if not path.exists():
            raise FileNotFoundError(f"Evaluation file not found: {path}")
        rows.extend(read_eval_file(
            path=path,
            model_override=model,
            category_override=category,
            count_error_as_zero=args.count_errors_as_zero,
        ))

    if not rows:
        raise RuntimeError("No valid evaluation records were loaded.")
    rows = keep_latest_task_rows(rows)

    # Keep only categories that appear in at least one row, plus any custom category.
    present = {row["category"] for row in rows if row["category"] != "unknown"}
    category_order = [cat for cat in category_order if cat in present]
    for cat in sorted(present):
        if cat not in category_order:
            category_order.append(cat)

    leaderboard, per_model_category_scores = aggregate_scores(
        rows=rows,
        category_order=category_order,
        category_weights=category_weights or None,
    )

    write_raw_scores_csv(rows, out_dir / "raw_scores.csv")
    write_leaderboard_csv(leaderboard, category_order, out_dir / "leaderboard.csv")
    write_leaderboard_md(leaderboard, category_order, out_dir / "leaderboard.md")
    write_audit_summary(rows, out_dir / "score_audit.json")
    plot_overall_bar(leaderboard, out_dir / "overall_bar.png")
    plot_radar(per_model_category_scores, category_order, out_dir / "radar_chart.png")

    print(f"Report written to: {out_dir}")
    for name in ("leaderboard.md", "leaderboard.csv", "raw_scores.csv", "score_audit.json", "overall_bar.png", "radar_chart.png"):
        path = out_dir / name
        if path.exists():
            print(f"- {path}")


if __name__ == "__main__":
    main()
