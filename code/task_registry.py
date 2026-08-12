from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class TaskSpec:
    name: str
    display_name: str
    dataset_dir: str
    data_file: str
    id_field: str
    expected_count: int
    loader_kind: str
    evaluator: str
    category: str
    filter_fn: Callable[[dict[str, Any]], bool] | None = None


def _civil_figure(row: dict[str, Any]) -> bool:
    return int(str(row["id"])) <= 394


def _civil_spatial(row: dict[str, Any]) -> bool:
    return int(str(row["id"])) >= 395


TASKS: dict[str, TaskSpec] = {
    "figure_completion": TaskSpec(
        "figure_completion", "Figure Completion", "datasets/civil_service", "data.jsonl", "id", 394,
        "civil_service", "civil_service", "figure_completion", _civil_figure,
    ),
    "spatial_generation": TaskSpec(
        "spatial_generation", "Spatial Generation", "datasets/civil_service", "data.jsonl", "id", 56,
        "civil_service", "civil_service", "spatial_generation", _civil_spatial,
    ),
    "maze_beginner": TaskSpec(
        "maze_beginner", "Maze - Beginner", "datasets/maze/beginner", "data.jsonl", "id", 64,
        "maze", "maze", "maze_beginner",
    ),
    "maze_intermediate": TaskSpec(
        "maze_intermediate", "Maze - Intermediate", "datasets/maze/intermediate", "data.jsonl", "id", 64,
        "maze", "maze", "maze_intermediate",
    ),
    "maze_advanced": TaskSpec(
        "maze_advanced", "Maze - Advanced", "datasets/maze/advanced", "data.jsonl", "id", 64,
        "maze", "maze", "maze_advanced",
    ),
    "sudoku_reasoning": TaskSpec(
        "sudoku_reasoning", "Sudoku", "datasets/sudoku", "data.jsonl", "id", 78,
        "sudoku", "sudoku", "sudoku_reasoning",
    ),
    "nonogram_reasoning": TaskSpec(
        "nonogram_reasoning", "Nonogram", "datasets/nonogram", "data.jsonl", "id", 150,
        "nonogram", "nonogram", "nonogram_reasoning",
    ),
    "tangram_reasoning": TaskSpec(
        "tangram_reasoning", "Tangram", "datasets/tangram", "data.jsonl", "id", 150,
        "tangram", "tangram", "tangram_reasoning",
    ),
    "board_game_reasoning": TaskSpec(
        "board_game_reasoning", "Board Games", "datasets/board_game", "data/dataset_board_game.jsonl", "task_id", 300,
        "structured", "board_game", "board_game_reasoning",
    ),
    "matchstick_reasoning": TaskSpec(
        "matchstick_reasoning", "Matchsticks", "datasets/matchsticks", "data/dataset_mathsticks.jsonl", "task_id", 300,
        "structured", "matchsticks", "matchstick_reasoning",
    ),
    "orthographic_reasoning": TaskSpec(
        "orthographic_reasoning", "Orthographic", "datasets/orthographic", "data/dataset_orthographic.jsonl", "task_id", 90,
        "structured", "orthographic", "orthographic_reasoning",
    ),
    "math_visual_reasoning": TaskSpec(
        "math_visual_reasoning", "Math Visual Proof", "datasets/mathematical_proof", "data/dataset.jsonl", "task_id", 295,
        "structured", "mathematical_proof", "math_visual_reasoning",
    ),
}

TASK_ORDER = list(TASKS)

ALIASES = {
    "all": "all",
    "figure": "figure_completion",
    "spatial": "spatial_generation",
    "sudoku": "sudoku_reasoning",
    "nonogram": "nonogram_reasoning",
    "picross": "nonogram_reasoning",
    "tangram": "tangram_reasoning",
    "board": "board_game_reasoning",
    "board_game": "board_game_reasoning",
    "matchstick": "matchstick_reasoning",
    "mathsticks": "matchstick_reasoning",
    "ortho": "orthographic_reasoning",
    "orthographic": "orthographic_reasoning",
    "proof": "math_visual_reasoning",
    "math": "math_visual_reasoning",
}


def parse_tasks(value: str) -> list[str]:
    raw = [x.strip() for x in value.split(",") if x.strip()]
    if not raw or raw == ["all"]:
        return TASK_ORDER.copy()
    out: list[str] = []
    for name in raw:
        name = ALIASES.get(name, name)
        if name == "all":
            return TASK_ORDER.copy()
        if name not in TASKS:
            raise ValueError(f"Unknown task: {name}. Choices: {', '.join(TASK_ORDER)}")
        if name not in out:
            out.append(name)
    return out


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_no}")
            rows.append(row)
    return rows


def dataset_root(suite_root: Path, spec: TaskSpec) -> Path:
    return (suite_root / spec.dataset_dir).resolve()


def load_raw_rows(suite_root: Path, spec: TaskSpec) -> list[dict[str, Any]]:
    root = dataset_root(suite_root, spec)
    rows = read_jsonl(root / spec.data_file)
    if spec.filter_fn is not None:
        rows = [r for r in rows if spec.filter_fn(r)]
    return rows


def build_matchsticks_generation_prompt(row: dict[str, Any]) -> str:
    prompt = str(row.get("user_prompt") or "").strip()
    mapping = {
        "one_stick_move": 1,
        "two_stick_move": 2,
        "three_stick_move": 3,
        "four_stick_move": 4,
    }
    count = mapping.get(str(row.get("sub_category") or ""))
    move_text = (
        f"Move exactly {count} matchstick" + ("" if count == 1 else "s")
        if count is not None
        else "Move exactly the required number of matchsticks"
    )
    checks = [
        (move_text.lower(), move_text + "."),
        ("do not add or remove", "Preserve the total number of matchsticks; do not add or remove sticks."),
        ("render only", "Render only the final corrected equation."),
        ("do not copy segment labels", "Do not copy segment labels such as A0/B1/C2, dashed ghost sticks, candidate-position markers, or other annotations from the input image."),
        ("arithmetically true", "The final equation must be arithmetically true, clean, and legible."),
    ]
    prompt_lower = prompt.lower()
    missing = [line for marker, line in checks if marker not in prompt_lower]
    if missing:
        prompt = prompt.rstrip() + "\n\nHard constraints:\n- " + "\n- ".join(missing)
    return prompt.strip()


def board_game_visual_guardrails(row: dict[str, Any]) -> list[str]:
    task_id = str(row.get("task_id") or "")
    sub_category = str(row.get("sub_category") or "")
    guardrails = [
        "Use only the board game shown in the input image; do not introduce piece icons, notation, labels, symbols, or visual conventions from any other board game.",
        "Preserve the input board geometry, coordinate labels, piece encoding, colors, and visual style except for the required answer move/placement/highlight.",
        "Render a clean board-state answer diagram, not a decorative reinterpretation.",
    ]

    if task_id.startswith("SHOGI_") or "shogi" in sub_category:
        guardrails.extend([
            "For Shogi, this dataset uses a simplified 9x9 board with single-letter pieces K/R/B/G/S/N/L/P in the same colors as the input.",
            "Keep the simplified letter-piece notation; do not draw western chess king/rook/bishop/knight icons, chess-style check badges, checkmate labels, captured-piece stands, drops, or promotion symbols.",
        ])
    elif task_id.startswith("XQ_") or "xiangqi" in sub_category:
        guardrails.append(
            "For Xiangqi, preserve the river, palace lines, intersections, and the original piece notation; do not draw western chess or Shogi pieces."
        )
    elif task_id.startswith("CHESS_") or "chess" in sub_category:
        guardrails.append(
            "For Chess, keep an 8x8 chessboard and chess pieces only; do not use Shogi, Xiangqi, checkers, Go, or abstract puzzle symbols."
        )
    elif task_id.startswith(("GO_", "GOMOKU_", "OTHELLO_")):
        guardrails.append(
            "For stone games, use only black/white stones on the correct grid or board; do not introduce chess-like pieces, arrows as pieces, or unrelated game symbols."
        )
    return guardrails


def build_board_game_generation_prompt(row: dict[str, Any]) -> str:
    prompt = str(row.get("user_prompt") or "").strip()
    prompt_lower = prompt.lower()
    missing = [
        line for line in board_game_visual_guardrails(row)
        if line.lower() not in prompt_lower
    ]
    if missing:
        prompt = prompt.rstrip() + "\n\nVisual-style constraints:\n- " + "\n- ".join(missing)
    return prompt.strip()


def normalize_item(suite_root: Path, spec: TaskSpec, row: dict[str, Any]) -> dict[str, Any]:
    root = dataset_root(suite_root, spec)
    item_id = str(row[spec.id_field])

    if spec.loader_kind == "civil_service":
        from prompts import build_civil_generation_prompt
        prompt = build_civil_generation_prompt(row, explanation_mode="none")
        input_images = [str(row["image"])]
    elif spec.loader_kind == "maze":
        from prompts import MAZE_PROMPT
        prompt = MAZE_PROMPT.strip()
        input_images = [str(row["image"])]
    elif spec.loader_kind == "sudoku":
        from prompts import SUDOKU_PROMPT
        prompt = SUDOKU_PROMPT.strip()
        input_images = [str(row["image"])]
    elif spec.loader_kind == "nonogram":
        from prompts import NONOGRAM_PROMPT
        prompt = NONOGRAM_PROMPT.strip()
        input_images = [str(row["image"])]
    elif spec.loader_kind == "tangram":
        from prompts import TANGRAM_PROMPT
        prompt = TANGRAM_PROMPT.strip()
        input_images = [str(row["image"])]
    elif spec.loader_kind == "structured":
        if spec.name == "matchstick_reasoning":
            prompt = build_matchsticks_generation_prompt(row)
        elif spec.name == "board_game_reasoning":
            prompt = build_board_game_generation_prompt(row)
        else:
            prompt = str(row.get("user_prompt") or "").strip()
        input_images = [str(x["path"]) for x in (row.get("input_images") or [])]
    else:
        raise ValueError(f"Unknown loader kind: {spec.loader_kind}")

    return {
        "task": spec.name,
        "id": item_id,
        "prompt": prompt,
        "input_images": input_images,
        "dataset_root": root,
        "raw": row,
    }


def load_items(suite_root: Path, spec: TaskSpec) -> list[dict[str, Any]]:
    return [normalize_item(suite_root, spec, row) for row in load_raw_rows(suite_root, spec)]
