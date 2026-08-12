from __future__ import annotations

import re

DEFAULT_MISSING_FIGURE_PROMPT = "This is a graphic reasoning question. Observe the picture of the question, analyze the rules within it, and generate the answer figure suitable for filling in the question mark."

_LEGACY_OUTPUT_CLAUSES = [
    "Only generate the final answer graph; do not generate multiple-choice options or textual explanations.",
    "Do not generate multiple-choice options or textual explanations.",
    "Only generate the missing answer graphics, do not generate complete questions, and do not add textual explanations.",
]


def build_civil_generation_prompt(item: dict, explanation_mode: str = "optional") -> str:
    base = str(item.get("prompt") or DEFAULT_MISSING_FIGURE_PROMPT).strip()
    for clause in _LEGACY_OUTPUT_CLAUSES:
        base = base.replace(clause, "")
    base = re.sub(r"\s+", " ", base).strip(" ，。") + "。"
    contract = "Generate the final answer graphic; do not generate multiple-choice options. The answer image shall only contain graphical content required to complete the task."
    if explanation_mode == "optional":
        contract += "Alternatively, a short explanatory text can be returned; the explanation must be output as independent text and not included in the answer image."
    return base + contract


def benchmark_task_text(item: dict) -> str:
    if item.get("prompt"):
        return build_civil_generation_prompt(item, explanation_mode="none")
    return DEFAULT_MISSING_FIGURE_PROMPT


# User's latest stricter prompts from maze_and_sudoku.zip.
MAZE_PROMPT = """
You need to complete a visual maze reasoning and image editing task.

Treat the input image as the base map that must be faithfully preserved, and you are only allowed to draw the solution path on the original image.
Examine the maze in the input image and identify the entrance and the destination. The entrance is indicated by an arrow, and the destination is marked by a small figure.

First plan a valid route inside the maze, then generate the final answer image. Mandatory requirements:
1. Directly edit the input image. Keep the original canvas proportions, perspective, boundaries, wall positions, line shapes, entrance arrow, destination figure and all background elements unchanged.
2. Do not redraw the maze, and do not convert the maze into photographs, paper textures, perspective views, hand-drawn styles, 3D styles or new layouts.
3. Do not move, delete, bold, deform, complete or modify any black walls; do not alter passages to make a viable path.
4. Only overlay one clear, continuous, semi-transparent or solid red path connecting the entrance to the destination.
5. The red path must stay entirely within the original passages. It must not pass through walls, cover walls in a way that may be misread as crossing walls, and must be unbroken with no jumps.
6. The output shall only contain the maze image with the solution. No explanatory text, titles, borders, options or extra decorations shall be added.
"""

SUDOKU_PROMPT = """
You need to complete a 4×4 Sudoku reasoning and image editing task.

Treat the input image as the base image that must be preserved faithfully. You are only allowed to fill in answers in blank cells.
Observe the 4×4 Sudoku in the input image. Keep the original grid, canvas, background color, lines and all given numbers unchanged, and fill in all blank cells to satisfy the following rules:
1. Each row shall contain exactly the numbers 1, 2, 3, 4 without repetition;
2. Each column shall contain exactly the numbers 1, 2, 3, 4 without repetition;
3. Each 2×2 subgrid shall contain exactly the numbers 1, 2, 3, 4 without repetition.

Mandatory requirements:
1. Directly edit the input image. Do not redraw the grid, change the perspective, convert it to photo/paper/hand-drawn style, or reformat the layout.
2. Keep all original numbers unchanged cell by cell. Do not move, modify, cover or redraw them.
3. Add clear red numbers only inside the originally blank cells; do not add any text, titles, options or decorations outside the grid.
4. New numbers shall be centered and legible without covering grid lines.
"""

EXPLANATION_SUFFIX = "\nAlternatively, a brief explanatory text may be returned; the explanation must be output as an independent text and shall not be included in the answer image."
NONOGRAM_PROMPT = """
You need to complete a Nonogram visual logical reasoning and image editing task.

Treat the input image as the base map that must be faithfully preserved. Only filling squares inside the grid is permitted.

The numerical clues indicate the lengths of consecutive black square blocks in the corresponding rows or columns, arranged in order from left to right or top to bottom.

Complete all logical reasoning internally first, then generate the final answer image. Mandatory requirements:
1. Directly edit the input image, keeping the original canvas, grid positions, grid lines, all row and column numerical clues and the overall layout unchanged.
2. Redrawing the puzzle is prohibited. Do not alter the perspective, scale, fonts, numerical clues, grid size or layout.
3. Fill the confirmed squares with solid black; leave all other squares white.
4. Fill in the entire grid and satisfy all row and column clues simultaneously.
5. Do not add any explanatory text, titles, options, legends or extra decorations outside the grid.
"""

TANGRAM_PROMPT = """
You need to complete a Tangram spatial combination reasoning and image editing task.

A grey target outline is provided at the top of the input image, and seven scattered, coloured standard tangram pieces are placed below.
Please first perform spatial reasoning internally, then directly edit the input image to move and rotate all seven pieces into the target outline.

Mandatory requirements:
1. All seven pieces must be used, and each piece can only be used once.
2. Retain the original shape, size and colour of each piece.
3. Only translation and rotation are permitted; stretching, compression, cutting, merging, adding or removing pieces are prohibited.
4. No overlapping between pieces.
5. No piece shall extend beyond the target outline.
6. The seven pieces must fully fill the target outline without obvious gaps.
7. Keep the original canvas proportion, the position of the target outline and the overall layout unchanged.
8. After completion, remove the originally scattered pieces at the bottom and only keep the final assembled result at the top.
9. Do not add explanatory text, titles, borders, options or other decorations.
"""

