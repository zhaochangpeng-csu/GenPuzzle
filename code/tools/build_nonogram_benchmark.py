from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# -----------------------------------------------------------------------------
# Clues and solver
# -----------------------------------------------------------------------------


def line_clues(values: Iterable[int]) -> tuple[int, ...]:
    out: list[int] = []
    run = 0
    for value in values:
        if int(value):
            run += 1
        elif run:
            out.append(run)
            run = 0
    if run:
        out.append(run)
    return tuple(out)


def grid_clues(grid: np.ndarray) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    rows = tuple(line_clues(row) for row in grid)
    cols = tuple(line_clues(grid[:, c]) for c in range(grid.shape[1]))
    return rows, cols


@lru_cache(maxsize=None)
def line_patterns(length: int, clues: tuple[int, ...]) -> tuple[int, ...]:
    """All bitmasks of a line that satisfy clues. Bit 0 is the left/top cell."""
    if not clues:
        return (0,)

    patterns: list[int] = []
    suffix_min = [0] * (len(clues) + 1)
    for i in range(len(clues) - 1, -1, -1):
        suffix_min[i] = suffix_min[i + 1] + clues[i] + (1 if i < len(clues) - 1 else 0)

    def rec(idx: int, pos: int, mask: int) -> None:
        block = clues[idx]
        latest_start = length - suffix_min[idx]
        for start in range(pos, latest_start + 1):
            block_mask = ((1 << block) - 1) << start
            next_mask = mask | block_mask
            if idx == len(clues) - 1:
                patterns.append(next_mask)
            else:
                rec(idx + 1, start + block + 1, next_mask)

    rec(0, 0, 0)
    return tuple(patterns)


@dataclass
class SolveStats:
    nodes: int = 0
    branches: int = 0
    propagation_rounds: int = 0
    domain_reductions: int = 0
    max_depth: int = 0
    elapsed_ms: float = 0.0


class SolveAbort(RuntimeError):
    pass


class NonogramSolver:
    def __init__(self, row_clues: tuple[tuple[int, ...], ...], col_clues: tuple[tuple[int, ...], ...]):
        self.row_clues = row_clues
        self.col_clues = col_clues
        self.h = len(row_clues)
        self.w = len(col_clues)
        self.full_row_mask = (1 << self.w) - 1
        self.full_col_mask = (1 << self.h) - 1

    def solve_count(self, limit: int = 2, max_nodes: int = 4000) -> tuple[int, np.ndarray | None, SolveStats]:
        stats = SolveStats()
        started = time.perf_counter()
        row_domains = [list(line_patterns(self.w, clues)) for clues in self.row_clues]
        col_domains = [list(line_patterns(self.h, clues)) for clues in self.col_clues]
        if any(not d for d in row_domains) or any(not d for d in col_domains):
            return 0, None, stats

        count = 0
        first_solution: np.ndarray | None = None

        def recurse(rows: list[list[int]], cols: list[list[int]], depth: int) -> None:
            nonlocal count, first_solution
            if count >= limit:
                return
            stats.nodes += 1
            if stats.nodes > max_nodes:
                raise SolveAbort(f"solver exceeded {max_nodes} nodes")
            stats.max_depth = max(stats.max_depth, depth)

            ok, grid = self._propagate(rows, cols, stats)
            if not ok:
                return

            # Solved if every line domain is singleton.
            if all(len(d) == 1 for d in rows) and all(len(d) == 1 for d in cols):
                solution = np.zeros((self.h, self.w), dtype=np.uint8)
                for r, domain in enumerate(rows):
                    pattern = domain[0]
                    for c in range(self.w):
                        solution[r, c] = (pattern >> c) & 1
                # Defensive cross-check against column singletons.
                for c, domain in enumerate(cols):
                    pattern = domain[0]
                    for r in range(self.h):
                        if int(solution[r, c]) != ((pattern >> r) & 1):
                            return
                count += 1
                if first_solution is None:
                    first_solution = solution
                return

            # Choose the smallest remaining line domain.
            choice_kind = "row"
            choice_idx = -1
            choice_size = 10**9
            for i, domain in enumerate(rows):
                if 1 < len(domain) < choice_size:
                    choice_kind, choice_idx, choice_size = "row", i, len(domain)
            for i, domain in enumerate(cols):
                if 1 < len(domain) < choice_size:
                    choice_kind, choice_idx, choice_size = "col", i, len(domain)

            if choice_idx < 0:
                return
            stats.branches += 1
            domain = rows[choice_idx] if choice_kind == "row" else cols[choice_idx]
            for pattern in list(domain):
                if count >= limit:
                    break
                new_rows = [d.copy() for d in rows]
                new_cols = [d.copy() for d in cols]
                if choice_kind == "row":
                    new_rows[choice_idx] = [pattern]
                else:
                    new_cols[choice_idx] = [pattern]
                recurse(new_rows, new_cols, depth + 1)

        try:
            recurse(row_domains, col_domains, 0)
        except SolveAbort:
            count = -1
            first_solution = None
        stats.elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
        return count, first_solution, stats

    def _propagate(
        self,
        rows: list[list[int]],
        cols: list[list[int]],
        stats: SolveStats,
    ) -> tuple[bool, np.ndarray]:
        # -1 unknown, 0 white, 1 black
        grid = np.full((self.h, self.w), -1, dtype=np.int8)
        changed = True
        while changed:
            changed = False
            stats.propagation_rounds += 1

            # Filter row domains by known cells, then infer forced cells.
            for r in range(self.h):
                before = len(rows[r])
                filtered = [p for p in rows[r] if self._pattern_matches_row(p, grid[r])]
                if not filtered:
                    return False, grid
                if len(filtered) != before:
                    rows[r] = filtered
                    stats.domain_reductions += before - len(filtered)
                    changed = True
                and_mask = self.full_row_mask
                or_mask = 0
                for p in rows[r]:
                    and_mask &= p
                    or_mask |= p
                for c in range(self.w):
                    if (and_mask >> c) & 1:
                        if grid[r, c] == 0:
                            return False, grid
                        if grid[r, c] != 1:
                            grid[r, c] = 1
                            changed = True
                    elif not ((or_mask >> c) & 1):
                        if grid[r, c] == 1:
                            return False, grid
                        if grid[r, c] != 0:
                            grid[r, c] = 0
                            changed = True

            # Filter column domains and infer forced cells.
            for c in range(self.w):
                before = len(cols[c])
                filtered = [p for p in cols[c] if self._pattern_matches_col(p, grid[:, c])]
                if not filtered:
                    return False, grid
                if len(filtered) != before:
                    cols[c] = filtered
                    stats.domain_reductions += before - len(filtered)
                    changed = True
                and_mask = self.full_col_mask
                or_mask = 0
                for p in cols[c]:
                    and_mask &= p
                    or_mask |= p
                for r in range(self.h):
                    if (and_mask >> r) & 1:
                        if grid[r, c] == 0:
                            return False, grid
                        if grid[r, c] != 1:
                            grid[r, c] = 1
                            changed = True
                    elif not ((or_mask >> r) & 1):
                        if grid[r, c] == 1:
                            return False, grid
                        if grid[r, c] != 0:
                            grid[r, c] = 0
                            changed = True

        return True, grid

    @staticmethod
    def _pattern_matches_row(pattern: int, known: np.ndarray) -> bool:
        for idx, value in enumerate(known):
            if value >= 0 and ((pattern >> idx) & 1) != int(value):
                return False
        return True

    @staticmethod
    def _pattern_matches_col(pattern: int, known: np.ndarray) -> bool:
        for idx, value in enumerate(known):
            if value >= 0 and ((pattern >> idx) & 1) != int(value):
                return False
        return True


# -----------------------------------------------------------------------------
# Candidate generation
# -----------------------------------------------------------------------------


def _neighbors(r: int, c: int, n: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < n and 0 <= cc < n:
            out.append((rr, cc))
    return out


def random_walk_mask(n: int, rng: random.Random) -> np.ndarray:
    grid = np.zeros((n, n), dtype=np.uint8)
    target = rng.randint(max(3, int(n * n * 0.28)), max(4, int(n * n * 0.62)))
    r, c = rng.randrange(n), rng.randrange(n)
    grid[r, c] = 1
    frontier = [(r, c)]
    while int(grid.sum()) < target:
        if rng.random() < 0.78 and frontier:
            r, c = rng.choice(frontier)
        else:
            r, c = rng.randrange(n), rng.randrange(n)
        for _ in range(rng.randint(1, 4)):
            r, c = rng.choice(_neighbors(r, c, n))
            grid[r, c] = 1
            frontier.append((r, c))
            if int(grid.sum()) >= target:
                break
    # Mild local smoothing.
    for _ in range(max(1, n // 5)):
        rr, cc = rng.randrange(n), rng.randrange(n)
        neigh = sum(grid[a, b] for a, b in _neighbors(rr, cc, n))
        if neigh >= 3:
            grid[rr, cc] = 1
        elif neigh == 0:
            grid[rr, cc] = 0
    return grid


def primitive_mask(n: int, rng: random.Random) -> np.ndarray:
    img = Image.new("1", (n, n), 0)
    draw = ImageDraw.Draw(img)
    primitive_count = rng.randint(2, max(3, n // 3 + 1))
    for _ in range(primitive_count):
        kind = rng.choice(["rect", "ellipse", "line", "triangle"])
        x0, x1 = sorted((rng.randrange(n), rng.randrange(n)))
        y0, y1 = sorted((rng.randrange(n), rng.randrange(n)))
        if x0 == x1:
            x1 = min(n - 1, x0 + 1)
        if y0 == y1:
            y1 = min(n - 1, y0 + 1)
        if kind == "rect":
            draw.rectangle([x0, y0, x1, y1], fill=1)
        elif kind == "ellipse":
            draw.ellipse([x0, y0, x1, y1], fill=1)
        elif kind == "line":
            width = max(1, n // 8)
            draw.line([x0, y0, x1, y1], fill=1, width=width)
        else:
            x2, y2 = rng.randrange(n), rng.randrange(n)
            draw.polygon([(x0, y0), (x1, y1), (x2, y2)], fill=1)
    grid = np.array(img, dtype=np.uint8)
    if rng.random() < 0.35:
        grid = np.fliplr(grid)
    if rng.random() < 0.35:
        grid = np.flipud(grid)
    return grid.copy()


def structured_mask(n: int, rng: random.Random) -> np.ndarray:
    # Generate one half and reflect, then perturb a few cells.
    half = np.zeros((n, (n + 1) // 2), dtype=np.uint8)
    density = rng.uniform(0.28, 0.62)
    half[:] = np.array(
        [[1 if rng.random() < density else 0 for _ in range(half.shape[1])] for _ in range(n)],
        dtype=np.uint8,
    )
    if rng.random() < 0.5:
        left = half[:, : n // 2]
        grid = np.concatenate([left, np.fliplr(left)], axis=1) if n % 2 == 0 else np.concatenate([left, half[:, -1:], np.fliplr(left)], axis=1)
    else:
        top = half[: (n + 1) // 2, :]
        base = np.zeros((n, n), dtype=np.uint8)
        # use a resized random core to create horizontal symmetry
        core = np.array(Image.fromarray((half * 255).astype(np.uint8)).resize((n, (n + 1) // 2), Image.Resampling.NEAREST)) > 0
        core = core.astype(np.uint8)
        if n % 2 == 0:
            grid = np.concatenate([core[: n // 2], np.flipud(core[: n // 2])], axis=0)
        else:
            grid = np.concatenate([core[: n // 2], core[n // 2 : n // 2 + 1], np.flipud(core[: n // 2])], axis=0)
    for _ in range(max(1, n // 4)):
        if rng.random() < 0.5:
            r, c = rng.randrange(n), rng.randrange(n)
            grid[r, c] ^= 1
    return grid.astype(np.uint8)


def random_matrix_mask(n: int, rng: random.Random) -> np.ndarray:
    density = rng.uniform(0.28, 0.64)
    grid = np.array([[1 if rng.random() < density else 0 for _ in range(n)] for _ in range(n)], dtype=np.uint8)
    # Reduce isolated noise by flipping some isolated cells.
    for r in range(n):
        for c in range(n):
            neigh = sum(grid[a, b] for a, b in _neighbors(r, c, n))
            if grid[r, c] and neigh == 0 and rng.random() < 0.65:
                grid[r, c] = 0
    return grid


def generate_mask(n: int, rng: random.Random) -> tuple[str, np.ndarray]:
    source = rng.choices(
        ["random_walk", "primitives", "structured", "random_matrix"],
        weights=[0.35, 0.30, 0.20, 0.15],
        k=1,
    )[0]
    if source == "random_walk":
        return source, random_walk_mask(n, rng)
    if source == "primitives":
        return source, primitive_mask(n, rng)
    if source == "structured":
        return source, structured_mask(n, rng)
    return source, random_matrix_mask(n, rng)


def is_reasonable_mask(grid: np.ndarray) -> bool:
    n = grid.shape[0]
    total = n * n
    black = int(grid.sum())
    ratio = black / total
    if not (0.20 <= ratio <= 0.72):
        return False
    # Avoid too many empty/full lines.
    row_sums = grid.sum(axis=1)
    col_sums = grid.sum(axis=0)
    extreme_lines = int(np.sum((row_sums == 0) | (row_sums == n))) + int(np.sum((col_sums == 0) | (col_sums == n)))
    if extreme_lines > max(2, n // 3):
        return False
    # Avoid pathological fragmentation.
    row_cl, col_cl = grid_clues(grid)
    block_count = sum(len(x) for x in row_cl) + sum(len(x) for x in col_cl)
    if block_count > int(1.15 * n * n / 2):
        return False
    return True


def complexity_score(n: int, row_clues: tuple[tuple[int, ...], ...], col_clues: tuple[tuple[int, ...], ...], stats: SolveStats) -> float:
    line_blocks = sum(len(x) for x in row_clues) + sum(len(x) for x in col_clues)
    return (
        n * 1.5
        + math.log1p(stats.nodes) * 8.0
        + math.log1p(stats.branches) * 10.0
        + stats.max_depth * 4.0
        + math.log1p(stats.domain_reductions) * 2.0
        + (line_blocks / max(1, 2 * n)) * 3.0
    )


# -----------------------------------------------------------------------------
# Rendering
# -----------------------------------------------------------------------------


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        "arialbd.ttf",
        "arial.ttf",
    ]
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def render_nonogram(
    out_path: Path,
    grid: np.ndarray,
    row_clues: tuple[tuple[int, ...], ...],
    col_clues: tuple[tuple[int, ...], ...],
    *,
    solved: bool,
    canvas_size: int = 1024,
) -> dict[str, list[int] | int]:
    n = grid.shape[0]
    max_row_tokens = max(1, max((len(x) for x in row_clues), default=1))
    max_col_tokens = max(1, max((len(x) for x in col_clues), default=1))

    # Reserve enough room for clues while keeping the grid large.
    margin = 36
    clue_unit = max(22, min(54, int(canvas_size / (n + max(max_row_tokens, max_col_tokens) + 5))))
    left_clue_w = max(120, max_row_tokens * clue_unit + 24)
    top_clue_h = max(120, max_col_tokens * clue_unit + 24)
    available = min(canvas_size - left_clue_w - 2 * margin, canvas_size - top_clue_h - 2 * margin)
    cell = max(24, available // n)
    grid_px = cell * n
    x0 = (canvas_size - (left_clue_w + grid_px)) // 2 + left_clue_w
    y0 = (canvas_size - (top_clue_h + grid_px)) // 2 + top_clue_h

    img = Image.new("RGB", (canvas_size, canvas_size), "white")
    draw = ImageDraw.Draw(img)
    font_size = max(16, min(42, int(cell * 0.46)))
    font = load_font(font_size)

    # Answer fill first.
    if solved:
        inset = max(2, cell // 18)
        for r in range(n):
            for c in range(n):
                if int(grid[r, c]) == 1:
                    draw.rectangle(
                        [x0 + c * cell + inset, y0 + r * cell + inset,
                         x0 + (c + 1) * cell - inset, y0 + (r + 1) * cell - inset],
                        fill="black",
                    )

    # Grid lines. Thicker every five cells for larger puzzles.
    for i in range(n + 1):
        width = 4 if i in (0, n) or (n >= 10 and i % 5 == 0) else 2
        draw.line([x0, y0 + i * cell, x0 + grid_px, y0 + i * cell], fill="black", width=width)
        draw.line([x0 + i * cell, y0, x0 + i * cell, y0 + grid_px], fill="black", width=width)

    def text_size(text: str) -> tuple[int, int]:
        box = draw.textbbox((0, 0), text, font=font)
        return box[2] - box[0], box[3] - box[1]

    # Row clues, right-aligned near grid.
    for r, clues in enumerate(row_clues):
        tokens = list(clues) if clues else [0]
        cy = y0 + r * cell + cell / 2
        cursor_x = x0 - 14
        for value in reversed(tokens):
            text = str(value)
            tw, th = text_size(text)
            cursor_x -= tw
            draw.text((cursor_x, cy - th / 2 - 1), text, fill="black", font=font)
            cursor_x -= max(10, clue_unit // 3)

    # Column clues, bottom-aligned above grid.
    for c, clues in enumerate(col_clues):
        tokens = list(clues) if clues else [0]
        cx = x0 + c * cell + cell / 2
        cursor_y = y0 - 14
        for value in reversed(tokens):
            text = str(value)
            tw, th = text_size(text)
            cursor_y -= th
            draw.text((cx - tw / 2, cursor_y - 1), text, fill="black", font=font)
            cursor_y -= max(7, clue_unit // 4)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG", optimize=True)
    return {
        "canvas_size": canvas_size,
        "grid_bbox": [int(x0), int(y0), int(x0 + grid_px), int(y0 + grid_px)],
        "cell_size": int(cell),
    }


# -----------------------------------------------------------------------------
# Benchmark builder
# -----------------------------------------------------------------------------


@dataclass
class Candidate:
    size: int
    source_type: str
    seed: int
    grid: np.ndarray
    row_clues: tuple[tuple[int, ...], ...]
    col_clues: tuple[tuple[int, ...], ...]
    stats: SolveStats
    complexity: float
    hash: str


def collect_candidates(
    *,
    size: int,
    target_pool: int,
    master_rng: random.Random,
    seen_hashes: set[str],
    max_attempts: int,
) -> list[Candidate]:
    accepted: list[Candidate] = []
    attempts = 0
    while len(accepted) < target_pool and attempts < max_attempts:
        attempts += 1
        seed = master_rng.randrange(1, 2**31 - 1)
        rng = random.Random(seed)
        source, grid = generate_mask(size, rng)
        if not is_reasonable_mask(grid):
            continue
        digest = hashlib.sha256(grid.tobytes()).hexdigest()
        if digest in seen_hashes:
            continue
        rows, cols = grid_clues(grid)
        solver = NonogramSolver(rows, cols)
        count, solution, stats = solver.solve_count(limit=2, max_nodes=4000)
        if count != 1 or solution is None:
            continue
        if not np.array_equal(solution, grid):
            # Unique solution should be exactly the source mask that generated the clues.
            continue
        complexity = complexity_score(size, rows, cols, stats)
        accepted.append(Candidate(size, source, seed, grid, rows, cols, stats, complexity, digest))
        seen_hashes.add(digest)
        if len(accepted) % 10 == 0 or len(accepted) == target_pool:
            print(f"  size {size}: accepted {len(accepted)}/{target_pool} after {attempts} attempts")
    if len(accepted) < target_pool:
        raise RuntimeError(
            f"Could only collect {len(accepted)}/{target_pool} unique-solvable candidates for size {size} "
            f"after {attempts} attempts. Increase --max-attempts-per-size or lower --pool-multiplier."
        )
    return accepted


def evenly_spaced_select(items: list[Candidate], count: int, low_q: float, high_q: float) -> list[Candidate]:
    items = sorted(items, key=lambda x: x.complexity)
    if count >= len(items):
        return items[:count]
    lo = int(round((len(items) - 1) * low_q))
    hi = int(round((len(items) - 1) * high_q))
    hi = max(lo, hi)
    positions = np.linspace(lo, hi, count)
    used: set[int] = set()
    selected: list[Candidate] = []
    for pos in positions:
        idx = int(round(float(pos)))
        if idx in used:
            # Find nearest unused index.
            for delta in range(1, len(items)):
                choices = [idx - delta, idx + delta]
                found = next((j for j in choices if 0 <= j < len(items) and j not in used), None)
                if found is not None:
                    idx = found
                    break
        used.add(idx)
        selected.append(items[idx])
    return selected


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def build_benchmark(args: argparse.Namespace) -> None:
    if args.easy_count + args.medium_count + args.hard_count != args.total:
        raise ValueError("easy + medium + hard counts must equal --total")

    out = args.output.resolve()
    if out.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {out}. Use --overwrite to replace it.")
        shutil.rmtree(out)
    (out / "questions").mkdir(parents=True, exist_ok=True)
    (out / "answers").mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    seen_hashes: set[str] = set()
    targets = {
        "easy": (5, args.easy_count),
        "medium": (10, args.medium_count),
        "hard": (15, args.hard_count),
    }
    pools: dict[str, list[Candidate]] = {}

    print("Collecting unique-solvable candidate puzzles...")
    for difficulty, (size, count) in targets.items():
        pool_target = max(count, int(math.ceil(count * args.pool_multiplier)))
        pools[difficulty] = collect_candidates(
            size=size,
            target_pool=pool_target,
            master_rng=rng,
            seen_hashes=seen_hashes,
            max_attempts=args.max_attempts_per_size,
        )

    selected: dict[str, list[Candidate]] = {
        "easy": evenly_spaced_select(pools["easy"], args.easy_count, 0.00, 0.55),
        "medium": evenly_spaced_select(pools["medium"], args.medium_count, 0.20, 0.80),
        "hard": evenly_spaced_select(pools["hard"], args.hard_count, 0.45, 1.00),
    }

    data_rows: list[dict] = []
    meta_rows: list[dict] = []
    summary_items: list[dict] = []
    idx = 1
    for difficulty in ("easy", "medium", "hard"):
        for candidate in selected[difficulty]:
            item_id = f"{idx:06d}"
            question_rel = f"questions/{item_id}.png"
            answer_rel = f"answers/{item_id}.png"
            render_meta = render_nonogram(
                out / question_rel,
                candidate.grid,
                candidate.row_clues,
                candidate.col_clues,
                solved=False,
                canvas_size=args.canvas_size,
            )
            render_nonogram(
                out / answer_rel,
                candidate.grid,
                candidate.row_clues,
                candidate.col_clues,
                solved=True,
                canvas_size=args.canvas_size,
            )
            data_rows.append({"id": item_id, "image": question_rel, "answer": answer_rel})
            meta_rows.append({
                "id": item_id,
                "difficulty": difficulty,
                "size": candidate.size,
                "row_clues": [list(x) for x in candidate.row_clues],
                "column_clues": [list(x) for x in candidate.col_clues],
                "solution": candidate.grid.astype(int).tolist(),
                "solution_count": 1,
                "source_type": candidate.source_type,
                "candidate_seed": candidate.seed,
                "solution_sha256": candidate.hash,
                "black_ratio": round(float(candidate.grid.mean()), 6),
                "complexity_score": round(candidate.complexity, 6),
                "solver_stats": asdict(candidate.stats),
                "render": render_meta,
            })
            summary_items.append({
                "id": item_id,
                "difficulty": difficulty,
                "size": candidate.size,
                "source_type": candidate.source_type,
                "complexity_score": round(candidate.complexity, 3),
            })
            idx += 1

    write_jsonl(out / "data.jsonl", data_rows)
    write_jsonl(out / "eval_meta.jsonl", meta_rows)
    summary = {
        "name": "Nonogram Benchmark",
        "total": args.total,
        "difficulty_split": {
            "easy": args.easy_count,
            "medium": args.medium_count,
            "hard": args.hard_count,
        },
        "grid_sizes": {"easy": 5, "medium": 10, "hard": 15},
        "seed": args.seed,
        "pool_multiplier": args.pool_multiplier,
        "unique_solution_required": True,
        "items": summary_items,
    }
    (out / "build_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "README.md").write_text(
        """# Nonogram Benchmark (150 items)\n\n"
        "This dataset is generated locally by `build_nonogram_benchmark.py`.\n\n"
        "- Public benchmark rows: `data.jsonl`\n"
        "- Hidden deterministic evaluation metadata: `eval_meta.jsonl`\n"
        "- Question images: `questions/`\n"
        "- Reference answer images: `answers/`\n"
        "- Every puzzle is checked to have exactly one solution.\n"
        "- Default split: 45 easy (5x5), 60 medium (10x10), 45 hard (15x15), i.e. 30%/40%/30%.\n"
        """,
        encoding="utf-8",
    )
    print(f"\nBuilt {len(data_rows)} items at: {out}")
    print(f"  easy={args.easy_count}, medium={args.medium_count}, hard={args.hard_count}")
    print("  all puzzles: unique solution verified")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="One-click generator for a 150-item Nonogram benchmark.")
    p.add_argument("--output", type=Path, default=Path("datasets/nonogram"))
    p.add_argument("--total", type=int, default=150)
    # User requested 30/40/30; for 150 this is interpreted as 30%/40%/30%.
    p.add_argument("--easy-count", type=int, default=45)
    p.add_argument("--medium-count", type=int, default=60)
    p.add_argument("--hard-count", type=int, default=45)
    p.add_argument("--seed", type=int, default=20260709)
    p.add_argument("--pool-multiplier", type=float, default=1.8)
    p.add_argument("--max-attempts-per-size", type=int, default=20000)
    p.add_argument("--canvas-size", type=int, default=1024)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    build_benchmark(parse_args())
