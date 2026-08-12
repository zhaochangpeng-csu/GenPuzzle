from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw
from shapely.affinity import rotate as shp_rotate, translate as shp_translate
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union


PIECE_ORDER = [
    "large_triangle_1",
    "large_triangle_2",
    "medium_triangle",
    "small_triangle_1",
    "small_triangle_2",
    "square",
    "parallelogram",
]

# Standard seven-piece tangram with total area 8 in arbitrary geometric units.
PIECE_VERTICES: dict[str, list[tuple[float, float]]] = {
    "large_triangle_1": [(0, 0), (2, 0), (0, 2)],
    "large_triangle_2": [(0, 0), (2, 0), (0, 2)],
    "medium_triangle": [(0, 0), (1, 1), (0, 2)],
    "small_triangle_1": [(0, 0), (1, 0), (0, 1)],
    "small_triangle_2": [(0, 0), (1, 0), (0, 1)],
    "square": [(0, 0), (1, 0), (1, 1), (0, 1)],
    "parallelogram": [(0, 0), (1, 0), (2, 1), (1, 1)],
}

PIECE_TYPES = {
    "large_triangle_1": "large_triangle",
    "large_triangle_2": "large_triangle",
    "medium_triangle": "medium_triangle",
    "small_triangle_1": "small_triangle",
    "small_triangle_2": "small_triangle",
    "square": "square",
    "parallelogram": "parallelogram",
}

PIECE_COLORS: dict[str, tuple[int, int, int]] = {
    "large_triangle_1": (225, 87, 89),
    "large_triangle_2": (78, 121, 167),
    "medium_triangle": (89, 161, 79),
    "small_triangle_1": (242, 142, 43),
    "small_triangle_2": (237, 201, 72),
    "square": (176, 122, 161),
    "parallelogram": (118, 183, 178),
}

EQUIVALENT_PIECE_GROUPS = [
    ["large_triangle_1", "large_triangle_2"],
    ["small_triangle_1", "small_triangle_2"],
]


@dataclass
class PlacedPiece:
    name: str
    polygon: Polygon
    angle_deg: float


@dataclass
class Candidate:
    seed: int
    placed: list[PlacedPiece]
    silhouette: Polygon
    canonical_hash: str
    metrics: dict[str, float]
    difficulty_score: float = 0.0
    difficulty: str = ""


def polygon_edges(poly: Polygon) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    coords = list(poly.exterior.coords)[:-1]
    return [(coords[i], coords[(i + 1) % len(coords)]) for i in range(len(coords))]


def edge_length(edge: tuple[tuple[float, float], tuple[float, float]]) -> float:
    (x0, y0), (x1, y1) = edge
    return math.hypot(x1 - x0, y1 - y0)


def align_edge(
    poly: Polygon,
    source_edge: tuple[tuple[float, float], tuple[float, float]],
    target_edge: tuple[tuple[float, float], tuple[float, float]],
    reverse: bool,
) -> tuple[Polygon, float]:
    q0, q1 = source_edge
    p0, p1 = target_edge
    if reverse:
        p0, p1 = p1, p0
    aq = math.atan2(q1[1] - q0[1], q1[0] - q0[0])
    ap = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    angle = math.degrees(ap - aq)
    rotated = shp_rotate(poly, angle, origin=q0, use_radians=False)
    return shp_translate(rotated, p0[0] - q0[0], p0[1] - q0[1]), angle


def exposed_edges(placed: list[PlacedPiece], union: Polygon) -> list[tuple[str, tuple[tuple[float, float], tuple[float, float]]]]:
    boundary = union.boundary
    out: list[tuple[str, tuple[tuple[float, float], tuple[float, float]]]] = []
    for piece in placed:
        for edge in polygon_edges(piece.polygon):
            mx = (edge[0][0] + edge[1][0]) / 2
            my = (edge[0][1] + edge[1][1]) / 2
            if boundary.distance(Point(mx, my)) < 1e-7:
                out.append((piece.name, edge))
    return out


def generate_arrangement(seed: int, max_branch_candidates: int = 5000) -> tuple[list[PlacedPiece], Polygon] | None:
    rng = random.Random(seed)
    first_name = rng.choice(["large_triangle_1", "large_triangle_2", "medium_triangle", "square", "parallelogram"])
    first_angle = rng.choice(range(0, 360, 45))
    first_poly = shp_rotate(Polygon(PIECE_VERTICES[first_name]), first_angle, origin=(0, 0), use_radians=False)
    placed = [PlacedPiece(first_name, first_poly, float(first_angle))]
    unplaced = [name for name in PIECE_ORDER if name != first_name]

    for _ in range(6):
        union = unary_union([p.polygon for p in placed])
        if union.geom_type != "Polygon":
            return None
        exp = exposed_edges(placed, union)
        candidates: list[tuple[str, Polygon, float, float]] = []

        for _, target_edge in exp:
            target_len = edge_length(target_edge)
            for name in unplaced:
                base = Polygon(PIECE_VERTICES[name])
                for source_edge in polygon_edges(base):
                    if abs(edge_length(source_edge) - target_len) > 1e-6:
                        continue
                    for reverse in (False, True):
                        candidate_poly, angle = align_edge(base, source_edge, target_edge, reverse)
                        if candidate_poly.intersection(union).area > 1e-7:
                            continue
                        shared = candidate_poly.boundary.intersection(union.boundary).length
                        if shared < min(0.80, target_len * 0.80):
                            continue
                        merged = unary_union([union, candidate_poly])
                        if merged.geom_type != "Polygon" or len(merged.interiors) != 0:
                            continue
                        candidates.append((name, candidate_poly, angle, shared))
                        if len(candidates) >= max_branch_candidates:
                            break
                    if len(candidates) >= max_branch_candidates:
                        break
                if len(candidates) >= max_branch_candidates:
                    break
            if len(candidates) >= max_branch_candidates:
                break

        if not candidates:
            return None

        # Mix compact and exploratory choices to create a broad silhouette distribution.
        if rng.random() < 0.45:
            scored: list[tuple[float, tuple[str, Polygon, float, float]]] = []
            for cand in candidates:
                merged = unary_union([union, cand[1]])
                compactness = 4 * math.pi * merged.area / max(1e-9, merged.length ** 2)
                scored.append((compactness + rng.random() * 0.08, cand))
            scored.sort(key=lambda x: x[0], reverse=True)
            name, poly, angle, _ = rng.choice(scored[: max(1, min(12, len(scored)))])[1]
        else:
            name, poly, angle, _ = rng.choice(candidates)

        placed.append(PlacedPiece(name, poly, angle % 360.0))
        unplaced.remove(name)

    silhouette = unary_union([p.polygon for p in placed])
    if silhouette.geom_type != "Polygon" or len(silhouette.interiors) != 0:
        return None
    return placed, silhouette


def rasterize_polygon(poly: Polygon, size: int = 128, margin: int = 8) -> np.ndarray:
    minx, miny, maxx, maxy = poly.bounds
    w = maxx - minx
    h = maxy - miny
    if w <= 0 or h <= 0:
        return np.zeros((size, size), dtype=np.uint8)
    scale = min((size - 2 * margin) / w, (size - 2 * margin) / h)
    ox = margin + ((size - 2 * margin) - w * scale) / 2
    oy = margin + ((size - 2 * margin) - h * scale) / 2
    pts = [(ox + (x - minx) * scale, oy + (maxy - y) * scale) for x, y in list(poly.exterior.coords)[:-1]]
    img = Image.new("1", (size, size), 0)
    ImageDraw.Draw(img).polygon(pts, fill=1)
    return np.array(img, dtype=np.uint8)


def canonical_silhouette_hash(poly: Polygon) -> str:
    variants: list[bytes] = []
    center = poly.centroid
    for angle in range(0, 360, 45):
        rotated = shp_rotate(poly, angle, origin=center, use_radians=False)
        arr = rasterize_polygon(rotated, 128, 8)
        variants.append(arr.tobytes())
        # Reflection is considered duplicate for dataset diversity even though generation does not require reflection.
        variants.append(np.fliplr(arr).tobytes())
    canonical = min(variants)
    return hashlib.sha256(canonical).hexdigest()


def concavity_count(poly: Polygon) -> int:
    p = poly.simplify(1e-7, preserve_topology=True)
    coords = list(p.exterior.coords)[:-1]
    if len(coords) < 4:
        return 0
    ccw = p.exterior.is_ccw
    count = 0
    for i in range(len(coords)):
        a = coords[i - 1]
        b = coords[i]
        c = coords[(i + 1) % len(coords)]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(cross) < 1e-8:
            continue
        if (ccw and cross < 0) or ((not ccw) and cross > 0):
            count += 1
    return count


def symmetry_score(poly: Polygon) -> float:
    arr = rasterize_polygon(poly, 128, 8).astype(bool)
    variants = [np.fliplr(arr), np.flipud(arr), np.rot90(arr, 2)]
    scores = []
    for v in variants:
        inter = np.logical_and(arr, v).sum()
        union = np.logical_or(arr, v).sum()
        scores.append(float(inter / union) if union else 1.0)
    return max(scores)


def pair_contact_count(placed: list[PlacedPiece]) -> int:
    count = 0
    for i in range(len(placed)):
        for j in range(i + 1, len(placed)):
            shared = placed[i].polygon.boundary.intersection(placed[j].polygon.boundary).length
            if shared > 0.18:
                count += 1
    return count


def candidate_metrics(placed: list[PlacedPiece], silhouette: Polygon) -> dict[str, float]:
    minx, miny, maxx, maxy = silhouette.bounds
    w, h = maxx - minx, maxy - miny
    aspect = max(w / max(h, 1e-9), h / max(w, 1e-9))
    compactness = 4 * math.pi * silhouette.area / max(1e-9, silhouette.length ** 2)
    hull_fill = silhouette.area / max(1e-9, silhouette.convex_hull.area)
    vertex_count = len(list(silhouette.simplify(1e-7, preserve_topology=True).exterior.coords)) - 1
    concavities = concavity_count(silhouette)
    symmetry = symmetry_score(silhouette)
    orientations = {int(round((p.angle_deg % 180) / 45.0)) % 4 for p in placed}
    contacts = pair_contact_count(placed)
    return {
        "aspect_ratio": float(aspect),
        "compactness": float(compactness),
        "hull_fill": float(hull_fill),
        "outline_vertex_count": float(vertex_count),
        "concavity_count": float(concavities),
        "symmetry_score": float(symmetry),
        "orientation_diversity": float(len(orientations)),
        "piece_contact_count": float(contacts),
    }


def candidate_is_reasonable(metrics: dict[str, float]) -> bool:
    return (
        metrics["aspect_ratio"] <= 2.85
        and metrics["compactness"] >= 0.13
        and metrics["hull_fill"] >= 0.42
        and metrics["outline_vertex_count"] >= 5
        and metrics["piece_contact_count"] >= 6
    )


def minmax(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def assign_difficulty_scores(candidates: list[Candidate]) -> None:
    fields = {
        "vertex": minmax([c.metrics["outline_vertex_count"] for c in candidates]),
        "concavity": minmax([c.metrics["concavity_count"] for c in candidates]),
        "noncompact": minmax([1.0 - c.metrics["compactness"] for c in candidates]),
        "hull_gap": minmax([1.0 - c.metrics["hull_fill"] for c in candidates]),
        "asymmetry": minmax([1.0 - c.metrics["symmetry_score"] for c in candidates]),
        "orientation": minmax([c.metrics["orientation_diversity"] for c in candidates]),
        "sparse_contacts": minmax([-c.metrics["piece_contact_count"] for c in candidates]),
        "aspect": minmax([math.log(max(1.0, c.metrics["aspect_ratio"])) for c in candidates]),
    }
    for i, c in enumerate(candidates):
        c.difficulty_score = round(
            0.24 * fields["vertex"][i]
            + 0.22 * fields["concavity"][i]
            + 0.12 * fields["noncompact"][i]
            + 0.10 * fields["hull_gap"][i]
            + 0.10 * fields["asymmetry"][i]
            + 0.08 * fields["orientation"][i]
            + 0.07 * fields["sparse_contacts"][i]
            + 0.07 * fields["aspect"][i],
            6,
        )


def spread_select(items: list[Candidate], count: int) -> list[Candidate]:
    if count >= len(items):
        return list(items)
    if count <= 1:
        return [items[len(items) // 2]] if count else []
    indices = [round(i * (len(items) - 1) / (count - 1)) for i in range(count)]
    return [items[i] for i in indices]


def select_difficulty_split(candidates: list[Candidate], easy_n: int, medium_n: int, hard_n: int) -> list[Candidate]:
    ordered = sorted(candidates, key=lambda c: c.difficulty_score)
    n = len(ordered)
    easy_pool = ordered[: max(easy_n, int(n * 0.34))]
    hard_pool = ordered[min(n - hard_n, int(n * 0.66)) :]
    mid_lo = int(n * 0.22)
    mid_hi = max(mid_lo + medium_n, int(n * 0.78))
    medium_pool = ordered[mid_lo:mid_hi]

    selected: list[Candidate] = []
    for c in spread_select(easy_pool, easy_n):
        c.difficulty = "easy"
        selected.append(c)
    for c in spread_select(medium_pool, medium_n):
        c.difficulty = "medium"
        selected.append(c)
    for c in spread_select(hard_pool, hard_n):
        c.difficulty = "hard"
        selected.append(c)

    # Defensive uniqueness if overlapping pools selected the same candidate.
    unique: dict[str, Candidate] = {}
    for c in selected:
        unique[c.canonical_hash] = c
    if len(unique) != len(selected):
        used = set(unique)
        result = list(unique.values())
        targets = {"easy": easy_n, "medium": medium_n, "hard": hard_n}
        counts = {k: sum(c.difficulty == k for c in result) for k in targets}
        for difficulty, target in targets.items():
            pool = ordered if difficulty == "medium" else (ordered if difficulty == "easy" else list(reversed(ordered)))
            for c in pool:
                if counts[difficulty] >= target:
                    break
                if c.canonical_hash in used:
                    continue
                c.difficulty = difficulty
                result.append(c)
                used.add(c.canonical_hash)
                counts[difficulty] += 1
        selected = result

    order = {"easy": 0, "medium": 1, "hard": 2}
    selected.sort(key=lambda c: (order[c.difficulty], c.difficulty_score, c.canonical_hash))
    return selected


def transform_points(poly: Polygon, scale: float, ox: float, oy: float, maxy: float | None = None) -> list[tuple[float, float]]:
    if maxy is None:
        return [(ox + x * scale, oy - y * scale) for x, y in list(poly.exterior.coords)[:-1]]
    return [(ox + x * scale, oy + (maxy - y) * scale) for x, y in list(poly.exterior.coords)[:-1]]


def render_candidate(candidate: Candidate, out_root: Path, item_id: str, canvas_size: int, scatter_seed: int) -> dict[str, Any]:
    silhouette = candidate.silhouette
    minx, miny, maxx, maxy = silhouette.bounds
    w, h = maxx - minx, maxy - miny
    scale = min(72.0, 490.0 / max(w, 1e-9), 500.0 / max(h, 1e-9))
    top_box = (80, 35, canvas_size - 80, 575)
    target_w, target_h = w * scale, h * scale
    target_left = (canvas_size - target_w) / 2
    target_top = top_box[1] + (top_box[3] - top_box[1] - target_h) / 2

    def to_target(poly: Polygon) -> list[tuple[float, float]]:
        return [
            (target_left + (x - minx) * scale, target_top + (maxy - y) * scale)
            for x, y in list(poly.exterior.coords)[:-1]
        ]

    question = Image.new("RGB", (canvas_size, canvas_size), "white")
    qdraw = ImageDraw.Draw(question)
    qdraw.polygon(to_target(silhouette), fill=(224, 224, 224), outline=(20, 20, 20), width=5)

    # Fixed tray slots make piece identity legible while randomizing assignment/orientation/jitter.
    rng = random.Random(scatter_seed)
    slots = [
        (145, 710), (375, 710), (645, 710), (875, 710),
        (225, 900), (510, 900), (800, 900),
    ]
    rng.shuffle(slots)
    piece_question_vertices: dict[str, list[list[float]]] = {}
    for name, (cx, cy) in zip(PIECE_ORDER, slots):
        base = Polygon(PIECE_VERTICES[name])
        angle = rng.choice(range(0, 360, 45))
        p = shp_rotate(base, angle, origin=base.centroid, use_radians=False)
        bx0, by0, bx1, by1 = p.bounds
        pw, ph = (bx1 - bx0) * scale, (by1 - by0) * scale
        jitter_x = rng.randint(-12, 12)
        jitter_y = rng.randint(-8, 8)
        left = cx - pw / 2 + jitter_x
        top = cy - ph / 2 + jitter_y
        pts = [(left + (x - bx0) * scale, top + (by1 - y) * scale) for x, y in list(p.exterior.coords)[:-1]]
        qdraw.polygon(pts, fill=PIECE_COLORS[name], outline=(18, 18, 18), width=4)
        piece_question_vertices[name] = [[round(x, 3), round(y, 3)] for x, y in pts]

    answer = Image.new("RGB", (canvas_size, canvas_size), "white")
    adraw = ImageDraw.Draw(answer)
    for piece in candidate.placed:
        pts = to_target(piece.polygon)
        adraw.polygon(pts, fill=PIECE_COLORS[piece.name], outline=(18, 18, 18), width=4)

    mask = Image.new("L", (canvas_size, canvas_size), 0)
    mdraw = ImageDraw.Draw(mask)
    mdraw.polygon(to_target(silhouette), fill=255)

    question_path = out_root / "questions" / f"{item_id}.png"
    answer_path = out_root / "answers" / f"{item_id}.png"
    mask_path = out_root / "masks" / f"{item_id}.png"
    question.save(question_path)
    answer.save(answer_path)
    mask.save(mask_path)

    piece_solution_vertices = {
        p.name: [[round(x, 3), round(y, 3)] for x, y in to_target(p.polygon)]
        for p in candidate.placed
    }
    target_bbox = mask.getbbox() or (0, 0, 0, 0)
    return {
        "canvas_size": canvas_size,
        "target_bbox": [int(x) for x in target_bbox],
        "tray_y_min": 610,
        "scale_px_per_unit": round(scale, 6),
        "piece_question_vertices": piece_question_vertices,
        "piece_solution_vertices": piece_solution_vertices,
    }


def build_contact_sheet(dataset_root: Path, selected: list[Candidate], out_path: Path) -> None:
    by_diff = {d: [] for d in ("easy", "medium", "hard")}
    for idx, c in enumerate(selected, 1):
        if len(by_diff[c.difficulty]) < 4:
            by_diff[c.difficulty].append(f"{idx:06d}")
    ids = by_diff["easy"] + by_diff["medium"] + by_diff["hard"]
    thumb_w, thumb_h = 250, 250
    sheet = Image.new("RGB", (thumb_w * 4, thumb_h * 6), "white")
    draw = ImageDraw.Draw(sheet)
    for j, item_id in enumerate(ids):
        q = Image.open(dataset_root / "questions" / f"{item_id}.png").convert("RGB").resize((thumb_w, thumb_h - 22))
        a = Image.open(dataset_root / "answers" / f"{item_id}.png").convert("RGB").resize((thumb_w, thumb_h - 22))
        col = j % 4
        row_base = (j // 4) * 2
        sheet.paste(q, (col * thumb_w, row_base * thumb_h))
        sheet.paste(a, (col * thumb_w, (row_base + 1) * thumb_h))
        draw.text((col * thumb_w + 4, row_base * thumb_h + thumb_h - 20), f"{item_id} question", fill="black")
        draw.text((col * thumb_w + 4, (row_base + 1) * thumb_h + thumb_h - 20), f"{item_id} answer", fill="black")
    sheet.save(out_path, quality=90)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="One-click generator for a 150-item Tangram benchmark.")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--count", type=int, default=150)
    p.add_argument("--seed", type=int, default=20260709)
    p.add_argument("--pool-multiplier", type=float, default=4.0)
    p.add_argument("--max-attempts", type=int, default=12000)
    p.add_argument("--canvas-size", type=int, default=1024)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.count < 3:
        raise ValueError("--count must be >= 3")
    out_root = args.output.resolve()
    if out_root.exists() and any(out_root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output is not empty: {out_root}. Use --overwrite.")
        shutil.rmtree(out_root)
    for sub in ("questions", "answers", "masks", "geometry"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    easy_n = round(args.count * 0.30)
    medium_n = round(args.count * 0.40)
    hard_n = args.count - easy_n - medium_n
    target_pool = max(args.count, math.ceil(args.count * args.pool_multiplier))

    candidates: list[Candidate] = []
    seen: set[str] = set()
    attempts = 0
    seed_rng = random.Random(args.seed)
    while len(candidates) < target_pool and attempts < args.max_attempts:
        candidate_seed = seed_rng.randrange(1, 2**31 - 1)
        attempts += 1
        generated = generate_arrangement(candidate_seed)
        if generated is None:
            continue
        placed, silhouette = generated
        metrics = candidate_metrics(placed, silhouette)
        if not candidate_is_reasonable(metrics):
            continue
        h = canonical_silhouette_hash(silhouette)
        if h in seen:
            continue
        seen.add(h)
        candidates.append(Candidate(candidate_seed, placed, silhouette, h, metrics))
        if len(candidates) % 50 == 0:
            print(f"accepted {len(candidates)}/{target_pool} unique candidates after {attempts} attempts")

    if len(candidates) < args.count:
        raise RuntimeError(f"Only generated {len(candidates)} valid unique candidates after {attempts} attempts")

    assign_difficulty_scores(candidates)
    selected = select_difficulty_split(candidates, easy_n, medium_n, hard_n)
    if len(selected) != args.count:
        raise RuntimeError(f"Selection produced {len(selected)} items, expected {args.count}")

    data_rows: list[dict[str, Any]] = []
    meta_rows: list[dict[str, Any]] = []
    for idx, candidate in enumerate(selected, 1):
        item_id = f"{idx:06d}"
        render_meta = render_candidate(candidate, out_root, item_id, args.canvas_size, args.seed ^ candidate.seed)
        geometry = {
            "id": item_id,
            "pieces": [
                {
                    "id": p.name,
                    "type": PIECE_TYPES[p.name],
                    "color_rgb": list(PIECE_COLORS[p.name]),
                    "canonical_vertices": [[float(x), float(y)] for x, y in PIECE_VERTICES[p.name]],
                    "solution_vertices": [[round(float(x), 8), round(float(y), 8)] for x, y in list(p.polygon.exterior.coords)[:-1]],
                    "solution_angle_deg": round(p.angle_deg % 360.0, 6),
                }
                for p in candidate.placed
            ],
            "target_outline": [[round(float(x), 8), round(float(y), 8)] for x, y in list(candidate.silhouette.exterior.coords)[:-1]],
            "total_area": round(candidate.silhouette.area, 8),
        }
        (out_root / "geometry" / f"{item_id}.json").write_text(
            json.dumps(geometry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        data_rows.append({"id": item_id, "image": f"questions/{item_id}.png", "answer": f"answers/{item_id}.png"})
        meta_rows.append({
            "id": item_id,
            "difficulty": candidate.difficulty,
            "candidate_seed": candidate.seed,
            "canonical_hash": candidate.canonical_hash,
            "difficulty_score": candidate.difficulty_score,
            "metrics": {k: round(v, 6) for k, v in candidate.metrics.items()},
            "target_mask": f"masks/{item_id}.png",
            "geometry": f"geometry/{item_id}.json",
            "allow_reflection": False,
            "equivalent_piece_groups": EQUIVALENT_PIECE_GROUPS,
            "piece_colors_rgb": {k: list(v) for k, v in PIECE_COLORS.items()},
            "render": render_meta,
        })

    with (out_root / "data.jsonl").open("w", encoding="utf-8") as f:
        for row in data_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out_root / "eval_meta.jsonl").open("w", encoding="utf-8") as f:
        for row in meta_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = {d: sum(r["difficulty"] == d for r in meta_rows) for d in ("easy", "medium", "hard")}
    summary = {
        "count": args.count,
        "difficulty_counts": counts,
        "seed": args.seed,
        "pool_multiplier": args.pool_multiplier,
        "candidate_pool_size": len(candidates),
        "attempts": attempts,
        "all_unique_canonical_hashes": len({r["canonical_hash"] for r in meta_rows}) == args.count,
        "expected_total_area": 8.0,
        "difficulty_score": {
            "min": min(r["difficulty_score"] for r in meta_rows),
            "median": statistics.median(r["difficulty_score"] for r in meta_rows),
            "max": max(r["difficulty_score"] for r in meta_rows),
        },
    }
    (out_root / "build_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    build_contact_sheet(out_root, selected, out_root / "sample_pairs.jpg")

    readme = f"""# Tangram Benchmark ({args.count} items)\n\n- Easy: {counts['easy']}\n- Medium: {counts['medium']}\n- Hard: {counts['hard']}\n- Every item uses the same seven standard Tangram pieces.\n- Candidate silhouettes are generated procedurally, filtered for connectivity/no holes, canonicalized under rotation/reflection for duplicate removal, and ranked by geometric complexity.\n- `data.jsonl` is public benchmark metadata.\n- `eval_meta.jsonl`, `masks/`, and `geometry/` are evaluation-only metadata and must never be sent to the generation model.\n- A reference answer is one known valid arrangement; alternative valid arrangements should also receive full credit.\n\nRegenerate from the integrated suite with:\n\n```bash\npython benchmark.py build-tangram --overwrite\n```\n"""
    (out_root / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
