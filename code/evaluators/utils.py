import base64
import json
import mimetypes
from pathlib import Path
from typing import Any

from common import image_payload_for_eval


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return records


def select_records(
    records: list[dict[str, Any]],
    limit: int | None = None,
    sample_every: int | None = None,
    sample_offset: int = 0,
) -> list[dict[str, Any]]:
    if sample_every is not None:
        if sample_every <= 0:
            raise ValueError("--sample-every must be a positive integer")
        if sample_offset < 0 or sample_offset >= sample_every:
            raise ValueError("--sample-offset must satisfy 0 <= offset < sample_every")
        records = [record for index, record in enumerate(records) if index % sample_every == sample_offset]

    if limit is not None:
        if limit < 0:
            raise ValueError("--limit must be non-negative")
        records = records[:limit]

    return records


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def image_to_data_url(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    payload, mime_type = image_payload_for_eval(path)
    if mime_type is None:
        mime_type, _ = mimetypes.guess_type(str(path))
    if mime_type is None:
        mime_type = "image/png"

    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def decode_b64_image_to_file(b64_image: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(base64.b64decode(b64_image))


def resolve_path(dataset_root: Path, path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (dataset_root / path).resolve()


def get_first_image_path(record: dict[str, Any], key: str, dataset_root: Path) -> Path:
    images = record.get(key)
    if not isinstance(images, list) or not images:
        raise ValueError(f"{record.get('task_id', '<unknown>')} has no {key}[0]")

    path_value = images[0].get("path")
    if not path_value:
        raise ValueError(f"{record.get('task_id', '<unknown>')} has no {key}[0].path")

    return resolve_path(dataset_root, str(path_value))


def safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)


def infer_dataset_root(dataset_path: Path, explicit_root: str | None) -> Path:
    if explicit_root:
        return Path(explicit_root).resolve()
    if dataset_path.parent.name.lower() == "data":
        return dataset_path.parent.parent.resolve()
    return dataset_path.parent.resolve()
