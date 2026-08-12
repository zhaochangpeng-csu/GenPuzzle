from __future__ import annotations

import base64
import json
import mimetypes
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar

T = TypeVar("T")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected object at {path}:{line_no}")
            rows.append(obj)
    return rows


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
        f.flush()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def image_mime(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    return mime or "image/png"


def image_data_url(path: Path) -> str:
    payload, mime = image_payload_for_eval(path)
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def image_payload_for_eval(path: Path) -> tuple[bytes, str]:
    return path.read_bytes(), image_mime(path)


def resolve_asset(dataset_root: Path, relative_path: str) -> Path:
    root = dataset_root.resolve()
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Asset escapes dataset root: {relative_path}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"Missing asset: {path}")
    return path


def call_with_retry(fn: Callable[[], T], *, max_retries: int, base_delay: float) -> T:
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt >= max_retries:
                break
            time.sleep(base_delay * (2**attempt) + random.uniform(0, max(0.05, 0.25 * base_delay)))
    assert last_exc is not None
    raise last_exc


def select_items(
    items: Iterable[dict[str, Any]], start_id: str | None, end_id: str | None,
    ids: set[str] | None, limit: int | None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in items:
        item_id = str(item.get("id", item.get("task_id", "")))
        if start_id and item_id < start_id:
            continue
        if end_id and item_id > end_id:
            continue
        if ids is not None and item_id not in ids:
            continue
        out.append(item)
        if limit is not None and len(out) >= limit:
            break
    return out


def latest_records(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if path.exists():
        for row in load_jsonl(path):
            latest[str(row["id"])] = row
    return latest


def successful_generation_ids(records_path: Path, image_dir: Path) -> set[str]:
    return {
        item_id for item_id, row in latest_records(records_path).items()
        if row.get("status") == "success" and (image_dir / f"{item_id}.png").is_file()
    }


def successful_ids(records_path: Path, image_dir: Path) -> set[str]:
    return successful_generation_ids(records_path, image_dir)


def _response_output_text(response: Any) -> str:
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


def _chat_output_text(response: Any) -> str:
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
            if isinstance(item, dict) and item.get("type") in {"text", "input_text"}:
                parts.append(str(item.get("text", "")))
        return "\n".join(parts)
    return str(content)


def _extract_json_object(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end >= start:
        return text[start : end + 1]
    return text


def _chat_content_from_responses_content(content: Any) -> Any:
    if isinstance(content, str):
        return content
    out: list[dict[str, Any]] = []
    for item in content or []:
        typ = item.get("type") if isinstance(item, dict) else None
        if typ in {"input_text", "text"}:
            out.append({"type": "text", "text": str(item.get("text", ""))})
        elif typ in {"input_image", "image_url"}:
            image_url = item.get("image_url")
            if isinstance(image_url, dict):
                out.append({"type": "image_url", "image_url": image_url})
            else:
                out.append({"type": "image_url", "image_url": {"url": image_url}})
    return out


def _chat_messages_from_responses_input(input_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for message in input_messages:
        messages.append({
            "role": message["role"],
            "content": _chat_content_from_responses_content(message.get("content")),
        })
    return messages


def _append_text_to_last_user(messages: list[dict[str, Any]], text: str) -> list[dict[str, Any]]:
    out = [dict(message) for message in messages]
    for message in reversed(out):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = content + "\n\n" + text
        elif isinstance(content, list):
            message["content"] = content + [{"type": "text", "text": text}]
        else:
            message["content"] = text
        break
    return out


def _append_input_text_to_last_user(messages: list[dict[str, Any]], text: str) -> list[dict[str, Any]]:
    out = [dict(message) for message in messages]
    for message in reversed(out):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = content + "\n\n" + text
        elif isinstance(content, list):
            message["content"] = content + [{"type": "input_text", "text": text}]
        else:
            message["content"] = text
        break
    return out


def _responses_compatible_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for message in messages:
        new_message = dict(message)
        content = new_message.get("content")
        if isinstance(content, list):
            new_content: list[dict[str, Any]] = []
            for item in content:
                if not isinstance(item, dict):
                    new_content.append(item)
                    continue
                new_item = dict(item)
                if new_item.get("type") in {"input_image", "image_url"}:
                    # Some OpenAI-compatible providers, including Ark, reject
                    # nonstandard detail values such as "original".
                    new_item.pop("detail", None)
                new_content.append(new_item)
            new_message["content"] = new_content
        out.append(new_message)
    return out


def _chat_extra_body() -> dict[str, Any] | None:
    extra: dict[str, Any] = {}
    enable_thinking = os.getenv("OPENAI_ENABLE_THINKING", "").strip().lower()
    if enable_thinking in {"1", "true", "yes", "on"}:
        extra["enable_thinking"] = True
    elif enable_thinking in {"0", "false", "no", "off"}:
        extra["enable_thinking"] = False

    budget = os.getenv("OPENAI_THINKING_BUDGET", "").strip()
    if budget:
        try:
            extra["thinking_budget"] = int(budget)
        except ValueError:
            pass
    return extra or None


def _schema_instruction(schema_name: str, schema: dict[str, Any]) -> str:
    return (
        "Return exactly one JSON object. It must match this JSON Schema, including all required keys. "
        "Do not use Markdown or omit nullable fields; use null when needed.\n"
        + json.dumps({"name": schema_name, "schema": schema}, ensure_ascii=False)
    )


def _schema_from_pydantic(model_cls: Any) -> dict[str, Any]:
    if hasattr(model_cls, "model_json_schema"):
        return model_cls.model_json_schema()
    return model_cls.schema()


def _coerce_pydantic_payload(obj: dict[str, Any], model_cls: Any) -> dict[str, Any]:
    fields = getattr(model_cls, "model_fields", None) or getattr(model_cls, "__fields__", {})
    if "reason" in fields and "reason" not in obj:
        obj["reason"] = str(obj.get("rationale") or obj.get("explanation") or obj.get("note") or "")
    if "review_reason" in fields and "review_reason" not in obj:
        obj["review_reason"] = None
    if "failure_tags" in fields and "failure_tags" not in obj:
        value = obj.get("error_tags") or obj.get("errors") or []
        obj["failure_tags"] = value if isinstance(value, list) else [str(value)]
    if "confidence" in fields and "confidence" not in obj:
        obj["confidence"] = 0.7
    if "needs_human_review" in fields and "needs_human_review" not in obj:
        obj["needs_human_review"] = False
    if "alternative_valid_solution" in fields and "alternative_valid_solution" not in obj:
        obj["alternative_valid_solution"] = False
    if "matches_reference_solution" in fields and "matches_reference_solution" not in obj:
        try:
            obj["matches_reference_solution"] = int(obj.get("score", 0)) == 3
        except Exception:
            obj["matches_reference_solution"] = False
    if "verdict" in fields and "verdict" not in obj:
        try:
            score = int(obj.get("score", 0))
        except Exception:
            score = 0
        obj["verdict"] = {
            3: "fully_correct",
            2: "mostly_correct",
            1: "partially_correct",
            0: "incorrect",
        }.get(score, "unjudgeable")
    return obj


def openai_json_call(
    client: Any,
    *,
    model: str,
    input_messages: list[dict[str, Any]],
    schema: dict[str, Any],
    schema_name: str,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    use_chat = os.getenv("OPENAI_JUDGE_API", "").lower() in {"chat", "chat_completions", "chat-completions"}
    schema_instruction = _schema_instruction(schema_name, schema)
    if not use_chat:
        try:
            response_input = _responses_compatible_input(input_messages)
            kwargs: dict[str, Any] = {"model": model, "input": response_input}
            kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                }
            }
            if reasoning_effort and reasoning_effort != "none":
                kwargs["reasoning"] = {"effort": reasoning_effort}
            response = client.responses.create(**kwargs)
            return json.loads(_extract_json_object(_response_output_text(response)))
        except Exception as exc:
            message = str(exc)
            fallback_markers = [
                "convert_request_failed",
                "not implemented",
                "/v1/responses",
                "responses",
            ]
            if not any(marker in message for marker in fallback_markers):
                raise

    messages = _append_text_to_last_user(_chat_messages_from_responses_input(input_messages), schema_instruction)
    use_strict_chat_schema = not model.lower().startswith("gemini")
    try:
        extra_body = _chat_extra_body()
        extra_kwargs = {"extra_body": extra_body} if extra_body else {}
        if use_strict_chat_schema:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": schema_name, "schema": schema, "strict": True},
                },
                **extra_kwargs,
            )
        else:
            response = client.chat.completions.create(model=model, messages=messages, **extra_kwargs)
    except Exception:
        messages = messages.copy()
        messages[-1] = {
            **messages[-1],
            "content": _chat_content_from_responses_content(input_messages[-1].get("content")) + [
                {"type": "text", "text": "\nReturn strict JSON only. Do not use Markdown."}
            ],
        }
        extra_body = _chat_extra_body()
        extra_kwargs = {"extra_body": extra_body} if extra_body else {}
        response = client.chat.completions.create(model=model, messages=messages, **extra_kwargs)
    return json.loads(_extract_json_object(_chat_output_text(response)))


def openai_pydantic_call(
    client: Any,
    *,
    model: str,
    input_messages: list[dict[str, Any]],
    output_model: Any,
    schema_name: str,
    reasoning_effort: str | None = None,
) -> Any:
    use_chat = os.getenv("OPENAI_JUDGE_API", "").lower() in {"chat", "chat_completions", "chat-completions"}
    if not use_chat:
        try:
            kwargs: dict[str, Any] = {"model": model, "input": input_messages, "text_format": output_model, "store": False}
            if reasoning_effort and reasoning_effort != "none":
                kwargs["reasoning"] = {"effort": reasoning_effort}
            response = client.responses.parse(**kwargs)
            result = response.output_parsed
            if result is None:
                raise RuntimeError("Judge returned no parsed result")
            return result
        except Exception as exc:
            message = str(exc)
            fallback_markers = [
                "convert_request_failed",
                "not implemented",
                "/v1/responses",
                "responses",
            ]
            if not any(marker in message for marker in fallback_markers):
                raise

    obj = openai_json_call(
        client,
        model=model,
        input_messages=input_messages,
        schema=_schema_from_pydantic(output_model),
        schema_name=schema_name,
        reasoning_effort=None,
    )
    obj = _coerce_pydantic_payload(obj, output_model)
    if hasattr(output_model, "model_validate"):
        return output_model.model_validate(obj)
    return output_model.parse_obj(obj)
