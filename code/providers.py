from __future__ import annotations

import base64
import os
import urllib.request
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Protocol

from common import image_data_url, image_mime


class ImageGenerator(Protocol):
    provider: str
    model: str

    def generate(self, input_images: list[Path], prompt: str) -> tuple[bytes, str | None, dict[str, Any]]:
        ...


class OpenAIImageGenerator:
    provider = "openai"

    def __init__(self, model: str, size: str, quality: str, input_fidelity: str):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install dependencies: pip install -r requirements.txt") from exc
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        kwargs: dict[str, Any] = {"api_key": key}
        if os.getenv("OPENAI_BASE_URL"):
            kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]
        self.client = OpenAI(**kwargs)
        self.model = model
        self.size = size
        self.quality = quality
        self.input_fidelity = input_fidelity

    def generate(self, input_images: list[Path], prompt: str) -> tuple[bytes, str | None, dict[str, Any]]:
        if not input_images:
            response = self.client.images.generate(
                model=self.model,
                prompt=prompt,
                size=self.size,
                quality=self.quality,
                output_format="png",
            )
        else:
            with ExitStack() as stack:
                files = [stack.enter_context(p.open("rb")) for p in input_images]
                image_arg: Any = files[0] if len(files) == 1 else files
                response = self.client.images.edit(
                    model=self.model,
                    image=image_arg,
                    prompt=prompt,
                    size=self.size,
                    quality=self.quality,
                    input_fidelity=self.input_fidelity,
                    output_format="png",
                )
        if not response.data or not response.data[0].b64_json:
            raise RuntimeError("OpenAI returned no base64 image data")
        return (
            base64.b64decode(response.data[0].b64_json),
            None,
            {"size": self.size, "quality": self.quality, "input_fidelity": self.input_fidelity},
        )


class OpenAIImageGenerationGenerator:
    provider = "openai-generation"

    def __init__(self, model: str, size: str):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install dependencies: pip install -r requirements.txt") from exc
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        kwargs: dict[str, Any] = {"api_key": key}
        if os.getenv("OPENAI_BASE_URL"):
            kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]
        self.client = OpenAI(**kwargs)
        self.model = model
        self.size = size

    def generate(self, input_images: list[Path], prompt: str) -> tuple[bytes, str | None, dict[str, Any]]:
        extra_body: dict[str, Any] = {}
        if input_images:
            images = [image_data_url(p) for p in input_images]
            extra_body["image"] = images[0] if len(images) == 1 else images

        response = self.client.images.generate(
            model=self.model,
            prompt=prompt,
            size=self.size,
            response_format="url",
            extra_body=extra_body or None,
        )
        if not response.data:
            raise RuntimeError("OpenAI-compatible generation endpoint returned no image data")
        item = response.data[0]
        if getattr(item, "b64_json", None):
            image_bytes = base64.b64decode(item.b64_json)
        elif getattr(item, "url", None):
            with urllib.request.urlopen(item.url, timeout=120) as resp:
                image_bytes = resp.read()
        else:
            raise RuntimeError("OpenAI-compatible generation endpoint returned neither image URL nor base64 data")
        return image_bytes, None, {"size": self.size, "endpoint": "images.generate"}


class GeminiImageGenerator:
    provider = "google"

    def __init__(self, model: str, aspect_ratio: str, image_size: str, explanation_mode: str):
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError("Install dependencies: pip install -r requirements.txt") from exc
        if not os.getenv("GEMINI_API_KEY"):
            raise RuntimeError("GEMINI_API_KEY is not set")
        self.client = genai.Client()
        self.model = model
        self.aspect_ratio = aspect_ratio
        self.image_size = image_size
        self.explanation_mode = explanation_mode

    def generate(self, input_images: list[Path], prompt: str) -> tuple[bytes, str | None, dict[str, Any]]:
        inputs: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for p in input_images:
            inputs.append({
                "type": "image",
                "data": base64.b64encode(p.read_bytes()).decode("ascii"),
                "mime_type": image_mime(p),
            })

        image_format = {
            "type": "image",
            "mime_type": "image/png",
            "aspect_ratio": self.aspect_ratio,
            "image_size": self.image_size,
        }
        response_format: Any = (
            [{"type": "text"}, image_format]
            if self.explanation_mode == "optional"
            else image_format
        )
        interaction = self.client.interactions.create(
            model=self.model,
            input=inputs,
            response_format=response_format,
        )
        if interaction.output_image is None or not interaction.output_image.data:
            raise RuntimeError("Gemini returned no image data")
        explanation = (interaction.output_text or "").strip() or None
        return (
            base64.b64decode(interaction.output_image.data),
            explanation,
            {"aspect_ratio": self.aspect_ratio, "image_size": self.image_size},
        )


class ArkSeedreamImageGenerator:
    provider = "ark"

    def __init__(self, model: str, size: str, watermark: bool, base_url: str | None):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install dependencies: pip install -r requirements.txt") from exc
        key = os.getenv("ARK_API_KEY")
        if not key:
            raise RuntimeError("ARK_API_KEY is not set")
        self.client = OpenAI(
            api_key=key,
            base_url=base_url or os.getenv("ARK_BASE_URL") or "https://ark.cn-beijing.volces.com/api/v3",
        )
        self.model = model
        self.size = size
        self.watermark = watermark

    def generate(self, input_images: list[Path], prompt: str) -> tuple[bytes, str | None, dict[str, Any]]:
        extra_body: dict[str, Any] = {"watermark": self.watermark}
        if input_images:
            images = [image_data_url(p) for p in input_images]
            extra_body["image"] = images[0] if len(images) == 1 else images

        response = self.client.images.generate(
            model=self.model,
            prompt=prompt,
            size=self.size,
            response_format="url",
            extra_body=extra_body,
        )
        if not response.data:
            raise RuntimeError("Ark returned no image data")

        item = response.data[0]
        if getattr(item, "b64_json", None):
            image_bytes = base64.b64decode(item.b64_json)
        elif getattr(item, "url", None):
            with urllib.request.urlopen(item.url, timeout=120) as resp:
                image_bytes = resp.read()
        else:
            raise RuntimeError("Ark returned neither image URL nor base64 data")
        return image_bytes, None, {"size": self.size, "watermark": self.watermark}


def create_generator(args: Any) -> ImageGenerator:
    if args.provider == "openai":
        return OpenAIImageGenerator(
            model=args.model or "gpt-image-2",
            size=args.openai_size,
            quality=args.openai_quality,
            input_fidelity=args.openai_input_fidelity,
        )
    if args.provider == "openai-generation":
        return OpenAIImageGenerationGenerator(
            model=args.model or "flux.1-kontext-pro",
            size=args.openai_size,
        )
    if args.provider == "google":
        return GeminiImageGenerator(
            model=args.model or "gemini-3.1-flash-image",
            aspect_ratio=args.gemini_aspect_ratio,
            image_size=args.gemini_image_size,
            explanation_mode=args.explanation_mode,
        )
    if args.provider == "ark":
        return ArkSeedreamImageGenerator(
            model=args.model or "doubao-seedream-5-0-pro-260628",
            size=args.ark_size,
            watermark=args.ark_watermark,
            base_url=args.ark_base_url,
        )
    raise ValueError(f"Unknown provider: {args.provider}")
