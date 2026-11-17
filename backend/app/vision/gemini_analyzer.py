"""Image understanding and UI element classification through the Gemini API."""

import asyncio
import base64
from io import BytesIO
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from PIL import Image

from app.vision.openai_analyzer import _clean_result, _image_data_url


class GeminiAnalyzerError(RuntimeError):
    """Safe-to-display configuration or upstream API error."""


class GeminiAnalyzerConfigError(GeminiAnalyzerError):
    """Missing or invalid local configuration for the provider."""


_PROMPT_FILE = Path(__file__).parent / "prompts" / "sketch_analysis.txt"
_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "recognized_text": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "text": {"type": "STRING"},
                    "bbox": {"type": "ARRAY", "items": {"type": "INTEGER"}},
                    "confidence": {"type": "NUMBER"},
                },
                "required": ["text", "bbox", "confidence"],
            },
        },
        "shapes": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "shape_index": {"type": "INTEGER", "nullable": True},
                    "bbox": {"type": "ARRAY", "items": {"type": "INTEGER"}},
                    "type": {
                        "type": "STRING",
                        "enum": ["button", "input", "text", "image", "container"],
                    },
                    "text": {"type": "STRING"},
                    "input_type": {
                        "type": "STRING",
                        "enum": ["text", "email", "password", "search", "number", "tel", "url"],
                        "nullable": True,
                    },
                },
                "required": ["shape_index", "bbox", "type", "text", "input_type"],
            },
        },
    },
    "required": ["recognized_text", "shapes"],
}


def _send_request(payload, api_key, model):
    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{quote(model, safe='')}:generateContent"
    )
    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=90) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace").strip()
        if exc.code in {400, 401, 403}:
            message = f"Gemini API returned HTTP {exc.code}. Check the API key and model access."
            if detail:
                message = f"{message} API detail: {detail[:500]}"
            raise GeminiAnalyzerConfigError(
                message
            ) from exc
        raise GeminiAnalyzerError(f"Gemini API request failed with HTTP {exc.code}.") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise GeminiAnalyzerError("Could not reach Gemini API. Check internet access and try again.") from exc


def _response_text(response_data):
    try:
        for part in response_data["candidates"][0]["content"]["parts"]:
            if isinstance(part.get("text"), str):
                return part["text"]
    except (KeyError, IndexError, TypeError):
        pass
    raise GeminiAnalyzerError("Gemini returned no structured analysis. Try another image.")


async def analyze_with_gemini(image_bytes, image_format, image_width, image_height, candidate_boxes):
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise GeminiAnalyzerConfigError(
            "GEMINI_API_KEY is missing. Add it to the project .env file, then restart the backend."
        )

    model = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest").strip() or "gemini-flash-lite-latest"
    prompt = _PROMPT_FILE.read_text(encoding="utf-8")
    prompt = prompt.replace("{canvas_width}", str(image_width))
    prompt = prompt.replace("{canvas_height}", str(image_height))
    prompt = prompt.replace(
        "{candidate_boxes}", json.dumps(candidate_boxes, separators=(",", ":"))
    )
    data_url = _image_data_url(image_bytes, image_format)
    media_type, encoded = data_url[5:].split(";base64,", 1)
    payload = {
        "contents": [{
            "role": "user",
            "parts": [
                {"text": prompt},
                {"inlineData": {"mimeType": media_type, "data": encoded}},
            ],
        }],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": _RESPONSE_SCHEMA,
        },
    }
    response_data = await asyncio.to_thread(_send_request, payload, api_key, model)
    try:
        parsed = json.loads(_response_text(response_data))
    except json.JSONDecodeError as exc:
        raise GeminiAnalyzerError("Gemini returned invalid structured data. Try again.") from exc
    return _clean_result(parsed, image_width, image_height, len(candidate_boxes))