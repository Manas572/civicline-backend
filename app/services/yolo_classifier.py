"""Run the civic image classifier hosted as a public Gradio Space."""
import asyncio
import json
import re
import tempfile
import time
from pathlib import Path
from threading import Lock
from typing import Any

from app.core.config import get_settings

_client = None
_lock = Lock()


def _predict(image: bytes, mime: str, space: str):
    from gradio_client import Client, handle_file
    global _client
    with tempfile.NamedTemporaryFile(suffix=Path('image.' + mime.split('/')[-1]).suffix) as file:
        file.write(image)
        file.flush()
        with _lock:
            if _client is None:
                _client = Client(space, verbose=False)
            client = _client
        return client.predict(image=handle_file(file.name), api_name="/predict")


def _json_value(value: Any, depth=0):
    if depth > 5:
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value[:1000] if isinstance(value, str) else value
    if isinstance(value, dict):
        return {str(k): _json_value(v, depth + 1) for k, v in list(value.items())[:50]
                if str(k).lower() not in {"path", "url", "filepath", "name"}}
    if isinstance(value, (list, tuple)):
        return [_json_value(v, depth + 1) for v in value[:100]]
    return str(type(value).__name__)


def _label(value):
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if parsed != value:
                return _label(parsed)
        except (json.JSONDecodeError, TypeError):
            text = value.strip()
            if text.startswith(("/tmp/", "http://", "https://", "file=")):
                return None
            for label in ("bridge crack", "bridge_crack", "water leak", "water_leak", "streetlight", "street light", "pothole", "garbage", "trash"):
                if re.search(rf"\b{re.escape(label)}\b", text, re.I):
                    return label
            return text[:160]
    if isinstance(value, dict):
        for key in ("label", "class", "class_name", "predicted_class", "prediction"):
            if value.get(key) is not None:
                return _label(value[key])
        if value:
            return str(max(value, key=lambda k: float(value[k]) if isinstance(value[k], (int, float)) else -1))[:160]
    if isinstance(value, (list, tuple)) and value:
        for item in value:
            label = _label(item)
            if label:
                return label
    return None


async def classify_image(image: bytes, mime: str):
    settings = get_settings()
    start = time.monotonic()
    try:
        result = await asyncio.wait_for(asyncio.to_thread(_predict, image, mime, settings.yolo_space), timeout=90)
        label = _label(result)
        if not label:
            raise ValueError("classifier returned no class label")
        normalized = label.lower().replace(" ", "_").replace("-", "_")
        issue_aliases = {"pothole": "pothole", "water_leak": "water_leak", "leak": "water_leak",
                         "garbage": "garbage", "trash": "garbage", "streetlight": "streetlight",
                         "street_light": "streetlight", "bridge_crack": "bridge_crack", "bridge": "bridge_crack"}
        return {"label": label, "issue_type": issue_aliases.get(normalized), "raw": _json_value(result)}, {
            "provider": "gradio", "model": settings.yolo_space, "task": "yolo_classification",
            "outcome": "ok", "latency_ms": int((time.monotonic() - start) * 1000), "error_code": None,
        }
    except Exception as exc:
        return None, {"provider": "gradio", "model": settings.yolo_space, "task": "yolo_classification",
                      "outcome": "error", "latency_ms": int((time.monotonic() - start) * 1000),
                      "error_code": type(exc).__name__}
