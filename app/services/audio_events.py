"""Audio event inference through the configured Gradio SED Space.

Audio labels are stored as supporting observations only. They do not affect
routing or severity because the Space's coverage/performance is not verified
for every civic sound class.
"""
import asyncio
import json
import tempfile
import time
from pathlib import Path
from threading import Lock
from typing import Any

from app.core.config import get_settings

_client = None
_client_lock = Lock()


def _get_client(space: str):
    global _client
    with _client_lock:
        if _client is None:
            from gradio_client import Client
            _client = Client(space, verbose=False)
        return _client


def _clean_output(value: Any, depth: int = 0):
    """Keep JSON-safe model output while excluding remote/local file handles."""
    if depth > 5:
        return None
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("/tmp/", "/home/", "file=", "http://", "https://")):
            return "[remote visual output omitted]"
        return value[:5000]
    if isinstance(value, (list, tuple)):
        return [_clean_output(item, depth + 1) for item in value[:200]]
    if isinstance(value, dict):
        return {str(key): _clean_output(item, depth + 1) for key, item in list(value.items())[:100]
                if str(key).lower() not in {"path", "url", "filepath", "name"}}
    # Avoid serializing SDK objects whose repr can include temporary file paths.
    return {"type": type(value).__name__}


def _extract_tags(value: Any) -> list[str]:
    if isinstance(value, dict):
        for key in ("events", "detections", "tags", "labels", "sounds"):
            found = value.get(key)
            if found is not None:
                return _extract_tags(found)
        label = value.get("label") or value.get("class") or value.get("sound")
        return [str(label)[:160]] if label else []
    if isinstance(value, (list, tuple)):
        labels = []
        for item in value:
            labels.extend(_extract_tags(item))
        return list(dict.fromkeys(labels))[:100]
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
            if decoded != value:
                return _extract_tags(decoded)
        except (json.JSONDecodeError, TypeError):
            pass
        # Preserve the Space's plain-text class rows without pretending the
        # labels are calibrated probabilities.
        return list(dict.fromkeys(line.strip(" -*\t")[:160] for line in value.splitlines()
                                  if line.strip()))[:100]
    return []


def _predict(audio: bytes, filename: str, space: str):
    from gradio_client import handle_file
    suffix = Path(filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix) as temp:
        temp.write(audio)
        temp.flush()
        client = _get_client(space)
        # Keep this in sync with the Space's published /predict controls.
        return client.predict(
            handle_file(temp.name),
            0.0,       # sensitivity offset
            2.0,       # window seconds
            0.25,      # hop seconds
            2,         # agreeing model families
            True,      # Silero VAD
            False,     # denoising
            True,      # discover AudioSet sounds
            "",        # extra CLAP categories
            api_name="/predict",
        )


async def detect_audio_events(audio: bytes, filename: str):
    settings = get_settings()
    space = settings.sed_space
    start = time.monotonic()
    try:
        result = await asyncio.wait_for(asyncio.to_thread(_predict, audio, filename, space), timeout=120)
        if not isinstance(result, (tuple, list)) or len(result) < 2:
            raise ValueError("SED returned an unexpected result")
        detection, timeline = _clean_output(result[0]), _clean_output(result[1])
        trace = {"provider": "gradio", "model": space, "task": "audio_events", "outcome": "ok",
                 "latency_ms": int((time.monotonic() - start) * 1000), "error_code": None}
        return _extract_tags(detection), detection, timeline, trace
    except Exception as exc:
        trace = {"provider": "gradio", "model": space, "task": "audio_events",
                 "outcome": "error", "latency_ms": int((time.monotonic() - start) * 1000),
                 "error_code": type(exc).__name__}
        return [], None, None, trace
