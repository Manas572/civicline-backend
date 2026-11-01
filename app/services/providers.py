import asyncio
import base64
import json
import time
from dataclasses import dataclass

import httpx

from app.core.config import get_settings
from app.schemas import VisionObservation

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_AUDIO_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
VISION_PROMPT = '''Inspect this civic report image and compare it with the citizen statement. Return JSON only with fields: issue_type (pothole|water_leak|garbage|streetlight|bridge_crack|other|not_civic), visual_evidence (visible facts only), visual_risk_factors (zero or more of: large_pothole, deep_pothole, road_collapse, bridge_crack, structural_damage, water_pooling, flooding, exposed_wire, live_wire, active_sparking, broken_streetlight, garbage_spread, faded_marking), pothole_size (small|medium|large|unknown), crack_extent (short|long|structural|unknown), water_pooling (none|minor|extensive|unknown), suggested_severity (integer 1-5), justification, hazards (visible hazards only), image_quality (good|poor|unusable), consistent_with_citizen_report (yes|no|unclear|no_report), model_confidence_uncalibrated (number or null). Estimate pothole size, visible crack extent, and water pooling qualitatively from the frame. Assess visible severity: distinguish a small/faded marking from a deep or large pothole, pooling/flooding from a dry stain, and a bridge crack/structural break from surface discoloration. Do not infer exact physical dimensions from a single image. Never put spoken or typed claims in visual_evidence or visual_risk_factors. Severity and confidence are preliminary observations; backend rules calculate the final score. Citizen statement is unverified and may be used only for consistency: {report}'''


@dataclass
class ProviderResult:
    observation: VisionObservation | None
    provider: str | None
    model: str | None
    traces: list[dict]


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("No JSON object returned")
    return json.loads(text[start:end + 1])


async def _gemini(image: bytes, mime: str, report: str | None, model: str) -> str:
    settings = get_settings()
    payload = {"contents": [{"parts": [
        {"text": VISION_PROMPT.format(report=report or "No citizen statement provided.")},
        {"inline_data": {"mime_type": mime, "data": base64.b64encode(image).decode("ascii")}},
    ]}], "generationConfig": {"responseMimeType": "application/json"}}
    # Keep synchronous report intake inside the public tunnel's request budget.
    async with httpx.AsyncClient(timeout=httpx.Timeout(15, connect=5)) as client:
        response = await client.post(GEMINI_URL.format(model=model), params={"key": settings.gemini_api_key}, json=payload)
        response.raise_for_status()
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]


async def _groq_vision(image: bytes, mime: str, report: str | None, model: str) -> str:
    settings = get_settings()
    image_url = f"data:{mime};base64,{base64.b64encode(image).decode('ascii')}"
    payload = {"model": model, "temperature": 0, "response_format": {"type": "json_object"}, "messages": [{
        "role": "user", "content": [
            {"type": "text", "text": VISION_PROMPT.format(report=report or "No citizen statement provided.")},
            {"type": "image_url", "image_url": {"url": image_url}},
        ],
    }]}
    async with httpx.AsyncClient(timeout=httpx.Timeout(15, connect=5)) as client:
        response = await client.post(GROQ_CHAT_URL, headers={"Authorization": f"Bearer {settings.groq_api_key}"}, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


async def _attempt(provider: str, model: str, image: bytes, mime: str, report: str | None):
    call = _gemini if provider == "gemini" else _groq_vision
    start = time.monotonic()
    error = "invalid_response"
    # Do not retry vision in the report request. Slow retries caused the
    # browser's tunneled request to time out before the ticket was saved.
    for retry in range(1):
        try:
            raw = await call(image, mime, report, model)
            observation = VisionObservation.model_validate(_parse_json(raw))
            trace = {"provider": provider, "model": model, "task": "vision", "outcome": "ok",
                     "latency_ms": int((time.monotonic() - start) * 1000), "error_code": None}
            return observation, trace
        except Exception as exc:
            error = type(exc).__name__
    return None, {"provider": provider, "model": model, "task": "vision", "outcome": "error",
                  "latency_ms": int((time.monotonic() - start) * 1000), "error_code": error}


async def observe_image(image: bytes, mime: str, report: str | None) -> ProviderResult:
    settings = get_settings()
    providers = []
    if settings.gemini_api_key:
        providers.append(("gemini", settings.gemini_vision_model))
    # Gemini is the configured live vision classifier. Groq remains available
    # for audio transcription; a later HF model can be integrated separately.
    traces = []
    for provider, model in providers:
        observation, trace = await _attempt(provider, model, image, mime, report)
        traces.append(trace)
        if observation:
            return ProviderResult(observation, provider, model, traces)
    return ProviderResult(None, None, None, traces)


async def transcribe_audio(audio: bytes, filename: str, mime: str):
    settings = get_settings()
    model = settings.groq_asr_model
    if not settings.groq_api_key:
        return None, None, None, {"provider": "groq", "model": model, "task": "asr", "outcome": "unavailable",
                                  "latency_ms": None, "error_code": "missing_api_key"}
    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(GROQ_AUDIO_URL,
                headers={"Authorization": f"Bearer {settings.groq_api_key}"},
                files={"file": (filename, audio, mime)},
                data={"model": model, "response_format": "verbose_json"})
            response.raise_for_status()
            result = response.json()
            return result.get("text"), result.get("language"), "groq", {
                "provider": "groq", "model": model, "task": "asr", "outcome": "ok",
                "latency_ms": int((time.monotonic() - start) * 1000), "error_code": None}
    except Exception as exc:
        return None, None, None, {"provider": "groq", "model": model, "task": "asr", "outcome": "error",
                                 "latency_ms": int((time.monotonic() - start) * 1000), "error_code": type(exc).__name__}
