import asyncio
import json
import math
import mimetypes
import re
import statistics
import shutil
import subprocess
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from app.core.config import get_settings
from app.auth import current_user, require_admin
from app.db import connection, dumps, row_dict, utc_now
from app.schemas import CommentCreate, RatingCreate, RoutingRuleCreate, TicketPatch, VoteCreate
from app.services.audio_events import detect_audio_events
from app.services.decision import DEPARTMENTS, decide
from app.services.providers import ProviderResult, observe_image, transcribe_audio
from app.services.text_understanding import analyze_report_text
from app.services.notifications import notify_ticket
from app.services.routing import create_routing_rule, find_ward, resolve_route

router = APIRouter(prefix="/api")
IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
AUDIO_TYPES = {"audio/mpeg", "audio/wav", "audio/x-wav", "audio/webm", "audio/ogg", "audio/mp4", "audio/m4a"}
VIDEO_TYPES = {"video/mp4", "video/webm", "video/quicktime", "video/x-matroska"}
STATUS_ORDER = {"awaiting_review": 0, "submitted": 1, "assigned": 2, "in_progress": 3, "resolved": 4, "reopened": 5}


@router.get("/auth/me")
def authenticated_identity(user=Depends(current_user)):
    """Return the app role from PostgreSQL alongside the verified Neon identity."""
    return {"user_id": user["user_id"], "role": user["role"]}


async def _read_upload(file: UploadFile, allowed: set[str], max_mb: int | None = None) -> tuple[bytes, str]:
    mime = (file.content_type or "").lower().split(";")[0].strip()
    if mime not in allowed:
        raise HTTPException(415, f"Unsupported media type: {mime or 'missing'}")
    max_mb = max_mb or get_settings().max_upload_mb
    limit = max_mb * 1024 * 1024
    data = await file.read(limit + 1)
    if not data:
        raise HTTPException(400, "Uploaded file is empty")
    if len(data) > limit:
        raise HTTPException(413, f"File exceeds {max_mb} MB limit")
    signatures = {
        "image/jpeg": lambda b: b.startswith(b"\xff\xd8\xff"),
        "image/png": lambda b: b.startswith(b"\x89PNG\r\n\x1a\n"),
        "image/webp": lambda b: b.startswith(b"RIFF") and b[8:12] == b"WEBP",
        "audio/wav": lambda b: b.startswith(b"RIFF") and b[8:12] == b"WAVE",
        "audio/x-wav": lambda b: b.startswith(b"RIFF") and b[8:12] == b"WAVE",
        "audio/ogg": lambda b: b.startswith(b"OggS"),
        "audio/webm": lambda b: b.startswith(b"\x1aE\xdf\xa3"),
        "audio/mp4": lambda b: len(b) > 12 and b[4:8] == b"ftyp",
        "audio/m4a": lambda b: len(b) > 12 and b[4:8] == b"ftyp",
        "audio/mpeg": lambda b: b.startswith(b"ID3") or (len(b) > 1 and b[0] == 0xFF and b[1] & 0xE0 == 0xE0),
        "video/mp4": lambda b: len(b) > 12 and b[4:8] == b"ftyp",
        "video/quicktime": lambda b: len(b) > 12 and b[4:8] == b"ftyp",
        "video/webm": lambda b: b.startswith(b"\x1aE\xdf\xa3"),
        "video/x-matroska": lambda b: b.startswith(b"\x1aE\xdf\xa3"),
    }
    if mime in signatures and not signatures[mime](data):
        raise HTTPException(415, "File content does not match its declared media type")
    return data, mime


def _extract_video(video: bytes) -> tuple[bytes, bytes | None]:
    """Extract a representative JPEG frame and up to 60 seconds of mono audio."""
    if not shutil.which("ffmpeg"):
        raise HTTPException(503, "Video uploads require ffmpeg on the backend server")
    with tempfile.TemporaryDirectory(prefix="civic-video-") as temp_dir:
        source = Path(temp_dir) / "source-video"
        frame = Path(temp_dir) / "frame.jpg"
        audio = Path(temp_dir) / "audio.wav"
        source.write_bytes(video)
        frame_run = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(source), "-ss", "0.5", "-frames:v", "1", str(frame)],
            capture_output=True, timeout=30,
        )
        if frame_run.returncode or not frame.is_file():
            raise HTTPException(422, "Could not extract an image frame from this video")
        audio_run = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(source), "-map", "0:a:0", "-vn", "-t", "60",
             "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(audio)],
            capture_output=True, timeout=30,
        )
        frame_bytes = frame.read_bytes()
        audio_bytes = audio.read_bytes() if audio_run.returncode == 0 and audio.is_file() else None
        return frame_bytes, audio_bytes


def _save_media(data: bytes, mime: str, prefix: str) -> str:
    ext = mimetypes.guess_extension(mime) or ".bin"
    if ext == ".jpe":
        ext = ".jpg"
    settings = get_settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    name = f"{prefix}_{uuid.uuid4().hex}{ext}"
    (settings.upload_dir / name).write_bytes(data)
    return name


def _distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6_371_000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def _configured_location_contexts(latitude: float | None, longitude: float | None) -> list[dict]:
    if latitude is None or longitude is None:
        return []
    try:
        configured = json.loads(get_settings().location_contexts_json)
    except json.JSONDecodeError:
        configured = []
    matches = []
    for place in configured if isinstance(configured, list) else []:
        try:
            distance = _distance_m(latitude, longitude, float(place["latitude"]), float(place["longitude"]))
            radius = float(place.get("radius_m", 300))
            if distance <= radius:
                matches.append({"name": str(place.get("name", place.get("type", "sensitive location")))[:120],
                               "type": str(place.get("type", "sensitive"))[:40],
                               "distance_m": round(distance), "radius_m": round(radius)})
        except (KeyError, TypeError, ValueError):
            continue
    return matches


def _find_duplicate(issue_type: str, latitude: float | None, longitude: float | None) -> dict | None:
    if latitude is None or longitude is None:
        return None
    settings = get_settings()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=settings.duplicate_window_days)).isoformat()
    with connection() as conn:
        rows = conn.execute("""SELECT id,duplicate_cluster_id,duplicate_count,latitude,longitude
            FROM tickets WHERE issue_type=%s AND is_simulated=FALSE AND created_at>=%s
            AND latitude IS NOT NULL AND longitude IS NOT NULL""", (issue_type, cutoff)).fetchall()
        matches = []
        for row in rows:
            distance = _distance_m(latitude, longitude, row["latitude"], row["longitude"])
            if distance <= settings.duplicate_radius_m:
                matches.append((distance, row))
    if not matches:
        return None
    distance, closest = min(matches, key=lambda item: item[0])
    return {"cluster_id": closest["duplicate_cluster_id"] or closest["id"],
            "prior_reports": int(closest["duplicate_count"] or 1), "distance_m": round(distance)}


def _ticket_dict(row) -> dict:
    result = row_dict(row)
    if not result:
        return {}
    for field in ("visual_evidence", "visual_risk_factors", "hazards", "rules_applied", "review_reasons",
                  "ast_tags", "audio_events", "audio_timeline", "severity_factors", "location_context",
                  "text_analysis", "escalation_path", "notification_channels"):
        result[field] = json.loads(result[field] or "[]")
    result["needs_human_review"] = bool(result["needs_human_review"])
    result["is_simulated"] = bool(result["is_simulated"])
    # Never expose server filesystem paths or raw voice recordings in public ticket data.
    image = result.pop("image_path", None)
    result.pop("audio_path", None)
    video = result.pop("video_path", None)
    result.pop("model_confidence_uncalibrated", None)
    result.pop("reporter_email", None)
    result.pop("reporter_phone", None)
    result.pop("notification_channels", None)
    result["image_url"] = f"/api/tickets/{result['id']}/image" if image else None
    result["video_url"] = f"/api/tickets/{result['id']}/video" if video else None
    result["sla_breached"] = False
    result["escalated_to"] = None
    if result.get("sla_hours") and result.get("status") != "resolved":
        elapsed = max(0, (datetime.now(timezone.utc) - _parse_time(result["created_at"])).total_seconds() / 3600)
        result["sla_breached"] = elapsed > float(result["sla_hours"])
        if result["sla_breached"] and result.get("escalation_path"):
            stage = min(len(result["escalation_path"]) - 1, int(elapsed // float(result["sla_hours"])) - 1)
            result["escalated_to"] = result["escalation_path"][stage]
    return result


def _owned_ticket_query(ticket_id: str, user: dict) -> tuple[str, tuple]:
    """Build a ticket lookup scoped to the authenticated user unless they are staff."""
    if user.get("role") == "ADMIN":
        return "SELECT * FROM tickets WHERE id=%s", (ticket_id,)
    return "SELECT * FROM tickets WHERE id=%s AND reporter_user_id=%s", (ticket_id, user["user_id"])


def _require_ticket_owner(conn, ticket_id: str, user: dict):
    query, params = _owned_ticket_query(ticket_id, user)
    row = conn.execute(query, params).fetchone()
    if not row:
        raise HTTPException(404, "Ticket not found")
    return row


def _require_interactable_ticket(conn, ticket_id: str):
    """Public issue interactions follow the existing public ticket business rules."""
    row = conn.execute("SELECT id,status FROM tickets WHERE id=%s", (ticket_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Ticket not found")
    return row


@router.post("/reports", status_code=201)
async def create_report(
    user=Depends(current_user),
    image: UploadFile | None = File(default=None),
    video: UploadFile | None = File(default=None),
    description: str | None = Form(default=None, max_length=4000),
    audio: UploadFile | None = File(default=None),
    # Leaflet/OSM use Web Mercator, whose usable latitude range is about ±85.0511°.
    latitude: float | None = Form(default=None, ge=-85.0511, le=85.0511),
    longitude: float | None = Form(default=None, ge=-180, le=180),
    reporter_label: str | None = Form(default=None, max_length=60),
    reporter_email: str | None = Form(default=None, max_length=254),
    reporter_phone: str | None = Form(default=None, max_length=20),
    notification_channels: str = Form(default=""),
):
    if (image is None or not image.filename) and (video is None or not video.filename):
        raise HTTPException(400, "Upload a photo or video as visual evidence")
    reporter_email = (reporter_email or "").strip() or None
    reporter_phone = (reporter_phone or "").strip() or None
    if reporter_email and not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", reporter_email):
        raise HTTPException(422, "Enter a valid email address")
    if reporter_phone and not re.fullmatch(r"\+?[0-9]{7,15}", reporter_phone):
        raise HTTPException(422, "Phone number must use international format, for example +919876543210")
    notification_channels = [channel for channel in notification_channels.split(",")
                             if channel in {"email", "sms", "whatsapp"}]
    if "email" in notification_channels and not reporter_email:
        raise HTTPException(422, "Email notifications require an email address")
    if any(channel in notification_channels for channel in ("sms", "whatsapp")) and not reporter_phone:
        raise HTTPException(422, "SMS or WhatsApp notifications require a phone number")
    video_bytes = video_mime = None
    if video is not None and video.filename:
        video_bytes, video_mime = await _read_upload(video, VIDEO_TYPES, get_settings().max_video_upload_mb)
    extracted_image = extracted_audio = None
    if video_bytes:
        extracted_image, extracted_audio = await asyncio.to_thread(_extract_video, video_bytes)
    if image is not None and image.filename:
        image_bytes, image_mime = await _read_upload(image, IMAGE_TYPES)
    else:
        image_bytes, image_mime = extracted_image, "image/jpeg"
    audio_bytes = None
    audio_mime = None
    if audio is not None and audio.filename:
        audio_bytes, audio_mime = await _read_upload(audio, AUDIO_TYPES)
    elif extracted_audio:
        audio_bytes, audio_mime = extracted_audio, "audio/wav"

    image_path = _save_media(image_bytes, image_mime, "image")
    audio_path = _save_media(audio_bytes, audio_mime, "audio") if audio_bytes else None
    video_path = _save_media(video_bytes, video_mime, "video") if video_bytes else None
    transcript = transcript_language = asr_provider = asr_model = None
    traces = []
    audio_filename = (Path(audio.filename or "voice-note").name if audio is not None and audio.filename
                      else "video-audio.wav")
    ast_tags, audio_events, audio_timeline = [], None, None
    if audio_bytes:
        transcript, transcript_language, asr_provider, asr_trace = await transcribe_audio(
            audio_bytes, audio_filename, audio_mime or "application/octet-stream")
        asr_model = asr_trace["model"]
        traces.append(asr_trace)
        ast_tags, audio_events, audio_timeline, event_trace = await detect_audio_events(audio_bytes, audio_filename)
        traces.append(event_trace)

    statements = [text for text in ((description or "").strip(), (transcript or "").strip()) if text]
    citizen_text = "\n".join(statements) or None
    text_analysis = analyze_report_text(citizen_text)
    if transcript_language and transcript:
        transcript_lang = transcript_language.lower()
        text_analysis["language"] = ("Hindi" if transcript_lang.startswith("hi") else
                                     "Marathi" if transcript_lang.startswith("mr") else
                                     "English" if transcript_lang.startswith("en") else transcript_language)
    # AI providers are optional enrichment. A provider/network stall must not
    # leave the citizen's report stuck before it is persisted.
    async def bounded(coro, seconds, label):
        try:
            return await asyncio.wait_for(coro, timeout=seconds)
        except asyncio.TimeoutError:
            return label

    inference = await bounded(observe_image(image_bytes, image_mime, citizen_text), 18, None)
    inference = inference or ProviderResult(None, None, None, [{
        "provider": "gemini", "model": get_settings().gemini_vision_model, "task": "vision",
        "outcome": "timeout", "latency_ms": 18000, "error_code": "timeout",
    }])
    traces.extend(inference.traces)
    vision_classification = ({
        "provider": inference.provider,
        "model": inference.model,
        "label": inference.observation.issue_type,
        "issue_type": inference.observation.issue_type,
    } if inference.observation else None)
    classified_issue = (inference.observation.issue_type if inference.observation else
                        (text_analysis.get("issue_type") or "other"))
    location_contexts = _configured_location_contexts(latitude, longitude)
    duplicate = _find_duplicate(classified_issue, latitude, longitude)
    decision = decide(
        inference.observation, has_gps=latitude is not None and longitude is not None,
        report_text=citizen_text, text_analysis=text_analysis, audio_tags=ast_tags,
        location_contexts=location_contexts, prior_duplicates=duplicate["prior_reports"] if duplicate else 0,
        vision_classification=vision_classification,
    )
    ward = find_ward(latitude, longitude)
    route = resolve_route(decision["issue_type"], ward["ward_id"] if ward else None)
    decision["department"] = route["department"]
    status = "awaiting_review" if decision["needs_human_review"] else "submitted"
    observation = inference.observation
    ticket_id = str(uuid.uuid4())
    cluster_id = duplicate["cluster_id"] if duplicate else None
    cluster_count = duplicate["prior_reports"] + 1 if duplicate else 1
    now = utc_now()

    with connection() as conn:
        if duplicate:
            conn.execute("UPDATE tickets SET duplicate_count=%s WHERE id=%s OR duplicate_cluster_id=%s",
                         (cluster_count, cluster_id, cluster_id))
        conn.execute("""INSERT INTO tickets (
            id,created_at,updated_at,issue_type,department,severity,status,description,transcript,
            transcript_language,image_path,audio_path,video_path,latitude,longitude,visual_evidence,suggested_severity,
            justification,hazards,image_quality,consistency,model_confidence_uncalibrated,rules_applied,
            needs_human_review,review_requested,review_reasons,is_simulated,vision_provider,vision_model,asr_provider,asr_model,
            ast_tags,audio_events,audio_timeline,severity_score,severity_factors,text_issue_type,detected_language,
            location_context,duplicate_cluster_id,duplicate_count,visual_risk_factors,text_analysis,ward_id,ward_name,
            assigned_officer,assigned_queue,sla_hours,escalation_path,routing_rule_id,reporter_label,reporter_email,
            reporter_phone,notification_channels,reporter_user_id
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""", (
            ticket_id, now, now, decision["issue_type"], decision["department"], decision["severity"],
            status, description, transcript, transcript_language, image_path, audio_path, video_path, latitude, longitude,
            dumps(observation.visual_evidence if observation else []),
            observation.suggested_severity if observation else None,
            observation.justification if observation else None,
            dumps(observation.hazards if observation else []), observation.image_quality if observation else None,
            observation.consistent_with_citizen_report if observation else None,
            observation.model_confidence_uncalibrated if observation else None, dumps(decision["rules_applied"]),
            decision["needs_human_review"], decision["needs_human_review"], dumps(decision["review_reasons"]), False,
            inference.provider, inference.model, asr_provider, asr_model, dumps(ast_tags), dumps(audio_events), dumps(audio_timeline),
            decision["severity_score"], dumps(decision["severity_factors"]), text_analysis.get("issue_type"),
            text_analysis.get("language"), dumps(location_contexts), cluster_id, cluster_count,
            dumps(observation.visual_risk_factors if observation else []), dumps(text_analysis),
            ward["ward_id"] if ward else None, ward["ward_name"] if ward else None,
            route["assigned_officer"], route["assigned_queue"], route["sla_hours"],
            dumps(route["escalation_path"]), route["routing_rule_id"],
            (reporter_label or "Anonymous").strip()[:60], reporter_email, reporter_phone, dumps(notification_channels), user["user_id"],
        ))
        conn.execute("INSERT INTO status_history(ticket_id,status,actor,reason,created_at) VALUES(%s,%s,%s,%s,%s)",
                     (ticket_id, status, "system", "Report created", now))
        for trace in traces:
            conn.execute("""INSERT INTO provider_traces(ticket_id,provider,model,task,outcome,latency_ms,error_code,created_at)
                         VALUES(%s,%s,%s,%s,%s,%s,%s,%s)""", (ticket_id, trace["provider"], trace["model"], trace["task"],
                         trace["outcome"], trace["latency_ms"], trace["error_code"], now))
        row = conn.execute("SELECT * FROM tickets WHERE id=%s", (ticket_id,)).fetchone()
    result = _ticket_dict(row)
    if notification_channels:
        notify_ticket(dict(row), f"Your CivicLine tracking ID is {ticket_id}. Current status: {status.replace('_', ' ')}.")
    return result


@router.get("/tickets")
def list_tickets(
    user=Depends(current_user),
    issue_type: str | None = None, status: str | None = None,
    department: str | None = None, simulated: bool | None = None,
    include_duplicates: bool = False,
    limit: int = Query(default=50, ge=1, le=200), offset: int = Query(default=0, ge=0),
):
    where, args = [], []
    if not include_duplicates:
        where.append("duplicate_cluster_id IS NULL")
    for column, value in (("issue_type", issue_type), ("status", status), ("department", department)):
        if value:
            where.append(f"{column}=%s")
            args.append(value)
    if simulated is not None:
        where.append("is_simulated=%s")
        args.append(simulated)
    if user.get("role") != "ADMIN":
        where.append("reporter_user_id=%s")
        args.append(user["user_id"])
    clause = " WHERE " + " AND ".join(where) if where else ""
    with connection() as conn:
        total = conn.execute(f"SELECT COUNT(*) AS count FROM tickets{clause}", args).fetchone()["count"]
        rows = conn.execute(f"SELECT * FROM tickets{clause} ORDER BY created_at DESC LIMIT %s OFFSET %s",
                            [*args, limit, offset]).fetchall()
    return {"items": [_ticket_dict(row) for row in rows], "total": total, "limit": limit, "offset": offset}


@router.get("/tickets/{ticket_id}")
def ticket_detail(ticket_id: str, user=Depends(current_user)):
    with connection() as conn:
        row = _require_ticket_owner(conn, ticket_id, user)
        ticket = _ticket_dict(row)
        cluster_root = ticket["duplicate_cluster_id"] or ticket["id"]
        duplicate_query = """SELECT id,created_at,status,description,latitude,longitude,duplicate_count
               FROM tickets WHERE (id=%s OR duplicate_cluster_id=%s)"""
        duplicate_params: tuple = (cluster_root, cluster_root)
        if user.get("role") != "ADMIN":
            duplicate_query += " AND reporter_user_id=%s"
            duplicate_params += (user["user_id"],)
        ticket["duplicate_reports"] = [dict(r) for r in conn.execute(
            duplicate_query + " ORDER BY created_at", duplicate_params)]
        ticket["status_history"] = [dict(r) for r in conn.execute(
            "SELECT status,actor,reason,created_at FROM status_history WHERE ticket_id=%s ORDER BY id", (ticket_id,))]
        ticket["audit_log"] = [dict(r) for r in conn.execute(
            "SELECT field,old_value,new_value,actor,reason,created_at FROM audit_events WHERE ticket_id=%s ORDER BY id", (ticket_id,))]
        ticket["provider_traces"] = [dict(r) for r in conn.execute(
            "SELECT provider,model,task,outcome,latency_ms,error_code,created_at FROM provider_traces WHERE ticket_id=%s ORDER BY id", (ticket_id,))]
        ticket["comments"] = [dict(r) for r in conn.execute(
            "SELECT id,author_label,body,created_at FROM ticket_comments WHERE ticket_id=%s ORDER BY id", (ticket_id,))]
        ticket["me_too_count"] = conn.execute(
            "SELECT COUNT(*) AS count FROM ticket_votes WHERE ticket_id=%s", (ticket_id,)).fetchone()["count"]
        rating = conn.execute("SELECT AVG(rating) average,COUNT(*) count FROM ticket_ratings WHERE ticket_id=%s",
                              (ticket_id,)).fetchone()
        ticket["satisfaction_average"] = round(rating["average"], 2) if rating["average"] is not None else None
        ticket["satisfaction_count"] = rating["count"]
    return ticket


@router.post("/tickets/{ticket_id}/comments", status_code=201)
def add_public_comment(ticket_id: str, payload: CommentCreate, user=Depends(current_user)):
    now = utc_now()
    with connection() as conn:
        _require_interactable_ticket(conn, ticket_id)
        cursor = conn.execute("INSERT INTO ticket_comments(ticket_id,author_label,body,created_at) VALUES(%s,%s,%s,%s) RETURNING id",
                              (ticket_id, payload.author_label, payload.body.strip(), now))
        return {"id": cursor.fetchone()["id"], "author_label": payload.author_label,
                "body": payload.body.strip(), "created_at": now}


@router.post("/tickets/{ticket_id}/votes")
def toggle_me_too(ticket_id: str, payload: VoteCreate, user=Depends(current_user)):
    now = utc_now()
    with connection() as conn:
        _require_interactable_ticket(conn, ticket_id)
        voter_id = user["user_id"]
        exists = conn.execute("SELECT 1 FROM ticket_votes WHERE ticket_id=%s AND voter_id=%s",
                              (ticket_id, voter_id)).fetchone()
        if exists:
            conn.execute("DELETE FROM ticket_votes WHERE ticket_id=%s AND voter_id=%s",
                         (ticket_id, voter_id))
            voted = False
        else:
            conn.execute("INSERT INTO ticket_votes(ticket_id,voter_id,created_at) VALUES(%s,%s,%s)",
                         (ticket_id, voter_id, now))
            voted = True
        count = conn.execute("SELECT COUNT(*) AS count FROM ticket_votes WHERE ticket_id=%s", (ticket_id,)).fetchone()["count"]
    return {"me_too_count": count, "voted": voted}


@router.post("/tickets/{ticket_id}/satisfaction")
def rate_resolution(ticket_id: str, payload: RatingCreate, user=Depends(current_user)):
    now = utc_now()
    with connection() as conn:
        ticket = _require_interactable_ticket(conn, ticket_id)
        voter_id = user["user_id"]
        if ticket["status"] != "resolved":
            raise HTTPException(409, "Satisfaction can be rated after resolution")
        conn.execute("""INSERT INTO ticket_ratings(ticket_id,voter_id,rating,comment,created_at)
            VALUES(%s,%s,%s,%s,%s) ON CONFLICT(ticket_id,voter_id) DO UPDATE SET
            rating=excluded.rating,comment=excluded.comment,created_at=excluded.created_at""",
            (ticket_id, voter_id, payload.rating, payload.comment, now))
        result = conn.execute("SELECT AVG(rating) average,COUNT(*) count FROM ticket_ratings WHERE ticket_id=%s",
                              (ticket_id,)).fetchone()
    return {"satisfaction_average": round(result["average"], 2), "satisfaction_count": result["count"]}


@router.get("/tickets/{ticket_id}/image")
def ticket_image(ticket_id: str, user=Depends(current_user)):
    with connection() as conn:
        row = _require_ticket_owner(conn, ticket_id, user)
    if not row or not row["image_path"]:
        raise HTTPException(404, "Image not found")
    path = get_settings().upload_dir / Path(row["image_path"]).name
    if not path.is_file():
        raise HTTPException(404, "Image file not found")
    return FileResponse(path)


@router.get("/tickets/{ticket_id}/video")
def ticket_video(ticket_id: str, user=Depends(current_user)):
    with connection() as conn:
        row = _require_ticket_owner(conn, ticket_id, user)
    if not row or not row["video_path"]:
        raise HTTPException(404, "Video not found")
    path = get_settings().upload_dir / Path(row["video_path"]).name
    if not path.is_file():
        raise HTTPException(404, "Video file not found")
    return FileResponse(path)


@router.get("/analytics/summary", dependencies=[Depends(require_admin)])
def analytics_summary():
    with connection() as conn:
        total = conn.execute("SELECT COUNT(*) AS count FROM tickets WHERE is_simulated=FALSE").fetchone()["count"]
        unique_issues = conn.execute("SELECT COUNT(DISTINCT COALESCE(duplicate_cluster_id,id)) AS count FROM tickets WHERE is_simulated=FALSE").fetchone()["count"]
        by_status = {r["status"]: r["count"] for r in conn.execute(
            "SELECT status,COUNT(*) AS count FROM tickets WHERE is_simulated=FALSE AND duplicate_cluster_id IS NULL GROUP BY status")}
        by_issue = {r["issue_type"]: r["count"] for r in conn.execute(
            "SELECT issue_type,COUNT(*) AS count FROM tickets WHERE is_simulated=FALSE AND duplicate_cluster_id IS NULL GROUP BY issue_type")}
        open_review = conn.execute("SELECT COUNT(*) AS count FROM tickets WHERE is_simulated=FALSE AND needs_human_review=TRUE").fetchone()["count"]
        high_total = conn.execute("SELECT COUNT(*) AS count FROM tickets WHERE is_simulated=FALSE AND duplicate_cluster_id IS NULL AND severity>=4").fetchone()["count"]
        high_review = conn.execute("SELECT COUNT(*) AS count FROM tickets WHERE is_simulated=FALSE AND duplicate_cluster_id IS NULL AND severity>=4 AND review_requested=TRUE").fetchone()["count"]
    return {"total_real_tickets": total, "unique_real_issues": unique_issues,
            "by_status": by_status, "by_issue_type": by_issue,
            "awaiting_human_review": open_review,
            "high_severity_review_coverage": high_review / high_total if high_total else None}


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


@router.get("/public/dashboard")
def public_dashboard(category: str | None = None, status: str | None = None, ward_id: str | None = None):
    with connection() as conn:
        all_tickets = [dict(r) for r in conn.execute(
            "SELECT * FROM tickets WHERE is_simulated=FALSE AND duplicate_cluster_id IS NULL ORDER BY created_at DESC")]
        status_rows = conn.execute("SELECT ticket_id,status,created_at FROM status_history ORDER BY id").fetchall()
    histories: dict[str, list[dict]] = {}
    for row in status_rows:
        histories.setdefault(row["ticket_id"], []).append(dict(row))
    for ticket in all_tickets:
        ticket["review_reasons"] = json.loads(ticket["review_reasons"] or "[]")
        ticket["visual_evidence"] = json.loads(ticket["visual_evidence"] or "[]")
        ticket["location_context"] = json.loads(ticket["location_context"] or "[]")
        ticket["image_url"] = f"/api/tickets/{ticket['id']}/image" if ticket.get("image_path") else None
        ticket["reporter_email"] = None
        ticket["reporter_phone"] = None
        ticket.pop("reporter_user_id", None)
        ticket.pop("image_path", None)
        ticket.pop("audio_path", None)
        ticket.pop("video_path", None)
    filtered = [t for t in all_tickets if (not category or t["issue_type"] == category)
                and (not status or t["status"] == status) and (not ward_id or t["ward_id"] == ward_id)]

    by_department: dict[str, list[dict]] = {}
    by_ward: dict[str, list[dict]] = {}
    for ticket in all_tickets:
        by_department.setdefault(ticket["department"], []).append(ticket)
        ward_key = ticket["ward_id"] or "unassigned"
        by_ward.setdefault(ward_key, []).append(ticket)

    department_metrics = []
    department_names = set(DEPARTMENTS.values()) | set(by_department)
    for department in sorted(department_names):
        items = by_department.get(department, [])
        resolved_durations = []
        breached = 0
        reopened = 0
        unresolved = 0
        for ticket in items:
            events = histories.get(ticket["id"], [])
            resolved_event = next((e for e in events if e["status"] == "resolved"), None)
            reopened += any(e["status"] == "reopened" for e in events)
            created = _parse_time(ticket["created_at"])
            sla_hours = float(ticket.get("sla_hours") or 72)
            if resolved_event:
                hours = max(0, (_parse_time(resolved_event["created_at"]) - created).total_seconds() / 3600)
                resolved_durations.append(hours)
                breached += hours > sla_hours
            else:
                unresolved += ticket["status"] != "resolved"
                breached += (datetime.now(timezone.utc) - created).total_seconds() / 3600 > sla_hours
        resolved_count = len(resolved_durations)
        department_metrics.append({
            "department": department,
            "median_resolve_hours": round(statistics.median(resolved_durations), 1) if resolved_durations else None,
            "sla_breach_rate": round(breached / len(items), 3) if items else 0,
            "reopen_rate": round(reopened / resolved_count, 3) if resolved_count else 0,
            "backlog": unresolved,
            "total_issues": len(items),
        })
    department_metrics.sort(key=lambda item: item["department"])

    ward_leaderboard = []
    for ward, items in by_ward.items():
        resolved_count = sum(1 for t in items if t["status"] == "resolved")
        ward_leaderboard.append({"ward_id": ward, "ward_name": next((t["ward_name"] for t in items if t["ward_name"]),
                                                                          "Unassigned" if ward == "unassigned" else ward),
                                 "issues": len(items), "resolved": resolved_count,
                                 "resolution_rate": round(resolved_count / len(items), 3),
                                 "backlog": len(items) - resolved_count})
    ward_leaderboard.sort(key=lambda item: (-item["resolution_rate"], -item["issues"], item["ward_name"]))
    oldest = sorted((t for t in all_tickets if t["status"] != "resolved"), key=lambda t: t["created_at"])[:10]
    public_items = [{key: ticket.get(key) for key in (
        "id", "created_at", "updated_at", "issue_type", "department", "severity", "severity_score", "status",
        "description", "latitude", "longitude", "ward_id", "ward_name", "duplicate_count", "image_url")}
        for ticket in filtered if ticket.get("latitude") is not None and ticket.get("longitude") is not None]
    try:
        ward_boundaries = json.loads(get_settings().ward_polygons_json)
    except json.JSONDecodeError:
        ward_boundaries = []
    return {"items": public_items, "department_metrics": department_metrics,
            "ward_leaderboard": ward_leaderboard,
            "ward_boundaries": ward_boundaries if isinstance(ward_boundaries, list) else [],
            "oldest_unresolved": [{key: t.get(key) for key in (
                "id", "created_at", "issue_type", "department", "severity", "severity_score", "status",
                "ward_name", "duplicate_count")} for t in oldest],
            "ward_options": [{"ward_id": key, "ward_name": rows[0].get("ward_name") or key}
                             for key, rows in by_ward.items() if key != "unassigned"]}


@router.get("/public/tickets/{ticket_id}")
def public_ticket_detail(ticket_id: str):
    """Return only fields intentionally published on the community issue map."""
    with connection() as conn:
        row = conn.execute(
            """SELECT id,created_at,updated_at,issue_type,department,severity,status,description,
                      latitude,longitude,ward_id,ward_name,duplicate_count,severity_score
               FROM tickets WHERE id=%s AND is_simulated=FALSE""",
            (ticket_id,),
        ).fetchone()
        if not row:
            raise HTTPException(404, "Public issue not found")
        result = dict(row)
        result["status_history"] = [dict(item) for item in conn.execute(
            "SELECT status,created_at FROM status_history WHERE ticket_id=%s ORDER BY id", (ticket_id,))]
        result["comments"] = [dict(item) for item in conn.execute(
            "SELECT id,author_label,body,created_at FROM ticket_comments WHERE ticket_id=%s ORDER BY id", (ticket_id,))]
        result["me_too_count"] = conn.execute(
            "SELECT COUNT(*) AS count FROM ticket_votes WHERE ticket_id=%s", (ticket_id,)).fetchone()["count"]
        rating = conn.execute("SELECT AVG(rating) AS average,COUNT(*) AS count FROM ticket_ratings WHERE ticket_id=%s",
                              (ticket_id,)).fetchone()
        result["satisfaction_average"] = round(rating["average"], 2) if rating["average"] is not None else None
        result["satisfaction_count"] = rating["count"]
    return result


@router.get("/staff/routing/rules", dependencies=[Depends(require_admin)])
def list_routing_rules():
    with connection() as conn:
        rows = conn.execute("SELECT * FROM routing_rules ORDER BY priority,id").fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["escalation_path"] = json.loads(item["escalation_path"] or "[]")
        item["active"] = bool(item["active"])
        result.append(item)
    return result


@router.get("/departments/{department}/queue", dependencies=[Depends(require_admin)])
def department_queue(department: str, limit: int = Query(default=100, ge=1, le=200)):
    with connection() as conn:
        rows = conn.execute("""SELECT * FROM tickets WHERE department=%s AND status!='resolved'
            AND duplicate_cluster_id IS NULL ORDER BY severity_score DESC,created_at ASC LIMIT %s""",
            (department, limit)).fetchall()
        total = conn.execute("""SELECT COUNT(*) AS count FROM tickets WHERE department=%s AND status!='resolved'
            AND duplicate_cluster_id IS NULL""", (department,)).fetchone()["count"]
    return {"department": department, "items": [_ticket_dict(row) for row in rows], "total": total}


@router.post("/staff/routing/rules", status_code=201, dependencies=[Depends(require_admin)])
def add_routing_rule(payload: RoutingRuleCreate):
    return create_routing_rule(payload.model_dump())


@router.delete("/staff/routing/rules/{rule_id}", dependencies=[Depends(require_admin)])
def disable_routing_rule(rule_id: int):
    with connection() as conn:
        cursor = conn.execute("UPDATE routing_rules SET active=FALSE WHERE id=%s", (rule_id,))
        if not cursor.rowcount:
            raise HTTPException(404, "Routing rule not found")
    return {"id": rule_id, "active": False}


@router.get("/staff/review-queue", dependencies=[Depends(require_admin)])
def review_queue(limit: int = Query(default=100, ge=1, le=200), offset: int = Query(default=0, ge=0)):
    with connection() as conn:
        rows = conn.execute("SELECT * FROM tickets WHERE needs_human_review=TRUE ORDER BY created_at LIMIT %s OFFSET %s",
                            (limit, offset)).fetchall()
        total = conn.execute("SELECT COUNT(*) AS count FROM tickets WHERE needs_human_review=TRUE").fetchone()["count"]
    return {"items": [_ticket_dict(row) for row in rows], "total": total, "limit": limit, "offset": offset}


@router.patch("/staff/tickets/{ticket_id}", dependencies=[Depends(require_admin)])
def update_ticket(ticket_id: str, patch: TicketPatch, user=Depends(require_admin)):
    values = patch.model_dump(exclude={"reason", "actor"}, exclude_none=True)
    if not values:
        raise HTTPException(400, "At least one ticket field must be provided")
    now = utc_now()
    with connection() as conn:
        row = conn.execute("SELECT * FROM tickets WHERE id=%s", (ticket_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Ticket not found")
        old = dict(row)
        if "status" in values:
            new_status = values["status"].value
            old_status = old["status"]
            allowed_reopen = old_status == "resolved" and new_status == "reopened"
            allowed_resume = old_status == "reopened" and new_status in ("assigned", "in_progress", "resolved")
            backwards = old_status != "reopened" and old_status != "resolved" and STATUS_ORDER.get(new_status, -1) < STATUS_ORDER[old_status]
            if new_status not in STATUS_ORDER or (new_status == "reopened" and not allowed_reopen) or \
                    (old_status == "resolved" and not allowed_reopen) or \
                    (old_status == "reopened" and not allowed_resume and new_status != "reopened") or backwards:
                raise HTTPException(422, "Status cannot move backwards")
            values["status"] = new_status
        if "issue_type" in values and "department" not in values:
            values["department"] = DEPARTMENTS[values["issue_type"]]
        for field, new_value in values.items():
            old_value = old[field]
            if old_value == new_value:
                continue
            conn.execute("UPDATE tickets SET " + field + "=%s,updated_at=%s WHERE id=%s", (new_value, now, ticket_id))
            conn.execute("""INSERT INTO audit_events(ticket_id,field,old_value,new_value,actor,reason,created_at)
                         VALUES(%s,%s,%s,%s,%s,%s,%s)""", (ticket_id, field, str(old_value), str(new_value), user["user_id"], patch.reason, now))
            if field == "status":
                conn.execute("INSERT INTO status_history(ticket_id,status,actor,reason,created_at) VALUES(%s,%s,%s,%s,%s)",
                             (ticket_id, new_value, user["user_id"], patch.reason, now))
        if old["needs_human_review"] and ("status" in values or any(k in values for k in ("issue_type", "severity", "department"))):
            # A staff action validates the ticket; retain review reasons and audit trail for accountability.
            conn.execute("UPDATE tickets SET needs_human_review=FALSE,updated_at=%s WHERE id=%s", (now, ticket_id))
            if "status" not in values and old["status"] == "awaiting_review":
                conn.execute("UPDATE tickets SET status='submitted',updated_at=%s WHERE id=%s", (now, ticket_id))
                conn.execute("INSERT INTO status_history(ticket_id,status,actor,reason,created_at) VALUES(%s,%s,%s,%s,%s)",
                             (ticket_id, "submitted", user["user_id"], patch.reason, now))
                conn.execute("""INSERT INTO audit_events(ticket_id,field,old_value,new_value,actor,reason,created_at)
                             VALUES(%s,%s,%s,%s,%s,%s,%s)""", (ticket_id, "status", "awaiting_review", "submitted", user["user_id"], patch.reason, now))
        result = conn.execute("SELECT * FROM tickets WHERE id=%s", (ticket_id,)).fetchone()
    result_data = dict(result)
    if result_data["status"] != old["status"]:
        try:
            notify_ticket(result_data, f"Issue {ticket_id[:8]} status updated to {result_data['status'].replace('_', ' ')}.")
        except Exception:
            pass
    return _ticket_dict(result)


@router.get("/health/providers")
def provider_health():
    settings = get_settings()
    try:
        import gradio_client  # noqa: F401
        sed_available = True
    except ImportError:
        sed_available = False
    return {"providers": {
        "gemini_vision": {"configured": bool(settings.gemini_api_key), "model": settings.gemini_vision_model},
        "groq_vision": {"configured": bool(settings.groq_api_key), "model": settings.groq_vision_model},
        "groq_transcription": {"configured": bool(settings.groq_api_key), "model": settings.groq_asr_model},
        "gemini_transcription_fallback": {"configured": bool(settings.gemini_api_key), "model": settings.gemini_transcribe_model},
        "audio_event_detection": {"configured": sed_available, "model": settings.sed_space},
        "email_notifications": {"configured": bool(settings.smtp_host and settings.smtp_from)},
        "sms_notifications": {"configured": bool(settings.twilio_account_sid and settings.twilio_auth_token and settings.twilio_sms_from)},
        "whatsapp_notifications": {"configured": bool(settings.twilio_account_sid and settings.twilio_auth_token and settings.twilio_whatsapp_from)},
    }}
