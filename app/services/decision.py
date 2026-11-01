from app.schemas import VisionObservation

DEPARTMENTS = {
    "pothole": "Roads", "water_leak": "Water", "garbage": "Sanitation",
    "streetlight": "Electrical", "bridge_crack": "Roads & Bridges",
    "other": "Triage desk", "not_civic": "Triage desk",
}
CATEGORY_WEIGHT = {
    "not_civic": 0, "other": 15, "garbage": 25, "streetlight": 38,
    "pothole": 42, "water_leak": 48, "bridge_crack": 95,
}
SEVERITY_BASE = {1: 10, 2: 25, 3: 45, 4: 70, 5: 90}
VISUAL_FACTORS = {
    "large_pothole": 65, "deep_pothole": 78, "road_collapse": 100,
    "bridge_crack": 88, "structural_damage": 92, "water_pooling": 52,
    "flooding": 78, "exposed_wire": 90, "live_wire": 98, "active_sparking": 100,
    "broken_streetlight": 38, "garbage_spread": 32, "faded_marking": 8,
}
SAFETY_TERMS = ("open manhole", "exposed wire", "live current", "live wire", "flooding", "sparking", "gas leak")


def _band(score: int) -> int:
    if score >= 80:
        return 5
    if score >= 60:
        return 4
    if score >= 40:
        return 3
    if score >= 20:
        return 2
    return 1


def _audio_signals(tags: list[str]) -> tuple[int, list[str]]:
    text = " ".join(str(tag).lower() for tag in tags)
    signals = []
    if "hiss" in text or ("gas" in text and any(term in text for term in ("leak", "smell", "flow"))):
        signals.append("possible_gas_hiss")
    if any(term in text for term in ("spark", "sizzle", "electric arc", "electrical buzzing")):
        signals.append("possible_sparking_or_electrical_hazard")
    if any(term in text for term in ("water", "pouring", "flow", "splash")):
        signals.append("possible_flowing_water")
    if any(term in text for term in ("impact", "crash", "collision")):
        signals.append("possible_impact")
    score = 0
    if "possible_gas_hiss" in signals or "possible_sparking_or_electrical_hazard" in signals:
        score = 100
    elif "possible_impact" in signals:
        score = 60
    elif "possible_flowing_water" in signals:
        score = 45
    return score, signals


def decide(
    observation: VisionObservation | None,
    *,
    has_gps: bool,
    report_text: str | None,
    text_analysis: dict | None = None,
    audio_tags: list[str] | None = None,
    location_contexts: list[dict] | None = None,
    prior_duplicates: int = 0,
    vision_classification: dict | None = None,
) -> dict:
    """Fuse model observations and context into a reviewable 0-100 risk score."""
    reasons: list[str] = []
    rules: list[str] = []
    text_analysis = text_analysis or {}
    location_contexts = location_contexts or []
    audio_tags = audio_tags or []

    visual_issue = observation.issue_type if observation else None
    text_issue = text_analysis.get("issue_type")
    yolo_issue = None
    if vision_classification:
        label = str(vision_classification.get("label", "")).lower().replace(" ", "_").replace("-", "_")
        aliases = {"pothole": "pothole", "water_leak": "water_leak", "leak": "water_leak",
                   "garbage": "garbage", "trash": "garbage", "streetlight": "streetlight",
                   "street_light": "streetlight", "bridge_crack": "bridge_crack", "bridge": "bridge_crack"}
        yolo_issue = vision_classification.get("issue_type") or aliases.get(label)
    issue = yolo_issue or visual_issue or text_issue or "other"
    text_categories = {item.get("issue_type") for item in text_analysis.get("categories", [])}
    category_conflict = bool(visual_issue and text_categories and visual_issue not in text_categories)
    yolo_conflict = bool(yolo_issue and ((visual_issue and yolo_issue != visual_issue) or (text_issue and yolo_issue != text_issue)))
    if yolo_conflict:
        reasons.append(f"Image model class ({yolo_issue}) conflicts with another report signal")
        rules.append("vision_cross_modal_category_conflict")
    if observation is None:
        reasons.append("Image model observation unavailable; text classification is preliminary")
        rules.append("model_failure_review")
    if category_conflict:
        reasons.append(f"Text suggests {', '.join(sorted(text_categories))}, but image model classified {visual_issue}")
        rules.append("cross_modal_category_conflict")
    elif observation and observation.consistent_with_citizen_report == "no":
        reasons.append("Image observations conflict with citizen report")
        rules.append("vision_text_conflict")

    visual_score = SEVERITY_BASE.get(observation.suggested_severity, 15) if observation else 10
    visual_factors = [str(factor).lower().strip().replace(" ", "_").replace("-", "_")
                      for factor in (observation.visual_risk_factors if observation else [])]
    evidence_text = " ".join((observation.visual_evidence if observation else []) +
                             (observation.hazards if observation else [])).lower()
    for factor, factor_score in VISUAL_FACTORS.items():
        if factor in visual_factors or factor.replace("_", " ") in evidence_text:
            visual_score = max(visual_score, factor_score)
    pothole_size = observation.pothole_size if observation else "unknown"
    crack_extent = observation.crack_extent if observation else "unknown"
    water_pooling = observation.water_pooling if observation else "unknown"
    visual_score = max(visual_score, {"small": 22, "medium": 42, "large": 68}.get(pothole_size, 0))
    visual_score = max(visual_score, {"short": 40, "long": 68, "structural": 92}.get(crack_extent, 0))
    visual_score = max(visual_score, {"minor": 35, "extensive": 72, "none": 0}.get(water_pooling, 0))
    if issue == "bridge_crack":
        visual_score = max(visual_score, 75)
    if observation and observation.image_quality == "poor":
        visual_score = round(visual_score * 0.75)
        reasons.append("Image quality is poor; score confidence is reduced")
        rules.append("poor_image_reduces_visual_component")
    elif observation and observation.image_quality == "unusable":
        visual_score = round(visual_score * 0.5)
        reasons.append("Image quality is unusable")
        rules.append("unusable_image_reduces_visual_component")

    category_weight = CATEGORY_WEIGHT.get(issue, 15)
    urgency_score = int(text_analysis.get("urgency_score", 15))
    audio_score, audio_signals = _audio_signals(audio_tags)
    location_score = 0
    for context in location_contexts:
        kind = str(context.get("type", "")).lower()
        location_score = max(location_score, 90 if kind in ("hospital", "school", "highway") else 65)
    duplicate_score = min(100, prior_duplicates * 25)

    score = round(
        visual_score * 0.38 + category_weight * 0.20 + urgency_score * 0.16 +
        location_score * 0.10 + duplicate_score * 0.08 + audio_score * 0.08
    )
    hazards = " ".join(observation.hazards if observation else []).lower()
    critical_safety = text_analysis.get("urgency") == "critical" or any(term in hazards for term in SAFETY_TERMS) or any(
        signal in audio_signals for signal in ("possible_gas_hiss", "possible_sparking_or_electrical_hazard"))
    if issue == "bridge_crack":
        score = max(score, 70)
    if critical_safety:
        score = max(score, 75)
        rules.append("critical_safety_signal_floors_score_at_75")
        if text_analysis.get("urgency") == "critical":
            reasons.append("Description includes critical safety or injury language")
    score = max(0, min(100, score))
    severity = _band(score)

    if issue in ("other", "not_civic"):
        reasons.append(f"Issue classified as {issue}")
    if not observation or (observation and observation.image_quality in ("poor", "unusable")):
        if not observation or not observation.visual_evidence:
            reasons.append("Visual evidence is unavailable or empty")
    if severity >= 4:
        reasons.append(f"Multimodal severity score is {score}/100")
    if not has_gps:
        reasons.append("GPS location is missing")
        rules.append("location_context_unavailable")
    if location_contexts:
        names = [str(item.get("name") or item.get("type") or "sensitive location") for item in location_contexts]
        reasons.append("Near sensitive location: " + ", ".join(names))
        rules.append("sensitive_location_context_applied")
    if prior_duplicates:
        reasons.append(f"{prior_duplicates} similar reports are already clustered nearby")
        rules.append("nearby_duplicate_cluster_applied")
    if audio_signals:
        reasons.append("Audio model signals: " + ", ".join(audio_signals))
        rules.append("audio_event_context_applied")
    if audio_score >= 90:
        reasons.append("Potential high-risk audio labels require staff verification")
    if text_analysis.get("urgency_terms"):
        rules.append("multilingual_text_urgency_applied")

    factors = {
        "visual": visual_score,
        "category": category_weight,
        "text_urgency": urgency_score,
        "location": location_score,
        "duplicate_reports": duplicate_score,
        "audio": audio_score,
        "weights": {"visual": 0.38, "category": 0.20, "text_urgency": 0.16,
                    "location": 0.10, "duplicate_reports": 0.08, "audio": 0.08},
        "audio_signals": audio_signals,
        "visual_risk_factors": visual_factors,
        "vision_classification": vision_classification,
        "pothole_size": pothole_size,
        "crack_extent": crack_extent,
        "water_pooling": water_pooling,
    }
    return {
        "issue_type": issue,
        "department": DEPARTMENTS[issue],
        "severity": severity,
        "severity_score": score,
        "severity_factors": factors,
        "needs_human_review": bool(reasons),
        "review_reasons": list(dict.fromkeys(reasons)),
        "rules_applied": rules,
    }
