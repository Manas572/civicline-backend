import json
from app.core.config import get_settings
from app.db import connection
from app.services.decision import DEPARTMENTS
from app.db import utc_now


def _point_in_polygon(lat: float, lon: float, polygon: list) -> bool:
    inside = False
    if len(polygon) < 3:
        return False
    j = len(polygon) - 1
    for i, point in enumerate(polygon):
        yi, xi = float(point[0]), float(point[1])
        yj, xj = float(polygon[j][0]), float(polygon[j][1])
        crosses = ((yi > lat) != (yj > lat)) and (lon < (xj - xi) * (lat - yi) / ((yj - yi) or 1e-12) + xi)
        if crosses:
            inside = not inside
        j = i
    return inside


def find_ward(latitude: float | None, longitude: float | None) -> dict | None:
    if latitude is None or longitude is None:
        return None
    try:
        wards = json.loads(get_settings().ward_polygons_json)
    except json.JSONDecodeError:
        return None
    for ward in wards if isinstance(wards, list) else []:
        try:
            if _point_in_polygon(latitude, longitude, ward["polygon"]):
                return {"ward_id": str(ward["ward_id"]), "ward_name": str(ward.get("name", ward["ward_id"]))}
        except (KeyError, TypeError, ValueError):
            continue
    return None


def resolve_route(issue_type: str, ward_id: str | None) -> dict:
    with connection() as conn:
        rules = conn.execute("SELECT * FROM routing_rules WHERE active=TRUE ORDER BY priority ASC,id ASC").fetchall()
    applicable = [r for r in rules if r["issue_type"] in (None, issue_type) and r["ward_id"] in (None, ward_id)]
    if applicable:
        rule = min(applicable, key=lambda r: (-int(r["issue_type"] == issue_type) - int(r["ward_id"] == ward_id and ward_id is not None),
                                               r["priority"], r["id"]))
        return {"department": rule["department"], "assigned_officer": rule["officer"],
                "assigned_queue": rule["queue"] or rule["department"], "sla_hours": rule["sla_hours"],
                "escalation_path": json.loads(rule["escalation_path"] or "[]"), "routing_rule_id": rule["id"]}
    defaults = {"Roads": 48, "Roads & Bridges": 24, "Water": 24, "Sanitation": 48,
                "Electrical": 24, "Triage desk": 72}
    try:
        configured_slas = json.loads(get_settings().sla_hours_json)
        if isinstance(configured_slas, dict):
            defaults.update({key: float(value) for key, value in configured_slas.items()})
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return {"department": DEPARTMENTS.get(issue_type, "Triage desk"), "assigned_officer": None,
            "assigned_queue": DEPARTMENTS.get(issue_type, "Triage desk"),
            "sla_hours": defaults.get(DEPARTMENTS.get(issue_type, "Triage desk"), 72),
            "escalation_path": ["JE", "AE", "EE", "Commissioner"], "routing_rule_id": None}


def create_routing_rule(payload: dict) -> dict:
    with connection() as conn:
        cursor = conn.execute("""INSERT INTO routing_rules
            (issue_type,ward_id,department,officer,queue,sla_hours,escalation_path,priority,active,created_at)
            VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
            (payload.get("issue_type"), payload.get("ward_id"), payload["department"],
             payload.get("officer"), payload.get("queue"), payload["sla_hours"],
             json.dumps(payload.get("escalation_path", ["JE", "AE", "EE", "Commissioner"])),
             payload.get("priority", 100), bool(payload.get("active", True)), utc_now()))
        new_id = cursor.fetchone()["id"]
        row = conn.execute("SELECT * FROM routing_rules WHERE id=%s", (new_id,)).fetchone()
    result = dict(row)
    result["escalation_path"] = json.loads(result["escalation_path"])
    result["active"] = bool(result["active"])
    return result
