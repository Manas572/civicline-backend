"""FastAPI route integration against isolated Neon PostgreSQL records."""
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from types import SimpleNamespace

from app.auth import current_user, ensure_public_role, require_admin
from app.db import connection, init_db
from app.main import app


def now():
    return datetime.now(timezone.utc).isoformat()


@pytest.fixture

def api_ticket():
    init_db()
    ticket_id = str(uuid4())
    ts = now()
    with connection() as conn:
        conn.execute("""INSERT INTO tickets(id,created_at,updated_at,issue_type,department,severity,status,
            needs_human_review,is_simulated,latitude,longitude,reporter_user_id)
            VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (ticket_id, ts, ts, "pothole", "Roads", 2, "submitted", False, False, 19.07, 72.87, "owner-a"))
    yield ticket_id
    with connection() as conn:
        for table in ("status_history", "audit_events", "provider_traces", "ticket_comments",
                      "ticket_votes", "ticket_ratings", "notification_outbox"):
            conn.execute(f"DELETE FROM {table} WHERE ticket_id=%s", (ticket_id,))
        conn.execute("DELETE FROM tickets WHERE id=%s", (ticket_id,))


def test_ticket_and_public_interaction_routes(api_ticket):
    init_db()
    app.dependency_overrides[require_admin] = lambda: {"user_id": "integration-admin", "role": "ADMIN"}
    app.dependency_overrides[current_user] = lambda: {"user_id": "integration-admin", "role": "ADMIN"}
    try:
        with TestClient(app) as client:
            identity = client.get("/api/auth/me")
            assert identity.status_code == 200
            assert identity.json() == {"user_id": "integration-admin", "role": "ADMIN"}
            detail = client.get(f"/api/tickets/{api_ticket}")
            assert detail.status_code == 200
            assert detail.json()["id"] == api_ticket
            public_detail = client.get(f"/api/public/tickets/{api_ticket}")
            assert public_detail.status_code == 200
            assert "reporter_user_id" not in public_detail.json()
            assert "audit_log" not in public_detail.json()
            listing = client.get("/api/tickets", params={"limit": 200})
            assert listing.status_code == 200
            assert any(row["id"] == api_ticket for row in listing.json()["items"])
            changed = client.patch(f"/api/staff/tickets/{api_ticket}", json={
                "status": "in_progress", "reason": "integration status transition", "actor": "pytest"})
            assert changed.status_code == 200, changed.text
            history = client.get(f"/api/tickets/{api_ticket}").json()["status_history"]
            assert any(row["status"] == "in_progress" for row in history)
            comment = client.post(f"/api/tickets/{api_ticket}/comments", json={"author_label": "pytest", "body": "integration comment"})
            assert comment.status_code == 201, comment.text
            assert isinstance(comment.json()["id"], int)
            vote_id = str(uuid4())
            vote = client.post(f"/api/tickets/{api_ticket}/votes", json={"voter_id": vote_id})
            assert vote.status_code == 200, vote.text
            assert vote.json()["voted"] is True
            resolved = client.patch(f"/api/staff/tickets/{api_ticket}", json={
                "status": "resolved", "reason": "integration resolution", "actor": "pytest"})
            assert resolved.status_code == 200, resolved.text
            rating = client.post(f"/api/tickets/{api_ticket}/satisfaction", json={
                "voter_id": str(uuid4()), "rating": 5, "comment": "integration rating"})
            assert rating.status_code == 200, rating.text
            assert rating.json()["satisfaction_count"] >= 1
            rules = client.get("/api/staff/routing/rules")
            assert rules.status_code == 200 and len(rules.json()) >= 7
            analytics = client.get("/api/analytics/summary")
            assert analytics.status_code == 200
            dashboard = client.get("/api/public/dashboard")
            assert dashboard.status_code == 200
    finally:
        app.dependency_overrides.pop(require_admin, None)
        app.dependency_overrides.pop(current_user, None)


def test_public_ticket_ownership_and_admin_access(api_ticket):
    try:
        with TestClient(app) as client:
            app.dependency_overrides[current_user] = lambda: {"user_id": "owner-a", "role": "PUBLIC"}
            own = client.get(f"/api/tickets/{api_ticket}")
            assert own.status_code == 200
            assert own.json()["reporter_user_id"] == "owner-a"
            own_list = client.get("/api/tickets")
            assert own_list.status_code == 200
            assert any(item["id"] == api_ticket for item in own_list.json()["items"])

            app.dependency_overrides[current_user] = lambda: {"user_id": "owner-b", "role": "PUBLIC"}
            assert client.get(f"/api/tickets/{api_ticket}").status_code == 404
            public_comment = client.post(f"/api/tickets/{api_ticket}/comments", json={
                "author_label": "owner-b", "body": "public issue discussion"})
            assert public_comment.status_code == 201, public_comment.text
            public_vote = client.post(f"/api/tickets/{api_ticket}/votes", json={
                "voter_id": str(uuid4())})
            assert public_vote.status_code == 200, public_vote.text
            other_list = client.get("/api/tickets")
            assert other_list.status_code == 200
            assert all(item["id"] != api_ticket for item in other_list.json()["items"])
            denied_patch = client.patch(f"/api/staff/tickets/{api_ticket}", json={
                "status": "resolved", "department": "Water", "reason": "attempt", "actor": "spoofed"})
            assert denied_patch.status_code == 403

            app.dependency_overrides[current_user] = lambda: {"user_id": "integration-admin", "role": "ADMIN"}
            app.dependency_overrides[require_admin] = lambda: {"user_id": "integration-admin", "role": "ADMIN"}
            admin_detail = client.get(f"/api/tickets/{api_ticket}")
            assert admin_detail.status_code == 200
            managed = client.patch(f"/api/staff/tickets/{api_ticket}", json={
                "department": "Water", "reason": "admin management", "actor": "spoofed"})
            assert managed.status_code == 200, managed.text
            with connection() as conn:
                audit = conn.execute("SELECT actor FROM audit_events WHERE ticket_id=%s ORDER BY id DESC LIMIT 1",
                                     (api_ticket,)).fetchone()
            assert audit["actor"] == "integration-admin"
    finally:
        app.dependency_overrides.pop(require_admin, None)
        app.dependency_overrides.pop(current_user, None)


def test_create_report_persists_verified_identity_and_ignores_owner_fields(monkeypatch):
    import app.api as api

    trace = {"provider": "test", "model": "stub", "task": "classification", "outcome": "ok",
             "latency_ms": 0, "error_code": None}
    async def fake_observe(*_args):
        return SimpleNamespace(observation=None, traces=[], provider="test", model="stub")
    async def fake_classify(*_args):
        return None, trace

    monkeypatch.setattr(api, "observe_image", fake_observe)
    monkeypatch.setattr(api, "classify_image", fake_classify)
    monkeypatch.setattr(api, "_save_media", lambda *_args: "ownership-test.jpg")
    verified_id = "verified-user-" + str(uuid4())
    app.dependency_overrides[current_user] = lambda: {"user_id": verified_id, "role": "PUBLIC"}
    ticket_id = None
    try:
        with TestClient(app) as client:
            response = client.post("/api/reports", data={
                "description": "Large pothole near the bus stop",
                "reporter_user_id": "attacker", "user_id": "attacker", "owner_id": "attacker",
            }, files={"image": ("pothole.jpg", b"\xff\xd8\xff", "image/jpeg")})
            assert response.status_code == 201, response.text
            body = response.json()
            ticket_id = body["id"]
            assert body["reporter_user_id"] == verified_id
            with connection() as conn:
                saved = conn.execute("SELECT reporter_user_id FROM tickets WHERE id=%s", (ticket_id,)).fetchone()
                assert saved["reporter_user_id"] == verified_id
    finally:
        if ticket_id:
            with connection() as conn:
                for table in ("status_history", "audit_events", "provider_traces", "ticket_comments",
                              "ticket_votes", "ticket_ratings", "notification_outbox"):
                    conn.execute(f"DELETE FROM {table} WHERE ticket_id=%s", (ticket_id,))
                conn.execute("DELETE FROM tickets WHERE id=%s", (ticket_id,))
        app.dependency_overrides.pop(current_user, None)


def test_role_initialization_does_not_downgrade_admin():
    public_id, admin_id = str(uuid4()), str(uuid4())
    try:
        assert ensure_public_role(public_id) == "PUBLIC"
        with connection() as conn:
            conn.execute("INSERT INTO user_roles(user_id,role,updated_at) VALUES(%s,'ADMIN',%s)", (admin_id, now()))
        assert ensure_public_role(admin_id) == "ADMIN"
    finally:
        with connection() as conn:
            conn.execute("DELETE FROM user_roles WHERE user_id IN (%s,%s)", (public_id, admin_id))
