"""Integration checks for the ForgeHackathon Neon PostgreSQL schema.

These tests use the configured Neon database. Every inserted ticket gets a
unique UUID, and teardown deletes only rows attached to that ticket.
"""
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from app.db import connection, init_db

TABLES = {
    "tickets", "user_roles", "status_history", "audit_events", "provider_traces",
    "routing_rules", "ticket_comments", "ticket_votes", "ticket_ratings",
    "notification_outbox",
}


def stamp():
    return datetime.now(timezone.utc).isoformat()


@pytest.fixture(scope="session", autouse=True)
def initialized_schema():
    init_db()


@pytest.fixture
def ticket_id():
    value = str(uuid4())
    now = stamp()
    with connection() as conn:
        conn.execute(
            """INSERT INTO tickets (
                id,created_at,updated_at,issue_type,department,severity,status,
                needs_human_review,is_simulated,reporter_label
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (value, now, now, "pothole", "Roads", 2, "submitted", False, False, "test"),
        )
    yield value
    with connection() as conn:
        for table in ("status_history", "audit_events", "provider_traces", "ticket_comments",
                      "ticket_votes", "ticket_ratings", "notification_outbox"):
            conn.execute(f"DELETE FROM {table} WHERE ticket_id=%s", (value,))
        conn.execute("DELETE FROM tickets WHERE id=%s", (value,))


def test_neon_connection_and_all_tables():
    with connection() as conn:
        assert conn.execute("SELECT 1 AS connected").fetchone()["connected"] == 1
        rows = conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"
        ).fetchall()
    assert TABLES <= {row["table_name"] for row in rows}


def test_default_routing_rules_are_present_and_stable():
    expected = {"pothole", "water_leak", "garbage", "streetlight", "bridge_crack", "other", "not_civic"}
    init_db()
    with connection() as conn:
        rows = conn.execute("SELECT issue_type,department FROM routing_rules WHERE issue_type IS NOT NULL").fetchall()
        counts = conn.execute("SELECT COUNT(*) AS n FROM routing_rules").fetchone()["n"]
    assert expected <= {row["issue_type"] for row in rows}
    assert counts >= 7


def test_ticket_insert_retrieve_update_and_persistence(ticket_id):
    with connection() as conn:
        row = conn.execute("SELECT id,status,issue_type FROM tickets WHERE id=%s", (ticket_id,)).fetchone()
        assert row == {"id": ticket_id, "status": "submitted", "issue_type": "pothole"}
        conn.execute("UPDATE tickets SET status=%s,updated_at=%s WHERE id=%s", ("in_progress", stamp(), ticket_id))
    with connection() as conn:
        row = conn.execute("SELECT status FROM tickets WHERE id=%s", (ticket_id,)).fetchone()
    assert row["status"] == "in_progress"


def test_generated_ids_and_ticket_relations(ticket_id):
    now = stamp()
    with connection() as conn:
        history_id = conn.execute(
            "INSERT INTO status_history(ticket_id,status,actor,reason,created_at) VALUES(%s,%s,%s,%s,%s) RETURNING id",
            (ticket_id, "submitted", "pytest", "created", now),
        ).fetchone()["id"]
        audit_id = conn.execute(
            "INSERT INTO audit_events(ticket_id,field,old_value,new_value,actor,reason,created_at) VALUES(%s,%s,%s,%s,%s,%s,%s) RETURNING id",
            (ticket_id, "status", "submitted", "in_progress", "pytest", "test update", now),
        ).fetchone()["id"]
        trace_id = conn.execute(
            "INSERT INTO provider_traces(ticket_id,provider,model,task,outcome,created_at) VALUES(%s,%s,%s,%s,%s,%s) RETURNING id",
            (ticket_id, "pytest", "fixture", "integration", "ok", now),
        ).fetchone()["id"]
        comment_id = conn.execute(
            "INSERT INTO ticket_comments(ticket_id,author_label,body,created_at) VALUES(%s,%s,%s,%s) RETURNING id",
            (ticket_id, "pytest", "integration comment", now),
        ).fetchone()["id"]
        conn.execute("INSERT INTO ticket_votes(ticket_id,voter_id,created_at) VALUES(%s,%s,%s)", (ticket_id, str(uuid4()), now))
        conn.execute("INSERT INTO ticket_ratings(ticket_id,voter_id,rating,comment,created_at) VALUES(%s,%s,%s,%s,%s)",
                     (ticket_id, str(uuid4()), 5, "integration rating", now))
        outbox_id = conn.execute(
            "INSERT INTO notification_outbox(ticket_id,channel,destination,message,outcome,created_at) VALUES(%s,%s,%s,%s,%s,%s) RETURNING id",
            (ticket_id, "email", "pytest@example.invalid", "test", "queued", now),
        ).fetchone()["id"]
        assert all(isinstance(v, int) and v > 0 for v in (history_id, audit_id, trace_id, comment_id, outbox_id))
        assert conn.execute("SELECT COUNT(*) AS n FROM ticket_votes WHERE ticket_id=%s", (ticket_id,)).fetchone()["n"] == 1
        assert conn.execute("SELECT COUNT(*) AS n FROM ticket_ratings WHERE ticket_id=%s", (ticket_id,)).fetchone()["n"] == 1
