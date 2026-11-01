"""Optional status notifications. Attempts are recorded even when delivery is unconfigured."""
import smtplib
import json
from email.message import EmailMessage

import httpx

from app.core.config import get_settings
from app.db import connection, utc_now


def _deliver_email(to: str, subject: str, body: str) -> None:
    settings = get_settings()
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = settings.smtp_from
    message["To"] = to
    message.set_content(body)
    with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=10) as server:
        server.starttls()
        if settings.smtp_username:
            server.login(settings.smtp_username, settings.smtp_password)
        server.send_message(message)


def _deliver_twilio(to: str, body: str, channel: str) -> None:
    settings = get_settings()
    from_number = settings.twilio_whatsapp_from if channel == "whatsapp" else settings.twilio_sms_from
    response = httpx.post(
        f"https://api.twilio.com/2010-04-01/Accounts/{settings.twilio_account_sid}/Messages.json",
        auth=(settings.twilio_account_sid, settings.twilio_auth_token),
        data={"From": from_number, "To": f"whatsapp:{to}" if channel == "whatsapp" else to, "Body": body},
        timeout=12,
    )
    response.raise_for_status()


def notify_ticket(ticket: dict, message: str) -> None:
    channels = ticket.get("notification_channels", [])
    if isinstance(channels, str):
        try:
            channels = json.loads(channels)
        except json.JSONDecodeError:
            channels = []
    destinations = {"email": ticket.get("reporter_email"), "sms": ticket.get("reporter_phone"),
                    "whatsapp": ticket.get("reporter_phone")}
    settings = get_settings()
    for channel in channels:
        destination = destinations.get(channel)
        if not destination:
            continue
        outcome, error = "unconfigured", "provider_not_configured"
        try:
            if channel == "email" and settings.smtp_host and settings.smtp_from:
                _deliver_email(destination, f"CivicLine update · {ticket['id'][:8]}", message)
                outcome, error = "sent", None
            elif channel in ("sms", "whatsapp") and settings.twilio_account_sid and settings.twilio_auth_token:
                if (channel == "sms" and settings.twilio_sms_from) or (channel == "whatsapp" and settings.twilio_whatsapp_from):
                    _deliver_twilio(destination, message, channel)
                    outcome, error = "sent", None
        except Exception as exc:
            outcome, error = "error", type(exc).__name__
        with connection() as conn:
            conn.execute("""INSERT INTO notification_outbox
                (ticket_id,channel,destination,message,outcome,error_code,created_at) VALUES(%s,%s,%s,%s,%s,%s,%s)""",
                (ticket["id"], channel, destination, message, outcome, error, utc_now()))
