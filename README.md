# Civic Triage backend

FastAPI backend for the Civic Triage roadmap. It accepts image or video evidence, optional text or voice, and optional GPS; stores tickets and evidence locally; calls hosted vision, transcription, and audio event detection providers; applies backend severity and routing rules; and provides public ticket, staff review, audit, and summary endpoints.

Google Colab notebooks and local model experiments are intentionally outside this backend.

## Run locally

```bash
cd backend_model
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API is available at `http://localhost:8000`; interactive docs are at `/docs`. When using a Dev Tunnel, keep `--host 0.0.0.0` so the tunnel can reach the server. Provider keys are optional for booting the API. Without a working vision provider, reports are still saved and routed to the review queue with an explicit model failure reason. SQLite and uploads default to `backend_model/data/` and `backend_model/uploads/`.

Set `GEMINI_API_KEY` and/or `GROQ_API_KEY` in the backend `.env`. Gemini is the primary vision provider for image classification, evidence extraction, and severity signals; Groq vision is the fallback. Gemini output participates in category selection and cross-modal disagreement flags. Vision inference is bounded so a slow provider does not keep a report request open indefinitely. The fine-tuned `lucider-2006/finetuneYolo` Space is retained as a future optional integration, but is not called in the live report path. Groq Whisper transcription and the `ManasKumar28/SED` Gradio Space are attempted for voice notes; provider failure does not block report creation. Video uploads require the `ffmpeg` executable on the backend server. Video is capped at 50 MB; the backend saves a representative JPEG frame and extracts up to 60 seconds of mono audio for inference. Never put provider credentials in frontend environment variables or commit `.env`.

Set `DUPLICATE_RADIUS_M=40` and `DUPLICATE_WINDOW_DAYS=14` to configure nearby report clustering. Sensitive place context is local and privacy preserving: provide verified places in `LOCATION_CONTEXTS_JSON`, for example `[ {"name":"Central School","type":"school","latitude":19.07,"longitude":72.87,"radius_m":300} ]`. Supported high priority types include `school`, `hospital`, and `highway`. No citizen coordinates are sent to a third party for place lookup.

Ward polygons go in `WARD_POLYGONS_JSON` as a list such as `[ {"ward_id":"W-1","name":"Ward 1","polygon":[[19.06,72.86],[19.08,72.86],[19.08,72.89],[19.06,72.89]]} ]`; each polygon coordinate is `[latitude, longitude]`. The same `ward_id` can be used by a routing rule. An environment example is in `.env.example`.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/reports` | Multipart `image` or `video` (one required), `description`, `audio`, `latitude`, `longitude` |
| `GET` | `/api/tickets` | Canonical issue list with filters and pagination; pass `include_duplicates=true` for each linked report |
| `GET` | `/api/tickets/{id}` | Ticket, immutable public status history, comments, vote and rating summaries |
| `POST` | `/api/tickets/{id}/comments` | Add a public comment |
| `POST` | `/api/tickets/{id}/votes` | Toggle a pseudonymous “me too” vote |
| `POST` | `/api/tickets/{id}/satisfaction` | Rate a resolved issue from 1–5 |
| `GET` | `/api/public/dashboard` | Public map points, department metrics, ward leaderboard and oldest unresolved list; accepts `category`, `status`, `ward_id` |
| `GET` | `/api/departments/{department}/queue` | Department issue queue ordered by risk and age |
| `GET` | `/api/staff/routing/rules` | View active and inactive routing rules |
| `POST` | `/api/staff/routing/rules` | Add a category/ward routing rule |
| `DELETE` | `/api/staff/routing/rules/{id}` | Disable a routing rule |
| `GET` | `/api/tickets/{id}/image` | Ticket image evidence |
| `GET` | `/api/tickets/{id}/video` | Original submitted video evidence |
| `GET` | `/api/analytics/summary` | Real ticket counts; simulated records excluded |
| `GET` | `/api/staff/review-queue` | Tickets with human review flags and reasons |
| `PATCH` | `/api/staff/tickets/{id}` | Staff status/issue/severity/department update; `reason` required. Resolved issues may be reopened. |
| `GET` | `/api/health` | API liveness |
| `GET` | `/api/health/providers` | Provider configuration readiness without exposing credentials |

The staff and routing-rule endpoints are not authenticated in this demo scaffold. Add an application auth layer before exposing them publicly; the department queue is likewise a functional queue endpoint, not a login system. Voice files are persisted for the report workflow but are never returned in public ticket responses; transcript and audio event observations are stored separately. Contact details for opt-in notifications are stripped from public ticket responses.

## Decision and provenance behavior

- The image model classifies potholes, water leaks, garbage, streetlights, bridge cracks, and other/not-civic images. It returns visible risk factors and a preliminary visual severity; it does not choose the final score by itself.
- The backend calculates a 0–100 score from visual risk (38%), category weight (20%), multilingual text urgency (16%), configured sensitive place context (10%), nearby duplicate count (8%), and audio labels (8%). The UI displays the components. Severity bands are 1 (0–19), 2 (20–39), 3 (40–59), 4 (60–79), and 5 (80–100); verified structural damage and critical safety signals have minimum floors.
- Text parsing recognizes English, Hindi, Hinglish, and common Marathi/Hindi category and urgency phrases. Whisper's detected language takes precedence for voice transcripts. Text/image category disagreement creates a staff review reason.
- Audio event labels and timeline output stay separate from transcript and visual evidence. Potential gas hiss or sparking labels raise the risk score and require staff verification; labels alone are not treated as confirmation.
- Same-category reports within the configured radius and time window are linked to one canonical issue. The list endpoint returns the canonical issue with a linked-report count; linked submissions remain in the audit data and can be retrieved with `include_duplicates=true`.
- Ward assignment, map boundaries, place context, routing rules and default SLAs require verified local configuration. Default escalation order is JE → AE → EE → Commissioner; the current overdue stage is calculated from the ticket age and SLA.
- Citizen email uses configured SMTP; SMS and WhatsApp use configured Twilio senders. Notifications are opt-in on report submission and delivery attempts are recorded in a private outbox. No message is sent when the channel is unconfigured.
- Public status history is append-only through the API. Public comments are append-only; a pseudonymous client token limits duplicate “me too” votes and satisfaction ratings in this demo.
- Poor/unusable images, empty visual evidence, high severity, cross-modal conflict, missing GPS, or failed image inference request staff review.
- Simulated tickets default to false and analytics excludes any rows explicitly marked simulated.
- Every staff change requires a reason and is written to the audit log; status changes are also recorded in status history.

## Scope notes

The roadmap marks Gemini transcription fallback and AST sound tags as optional. This scaffold includes the model configuration slots and keeps ASR observations separate, but only wires the primary Groq Whisper path. AST and Colab are excluded. The provider health endpoint reports configured credentials, not a live inference test.
