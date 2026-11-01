from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    neon_db: str = ""
    auth_url: str = ""
    jwks_url: str = ""
    admin_user_ids: str = ""
    admin_emails: str = ""
    gemini_api_key: str = ""
    groq_api_key: str = ""
    # 2.5 Flash is verified against the configured API key and civic image
    # workload; the newer 3.8 endpoint timed out in the live test.
    gemini_vision_model: str = "gemini-2.5-flash"
    groq_vision_model: str = "qwen/qwen3.8-27b"
    groq_asr_model: str = "whisper-large-v3-turbo"
    gemini_transcribe_model: str = "gemini-3.5-transcribe"
    database_path: Path = Path("./data/civic_triage.db")
    upload_dir: Path = Path("./uploads")
    max_upload_mb: int = 10
    max_video_upload_mb: int = 50
    sed_space: str = "ManasKumar28/SED"
    yolo_space: str = "lucider-2006/finetuneYolo"
    duplicate_radius_m: int = 40
    duplicate_window_days: int = 14
    location_contexts_json: str = "[]"
    ward_polygons_json: str = "[]"
    sla_hours_json: str = '{"Roads":48,"Roads & Bridges":24,"Water":24,"Sanitation":48,"Electrical":24,"Triage desk":72}'
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_from: str = ""
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_sms_from: str = ""
    twilio_whatsapp_from: str = ""
    cors_origins: str = "http://localhost:5173"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def allowed_origins(self) -> list[str]:
        return ["*"] if self.cors_origins.strip() == "*" else [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
