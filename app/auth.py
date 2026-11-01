"""Neon Auth JWT verification and backend role authorization."""
from functools import lru_cache
from urllib.parse import urlsplit

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import get_settings
from app.db import connection

bearer = HTTPBearer(auto_error=False)


@lru_cache(maxsize=1)
def _jwks_client(url: str):
    if not url:
        return None
    return jwt.PyJWKClient(url, cache_jwk_set=True, lifespan=300)


def ensure_public_role(user_id: str, email: str | None = None, name: str | None = None) -> str:
    """Give first-time users PUBLIC without overwriting a trusted existing role."""
    settings = get_settings()
    trusted_ids = {v.strip() for v in settings.admin_user_ids.split(',') if v.strip()}
    trusted_emails = {v.strip().lower() for v in settings.admin_emails.split(',') if v.strip()}
    initial_role = 'ADMIN' if str(user_id) in trusted_ids or (email and email.lower() in trusted_emails) else 'PUBLIC'
    with connection() as conn:
        conn.execute(
            "INSERT INTO user_roles(user_id,role,email,name,updated_at) VALUES(%s,%s,%s,%s,NOW()) "
            "ON CONFLICT(user_id) DO UPDATE SET email=COALESCE(EXCLUDED.email,user_roles.email), "
            "name=COALESCE(EXCLUDED.name,user_roles.name), "
            "role=CASE WHEN user_roles.role='ADMIN' OR EXCLUDED.role='ADMIN' THEN 'ADMIN' ELSE user_roles.role END,updated_at=NOW()",
            (str(user_id), initial_role, email, name),
        )
        row = conn.execute("SELECT role FROM user_roles WHERE user_id=%s", (str(user_id),)).fetchone()
    return row["role"]


def current_user(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
    settings = get_settings()
    if not credentials:
        raise HTTPException(401, "Authentication required", headers={"WWW-Authenticate": "Bearer"})
    client = _jwks_client(settings.jwks_url)
    if not client:
        raise HTTPException(503, "Authentication is not configured")
    try:
        key = client.get_signing_key_from_jwt(credentials.credentials).key
        auth_endpoint = urlsplit(settings.auth_url)
        issuer = f"{auth_endpoint.scheme}://{auth_endpoint.netloc}" if auth_endpoint.scheme and auth_endpoint.netloc else None
        claims = jwt.decode(credentials.credentials, key, algorithms=["EdDSA", "RS256", "ES256"],
                            options={"verify_aud": False}, issuer=issuer)
        user_id = claims.get("sub")
        if not isinstance(user_id, str) or not user_id.strip():
            raise jwt.InvalidTokenError("missing subject")
    except Exception as exc:
        raise HTTPException(401, "Invalid or expired session", headers={"WWW-Authenticate": "Bearer"}) from exc
    role = ensure_public_role(str(user_id), claims.get("email"), claims.get("name"))
    return {"user_id": str(user_id), "role": role, "claims": claims}


get_current_user = current_user


def get_current_user_role(user=Depends(current_user)):
    return user["role"]


def require_public(user=Depends(current_user)):
    return user


def require_admin(user=Depends(current_user)):
    if user["role"] != "ADMIN":
        raise HTTPException(403, "Admin role required")
    return user
