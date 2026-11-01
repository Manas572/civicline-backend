"""JWT verification behavior tests using generated keys and a fake JWKS transport.

These exercise signature/claim validation, not a live Neon account or Neon token.
"""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from app import auth


AUTH_URL = "https://auth.example.test/neondb/auth"
ISSUER = "https://auth.example.test"


@pytest.fixture
def jwt_setup(monkeypatch):
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    now = datetime.now(timezone.utc)

    class LocalJWKS:
        def get_signing_key_from_jwt(self, _token):
            return SimpleNamespace(key=public_key)

    monkeypatch.setattr(auth, "_jwks_client", lambda _url: LocalJWKS())
    monkeypatch.setattr(auth, "get_settings", lambda: SimpleNamespace(auth_url=AUTH_URL, jwks_url="https://jwks.test"))
    monkeypatch.setattr(auth, "ensure_public_role", lambda _user_id: "PUBLIC")

    def make_token(*, subject="user-123", expires=now + timedelta(minutes=5), issuer=ISSUER, key=private_key):
        return jwt.encode(
            {"sub": subject, "iss": issuer, "iat": now, "exp": expires},
            key,
            algorithm="EdDSA",
            headers={"kid": "local-test-key"},
        )

    def call(token):
        return auth.current_user(HTTPAuthorizationCredentials(scheme="Bearer", credentials=token))

    return make_token, call


def test_accepts_valid_neon_style_eddsa_jwt_and_uses_subject(jwt_setup):
    make_token, call = jwt_setup
    user = call(make_token())
    assert user["user_id"] == "user-123"
    assert user["role"] == "PUBLIC"


def test_rejects_bad_signature_expiry_and_wrong_issuer(jwt_setup):
    make_token, call = jwt_setup
    wrong_key = Ed25519PrivateKey.generate()
    invalid_cases = [
        make_token(key=wrong_key),
        make_token(expires=datetime.now(timezone.utc) - timedelta(minutes=1)),
        make_token(issuer="https://attacker.example.test"),
    ]
    for token in invalid_cases:
        with pytest.raises(HTTPException) as error:
            call(token)
        assert error.value.status_code == 401
