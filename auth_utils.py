"""
Delegated Microsoft OAuth2 helpers for FastAPI session auth.
"""
from __future__ import annotations

import secrets
import time
import os
from typing import Any, Dict, Optional

from fastapi import HTTPException, Request
from msal import ConfidentialClientApplication

from config import (
    SHAREPOINT_CLIENT_ID,
    SHAREPOINT_CLIENT_SECRET,
    SHAREPOINT_TENANT_ID,
)


# Least-privilege default for delegated login. Can be overridden via env var:
# GRAPH_DELEGATED_SCOPES=User.Read,Files.Read
# Do not include reserved OIDC scopes (openid/profile/offline_access) here.
_SCOPES_ENV = (os.environ.get("GRAPH_DELEGATED_SCOPES") or "").strip()
AUTH_SCOPES = [s.strip() for s in _SCOPES_ENV.split(",") if s.strip()] if _SCOPES_ENV else [
    "User.Read",
    "Files.Read",
]
# In-memory token store keyed by per-session ID.
# Keeps session cookies small; avoids browser cookie-size limits with OAuth tokens.
_SESSION_TOKEN_STORE: Dict[str, Dict[str, Any]] = {}


def _authority() -> str:
    if not SHAREPOINT_TENANT_ID:
        raise RuntimeError("SHAREPOINT_TENANT_ID is required for delegated OAuth2.")
    return f"https://login.microsoftonline.com/{SHAREPOINT_TENANT_ID}"


def build_msal_app() -> ConfidentialClientApplication:
    if not SHAREPOINT_CLIENT_ID or not SHAREPOINT_CLIENT_SECRET:
        raise RuntimeError("SHAREPOINT_CLIENT_ID and SHAREPOINT_CLIENT_SECRET are required.")
    return ConfidentialClientApplication(
        SHAREPOINT_CLIENT_ID,
        authority=_authority(),
        client_credential=SHAREPOINT_CLIENT_SECRET,
    )


def create_login_state() -> str:
    return secrets.token_urlsafe(24)


def auth_redirect_uri(request: Request) -> str:
    # Keep callback on same host/domain where app is served.
    return str(request.url_for("auth_callback"))


def auth_start_url(request: Request, state: str) -> str:
    app = build_msal_app()
    return app.get_authorization_request_url(
        scopes=AUTH_SCOPES,
        state=state,
        redirect_uri=auth_redirect_uri(request),
        prompt="select_account",
    )


def exchange_code_for_token(request: Request, code: str) -> Dict[str, Any]:
    app = build_msal_app()
    result = app.acquire_token_by_authorization_code(
        code=code,
        scopes=AUTH_SCOPES,
        redirect_uri=auth_redirect_uri(request),
    )
    if "access_token" not in result:
        raise HTTPException(status_code=401, detail=f"OAuth callback failed: {result.get('error_description', result)}")
    return result


def _token_expired(session_tokens: Dict[str, Any]) -> bool:
    expires_at = int(session_tokens.get("expires_at", 0) or 0)
    return time.time() >= max(0, expires_at - 60)


def refresh_if_needed(session_tokens: Dict[str, Any]) -> Dict[str, Any]:
    if not session_tokens:
        raise HTTPException(status_code=401, detail="Missing delegated token.")
    if not _token_expired(session_tokens):
        return session_tokens
    refresh_token = session_tokens.get("refresh_token")
    if not refresh_token:
        raise HTTPException(status_code=401, detail="Session expired; please login again.")
    app = build_msal_app()
    result = app.acquire_token_by_refresh_token(refresh_token, scopes=AUTH_SCOPES)
    if "access_token" not in result:
        raise HTTPException(status_code=401, detail="Session refresh failed; please login again.")
    return {
        "access_token": result["access_token"],
        "refresh_token": result.get("refresh_token", refresh_token),
        "expires_at": int(time.time()) + int(result.get("expires_in", 3600)),
        "id_token_claims": result.get("id_token_claims") or session_tokens.get("id_token_claims") or {},
    }


def get_current_user(request: Request) -> Dict[str, Any]:
    user = request.session.get("user")
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required.")
    return user


def save_session_tokens(request: Request, token_result: Dict[str, Any]) -> None:
    """Store OAuth tokens server-side and save only a small session key in cookie."""
    sid = request.session.get("sid") or secrets.token_urlsafe(24)
    request.session["sid"] = sid
    claims = token_result.get("id_token_claims") or {}
    _SESSION_TOKEN_STORE[sid] = {
        "access_token": token_result["access_token"],
        "refresh_token": token_result.get("refresh_token"),
        "expires_at": int(time.time()) + int(token_result.get("expires_in", 3600)),
        "id_token_claims": claims,
    }


def clear_session_tokens(request: Request) -> None:
    sid = request.session.pop("sid", None)
    if sid:
        _SESSION_TOKEN_STORE.pop(sid, None)


def get_current_access_token(request: Request) -> str:
    sid = request.session.get("sid")
    tokens = _SESSION_TOKEN_STORE.get(sid, {}) if sid else {}
    updated = refresh_if_needed(tokens)
    if sid:
        _SESSION_TOKEN_STORE[sid] = updated
    return updated["access_token"]

