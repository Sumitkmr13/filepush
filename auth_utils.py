"""
Delegated Microsoft OAuth2 helpers for FastAPI session auth.
"""
from __future__ import annotations

import secrets
import time
from typing import Any, Dict, Optional

from fastapi import HTTPException, Request
from msal import ConfidentialClientApplication

from config import (
    SHAREPOINT_CLIENT_ID,
    SHAREPOINT_CLIENT_SECRET,
    SHAREPOINT_TENANT_ID,
)


# MSAL adds OIDC reserved scopes internally; do not pass reserved values like
# offline_access/openid/profile here or get_authorization_request_url can fail.
AUTH_SCOPES = ["User.Read", "Sites.Read.All", "Files.Read.All"]


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


def get_current_access_token(request: Request) -> str:
    tokens = request.session.get("tokens") or {}
    updated = refresh_if_needed(tokens)
    request.session["tokens"] = updated
    return updated["access_token"]

