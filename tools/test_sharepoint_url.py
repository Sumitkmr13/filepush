"""
SharePoint URL diagnostic tool.

What it does
------------
- Inspects a pasted SharePoint URL (full, share link, or site URL).
- Tries to resolve it via the user's delegated token AND the app's .env credentials.
- Prints step-by-step what works and where it fails.

Usage
-----
1. Start the app, login via UI, set DEBUG_AUTH=1 in environment, then:
       curl -b cookies.txt -c cookies.txt http://localhost:8000/auth/debug/token
   Or open http://localhost:8000/auth/debug/token in the same browser session and copy the access_token.

2. Run this script (Windows PowerShell):
       $env:GEMRAG_USER_TOKEN = "<paste_token_here>"
       python tools/test_sharepoint_url.py "https://...sharepoint.com/..."

3. Optional flags:
       --token <jwt>          override token explicitly
       --app-only             test only with .env app credentials
       --folder-path <path>   verify a specific folder under the site
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from sharepoint_utils import (  # noqa: E402
    _get_site_id,
    get_access_token,
    list_contents_at_path,
    list_site_drives,
    verify_sharepoint_path_reachable,
)

GRAPH = "https://graph.microsoft.com/v1.0"
SHARE_LINK_RE = re.compile(r"/:[a-zA-Z]:/")


def _ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def _step(msg: str) -> None:
    print(f"\n>>> {msg}")


def _decode_jwt(token: str) -> dict:
    try:
        parts = token.split(".")
        payload = parts[1]
        padded = payload + "=" * (-len(payload) % 4)
        raw = base64.urlsafe_b64decode(padded)
        return json.loads(raw)
    except Exception as e:
        return {"_error": str(e)}


def _looks_like_share_link(url: str) -> bool:
    return bool(SHARE_LINK_RE.search(url or ""))


def _encode_share_url(share_url: str) -> str:
    encoded = base64.urlsafe_b64encode(share_url.encode("utf-8")).decode("ascii").rstrip("=")
    return f"u!{encoded}"


def _try_share_resolve(token: str, share_url: str, label: str) -> None:
    _step(f"[{label}] Resolve share-link via Graph /shares")
    encoded = _encode_share_url(share_url)
    url = f"{GRAPH}/shares/{encoded}/driveItem"
    try:
        r = requests.get(
            url,
            headers={"Authorization": f"Bearer {token}", "Prefer": "redeemSharingLink"},
            timeout=30,
        )
    except Exception as e:
        _fail(f"network error: {e}")
        return
    print(f"  status={r.status_code}")
    if r.status_code == 200:
        data = r.json() or {}
        parent = data.get("parentReference") or {}
        _ok(f"name={data.get('name')!r}")
        print(f"        driveId={parent.get('driveId')}")
        print(f"        parentPath={parent.get('path')}")
        print(f"        webUrl={data.get('webUrl')}")
    else:
        _fail(r.text[:300])


def _try_canonical_resolve(token: str, raw_url: str, folder_path: str, label: str) -> None:
    _step(f"[{label}] Resolve site_id from URL")
    try:
        site_id = _get_site_id(token, site_url=raw_url)
        _ok(f"site_id={site_id}")
    except Exception as e:
        _fail(f"_get_site_id failed: {e}")
        return

    _step(f"[{label}] List document libraries (drives)")
    try:
        drives = list_site_drives(token=token, site_id=site_id)
        if not drives:
            _fail("no drives returned")
        for d in drives:
            print(f"        - {d.get('name')!r} (id={d.get('id')})")
    except Exception as e:
        _fail(f"list_site_drives failed: {e}")

    if folder_path:
        _step(f"[{label}] Verify folder path: {folder_path}")
        ok2, msg = verify_sharepoint_path_reachable(
            token=token,
            site_url=raw_url,
            folder_path=folder_path,
        )
        if ok2:
            _ok(msg)
        else:
            _fail(msg)

        _step(f"[{label}] List first 5 items at: {folder_path}")
        try:
            items = list_contents_at_path(
                folder_path=folder_path,
                token=token,
                site_url=raw_url,
            )
            for it in items[:5]:
                kind = "DIR" if it.get("is_folder") else "FILE"
                print(f"        - [{kind}] {it.get('name')}")
            print(f"        ({len(items)} items total)")
        except Exception as e:
            _fail(f"list_contents_at_path failed: {e}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Test SharePoint URL access.")
    ap.add_argument("url", help="SharePoint URL (full, share link, or site URL)")
    ap.add_argument("--token", help="Delegated user token (default: env GEMRAG_USER_TOKEN)")
    ap.add_argument("--app-only", action="store_true", help="Test using only .env app credentials")
    ap.add_argument("--folder-path", default="", help="Optional folder path to verify under site")
    args = ap.parse_args()

    raw_url = args.url.strip()
    parsed = urlparse(raw_url)

    print("=" * 70)
    print(f"URL:       {raw_url}")
    print(f"Host:      {parsed.netloc}")
    print(f"Path:      {unquote(parsed.path)}")
    print(f"Share-link form: {_looks_like_share_link(raw_url)}")
    print("=" * 70)

    tokens: list[tuple[str, str]] = []

    user_token = (args.token or os.environ.get("GEMRAG_USER_TOKEN") or "").strip()
    if not args.app_only and user_token:
        _step("Inspect user token (delegated)")
        claims = _decode_jwt(user_token)
        print(f"  upn:    {claims.get('upn') or claims.get('preferred_username')}")
        print(f"  scopes: {claims.get('scp')}")
        print(f"  aud:    {claims.get('aud')}")
        print(f"  tid:    {claims.get('tid')}")
        tokens.append(("user", user_token))
    elif not args.app_only:
        print("\n(no user token provided; set --token or GEMRAG_USER_TOKEN env to test delegated)")

    try:
        app_tok = get_access_token()
        tokens.append(("app", app_tok))
        _ok("Acquired app token from .env credentials")
    except Exception as e:
        if args.app_only:
            _fail(f"app token unavailable: {e}")
            return 1
        print(f"\n(app token unavailable: {e})")

    if not tokens:
        _fail("No tokens available; aborting.")
        return 1

    is_share = _looks_like_share_link(raw_url)
    for label, tok in tokens:
        if is_share:
            _try_share_resolve(tok, raw_url, label)
        else:
            _try_canonical_resolve(tok, raw_url, args.folder_path, label)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
