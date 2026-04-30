"""
Check that the .env app credentials work and show what they can see.

What it does
------------
- Acquires an app-only token using SHAREPOINT_CLIENT_ID/SECRET/TENANT_ID from .env.
- Reports token claims (roles, expiry).
- If SHAREPOINT_SITE_URL is set, lists drives and root folder contents.

Usage
-----
    python tools/check_app_credentials.py
    python tools/check_app_credentials.py --site-url "https://tenant.sharepoint.com/sites/Foo"
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from sharepoint_utils import (  # noqa: E402
    _get_site_id,
    get_access_token,
    list_contents_at_path,
    list_site_drives,
)


def _decode_jwt(token: str) -> dict:
    try:
        parts = token.split(".")
        payload = parts[1]
        padded = payload + "=" * (-len(payload) % 4)
        raw = base64.urlsafe_b64decode(padded)
        return json.loads(raw)
    except Exception as e:
        return {"_error": str(e)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site-url", default=os.environ.get("SHAREPOINT_SITE_URL", ""))
    args = ap.parse_args()

    print("=" * 70)
    print("Acquiring app token (.env credentials)...")
    try:
        token = get_access_token()
    except Exception as e:
        print(f"  FAIL: {e}")
        return 1
    print("  OK")

    claims = _decode_jwt(token)
    print(f"  app_id (appid): {claims.get('appid')}")
    print(f"  tenant (tid):   {claims.get('tid')}")
    print(f"  roles:          {claims.get('roles')}")
    print(f"  expires:        {claims.get('exp')}")

    if not args.site_url:
        print("\nNo --site-url provided and SHAREPOINT_SITE_URL not set; skipping site checks.")
        return 0

    print("\n" + "=" * 70)
    print(f"Resolving site: {args.site_url}")
    try:
        site_id = _get_site_id(token, site_url=args.site_url)
        print(f"  OK site_id={site_id}")
    except Exception as e:
        print(f"  FAIL: {e}")
        return 1

    print("\nListing drives in site:")
    try:
        drives = list_site_drives(token=token, site_id=site_id)
        for d in drives:
            print(f"  - {d.get('name')!r} (id={d.get('id')})")
    except Exception as e:
        print(f"  FAIL: {e}")

    print("\nListing root contents (first 10):")
    try:
        items = list_contents_at_path(folder_path="", token=token, site_url=args.site_url)
        for it in items[:10]:
            kind = "DIR" if it.get("is_folder") else "FILE"
            print(f"  - [{kind}] {it.get('name')}")
        print(f"  ({len(items)} items total)")
    except Exception as e:
        print(f"  FAIL: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
