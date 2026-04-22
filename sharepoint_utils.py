"""
SharePoint integration via MSAL + Microsoft Graph API.
Authenticates with client credentials; lists PDFs and downloads file content as bytes (in-memory).
Uses Graph Search API for high-volume PDF discovery (with pagination and 429 retry).
"""
import logging
import sys
import time
from typing import Generator, List, Optional, Tuple
from urllib.parse import quote

import requests
from msal import ConfidentialClientApplication

from config import (
    SHAREPOINT_CLIENT_ID,
    SHAREPOINT_CLIENT_SECRET,
    SHAREPOINT_TENANT_ID,
    SHAREPOINT_SITE_URL,
    SHAREPOINT_DRIVE_ID,
    SHAREPOINT_DRIVE_PATH,
    SHAREPOINT_SCOPES,
)

logger = logging.getLogger(__name__)

GRAPH_BASE = "https://graph.microsoft.com/v1.0"

# Excel column name for the SharePoint file link (used in main.py).
SHAREPOINT_LINK_COLUMN = "SharePoint Link"


def _sharepoint_file_web_url(item_id: str, site_url: Optional[str] = None) -> str:
    """Build a SharePoint 'open in browser' URL when Graph API does not return webUrl."""
    base_site = (site_url or SHAREPOINT_SITE_URL or "").strip()
    if not item_id or not base_site:
        return ""
    base = base_site.rstrip("/")
    # Doc.aspx opens the file in browser; sourcedoc is the drive item id (optional braces).
    encoded_id = quote(item_id, safe="")
    return f"{base}/_layouts/15/Doc.aspx?sourcedoc={encoded_id}&action=default"
# Timeout for Graph API calls (list/segment resolution). Prevents indefinite hang on slow or blocked networks.
GRAPH_REQUEST_TIMEOUT = 20
# Retry 429 (throttling): max attempts and base delay in seconds.
GRAPH_RETRY_MAX_ATTEMPTS = 4
GRAPH_RETRY_BASE_DELAY = 2


def _graph_get_with_retry(
    url: str,
    token: str,
    timeout: int = GRAPH_REQUEST_TIMEOUT,
) -> requests.Response:
    """
    GET a Graph API URL with retry on 429 (throttling).
    Uses exponential backoff; respects Retry-After header when present.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(GRAPH_RETRY_MAX_ATTEMPTS):
        try:
            r = requests.get(
                url,
                headers={"Authorization": f"Bearer {token}"},
                timeout=timeout,
            )
            if r.status_code == 429:
                last_exc = RuntimeError(f"Graph throttled (429): {r.text[:200]}")
                retry_after = r.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    delay = int(retry_after)
                else:
                    delay = GRAPH_RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning("Graph 429 throttling, retry in %ss (attempt %s/%s)", delay, attempt + 1, GRAPH_RETRY_MAX_ATTEMPTS)
                time.sleep(delay)
                continue
            return r
        except requests.RequestException as e:
            last_exc = e
            if attempt + 1 < GRAPH_RETRY_MAX_ATTEMPTS:
                delay = GRAPH_RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning("Graph request failed, retry in %ss: %s", delay, e)
                time.sleep(delay)
            else:
                raise
    if last_exc:
        raise last_exc
    raise RuntimeError("Graph request failed after retries")


def is_sharepoint_configured() -> bool:
    """True if we have enough app config for SharePoint auth (site may be user-selected)."""
    return bool(
        SHAREPOINT_CLIENT_ID
        and SHAREPOINT_CLIENT_SECRET
        and SHAREPOINT_TENANT_ID
    )


def get_access_token() -> str:
    """Acquire token for Microsoft Graph using client credentials (app-only)."""
    if not all([SHAREPOINT_CLIENT_ID, SHAREPOINT_CLIENT_SECRET, SHAREPOINT_TENANT_ID]):
        raise ValueError(
            "Set SHAREPOINT_CLIENT_ID, SHAREPOINT_CLIENT_SECRET, SHAREPOINT_TENANT_ID"
        )
    authority = f"https://login.microsoftonline.com/{SHAREPOINT_TENANT_ID}"
    app = ConfidentialClientApplication(
        SHAREPOINT_CLIENT_ID,
        authority=authority,
        client_credential=SHAREPOINT_CLIENT_SECRET,
    )
    result = app.acquire_token_for_client(scopes=SHAREPOINT_SCOPES)
    if "access_token" not in result:
        raise RuntimeError(
            f"Failed to acquire token: {result.get('error_description', result)}"
        )
    return result["access_token"]


def _get_site_id(token: str, site_url: Optional[str] = None) -> str:
    """Resolve site ID from a site URL (e.g. https://tenant.sharepoint.com/sites/SiteName)."""
    from urllib.parse import urlparse
    resolved_site_url = (site_url or SHAREPOINT_SITE_URL or "").strip()
    if not resolved_site_url:
        raise ValueError("SharePoint site URL is required.")
    parsed = urlparse(resolved_site_url)
    hostname = parsed.netloc
    path = parsed.path.rstrip("/") or "/"
    # Graph: GET /sites/{hostname}:/{serverRelativePath}
    url = f"{GRAPH_BASE}/sites/{hostname}:{path}"
    r = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=GRAPH_REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()["id"]


def _get_drive_id(token: str, site_id: str) -> str:
    """Get default document library (drive) ID for the site."""
    url = f"{GRAPH_BASE}/sites/{site_id}/drive"
    r = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=GRAPH_REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()["id"]


def _resolve_drive_id(token: str, site_id: str, drive_id_override: Optional[str] = None) -> str:
    """Use explicit drive ID when provided, otherwise config/default drive."""
    if drive_id_override:
        return drive_id_override
    if SHAREPOINT_DRIVE_ID:
        return SHAREPOINT_DRIVE_ID
    return _get_drive_id(token, site_id)


def list_site_drives(
    token: Optional[str] = None,
    site_id: Optional[str] = None,
) -> List[dict]:
    """
    List all document libraries (drives) for the connected site.
    Use this to see available drives, then navigate into folders of the chosen drive.

    Returns:
        List of dicts: name, id, driveType. name is the library display name (e.g. "Documents").
    """
    if token is None:
        token = get_access_token()
    if site_id is None:
        site_id = _get_site_id(token)
    url = f"{GRAPH_BASE}/sites/{site_id}/drives"
    out: List[dict] = []
    while url:
        r = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=GRAPH_REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        for d in data.get("value", []):
            out.append({
                "name": (d.get("name") or "").strip(),
                "id": d["id"],
                "driveType": d.get("driveType", ""),
            })
        url = data.get("@odata.nextLink")
    return out


def _get_children_page(
    token: str, url: str
) -> Tuple[dict, Optional[str]]:
    """GET a children URL; returns (data, next_link)."""
    r = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=GRAPH_REQUEST_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    return data, data.get("@odata.nextLink")


def _normalize_folder_name(name: str) -> str:
    """Normalize for comparison: strip, collapse spaces, normalize dashes and spaces."""
    if not name:
        return ""
    s = (name or "").strip()
    # Normalize common Unicode dashes to ASCII hyphen
    for c in ("\u2013", "\u2014", "\u2212"):
        s = s.replace(c, "-")
    # Normalize non-breaking and other Unicode spaces to ASCII space
    for c in ("\u00a0", "\u2002", "\u2003", "\u202f", "\u205f"):
        s = s.replace(c, " ")
    return " ".join(s.split())  # collapse internal spaces


def _resolve_folder_by_segments(
    token: str, drive_id: str, folder_path: str
) -> Optional[str]:
    """
    Resolve a folder by walking path segments (e.g. "pmo/Solenis IT/Sub").
    Returns the driveItem id of the folder, or None if not found.
    Use when root:/path returns 404 (long paths or encoding).
    """
    segments = [s.strip() for s in (folder_path or "").strip().strip("/").split("/") if s.strip()]
    if not segments:
        return None
    current_id: Optional[str] = None
    for i, name in enumerate(segments):
        next_url = (
            f"{GRAPH_BASE}/drives/{drive_id}/root/children"
            if current_id is None
            else f"{GRAPH_BASE}/drives/{drive_id}/items/{current_id}/children"
        )
        found: Optional[str] = None
        want = _normalize_folder_name(name)
        folder_names_at_level: List[str] = []
        while next_url:
            data, odata_next = _get_children_page(token, next_url)
            for item in data.get("value", []):
                if item.get("folder") is not None:
                    raw_name = (item.get("name") or "").strip()
                    folder_names_at_level.append(raw_name)
                    item_name = _normalize_folder_name(raw_name)
                    if item_name == want or item_name.lower() == want.lower():
                        found = item["id"]
                        break
            if found is not None:
                break
            next_url = odata_next
        if found is None:
            level_desc = "drive root" if current_id is None else f"under segment {i}"
            actual = folder_names_at_level[:20] if len(folder_names_at_level) > 20 else folder_names_at_level
            msg = (
                f"SharePoint segment not found: looking for {name!r} at {level_desc}. "
                f"Actual folder names here: {actual}"
            )
            logger.warning("%s", msg)
            print(msg, file=sys.stderr)
            return None
        current_id = found
    return current_id


def _parent_folder_name_from_item(item: dict) -> str:
    """Derive immediate parent folder name from driveItem.parentReference (path or name)."""
    pref = item.get("parentReference") or {}
    name = (pref.get("name") or "").strip()
    if name:
        return name
    path = (pref.get("path") or "").strip()
    if not path:
        return ""
    # path is like "/drives/{id}/root" or "/drives/{id}/root:/Folder/SubFolder"
    if "root:/" in path:
        segment = path.split("root:/", 1)[-1].rstrip("/")
        return segment.split("/")[-1] if segment else ""
    if path.rstrip("/").endswith("/root"):
        return "root"
    return path.split("/")[-1] if path else ""


def _list_pdfs_via_search(
    token: str, drive_id: str, folder_path: str
) -> List[dict]:
    """
    List all PDFs under the given drive path using Microsoft Graph Search API.
    Always resolves folder_id via _resolve_folder_by_segments when path is set, then uses
    drives/{drive_id}/items/{folder_id}/search(q='.pdf') to avoid path-parsing 500 errors.
    Paginates through @odata.nextLink for thousands of files.
    Returns list of dicts: id, name, folder (parent folder name), path (relative path for display).
    """
    base_path = (folder_path or "").strip().strip("/")
    all_items: List[dict] = []

    # STEP 1: Always resolve the folder ID first when we have a path.
    # This avoids 500 errors caused by complex path strings in the search URL.
    if base_path:
        folder_id = _resolve_folder_by_segments(token, drive_id, base_path)
        if not folder_id:
            logger.warning("Search failed: Could not resolve folder ID for path: %s", base_path)
            return []
        search_url = f"{GRAPH_BASE}/drives/{drive_id}/items/{folder_id}/search(q='.pdf')"
    else:
        search_url = f"{GRAPH_BASE}/drives/{drive_id}/root/search(q='.pdf')"

    # STEP 2: Paginate through thousands of results (retry once on 5xx; some tenants return 500 for Search API)
    next_url: Optional[str] = search_url
    while next_url:
        r = _graph_get_with_retry(next_url, token, timeout=60)
        if r.status_code >= 500 and next_url == search_url:
            # One retry for transient 500/502/503 on first request
            logger.debug("Search API returned %s, retrying once in 3s", r.status_code)
            time.sleep(3)
            r = _graph_get_with_retry(next_url, token, timeout=60)
        r.raise_for_status()
        data = r.json()
        for item in data.get("value", []):
            if "folder" in item:
                continue
            name = (item.get("name") or "").strip()
            if not name.lower().endswith(".pdf"):
                continue
            parent_name = _parent_folder_name_from_item(item)
            pref = item.get("parentReference") or {}
            path_str = pref.get("path") or ""
            if path_str and "root:/" in path_str:
                rel = path_str.split("root:/", 1)[-1].rstrip("/")
                item_path = f"{rel}/{name}".strip("/") if rel else name
            else:
                item_path = f"{parent_name}/{name}".strip("/") if parent_name else name
            web_url = item.get("webUrl") or _sharepoint_file_web_url(item["id"])
            all_items.append({
                "id": item["id"],
                "name": name,
                "folder": parent_name,
                "path": item_path,
                "eTag": item.get("eTag"),
                "last_modified": item.get("lastModifiedDateTime"),
                "webUrl": web_url,
            })
        next_url = data.get("@odata.nextLink")
        if next_url:
            logger.debug("Search API next page, collected %s so far", len(all_items))

    return all_items


def _list_pdfs_in_drive(
    token: str, drive_id: str, folder_path: str
) -> List[dict]:
    """
    List all PDF items under drive path, including inside every nested subfolder.
    Path is relative to drive root (e.g. Applications List.../IT Contracts and Agreements).
    Recurses into all subfolders (e.g. project-name folders) so every PDF under the path is returned.
    Tries root:/path first; on 404, resolves path segment-by-segment.
    Returns list of dicts: id (UniqueID), name, path, folder (immediate parent folder name for Excel).
    """
    base_path = (folder_path or "").strip().strip("/")
    all_items: List[dict] = []
    _progress_log_interval = 100  # log every N PDFs so user sees activity

    def recurse(
        next_url: Optional[str],
        current_folder_name: str,
        path_prefix: str,
    ) -> None:
        if not next_url:
            return
        r = _graph_get_with_retry(next_url, token, GRAPH_REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        for item in data.get("value", []):
            if "folder" in item:
                child_url = f"{GRAPH_BASE}/drives/{drive_id}/items/{item['id']}/children"
                recurse(
                    child_url,
                    item.get("name", ""),
                    f"{path_prefix}/{item.get('name', '')}".strip("/"),
                )
            elif item.get("name", "").lower().endswith(".pdf"):
                web_url = item.get("webUrl") or _sharepoint_file_web_url(item["id"])
                all_items.append({
                    "id": item["id"],
                    "name": item["name"],
                    "folder": current_folder_name,
                    "path": f"{path_prefix}/{item['name']}".strip("/"),
                    "eTag": item.get("eTag"),
                    "last_modified": item.get("lastModifiedDateTime"),
                    "webUrl": web_url,
                })
                n = len(all_items)
                if n % _progress_log_interval == 0 and n > 0:
                    logger.info("SharePoint recursive list: %s PDF(s) found so far...", n)
        next_link = data.get("@odata.nextLink")
        if next_link:
            recurse(next_link, current_folder_name, path_prefix)

    # 1) Try direct path (root:/path:/children)
    start_url: Optional[str] = None
    if base_path:
        url = f"{GRAPH_BASE}/drives/{drive_id}/root:/{quote(base_path, safe='')}:/children"
        r = _graph_get_with_retry(url, token, GRAPH_REQUEST_TIMEOUT)
        if r.status_code == 404:
            # 2) Fallback: resolve by segments, then list from that folder id
            folder_id = _resolve_folder_by_segments(token, drive_id, base_path)
            if folder_id:
                start_url = f"{GRAPH_BASE}/drives/{drive_id}/items/{folder_id}/children"
                logger.info("SharePoint path resolved by segments (direct path returned 404)")
            else:
                logger.warning("SharePoint path not found: %s", base_path)
                return []
        else:
            r.raise_for_status()
            start_url = url
    else:
        start_url = f"{GRAPH_BASE}/drives/{drive_id}/root/children"

    if not start_url:
        return []

    # First page
    r = _graph_get_with_retry(start_url, token, GRAPH_REQUEST_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    top_folder = base_path.split("/")[-1] if base_path else "root"
    path_prefix = base_path or "root"
    for item in data.get("value", []):
        if "folder" in item:
            child_url = f"{GRAPH_BASE}/drives/{drive_id}/items/{item['id']}/children"
            recurse(child_url, item.get("name", ""), f"{path_prefix}/{item.get('name', '')}".strip("/"))
        elif item.get("name", "").lower().endswith(".pdf"):
            web_url = item.get("webUrl") or _sharepoint_file_web_url(item["id"])
            all_items.append({
                "id": item["id"],
                "name": item["name"],
                "folder": top_folder,
                "path": f"{path_prefix}/{item['name']}".strip("/"),
                "eTag": item.get("eTag"),
                "last_modified": item.get("lastModifiedDateTime"),
                "webUrl": web_url,
            })
    next_link = data.get("@odata.nextLink")
    while next_link:
        r = _graph_get_with_retry(next_link, token, GRAPH_REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        for item in data.get("value", []):
            if "folder" in item:
                child_url = f"{GRAPH_BASE}/drives/{drive_id}/items/{item['id']}/children"
                recurse(child_url, item.get("name", ""), f"{path_prefix}/{item.get('name', '')}".strip("/"))
            elif item.get("name", "").lower().endswith(".pdf"):
                web_url = item.get("webUrl") or _sharepoint_file_web_url(item["id"])
                all_items.append({
                    "id": item["id"],
                    "name": item["name"],
                    "folder": top_folder,
                    "path": f"{path_prefix}/{item['name']}".strip("/"),
                    "eTag": item.get("eTag"),
                    "last_modified": item.get("lastModifiedDateTime"),
                    "webUrl": web_url,
                })
        next_link = data.get("@odata.nextLink")

    return all_items


def _list_pdfs_in_drive_iter(
    token: str, drive_id: str, folder_path: str
) -> Generator[dict, None, None]:
    """
    Same as _list_pdfs_in_drive but yields PDF items as they are found (for streaming;
    processing can start while listing continues in another thread).
    """
    base_path = (folder_path or "").strip().strip("/")
    count = 0
    _progress_log_interval = 100

    def recurse(
        next_url: Optional[str],
        current_folder_name: str,
        path_prefix: str,
    ) -> Generator[dict, None, None]:
        if not next_url:
            return
        r = _graph_get_with_retry(next_url, token, GRAPH_REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        for item in data.get("value", []):
            if "folder" in item:
                child_url = f"{GRAPH_BASE}/drives/{drive_id}/items/{item['id']}/children"
                yield from recurse(
                    child_url,
                    item.get("name", ""),
                    f"{path_prefix}/{item.get('name', '')}".strip("/"),
                )
            elif item.get("name", "").lower().endswith(".pdf"):
                web_url = item.get("webUrl") or _sharepoint_file_web_url(item["id"])
                out = {
                    "id": item["id"],
                    "name": item["name"],
                    "folder": current_folder_name,
                    "path": f"{path_prefix}/{item['name']}".strip("/"),
                    "eTag": item.get("eTag"),
                    "last_modified": item.get("lastModifiedDateTime"),
                    "webUrl": web_url,
                }
                yield out
                nonlocal count
                count += 1
                if count % _progress_log_interval == 0:
                    logger.info("SharePoint list (this run): %s PDF(s) listed so far...", count)
        next_link = data.get("@odata.nextLink")
        if next_link:
            yield from recurse(next_link, current_folder_name, path_prefix)

    start_url: Optional[str] = None
    if base_path:
        url = f"{GRAPH_BASE}/drives/{drive_id}/root:/{quote(base_path, safe='')}:/children"
        r = _graph_get_with_retry(url, token, GRAPH_REQUEST_TIMEOUT)
        if r.status_code == 404:
            folder_id = _resolve_folder_by_segments(token, drive_id, base_path)
            if folder_id:
                start_url = f"{GRAPH_BASE}/drives/{drive_id}/items/{folder_id}/children"
                logger.info("SharePoint path resolved by segments (direct path returned 404)")
            else:
                logger.warning("SharePoint path not found: %s", base_path)
                return
        else:
            r.raise_for_status()
            start_url = url
    else:
        start_url = f"{GRAPH_BASE}/drives/{drive_id}/root/children"

    if not start_url:
        return

    r = _graph_get_with_retry(start_url, token, GRAPH_REQUEST_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    top_folder = base_path.split("/")[-1] if base_path else "root"
    path_prefix = base_path or "root"
    for item in data.get("value", []):
        if "folder" in item:
            child_url = f"{GRAPH_BASE}/drives/{drive_id}/items/{item['id']}/children"
            yield from recurse(child_url, item.get("name", ""), f"{path_prefix}/{item.get('name', '')}".strip("/"))
        elif item.get("name", "").lower().endswith(".pdf"):
            web_url = item.get("webUrl") or _sharepoint_file_web_url(item["id"])
            out = {
                "id": item["id"],
                "name": item["name"],
                "folder": top_folder,
                "path": f"{path_prefix}/{item['name']}".strip("/"),
                "eTag": item.get("eTag"),
                "last_modified": item.get("lastModifiedDateTime"),
                "webUrl": web_url,
            }
            yield out
            count += 1
            if count % _progress_log_interval == 0:
                logger.info("SharePoint list (this run): %s PDF(s) listed so far...", count)
    next_link = data.get("@odata.nextLink")
    while next_link:
        r = _graph_get_with_retry(next_link, token, GRAPH_REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        for item in data.get("value", []):
            if "folder" in item:
                child_url = f"{GRAPH_BASE}/drives/{drive_id}/items/{item['id']}/children"
                yield from recurse(child_url, item.get("name", ""), f"{path_prefix}/{item.get('name', '')}".strip("/"))
            elif item.get("name", "").lower().endswith(".pdf"):
                web_url = item.get("webUrl") or _sharepoint_file_web_url(item["id"])
                out = {
                    "id": item["id"],
                    "name": item["name"],
                    "folder": top_folder,
                    "path": f"{path_prefix}/{item['name']}".strip("/"),
                    "eTag": item.get("eTag"),
                    "last_modified": item.get("lastModifiedDateTime"),
                    "webUrl": web_url,
                }
                yield out
                count += 1
                if count % _progress_log_interval == 0:
                    logger.info("SharePoint list (this run): %s PDF(s) listed so far...", count)
        next_link = data.get("@odata.nextLink")


def list_pdf_items_streaming() -> Generator[Tuple[dict, str], None, None]:
    """
    Yields (item, drive_id) as PDFs are discovered (recursive crawl). Use for extraction
    so processing can start as soon as the first PDF is found; listing continues in parallel.
    """
    token = get_access_token()
    site_id = _get_site_id(token, site_url=SHAREPOINT_SITE_URL)
    drive_id = _resolve_drive_id(token, site_id, drive_id_override=SHAREPOINT_DRIVE_ID)
    folder_path = (SHAREPOINT_DRIVE_PATH or "").strip().strip("/")
    try:
        items = _list_pdfs_via_search(token, drive_id, folder_path)
        for item in items:
            yield (item, drive_id)
        return
    except Exception:
        pass
    for item in _list_pdfs_in_drive_iter(token, drive_id, folder_path):
        yield (item, drive_id)


def list_contents_at_path(
    folder_path: str = "",
    token: Optional[str] = None,
    drive_id: Optional[str] = None,
    site_url: Optional[str] = None,
) -> List[dict]:
    """
    List immediate children (folders and files) at a given path in the site's default drive.
    Use this to browse the SharePoint site and discover the correct folder path.

    Args:
        folder_path: Path relative to drive root, e.g. "" for root, "pmo" for folder pmo,
                     "pmo/Solenis IT Contracts/Sub" for nested. Use "" to see top-level.
        token: Optional; if not provided, acquired via get_access_token().
        drive_id: Optional; if not provided, resolved from site.

    Returns:
        List of dicts: name, id, is_folder, path (full path from root for use in SHAREPOINT_DRIVE_PATH).
    """
    if token is None:
        token = get_access_token()
    if drive_id is None:
        site_id = _get_site_id(token, site_url=site_url)
        drive_id = _resolve_drive_id(token, site_id)
    base_path = (folder_path or "").strip().strip("/")
    if not base_path:
        url = f"{GRAPH_BASE}/drives/{drive_id}/root/children"
    else:
        url = f"{GRAPH_BASE}/drives/{drive_id}/root:/{quote(base_path, safe='')}:/children"
        r = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=GRAPH_REQUEST_TIMEOUT)
        if r.status_code == 404:
            folder_id = _resolve_folder_by_segments(token, drive_id, base_path)
            if folder_id is None:
                return []
            url = f"{GRAPH_BASE}/drives/{drive_id}/items/{folder_id}/children"
        else:
            r.raise_for_status()
    out: List[dict] = []
    next_url: Optional[str] = url
    while next_url:
        r = requests.get(next_url, headers={"Authorization": f"Bearer {token}"}, timeout=GRAPH_REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        for item in data.get("value", []):
            name = (item.get("name") or "").strip()
            item_path = f"{base_path}/{name}".strip("/") if base_path else name
            out.append({
                "name": name,
                "id": item["id"],
                "is_folder": "folder" in item,
                "path": item_path,
            })
        next_url = data.get("@odata.nextLink")
    return out


def browse_site_contents(
    folder_path: str = "",
    max_depth: int = 1,
    drive_id: Optional[str] = None,
    token: Optional[str] = None,
    site_url: Optional[str] = None,
) -> List[dict]:
    """
    Browse the connected SharePoint site: list contents at a path, optionally with one level of subfolders.

    Args:
        folder_path: Path relative to drive root; "" lists the root of the default document library.
        max_depth: 1 = only immediate children; 2 = include one level of subfolder contents (for each folder).
        drive_id: Optional; if not set, uses the site's default drive.

    Returns:
        List of items at folder_path. Each item: name, id, is_folder, path.
        If max_depth==2, folders also have "children" (list of their immediate contents).
    """
    if token is None:
        token = get_access_token()
    site_id = _get_site_id(token, site_url=site_url)
    if drive_id is None:
        drive_id = _get_drive_id(token, site_id)
    items = list_contents_at_path(folder_path, token=token, drive_id=drive_id, site_url=site_url)
    if max_depth < 2 or not items:
        return items
    result: List[dict] = []
    for it in items:
        row = dict(it)
        if it.get("is_folder"):
            row["children"] = list_contents_at_path(it["path"], token=token, drive_id=drive_id, site_url=site_url)
        result.append(row)
    return result


def get_site_connection():
    """
    Connect to the site and return drives plus default drive id.
    Use to list all drives, then navigate into a chosen drive's folders.

    Returns:
        Tuple of (token, site_id, drives_list, default_drive_id).
        drives_list: list of {name, id, driveType}.
    """
    token = get_access_token()
    site_id = _get_site_id(token)
    drives = list_site_drives(token=token, site_id=site_id)
    default_drive_id = _get_drive_id(token, site_id)
    return token, site_id, drives, default_drive_id


def _format_first_page_summary(value: list) -> str:
    """Build a short summary of item types on first page: folders, then counts by extension."""
    folders = sum(1 for item in value if item.get("folder") is not None)
    by_ext: dict = {}
    for item in value:
        if item.get("folder") is not None:
            continue
        name = (item.get("name") or "").strip()
        if not name:
            continue
        ext = ""
        if "." in name:
            ext = "." + name.rsplit(".", 1)[-1].lower()
        else:
            ext = "(no ext)"
        by_ext[ext] = by_ext.get(ext, 0) + 1
    parts = []
    if folders:
        parts.append(f"{folders} folder(s)")
    for ext in sorted(by_ext.keys(), key=lambda x: (-by_ext[x], x)):
        parts.append(f"{by_ext[ext]} {ext}")
    return ", ".join(parts) if parts else "0 items"


def verify_sharepoint_path_reachable(
    token: Optional[str] = None,
    site_url: Optional[str] = None,
    drive_id: Optional[str] = None,
    folder_path: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Lightweight check: verify the configured drive/path is reachable and list first page only (no recursion).
    Returns a summary of all item types (folders and file extensions) on the first page.
    Returns (success, message).
    """
    try:
        token = token or get_access_token()
        site_id = _get_site_id(token, site_url=site_url)
        drive_id = _resolve_drive_id(token, site_id, drive_id_override=drive_id)
        folder_path = (folder_path if folder_path is not None else SHAREPOINT_DRIVE_PATH or "").strip().strip("/")
        if not folder_path:
            url = f"{GRAPH_BASE}/drives/{drive_id}/root/children"
            r = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=GRAPH_REQUEST_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            value = data.get("value", [])
            summary = _format_first_page_summary(value)
            return True, f"drive root reachable, first page: {summary}"
        folder_id = _resolve_folder_by_segments(token, drive_id, folder_path)
        if not folder_id:
            return False, "path not found (segment resolution failed)"
        url = f"{GRAPH_BASE}/drives/{drive_id}/items/{folder_id}/children"
        r = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=GRAPH_REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        value = data.get("value", [])
        summary = _format_first_page_summary(value)
        return True, f"path reachable, first page: {summary}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def list_pdf_items(
    access_token: Optional[str] = None,
    site_url: Optional[str] = None,
    drive_id: Optional[str] = None,
    drive_path: Optional[str] = None,
) -> List[dict]:
    """
    List all PDF items in the configured SharePoint drive path.
    Returns list of dicts: id (UniqueID), name, folder, path.
    """
    items, _ = list_pdf_items_with_drive(
        access_token=access_token,
        site_url=site_url,
        drive_id=drive_id,
        drive_path=drive_path,
    )
    return items


def list_pdf_items_with_drive(
    access_token: Optional[str] = None,
    site_url: Optional[str] = None,
    drive_id: Optional[str] = None,
    drive_path: Optional[str] = None,
) -> Tuple[List[dict], str]:
    """
    List all PDF items and return (items, drive_id). Use drive_id when calling download_file_bytes.
    Uses SHAREPOINT_DRIVE_ID if set (same drive as in interactive browse), else the site's default drive.
    Uses Graph Search API first (fast, paginated); falls back to recursive folder crawl if Search fails (e.g. 404/path).
    """
    token = access_token or get_access_token()
    resolved_site_url = (site_url or SHAREPOINT_SITE_URL or "").strip()
    site_id = _get_site_id(token, site_url=resolved_site_url)
    resolved_drive_id = _resolve_drive_id(token, site_id, drive_id_override=drive_id)
    folder_path = (drive_path if drive_path is not None else SHAREPOINT_DRIVE_PATH or "").strip().strip("/")
    try:
        items = _list_pdfs_via_search(token, resolved_drive_id, folder_path)
        for item in items:
            if not item.get("webUrl"):
                item["webUrl"] = _sharepoint_file_web_url(item.get("id", ""), site_url=resolved_site_url)
        logger.info("SharePoint: listed %s PDF(s) via Search API", len(items))
        return items, resolved_drive_id
    except Exception as e:
        # Many tenants return 500 for the Graph Search API; recursive list is the reliable fallback
        logger.info(
            "SharePoint Search API unavailable (%s), using recursive folder list: %s",
            type(e).__name__, e,
        )
    items = _list_pdfs_in_drive(token, resolved_drive_id, folder_path)
    for item in items:
        if not item.get("webUrl"):
            item["webUrl"] = _sharepoint_file_web_url(item.get("id", ""), site_url=resolved_site_url)
    logger.info("SharePoint: listed %s PDF(s) via recursive crawl", len(items))
    return items, resolved_drive_id


def download_file_bytes(
    item_id: str,
    drive_id: Optional[str] = None,
    access_token: Optional[str] = None,
    site_url: Optional[str] = None,
) -> bytes:
    """
    Download file content as bytes. If drive_id is None, use SHAREPOINT_DRIVE_ID or default drive.
    """
    token = access_token or get_access_token()
    if not drive_id:
        site_id = _get_site_id(token, site_url=site_url)
        drive_id = _resolve_drive_id(token, site_id)
    url = f"{GRAPH_BASE}/drives/{drive_id}/items/{item_id}/content"
    r = requests.get(
        url,
        headers={"Authorization": f"Bearer {token}"},
        timeout=60,
        allow_redirects=True,
    )
    r.raise_for_status()
    return r.content


def list_pdf_items_streaming(
    access_token: Optional[str] = None,
    site_url: Optional[str] = None,
    drive_id: Optional[str] = None,
    drive_path: Optional[str] = None,
) -> Generator[Tuple[dict, str], None, None]:
    """
    Yields (item, drive_id) as PDFs are discovered (recursive crawl). Use for extraction
    so processing can start as soon as the first PDF is found; listing continues in parallel.
    Accepts delegated user token/context for user-level permission enforcement.
    """
    token = access_token or get_access_token()
    resolved_site_url = (site_url or SHAREPOINT_SITE_URL or "").strip()
    site_id = _get_site_id(token, site_url=resolved_site_url)
    resolved_drive_id = _resolve_drive_id(token, site_id, drive_id_override=drive_id)
    folder_path = (drive_path if drive_path is not None else SHAREPOINT_DRIVE_PATH or "").strip().strip("/")
    try:
        items = _list_pdfs_via_search(token, resolved_drive_id, folder_path)
        for item in items:
            if not item.get("webUrl"):
                item["webUrl"] = _sharepoint_file_web_url(item.get("id", ""), site_url=resolved_site_url)
            yield (item, resolved_drive_id)
        return
    except Exception:
        pass
    for item in _list_pdfs_in_drive_iter(token, resolved_drive_id, folder_path):
        if not item.get("webUrl"):
            item["webUrl"] = _sharepoint_file_web_url(item.get("id", ""), site_url=resolved_site_url)
        yield (item, resolved_drive_id)


# Run unit tests: python sharepoint_utils.py
if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main(["-v", "tests/test_sharepoint_utils.py"]))

