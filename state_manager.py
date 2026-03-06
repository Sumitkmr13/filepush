"""
Smart Resume: state manager for processed SharePoint items.
Tracks UniqueIDs and optional eTag/last_modified so we can reprocess when files change.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from config import EXTRACTION_STATE_PATH

logger = logging.getLogger(__name__)


def _load_state() -> dict:
    if not EXTRACTION_STATE_PATH.exists():
        return {"processed_ids": [], "processed_meta": {}, "last_updated": None}
    try:
        with open(EXTRACTION_STATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "processed_meta" not in data:
            data["processed_meta"] = {}
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Could not load extraction state: %s", e)
        return {"processed_ids": [], "processed_meta": {}, "last_updated": None}


def _save_state(data: dict) -> None:
    EXTRACTION_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    data["last_updated"] = datetime.utcnow().isoformat() + "Z"
    with open(EXTRACTION_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def get_processed_ids() -> Set[str]:
    """Return set of SharePoint UniqueIDs already in state (to skip on resume)."""
    data = _load_state()
    return set(data.get("processed_ids", []))


def get_processed_meta() -> Dict[str, Dict[str, str]]:
    """Return map id -> { eTag, last_modified } for change detection (reprocess when changed)."""
    data = _load_state()
    return dict(data.get("processed_meta", {}))


def should_process_item(item: dict, processed_ids: Set[str], processed_meta: Dict[str, Dict[str, str]]) -> bool:
    """
    True if item should be queued for processing: never processed, or processed but file changed (eTag/last_modified).
    item may have "eTag" and "last_modified" (from Graph lastModifiedDateTime).
    """
    return classify_item(item, processed_ids, processed_meta) != "up_to_date"


def classify_item(
    item: dict, processed_ids: Set[str], processed_meta: Dict[str, Dict[str, str]]
) -> str:
    """
    Return "new" | "changed" | "up_to_date" for scanning/reporting.
    "new" = never processed; "changed" = processed but eTag/last_modified changed; "up_to_date" = skip.
    """
    uid = item.get("id")
    if not uid:
        return "up_to_date"
    if uid not in processed_ids:
        return "new"
    meta = processed_meta.get(uid) or {}
    if not meta:
        return "up_to_date"
    cur_etag = (item.get("eTag") or "").strip()
    cur_mod = (item.get("last_modified") or "").strip()
    if cur_etag and meta.get("eTag") != cur_etag:
        return "changed"
    if cur_mod and meta.get("last_modified") != cur_mod:
        return "changed"
    return "up_to_date"


def filter_unprocessed(unique_ids: List[str]) -> List[str]:
    """Return only UniqueIDs that are not yet in state."""
    processed = get_processed_ids()
    return [uid for uid in unique_ids if uid not in processed]


def mark_processed(
    unique_id: str,
    eTag: Optional[str] = None,
    last_modified: Optional[str] = None,
    name: Optional[str] = None,
    folder: Optional[str] = None,
    doc_type: Optional[str] = None,
) -> None:
    """
    Append a UniqueID to state and persist.
    Stores eTag/last_modified for change detection, name/folder for readability, and doc_type for debug.
    doc_type: "sow" | "invoice" | raw Document Type string from LLM.
    """
    data = _load_state()
    ids = data.get("processed_ids", [])
    meta = data.get("processed_meta", {})
    if unique_id not in ids:
        ids.append(unique_id)
        data["processed_ids"] = ids
    entry = {k: v for k, v in (("eTag", eTag), ("last_modified", last_modified)) if v is not None}
    if name is not None:
        entry["name"] = name
    if folder is not None:
        entry["folder"] = folder
    if doc_type is not None:
        entry["doc_type"] = doc_type
    meta[unique_id] = entry
    data["processed_meta"] = meta
    _save_state(data)
    logger.debug("Marked as processed: %s (name=%s, doc_type=%s)", unique_id, name, doc_type)


def mark_processed_batch(unique_ids: List[str]) -> None:
    """Append multiple UniqueIDs to state and persist once."""
    if not unique_ids:
        return
    data = _load_state()
    ids = data.get("processed_ids", [])
    for uid in unique_ids:
        if uid not in ids:
            ids.append(uid)
    data["processed_ids"] = ids
    _save_state(data)
    logger.debug("Marked batch processed: %s", unique_ids)


def get_state_summary() -> dict:
    """Return summary for API (processed_count, last_updated, by_doc_type counts)."""
    data = _load_state()
    meta = data.get("processed_meta", {})
    sow_count = sum(1 for m in meta.values() if m.get("doc_type") == "sow")
    invoice_count = sum(1 for m in meta.values() if m.get("doc_type") == "invoice")
    unknown_count = len(meta) - sow_count - invoice_count
    return {
        "processed_count": len(data.get("processed_ids", [])),
        "last_updated": data.get("last_updated"),
        "meta_tracked": len(meta),
        "by_doc_type": {"sow": sow_count, "invoice": invoice_count, "unknown": unknown_count},
    }


def get_processed_items_by_type(doc_type: str) -> List[Dict[str, Any]]:
    """Return list of processed items filtered by doc_type ('sow' or 'invoice')."""
    data = _load_state()
    meta = data.get("processed_meta", {})
    return [
        {"id": uid, **info}
        for uid, info in meta.items()
        if info.get("doc_type") == doc_type
    ]


# Run unit tests: python state_manager.py
if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main(["-v", "tests/test_state_manager.py"]))
