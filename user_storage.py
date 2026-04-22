"""
User-scoped storage paths for per-user isolation.
"""
from pathlib import Path
import re
from typing import Dict

from config import DATA_DIR


def _safe_user_id(user_id: str) -> str:
    s = (user_id or "").strip()
    if not s:
        s = "anonymous"
    return re.sub(r"[^a-zA-Z0-9._-]", "_", s)


def user_base_dir(user_id: str) -> Path:
    uid = _safe_user_id(user_id)
    return DATA_DIR / "users" / uid


def user_paths(user_id: str) -> Dict[str, Path]:
    base = user_base_dir(user_id)
    return {
        "base_dir": base,
        "state_path": base / "extraction_state.json",
        "excel_sow_path": base / "contract_metrics.xlsx",
        "excel_invoice_path": base / "license_metrics.xlsx",
        "context_path": base / "context.json",
    }


def user_blob_prefix(user_id: str) -> str:
    return f"users/{_safe_user_id(user_id)}"

