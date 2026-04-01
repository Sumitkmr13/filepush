"""
Data cleaning and deduplication for SOW/Invoice extraction results.
"""
import hashlib
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def clean_date(value: str) -> str:
    """Normalize date strings into YYYY-MM-DD format if possible."""
    try:
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            return value
        return parsed.strftime("%Y-%m-%d")
    except Exception:
        return value


def clean_amount(value: str) -> str:
    """Normalize amount formatting while preserving explicit currency text/symbols."""
    if not isinstance(value, str) or not value.strip():
        return value
    raw = value.strip()
    has_currency_code = bool(re.search(r"\b[A-Z]{3}\b", raw))
    if has_currency_code:
        return raw
    prefix_match = re.search(r"(\$|€|£|¥|₹|AUD|CAD|USD|EUR|GBP|INR|JPY|CNY|RUB)", raw, flags=re.IGNORECASE)
    prefix = (prefix_match.group(1).upper() if prefix_match else "$")
    cleaned = re.sub(r"[^\d.,]", "", value)
    try:
        num = float(cleaned.replace(",", ""))
        return f"{prefix}{num:,.0f}" if prefix in {"$", "€", "£", "¥", "₹"} else f"{prefix} {num:,.0f}"
    except Exception:
        return value


def remove_duplicate_entries(
    df: pd.DataFrame,
    fields: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Remove duplicate rows (keep first).
    Uses *fields* as the dedup key; if None, falls back to all non-meta columns.
    Filename and SharePoint URL are always excluded from the key.
    """
    if df.empty:
        return df
    exclude = {"Filename", "SharePoint URL", "Error", "_web_url"}
    if fields is not None:
        check_cols = [f for f in fields if f in df.columns]
    else:
        check_cols = [c for c in df.columns if c not in exclude]
    if not check_cols:
        return df
    result_df = df.copy()
    duplicates = result_df.duplicated(subset=check_cols, keep="first")
    if duplicates.any():
        result_df = result_df[~duplicates]
        logger.info("Removed %s duplicate entries", duplicates.sum())
    return result_df


def _normalize_for_signature(value: object) -> str:
    s = "" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value)
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def content_signature(row: pd.Series, field_cols: List[str]) -> str:
    """Stable hash of business fields (same document content → same signature)."""
    parts = [_normalize_for_signature(row.get(c, "")) for c in field_cols if c in row.index]
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def parse_filename_base_and_revision(filename: str) -> Tuple[str, int]:
    """
    Strip common revision/version suffixes from the filename stem and return (base_key, revision_num).
    Higher revision_num means a newer revision when filenames differ only by rev markers.
    """
    stem = Path(filename or "").stem
    rev = 0
    s = stem
    # Apply from the end repeatedly (max 12 passes)
    end_patterns = [
        re.compile(r"(?i)[\s\-_]*(?:\(|\[)?(?:rev|revision|ver|version)\s*\.?\s*(\d+)(?:\)|\])?\s*$"),
        re.compile(r"(?i)[\s\-_]+r\s*(\d+)\s*$"),
        re.compile(r"(?i)[\s\-_]+v\s*(\d+)\s*$"),
        re.compile(r"(?i)[\s\-_]+(?:final|draft)\s*(\d+)\s*$"),
        re.compile(r"(?i)\s*\((\d{1,3})\)\s*$"),
    ]
    for _ in range(12):
        matched = False
        for pat in end_patterns:
            m = pat.search(s)
            if m:
                try:
                    rev = max(rev, int(m.group(1)))
                except (ValueError, IndexError):
                    pass
                s = s[: m.start()].rstrip(" -_")
                matched = True
                break
        if not matched:
            break
    s = re.sub(r"\s+", " ", s).strip(" -_").lower()
    return s, rev


def _end_date_for_sort(row: pd.Series) -> pd.Timestamp:
    d = pd.to_datetime(row.get("End Date"), errors="coerce")
    if pd.isna(d):
        return pd.Timestamp.min
    return d


def dedupe_keep_latest_revision(
    df: pd.DataFrame,
    output_fields: List[str],
) -> pd.DataFrame:
    """
    When multiple rows share the same logical document name (differing only by revision
    markers like 'Revision 0', 'Revision 1', 'v2', etc.), keep ONLY the rows from the
    highest revision — the latest revision always supersedes earlier ones.

    All PDFs are still processed by the agent (for comparison / debugging), but only the
    final revision's data appears in the output Excel.

    Grouping is by filename base only (not content). Within each group, the highest
    revision number wins; ties broken by latest End Date.
    """
    if df.empty or len(df) < 2:
        return df

    # Build buckets keyed by filename base (revision markers stripped).
    # Each bucket entry: (revision_num, row_index, row).
    buckets: Dict[str, list] = {}
    for idx, row in df.iterrows():
        base, rev = parse_filename_base_and_revision(str(row.get("Filename", "") or ""))
        buckets.setdefault(base, []).append((rev, idx, row))

    kept_indices: list = []
    removed = 0
    for base, items in buckets.items():
        if len(items) == 1:
            kept_indices.append(items[0][1])
            continue

        # Find the maximum revision number in the group.
        max_rev = max(r for r, _, _ in items)

        # Keep all rows from the highest revision (a multi-line-item invoice
        # produces multiple rows with the same filename — keep them all).
        best_items = [(r, i, row) for r, i, row in items if r == max_rev]
        dropped_items = [(r, i, row) for r, i, row in items if r != max_rev]

        for _, i, _ in best_items:
            kept_indices.append(i)
        removed += len(dropped_items)

        if dropped_items:
            dropped_names = sorted(set(str(row.get("Filename", "")) for _, _, row in dropped_items))
            kept_names = sorted(set(str(row.get("Filename", "")) for _, _, row in best_items))
            logger.info(
                "Revision dedup [%s]: keeping %s (rev %s), dropping %s older revision(s): %s",
                base, kept_names, max_rev, len(dropped_items), dropped_names,
            )

    if removed:
        logger.info(
            "Revision dedup total: dropped %s row(s) from older revisions across all filename groups.",
            removed,
        )
    return df.loc[kept_indices].reset_index(drop=True)


# Run unit tests: python data_utils.py
if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main(["-v", "tests/test_data_utils.py"]))
