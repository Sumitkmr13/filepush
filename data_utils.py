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
    """Normalize amounts into $###,### format; leave unchanged if not parseable."""
    if not isinstance(value, str) or not value.strip():
        return value
    cleaned = re.sub(r"[^\d.,]", "", value)
    try:
        num = float(cleaned.replace(",", ""))
        return f"${num:,.0f}"
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
        re.compile(r"(?i)\s*[\(_\-](\d{1,3})\s*[\)_\-]?\s*$"),
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


def dedupe_same_content_latest_revision(
    df: pd.DataFrame,
    output_fields: List[str],
) -> pd.DataFrame:
    """
    When multiple rows represent the same agreement (same extracted business fields) and the same
    logical document name differing only by revision markers in Filename, keep a single row:
    prefer the highest revision number parsed from Filename, then the latest End Date.

    Applies to both SOW and Invoice Excels (uses *output_fields* for the content signature).
    """
    if df.empty or len(df) < 2:
        return df
    field_cols = [c for c in output_fields if c in df.columns]
    if not field_cols:
        return df

    buckets: Dict[Tuple[str, str], list] = {}
    for _, row in df.iterrows():
        sig = content_signature(row, field_cols)
        base, rev = parse_filename_base_and_revision(str(row.get("Filename", "") or ""))
        key = (sig, base)
        buckets.setdefault(key, []).append((rev, row))

    kept: list[pd.Series] = []
    removed = 0
    for key, items in buckets.items():
        if len(items) == 1:
            kept.append(items[0][1])
            continue
        best_rev, best_row = max(
            items,
            key=lambda t: (t[0], _end_date_for_sort(t[1])),
        )
        kept.append(best_row)
        removed += len(items) - 1

    if removed:
        logger.info(
            "Deduped %s row(s) (same extracted content + same filename base): kept latest revision / latest end date.",
            removed,
        )
    return pd.DataFrame(kept).reset_index(drop=True)


# Run unit tests: python data_utils.py
if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main(["-v", "tests/test_data_utils.py"]))
