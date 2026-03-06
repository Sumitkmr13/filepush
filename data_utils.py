"""
Data cleaning and deduplication for SOW/Invoice extraction results.
"""
import re
import logging
from typing import List, Optional

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


# Run unit tests: python data_utils.py
if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main(["-v", "tests/test_data_utils.py"]))
