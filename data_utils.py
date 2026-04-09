"""
Data cleaning and deduplication for SOW/Invoice extraction results.
"""
import hashlib
import re
import logging
import calendar
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def clean_date(value: str, month_year_position: str = "start") -> str:
    """Normalize date strings into YYYY-MM-DD format if possible."""
    if not isinstance(value, str) or not value.strip():
        return value
    raw = value.strip()

    month_year_patterns = (
        (r"^(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})$", "%B %Y"),
        (r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\s+(\d{4})$", "%b %Y"),
        (r"^(\d{1,2})/(\d{4})$", None),
        (r"^(\d{1,2})-(\d{4})$", None),
    )
    for pattern, fmt in month_year_patterns:
        if re.match(pattern, raw, flags=re.IGNORECASE):
            try:
                if fmt:
                    parsed = pd.to_datetime(raw, format=fmt, errors="raise")
                    year = int(parsed.year)
                    month = int(parsed.month)
                else:
                    m = re.match(pattern, raw)
                    if not m:
                        break
                    month = int(m.group(1))
                    year = int(m.group(2))
                day = 1 if month_year_position == "start" else calendar.monthrange(year, month)[1]
                return f"{year:04d}-{month:02d}-{day:02d}"
            except Exception:
                return value
    try:
        parsed = pd.to_datetime(raw, errors="coerce")
        if pd.isna(parsed):
            return value
        return parsed.strftime("%Y-%m-%d")
    except Exception:
        return value


def _split_currency(raw: str) -> Tuple[str, str, str]:
    prefix_match = re.match(r"^\s*(\$|€|£|¥|₹|AUD\b|CAD\b|USD\b|EUR\b|GBP\b|INR\b|JPY\b|CNY\b|RUB\b)\s*", raw, flags=re.IGNORECASE)
    suffix_match = re.search(r"\s*(AUD|CAD|USD|EUR|GBP|INR|JPY|CNY|RUB)\s*$", raw, flags=re.IGNORECASE)
    prefix = prefix_match.group(1).upper() if prefix_match else ""
    suffix = suffix_match.group(1).upper() if suffix_match else ""
    core = raw
    if prefix_match:
        core = core[prefix_match.end():]
    if suffix_match:
        core = core[:suffix_match.start()]
    return prefix, core.strip(), suffix


def _parse_amount_core(number_text: str) -> Optional[Tuple[float, int]]:
    cleaned = re.sub(r"[^\d.,-]", "", number_text or "")
    if not cleaned or not re.search(r"\d", cleaned):
        return None

    decimal_places = 0
    normalized = cleaned
    if "." in cleaned and "," in cleaned:
        if cleaned.rfind(",") > cleaned.rfind("."):
            decimal_places = len(cleaned) - cleaned.rfind(",") - 1
            normalized = cleaned.replace(".", "").replace(",", ".")
        else:
            decimal_places = len(cleaned) - cleaned.rfind(".") - 1
            normalized = cleaned.replace(",", "")
    elif "," in cleaned:
        digits_after = len(cleaned) - cleaned.rfind(",") - 1
        if 0 < digits_after <= 2:
            decimal_places = digits_after
            normalized = cleaned.replace(".", "").replace(",", ".")
        else:
            normalized = cleaned.replace(",", "")
    elif "." in cleaned:
        digits_after = len(cleaned) - cleaned.rfind(".") - 1
        if 0 < digits_after <= 2:
            decimal_places = digits_after
            normalized = cleaned.replace(",", "")
        else:
            normalized = cleaned.replace(".", "")

    try:
        return float(normalized), decimal_places
    except Exception:
        return None


def clean_amount(value: str) -> str:
    """Normalize amount formatting while preserving currency and decimal precision."""
    if not isinstance(value, str) or not value.strip():
        return value
    raw = value.strip()
    prefix, number_text, suffix = _split_currency(raw)
    parsed = _parse_amount_core(number_text)
    if not parsed:
        return value
    num, decimal_places = parsed

    number_fmt = f"{{:,.{decimal_places}f}}" if decimal_places > 0 else "{:,.0f}"
    number_out = number_fmt.format(num)
    if prefix:
        return f"{prefix}{number_out}" if prefix in {"$", "€", "£", "¥", "₹"} else f"{prefix} {number_out}"
    if suffix:
        return f"{number_out} {suffix}"
    return f"${number_out}"


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


def _date_for_sort(row: pd.Series, col: str = "End Date") -> pd.Timestamp:
    d = pd.to_datetime(row.get(col), errors="coerce")
    if pd.isna(d):
        return pd.Timestamp.min
    return d


_CONTRACT_ID_SUFFIX_RE = re.compile(r"[-_](\d{1,3})$")


def normalize_contract_id(cid: Any) -> str:
    """
    Lowercase + strip whitespace for stable grouping.
    Does NOT strip suffixes — suffix stripping is context-dependent (only when a
    matching base ID exists in the dataset). See _build_contract_id_groups.
    """
    if cid is None:
        return ""
    # Pandas may pass NaN/float values when Excel cells are empty/mixed-type.
    if pd.isna(cid):
        return ""
    return str(cid).strip().lower()


def _strip_version_suffix(cid: str) -> Tuple[str, int]:
    """
    Try to split a Contract ID into (base, suffix_number).
    '4533232908-1' → ('4533232908', 1);  '4533232908' → ('4533232908', 0).
    """
    m = _CONTRACT_ID_SUFFIX_RE.search(cid)
    if m:
        return cid[: m.start()], int(m.group(1))
    return cid, 0


def _build_contract_id_groups(raw_ids: List[str]) -> Dict[str, str]:
    """
    Given a list of raw Contract IDs, build a mapping: raw_id → group_base.
    Two IDs group together only when one is a suffix-stripped version of the other
    AND that base actually exists (or another suffix variant of it exists).
    E.g. ['4533232908', '4533232908-1'] → both map to '4533232908'.
    But  ['11-005', '11-005-1'] → both map to '11-005' (not '11').
    """
    normalized = {rid: normalize_contract_id(rid) for rid in raw_ids}
    stripped = {rid: _strip_version_suffix(normalized[rid]) for rid in raw_ids}
    all_norm = set(normalized.values())

    mapping: Dict[str, str] = {}
    for rid in raw_ids:
        norm = normalized[rid]
        base, _ = stripped[rid]
        if base != norm and base in all_norm:
            mapping[rid] = base
        else:
            mapping[rid] = norm
    return mapping


def dedupe_by_contract_id(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    When multiple files produce rows with the same (or suffix-related) Contract ID,
    keep only the rows from the **latest version** of that contract.

    Versions are identified by **Filename** — each distinct PDF is one version.
    This handles:
      a) Different filenames, same Contract ID (e.g. two PDFs both say '11 005')
      b) Suffix variants (e.g. '4533232908' vs '4533232908-1' from different PDFs)

    Latest version is determined by:
      1. Most recent Start Date across all rows of that file (later = newer version).
      2. Most recent End Date as tiebreaker.
      3. Highest Contract ID suffix number (-2 > -1 > none) as final tiebreaker.
    """
    if df.empty or len(df) < 2 or "Contract ID" not in df.columns:
        return df, []

    raw_ids = [str(row.get("Contract ID", "") or "").strip() for _, row in df.iterrows()]
    id_to_group = _build_contract_id_groups(raw_ids)

    # buckets[group_base][filename] = list of row indices
    buckets: Dict[str, Dict[str, list]] = {}
    for idx, row in df.iterrows():
        raw_cid = str(row.get("Contract ID", "") or "").strip()
        group = id_to_group.get(raw_cid, normalize_contract_id(raw_cid))
        if not group:
            continue
        fname = str(row.get("Filename", "") or "").strip()
        buckets.setdefault(group, {}).setdefault(fname, []).append(idx)

    drop_indices: set = set()
    dedup_log: List[Dict[str, Any]] = []
    for group_base, file_versions in buckets.items():
        if len(file_versions) <= 1:
            continue

        def _version_sort_key(fname: str) -> Tuple:
            indices = file_versions[fname]
            rows_sub = df.loc[indices]
            best_start = max(
                (_date_for_sort(r, "Start Date") for _, r in rows_sub.iterrows()),
                default=pd.Timestamp.min,
            )
            best_end = max(
                (_date_for_sort(r, "End Date") for _, r in rows_sub.iterrows()),
                default=pd.Timestamp.min,
            )
            raw_cids = rows_sub["Contract ID"].astype(str).unique()
            suffix_num = max(
                (_strip_version_suffix(normalize_contract_id(c))[1] for c in raw_cids),
                default=0,
            )
            return (best_start, best_end, suffix_num)

        sorted_files = sorted(file_versions.keys(), key=_version_sort_key, reverse=True)
        winner = sorted_files[0]
        losers = sorted_files[1:]

        winner_key = _version_sort_key(winner)
        winner_start = winner_key[0].isoformat() if winner_key[0] != pd.Timestamp.min else ""

        loser_indices = []
        for fn in losers:
            loser_indices.extend(file_versions[fn])
            loser_key = _version_sort_key(fn)
            loser_start = loser_key[0].isoformat() if loser_key[0] != pd.Timestamp.min else ""
            loser_cids = sorted(set(str(df.loc[i, "Contract ID"]) for i in file_versions[fn]))
            reason_parts = []
            if winner_start and loser_start and winner_key[0] > loser_key[0]:
                reason_parts.append(f"kept file has later Start Date ({winner_start} vs {loser_start})")
            elif winner_key[1] > loser_key[1]:
                reason_parts.append("kept file has later End Date")
            elif winner_key[2] > loser_key[2]:
                reason_parts.append(f"kept file has higher Contract ID suffix ({winner_key[2]} vs {loser_key[2]})")
            else:
                reason_parts.append("kept file appeared first with same dates")
            dedup_log.append({
                "dropped_file": fn,
                "dropped_contract_ids": loser_cids,
                "kept_file": winner,
                "kept_contract_id": sorted(set(str(df.loc[i, "Contract ID"]) for i in file_versions[winner])),
                "group_base": group_base,
                "reason": "Contract ID duplicate: " + "; ".join(reason_parts),
                "dedup_type": "contract_id",
            })
        drop_indices.update(loser_indices)

        if loser_indices:
            logger.info(
                "Contract ID dedup [%s]: keeping '%s' (%s rows), dropping older version(s): %s",
                group_base, winner, len(file_versions[winner]),
                {fn: len(file_versions[fn]) for fn in losers},
            )

    if drop_indices:
        logger.info(
            "Contract ID dedup total: dropped %s row(s) from older contract versions.",
            len(drop_indices),
        )
        return df.drop(index=list(drop_indices)).reset_index(drop=True), dedup_log
    return df, dedup_log


def dedupe_keep_latest_revision(
    df: pd.DataFrame,
    output_fields: List[str],
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
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
        return df, []

    # Build buckets keyed by filename base (revision markers stripped).
    # Each bucket entry: (revision_num, row_index, row).
    buckets: Dict[str, list] = {}
    for idx, row in df.iterrows():
        base, rev = parse_filename_base_and_revision(str(row.get("Filename", "") or ""))
        buckets.setdefault(base, []).append((rev, idx, row))

    kept_indices: list = []
    removed = 0
    dedup_log: List[Dict[str, Any]] = []
    for base, items in buckets.items():
        if len(items) == 1:
            kept_indices.append(items[0][1])
            continue

        max_rev = max(r for r, _, _ in items)
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
            for dname in dropped_names:
                d_rev = max(r for r, _, row in dropped_items if str(row.get("Filename", "")) == dname)
                dedup_log.append({
                    "dropped_file": dname,
                    "dropped_contract_ids": [],
                    "kept_file": kept_names[0] if len(kept_names) == 1 else kept_names,
                    "kept_contract_id": [],
                    "group_base": base,
                    "reason": f"Filename revision duplicate: kept rev {max_rev}, dropped rev {d_rev}",
                    "dedup_type": "filename_revision",
                })

    if removed:
        logger.info(
            "Revision dedup total: dropped %s row(s) from older revisions across all filename groups.",
            removed,
        )
    return df.loc[kept_indices].reset_index(drop=True), dedup_log


# Run unit tests: python data_utils.py
if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main(["-v", "tests/test_data_utils.py"]))
