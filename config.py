"""
Application configuration from environment variables.
GCP: GOOGLE_APPLICATION_CREDENTIALS (path to service account JSON); project derived from it if not set.
SharePoint: MSAL client id, secret, tenant id.
"""
import json
import os
from pathlib import Path

# ------------------ Paths ------------------
APP_ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", APP_ROOT / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
EXTRACTION_STATE_PATH = DATA_DIR / "extraction_state.json"
EXCEL_OUTPUT_DIR = DATA_DIR

# Two separate cumulative Excels — one per document type
EXCEL_SOW_FILENAME = "contract_metrics.xlsx"
EXCEL_SOW_PATH = DATA_DIR / EXCEL_SOW_FILENAME

EXCEL_INVOICE_FILENAME = "license_metrics.xlsx"
EXCEL_INVOICE_PATH = DATA_DIR / EXCEL_INVOICE_FILENAME

# Backward-compat alias (used by older imports / GCS upload key)
EXCEL_OUTPUT_FILENAME = EXCEL_SOW_FILENAME
EXCEL_OUTPUT_PATH = EXCEL_SOW_PATH

# ------------------ Extraction fields ------------------
# Full extraction field list sent to the LLM for every PDF.
# "Document Type" is used to route each row to the correct Excel; it is not shown as a column.
FIELDS = [
    "Document Type",        # License Invoice | SOW Document — routing, not shown in Excel
    "Original Language",    # From OCR tag; not written to invoice Excel columns
    # SOW Document fields
    "Contract Name",
    "Commercial Value",
    "Owner/Contact",
    # License Invoice fields (invoice_results.xlsx)
    "Contract ID",
    "Vendor",
    "Customer",
    "Contract Type",
    "Billing Frequency",
    "Products / Modules",
    "Currency",
    "TCV",
    "Pricing Model",
    "Licenses Purchased",
    # Shared
    "Start Date",
    "End Date",
]

# Columns written to sow_results.xlsx  (excludes Document Type; Filename is always first)
SOW_FIELDS = [
    "Original Language",
    "Contract Name",
    "Start Date",
    "End Date",
    "Commercial Value",
    "Currency",        # currency for Commercial Value
    "Owner/Contact",
]

# Columns written to invoice_results.xlsx (Filename + SharePoint URL added by main.py)
# Parent fields are constant for the whole document; line-item fields vary per table/entry.
INVOICE_PARENT_FIELDS = [
    "Contract ID",
    "Vendor",
    "Customer",
    "Contract Type",
    "Billing Frequency",
    "Currency",
    "Start Date",
    "End Date",
]

INVOICE_LINE_FIELDS = [
    "Products / Modules",
    "TCV",
    "Pricing Model",
    "Licenses Purchased",
]

# Full ordered column list for the Excel
INVOICE_FIELDS = INVOICE_PARENT_FIELDS + INVOICE_LINE_FIELDS


def _get_gcp_project_from_credentials() -> str:
    """Read project_id from the service account JSON pointed to by GOOGLE_APPLICATION_CREDENTIALS."""
    path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not path or not os.path.isfile(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("project_id", "")
    except (json.JSONDecodeError, OSError):
        return ""


# ------------------ GCP / Vertex AI ------------------
# Project: from env or from service account JSON (prefer not setting GCP_PROJECT in env).
GCP_PROJECT = os.environ.get("GCP_PROJECT", "").strip() or _get_gcp_project_from_credentials()
GCP_LOCATION = os.environ.get("GCP_LOCATION", "us-central1")
# Default: Gemini 1.5 Flash for high-volume extraction (speed/cost). Override with VERTEX_MODEL for 2.5+.
VERTEX_MODEL = os.environ.get("VERTEX_MODEL", "gemini-1.5-flash")
VERTEX_EMBEDDING_MODEL = os.environ.get("VERTEX_EMBEDDING_MODEL", "text-embedding-004")

# ------------------ SharePoint ------------------
SHAREPOINT_CLIENT_ID = os.environ.get("SHAREPOINT_CLIENT_ID", "")
SHAREPOINT_CLIENT_SECRET = os.environ.get("SHAREPOINT_CLIENT_SECRET", "")
SHAREPOINT_TENANT_ID = os.environ.get("SHAREPOINT_TENANT_ID", "")
SHAREPOINT_SITE_URL = (os.environ.get("SHAREPOINT_SITE_URL", "") or "").strip().rstrip("/") or ""
# Optional: if set, use this drive (document library) instead of the site's default. Required when your folder is in a non-default library.
SHAREPOINT_DRIVE_ID = (os.environ.get("SHAREPOINT_DRIVE_ID", "") or "").strip()
SHAREPOINT_DRIVE_PATH = (os.environ.get("SHAREPOINT_DRIVE_PATH", "") or "").strip().strip("/")  # no leading/trailing slashes
SHAREPOINT_SCOPES = ["https://graph.microsoft.com/.default"]

# ------------------ GCP Cloud Storage (optional) ------------------
# If set, after each run we upload sow_results.xlsx to the bucket. PDFs are not uploaded (only sent to Gemini for OCR).
# Same credentials as Vertex AI. PDF source is always SharePoint only.
GCS_OUTPUT_BUCKET = os.environ.get("GCS_OUTPUT_BUCKET", "").strip()
GCS_PDF_STORAGE_PREFIX = os.environ.get("GCS_PDF_STORAGE_PREFIX", "pdfs").strip() or "pdfs"  # kept for compatibility; PDFs are no longer uploaded

# ------------------ App ------------------
BASE_PATH = Path(os.environ.get("BASE_PATH", APP_ROOT / "Data"))
INDEX_ROOT = APP_ROOT / "indexes"
# If > 0, background thread scans SharePoint every N minutes and runs extraction when new/changed files are found.
EXTRACTION_MONITOR_INTERVAL_MINUTES = int(os.environ.get("EXTRACTION_MONITOR_INTERVAL_MINUTES", "0") or "0")

# Run unit tests for this module: python config.py
if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main(["-v", "tests/test_config.py"]))
