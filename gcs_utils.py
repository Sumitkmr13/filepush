"""
GCP Cloud Storage: upload Excel output to a bucket after each run (PDFs are not stored in GCS).
Uses GOOGLE_APPLICATION_CREDENTIALS (same as Vertex AI). PDF source is SharePoint only; PDFs are sent to Gemini for OCR only.
"""
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _client():
    """Lazy client to avoid import at module load."""
    from google.cloud import storage
    return storage.Client()


def is_gcs_output_configured() -> bool:
    """True if GCS_OUTPUT_BUCKET is set (we upload Excel after each run)."""
    from config import GCS_OUTPUT_BUCKET
    return bool(GCS_OUTPUT_BUCKET)


def upload_file_to_bucket(
    local_path: Path,
    bucket_name: str,
    blob_name: Optional[str] = None,
) -> str:
    """
    Upload a local file to gs://bucket_name/blob_name.
    Returns gs://bucket_name/blob_name.
    """
    blob_name = blob_name or Path(local_path).name
    client = _client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(str(local_path), content_type=_content_type(path=local_path))
    return f"gs://{bucket_name}/{blob_name}"


def upload_bytes_to_bucket(
    bucket_name: str,
    blob_name: str,
    data: bytes,
    content_type: str = "application/pdf",
) -> str:
    """
    Upload raw bytes to gs://bucket_name/blob_name. Used for storing processed PDFs.
    Returns gs://bucket_name/blob_name.
    """
    client = _client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(data, content_type=content_type)
    return f"gs://{bucket_name}/{blob_name}"


def download_file_from_bucket(
    bucket_name: str,
    blob_name: str,
    local_path: Path,
) -> bool:
    """
    Download gs://bucket_name/blob_name to local_path.
    Returns True on success, False if blob doesn't exist or download fails.
    """
    try:
        client = _client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        if not blob.exists():
            logger.info("GCS blob gs://%s/%s does not exist; skipping download.", bucket_name, blob_name)
            return False
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_path))
        logger.info("Downloaded gs://%s/%s → %s", bucket_name, blob_name, local_path)
        return True
    except Exception as e:
        logger.warning("Failed to download gs://%s/%s: %s", bucket_name, blob_name, e)
        return False


def _content_type(path: Optional[Path] = None, suffix: Optional[str] = None) -> str:
    s = (path.suffix if path else suffix) or ""
    if s.lower() in (".xlsx",):
        return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    if s.lower() == ".pdf":
        return "application/pdf"
    return "application/octet-stream"


def _safe_blob_path(folder: str, filename: str, prefix: str) -> str:
    """Build a safe blob path: prefix/folder/filename (no leading slashes, safe chars)."""
    folder = re.sub(r"[^\w\-./]", "_", (folder or "").strip())
    filename = (filename or "document.pdf").strip()
    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"
    parts = [p for p in (prefix.strip().strip("/"), folder, filename) if p]
    return "/".join(parts)


# Run unit tests: pytest tests/test_gcs_utils.py -v
if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main(["-v", "tests/test_gcs_utils.py"]))
