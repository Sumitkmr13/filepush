"""
Single-shot field extraction using Gemini 2.5 Flash on Vertex AI.

Replaces the RAG/LlamaIndex chunking approach with one full-context request per document.
OCR is still performed via Gemini native multimodal (PDF bytes → text), identical to the
original module. The extracted text is then sent in a single generate_content call with
response_mime_type="application/json" to get structured field data back.

For invoice documents with multiple pricing tables the model returns an array of line-items;
for SOW documents or single-table invoices it returns a single object.
"""
import json
import re
import logging
from typing import Any, Dict, List, Optional, Tuple

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    FIELDS,
    INVOICE_LINE_FIELDS,
    INVOICE_PARENT_FIELDS,
    INVOICE_FIELDS,
    SOW_FIELDS,
    GCP_PROJECT,
    GCP_LOCATION,
    VERTEX_MODEL,
)

logger = logging.getLogger(__name__)

MAX_PDF_SIZE_BYTES = 50 * 1024 * 1024

EMPTY_RESPONSE_PHRASES = (
    "empty response",
    "information not available",
    "information not available.",
    "n/a",
    "na",
    "not found",
    "not available",
    "none",
    "-",
)


def _normalize_field_value(value: str) -> str:
    if not value or not value.strip():
        return ""
    v = value.strip()
    if v.lower() in EMPTY_RESPONSE_PHRASES:
        return ""
    return v


# ---------------------------------------------------------------------------
# Gemini client (Vertex AI via google-genai SDK)
# ---------------------------------------------------------------------------

def _get_genai_client():
    from google import genai
    from google.genai.types import HttpOptions
    return genai.Client(
        vertexai=True,
        project=GCP_PROJECT,
        location=GCP_LOCATION,
        http_options=HttpOptions(api_version="v1"),
    )


# ---------------------------------------------------------------------------
# OCR — identical to original module (Gemini native multimodal)
# ---------------------------------------------------------------------------

_OCR_PROMPT = """You are an expert OCR and document analysis engine.
We need to process all files: first produce text in English, then that text will be used for structured field extraction.

**Important:** At the very beginning of your response, on the first line, write the detected original language of the document in this exact format:
  DETECTED_LANGUAGE: <language name>
Examples: DETECTED_LANGUAGE: English   or   DETECTED_LANGUAGE: Russian   or   DETECTED_LANGUAGE: Spanish
Use the language name in English (e.g. Russian, not русский). Then leave a blank line and output the rest of the content.

- If the document is NOT in English: translate it into English while maintaining the layout, tables, and headers. Output the full translated text after the DETECTED_LANGUAGE line.
- If the document is already in English: perform a standard OCR transcription only.

In all cases: extract all visible text (including handwritten notes or scanned images); maintain layout, tables, and headers. Your output must always be in English (after the DETECTED_LANGUAGE line) so we can process every file the same way."""

_DETECTED_LANGUAGE_RE = re.compile(
    r"^\s*DETECTED_LANGUAGE:\s*(.+?)(?:\s*$)", re.IGNORECASE | re.MULTILINE
)


def _parse_ocr_response(full_text: str) -> Tuple[str, str]:
    if not full_text or not full_text.strip():
        return ("", "")
    text = full_text.strip()
    first_line, _, rest = text.partition("\n")
    match = _DETECTED_LANGUAGE_RE.match(first_line)
    if match:
        detected_lang = match.group(1).strip()
        body = rest.strip() if rest else ""
        logger.info("OCR detected language: %s", detected_lang)
        return (body, detected_lang)
    return (text, "")


def extract_text_from_scanned_pdf(pdf_bytes: bytes, file_name: str) -> Tuple[str, str]:
    if len(pdf_bytes) > MAX_PDF_SIZE_BYTES:
        logger.error("PDF %s exceeds 50 MB limit (%.1f MB). Skipping.", file_name, len(pdf_bytes) / (1024 * 1024))
        return ("", "")
    try:
        from google.genai.types import Part
    except ImportError:
        logger.error("google.genai not available for PDF OCR")
        return ("", "")
    try:
        logger.info("Processing PDF via Gemini native OCR: %s", file_name)
        client = _get_genai_client()
        pdf_part = Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")
        response = client.models.generate_content(
            model=VERTEX_MODEL,
            contents=[pdf_part, _OCR_PROMPT],
        )
        text = getattr(response, "text", None) if response else None
        if not text and response and getattr(response, "candidates", None):
            c = response.candidates[0]
            if c.content and c.content.parts:
                text = getattr(c.content.parts[0], "text", None) or ""
        full = (text or "").strip()
        return _parse_ocr_response(full)
    except Exception as e:
        logger.error("Gemini OCR failed for %s: %s", file_name, e)
        return ("", "")


def pdf_bytes_to_text(pdf_bytes: bytes, file_name: str = "") -> Tuple[str, str]:
    if len(pdf_bytes) > MAX_PDF_SIZE_BYTES:
        return ("", "")
    return extract_text_from_scanned_pdf(pdf_bytes, file_name or "PDF")


def pdf_diagnose_empty(pdf_bytes: bytes, file_name: str = "") -> str:
    if len(pdf_bytes) > MAX_PDF_SIZE_BYTES:
        return f"PDF too large ({len(pdf_bytes) / (1024 * 1024):.1f} MB > 50 MB). Skipped."
    try:
        import fitz
    except ImportError:
        return "PyMuPDF not available for diagnosis"
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        num_pages = len(doc)
        has_text = any(page.get_text().strip() for page in doc)
        num_images = sum(len(page.get_images()) for page in doc)
        doc.close()
        if not has_text and num_images > 0:
            return f"PDF has {num_images} image(s) and no embedded text (Gemini OCR should still process it)."
        if not has_text:
            return "PDF has no embedded text and no images — possibly empty or non-standard."
        return "PDF has embedded text; Gemini OCR may still have returned empty — check API/response."
    except Exception as e:
        return f"PDF corrupted or unreadable: {e}"


# ---------------------------------------------------------------------------
# Single-shot structured extraction (replaces RAG / VectorStoreIndex)
# ---------------------------------------------------------------------------

_PARENT_FIELDS_STR = ", ".join(f'"{f}"' for f in INVOICE_PARENT_FIELDS)
_LINE_FIELDS_STR = ", ".join(f'"{f}"' for f in INVOICE_LINE_FIELDS)
_SOW_FIELDS_STR = ", ".join(f'"{f}"' for f in SOW_FIELDS if f != "Original Language")

_EXTRACTION_PROMPT = """You are a financial analyst reviewing contract and invoice documents.

TASK: Extract structured data from the document text below.

STEP 1 — Classify the document:
  Decide if this is a "License Invoice" (software licence, subscription, vendor invoice for products/SKUs)
  or an "SOW Document" (Statement of Work, professional services contract).

STEP 2 — Extract fields based on document type.

For **License Invoice** documents, the document may contain ONE or MORE distinct pricing tables.
Each table has its own products/modules, total cost (TCV), pricing model, and licence units.
Return a JSON object with:
  "document_type": "License Invoice"
  "parent": {{ {parent_fields} }}
  "line_items": [
    {{ {line_fields} }}     ← one object per pricing table
  ]

Field definitions for invoices:
  "Contract ID": contract or agreement reference ID
  "Vendor": supplier / licensor legal name
  "Customer": buyer / licensee legal name
  "Contract Type": e.g. Addendum / Subscription, Perpetual, Service / PO
  "Billing Frequency": e.g. Annual, One-time, Project-based
  "Currency": 3-letter code (USD, EUR, RUB) or symbol ($, €)
  "Start Date": contract start date exactly as printed in the document (original format); parent-level dates apply to the whole agreement
  "End Date": contract end date exactly as printed in the document (original format)
  "Products / Modules": comma-separated product/module/SKU names for this table entry
  "TCV": total monetary value for this table entry (e.g. "$97,000")
  "Annual Value": per-year amount when applicable — use stated annual/yearly fee if printed; if only multi-year TCV and term length in years is clear (from dates or text), compute TCV divided by full years (e.g. $30,000 TCV over 3 years → "$10,000"); for a one-year term annual may equal TCV; null if not derivable
  "Pricing Model": e.g. Fixed, Hybrid (Fixed + User)
  "Licenses Purchased": quantity + unit (e.g. "10 Users (WebInsight)", "16 Seats")

If a document has TWO separate pricing tables with different costs, "line_items" MUST have two objects.
If only one table exists, "line_items" has one object.

For **SOW Document**, return:
  "document_type": "SOW Document"
  "fields": {{ {sow_fields} }}

Field definitions for SOW:
  "Contract ID": agreement or reference ID if shown (otherwise null)
  "Vendor": supplier / services provider / counterparty legal name if shown (otherwise null)
  "Contract Name": formal contract or Statement of Work title
  "Start Date": contract start date exactly as printed in the document (original format)
  "End Date": contract end date exactly as printed in the document (original format)
  "Commercial Value": total contract value (amount only)
  "Currency": 3-letter code or symbol
  "Owner/Contact": primary owner or contact name

RULES:
- Return a single valid JSON object. No markdown, no code fences, no conversational text.
- If a value is not found in the document, use null.
- Never invent values not supported by the document text.
- Dates: copy Start Date and End Date in the original format from the document; do not convert to a different date format.
- For amounts, include currency symbol if present (e.g. "$130,600").
""".format(
    parent_fields=_PARENT_FIELDS_STR,
    line_fields=_LINE_FIELDS_STR,
    sow_fields=_SOW_FIELDS_STR,
)


def extract_fields_from_full_text(
    full_text: str,
    detected_language: str = "",
    folder_name: str = "",
    file_name: str = "",
) -> List[Dict[str, str]]:
    """
    Single-shot extraction: send entire OCR text to Gemini with structured JSON output.
    Returns a list of row dicts (one per line-item for invoices, one for SOW).
    """
    from google.genai.types import GenerateContentConfig

    lang_note = ""
    if detected_language:
        lang_note = f"\n\nNOTE: The original document language is {detected_language}. The text below has been translated to English.\n"

    prompt = _EXTRACTION_PROMPT + lang_note + "\n--- DOCUMENT TEXT ---\n" + full_text

    client = _get_genai_client()
    try:
        response = client.models.generate_content(
            model=VERTEX_MODEL,
            contents=[prompt],
            config=GenerateContentConfig(
                response_mime_type="application/json",
                max_output_tokens=8192,
                temperature=0.0,
            ),
        )
        raw_json = getattr(response, "text", None) if response else None
        if not raw_json and response and getattr(response, "candidates", None):
            c = response.candidates[0]
            if c.content and c.content.parts:
                raw_json = getattr(c.content.parts[0], "text", None) or ""
        raw_json = (raw_json or "").strip()
        logger.info("  [extract] %s: raw JSON length = %s chars", file_name, len(raw_json))
        logger.debug("  [extract] %s: raw JSON: %s", file_name, raw_json[:1000] if len(raw_json) > 1000 else raw_json)
    except Exception as e:
        logger.error("  [extract] %s: Gemini call failed: %s", file_name, e)
        return [{"Folder": folder_name, "Filename": file_name, **{f: "" for f in FIELDS}}]

    return _parse_extraction_response(raw_json, detected_language, folder_name, file_name)


def _safe_str(val: Any) -> str:
    if val is None:
        return ""
    s = str(val).strip()
    if s.lower() in EMPTY_RESPONSE_PHRASES:
        return ""
    return s


def _parse_extraction_response(
    raw_json: str,
    detected_language: str,
    folder_name: str,
    file_name: str,
) -> List[Dict[str, str]]:
    """Parse the JSON from Gemini into one or more row dicts matching FIELDS."""
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as e:
        logger.error("  [extract] %s: JSON parse failed: %s", file_name, e)
        return [{"Folder": folder_name, "Filename": file_name, **{f: "" for f in FIELDS}}]

    doc_type_raw = _safe_str(data.get("document_type", ""))
    is_invoice = "invoice" in doc_type_raw.lower() or "licen" in doc_type_raw.lower()

    base = {"Folder": folder_name, "Filename": file_name}
    base["Document Type"] = doc_type_raw
    base["Original Language"] = _normalize_field_value(detected_language) if detected_language else ""

    if is_invoice:
        parent = data.get("parent") or {}
        for f in INVOICE_PARENT_FIELDS:
            base[f] = _safe_str(parent.get(f))

        # SOW-only fields are empty for invoices
        for f in ("Contract Name", "Commercial Value", "Owner/Contact"):
            base[f] = ""

        line_items = data.get("line_items") or []
        if not isinstance(line_items, list):
            line_items = [line_items]
        if not line_items:
            line_items = [{}]

        rows: List[Dict[str, str]] = []
        for item in line_items:
            row = dict(base)
            for f in INVOICE_LINE_FIELDS:
                v = _safe_str(item.get(f))
                if not v and f == "Annual Value":
                    v = _safe_str(item.get("annual_value"))
                row[f] = v
            rows.append(row)

        logger.info(
            "  [extract] %s: Invoice with %s line-item(s), parent fields: Contract ID=%s, Vendor=%s",
            file_name, len(rows),
            base.get("Contract ID", "")[:30], base.get("Vendor", "")[:30],
        )
        return rows

    # SOW Document
    fields_data = data.get("fields") or data
    for f in SOW_FIELDS:
        if f == "Original Language":
            continue
        base[f] = _safe_str(fields_data.get(f))

    # Invoice-only fields are empty for SOW
    for f in INVOICE_FIELDS:
        if f not in base:
            base[f] = ""

    # Dates might live in fields_data too
    for f in ("Start Date", "End Date"):
        if not base.get(f):
            base[f] = _safe_str(fields_data.get(f))

    logger.info(
        "  [extract] %s: SOW Document, Contract Name=%s",
        file_name, base.get("Contract Name", "")[:50],
    )
    return [base]


# ---------------------------------------------------------------------------
# Public API — same signature as original ai_processor
# ---------------------------------------------------------------------------

def _dict_from_fields(value: str) -> Dict[str, str]:
    return {f: value for f in FIELDS}


def process_pdf_bytes(
    pdf_bytes: bytes,
    folder_name: str,
    file_name: str,
) -> List[Dict[str, str]]:
    """
    Full pipeline: 50 MB check → Gemini native OCR → single-shot JSON extraction.
    Returns a list of row dicts (multi-row for multi-table invoices).
    """
    if len(pdf_bytes) > MAX_PDF_SIZE_BYTES:
        size_mb = len(pdf_bytes) / (1024 * 1024)
        logger.error("Skipping %s: PDF too large (%.1f MB > 50 MB limit).", file_name, size_mb)
        result = {"Folder": folder_name, "Filename": file_name, **_dict_from_fields("")}
        result["_debug_note"] = f"PDF too large ({size_mb:.1f} MB > 50 MB). Skipped."
        return [result]

    text, detected_language = pdf_bytes_to_text(pdf_bytes, file_name=file_name)
    logger.info("  [text] %s: %s chars from Gemini OCR, detected language: %s", file_name, len(text), detected_language or "(none)")
    if not text.strip():
        reason = pdf_diagnose_empty(pdf_bytes, file_name)
        logger.warning("  [text] %s: Gemini OCR returned EMPTY — %s", file_name, reason)
        result = {"Folder": folder_name, "Filename": file_name, **_dict_from_fields("")}
        result["_debug_note"] = reason
        return [result]

    rows = extract_fields_from_full_text(
        full_text=text,
        detected_language=detected_language,
        folder_name=folder_name,
        file_name=file_name,
    )
    for row in rows:
        empty_fields = [f for f in FIELDS if not _normalize_field_value(str(row.get(f) or ""))]
        if empty_fields:
            logger.warning("  [fields] %s: empty response for: %s", file_name, ", ".join(empty_fields))
    return rows


def debug_pdf_bytes(pdf_bytes: bytes, file_name: str = "unknown.pdf") -> Dict[str, Any]:
    """Debug a single PDF: Gemini OCR → single-shot extraction; return everything for inspection."""
    result: Dict[str, Any] = {"file_name": file_name, "size_kb": round(len(pdf_bytes) / 1024, 1)}
    if len(pdf_bytes) > MAX_PDF_SIZE_BYTES:
        result["text_length"] = 0
        result["text_preview"] = "(skipped: PDF too large)"
        result["text_empty"] = True
        result["fields"] = _dict_from_fields("")
        result["diagnosis"] = f"PDF too large ({len(pdf_bytes) / (1024 * 1024):.1f} MB > 50 MB). Skipped."
        return result
    text, detected_language = pdf_bytes_to_text(pdf_bytes, file_name=file_name)
    result["text_length"] = len(text)
    result["text_preview"] = text[:2000] if text else "(empty)"
    result["text_empty"] = not bool(text.strip())
    result["detected_language_from_ocr"] = detected_language or ""
    if not text.strip():
        result["fields"] = _dict_from_fields("")
        result["diagnosis"] = "Gemini OCR returned no text. " + pdf_diagnose_empty(pdf_bytes, file_name)
        return result
    try:
        rows = extract_fields_from_full_text(
            full_text=text,
            detected_language=detected_language,
            folder_name="",
            file_name=file_name,
        )
        result["fields"] = {f: rows[0].get(f, "") for f in FIELDS} if rows else _dict_from_fields("")
        result["line_items"] = rows if len(rows) > 1 else None
        empty = [f for f in FIELDS if not (rows[0].get(f) or "").strip()] if rows else list(FIELDS)
        result["diagnosis"] = f"Fields with empty response: {', '.join(empty)}." if empty else "All fields extracted successfully."
    except Exception as e:
        result["fields"] = _dict_from_fields("")
        result["diagnosis"] = f"Extraction failed: {type(e).__name__}: {e}"
    return result
