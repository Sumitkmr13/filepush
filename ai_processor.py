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
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

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

In all cases:
- Extract ALL visible text (including handwritten notes, scanned images, and fine-print).
- Maintain layout, tables, and headers faithfully.
- For tables with columns (e.g. ITEM | MATERIAL DESCRIPTION | QUANTITY | UNIT | UNIT PRICE | AMOUNT):
  * First output the header row exactly as printed, with column names separated by " | ".
  * Then output each data row with values separated by " | " in the same column order.
  * Example output for a PO table:
    ITEM | MATERIAL DESCRIPTION | VENDOR PART NUMBER | QUANTITY | UNIT | UNIT PRICE | AMOUNT
    D0010 | Adobe Pro, Photoshop, InDesign C Cloud | | 49,963.160 | AU | 1.0000 | 45,963.36
    D0020 | Adobe Pro 100 Licenses | | 11,268.000 | AU | 1.0000 | 11,268.00
  * This column-separated format is critical — downstream extraction depends on correctly distinguishing QUANTITY from UNIT PRICE and UNIT from DESCRIPTION.
- For per-line-item annotations (e.g. "Delivery date: 01/26/2023", "Goods recipient: ...", "*** Item completely delivered ***"): include them on separate lines directly after the table row they belong to.
- Your output must always be in English (after the DETECTED_LANGUAGE line) so we can process every file the same way."""

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
  Decide if this is a "License Invoice" (software licence, subscription, vendor invoice for products/SKUs,
  purchase order) or an "SOW Document" (Statement of Work, professional services contract).

STEP 2 — Extract fields based on document type.

For **License Invoice** documents, the document may contain ONE or MORE distinct line items / rows in a pricing table.
Each line item has its own product name, quantity, unit, cost, and optionally its own dates.
Return a JSON object with:
  "document_type": "License Invoice"
  "parent": {{ {parent_fields}, "Start Date": "...", "End Date": "..." }}
  "line_items": [
    {{ {line_fields} }}     ← one object per distinct line item / row
  ]

Field definitions for invoices — PARENT (document-level):
  "Contract ID": contract, PO number, or agreement reference ID (look for labels like PO Number, Order Number, Contract ID)
  "Vendor": supplier / licensor legal name
  "Customer": buyer / licensee / "Bill To" legal name
  "Contract Type": e.g. Addendum / Subscription, Perpetual, Service / PO
  "Billing Frequency": e.g. Annual, One-time, Project-based
  "Currency": 3-letter code (USD, EUR, RUB) or symbol ($, €)
  "Start Date": the main document-level start date. Look for: Start Date, Order Date, PO Date, Agreement Date.
                Copy exactly as printed (original format). This is the default date for all line items.
  "End Date": the main document-level end date. Look for: End Date, Expiration Date, Due Date.
              Copy exactly as printed (original format). null if not present at document level.

Field definitions for invoices — LINE ITEMS (per row in the pricing table):
  "Products / Modules": product name, material description, or SKU for this line
  "Quantity": the value from the QUANTITY column of the table. This is typically a large number representing
              volume, count, or license units. Copy the number exactly as printed.
              IMPORTANT: Do NOT confuse QUANTITY with UNIT PRICE. In a typical PO table the columns are:
                ITEM | MATERIAL DESCRIPTION | VENDOR PART NUMBER | QUANTITY | UNIT | UNIT PRICE | AMOUNT
              The QUANTITY column is BEFORE the UNIT column. UNIT PRICE is a DIFFERENT column (usually 1.0000 or a per-unit cost).
              Examples of correct Quantity values: "49,963.160", "11,268.000", "2,500.000", "1,173.750", "10", "16".
              If there is no table (e.g. "10 Users" in prose) → Quantity="10".
  "Unit": the value from the UNIT column of the table. This is a short code like "AU", "EA", "PC", "LIC".
          IMPORTANT: Read the UNIT column directly — do NOT infer unit from the product name.
          In a PO table with columns QUANTITY | UNIT | UNIT PRICE, the UNIT column is between QUANTITY and UNIT PRICE.
          Examples of correct Unit values: "AU", "EA", "PC", "LIC", "Seats", "Users".
          Only if there is no UNIT column in the table, derive from context (e.g. "10 Users" → Unit="Users").
  "TCV": total monetary value / amount for this line item (the AMOUNT column, NOT the UNIT PRICE column).
         Example: if UNIT PRICE is 1.0000 and AMOUNT is 45,963.36 → TCV = "$45,963.36".
  "Annual Value": per-year amount. Compute as follows:
      - If billing is Annual/Yearly: Annual Value = TCV.
      - If billing is One-time and term is 1 year (or no multi-year term stated): Annual Value = TCV.
      - If billing is One-time and term spans N years: Annual Value = TCV / N.
      - If it cannot be determined: null.
  "Pricing Model": e.g. Fixed, Per Unit, Hybrid (Fixed + User), Paid Up
  "Start Date": date for this specific line item. If the line item has its own start/order date, use it.
                Otherwise, copy the parent-level Start Date. Must not be null if parent has a date.
  "End Date": end/delivery date for this specific line item. If the line item has its own delivery/end date
              (e.g. "Delivery date: 01/26/2023" printed below the row), use that.
              Otherwise, copy the parent-level End Date (or null if parent also has none).

IMPORTANT for dates:
  - ALWAYS include Start Date and End Date in the "parent" object if any document-level date exists.
  - Purchase orders: "PO Date" at the top → parent "Start Date"; per-line "Delivery date:" → that line's "End Date".
  - Subscription contracts: "Start Date" / "End Date" at document level → parent dates + copy to each line item.
  - Always copy the date exactly as it appears in the document. Do not reformat.

If a document has THREE line items in the table, "line_items" MUST have three objects — one per row.

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
- Dates: copy exactly as printed in the document; do not convert to a different format.
- For amounts, include currency symbol if present (e.g. "$130,600").
- CRITICAL for PO / invoice tables with columns like ITEM | DESCRIPTION | QUANTITY | UNIT | UNIT PRICE | AMOUNT:
    * Quantity = the QUANTITY column value (e.g. 49,963.160 or 11,268.000). NOT the UNIT PRICE.
    * Unit = the UNIT column value (e.g. "AU", "EA"). NOT derived from the product description.
    * TCV = the AMOUNT column value (the total for that row). NOT the UNIT PRICE.
    * UNIT PRICE is a per-unit cost — do not use it as Quantity or TCV.
- For Annual Value: always try to compute it from TCV and billing frequency / term length. Only use null as last resort.
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


def _first_non_empty(source: Dict[str, Any], keys: List[str]) -> str:
    for k in keys:
        v = _safe_str(source.get(k))
        if v:
            return v
    return ""


def _parse_amount_number(value: str) -> Optional[float]:
    if not value:
        return None
    cleaned = re.sub(r"[^\d.,]", "", value)
    if not cleaned:
        return None
    # Handle both 1,234.56 and 1.234,56 style inputs.
    if "," in cleaned and "." in cleaned:
        if cleaned.rfind(",") > cleaned.rfind("."):
            cleaned = cleaned.replace(".", "").replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")
    elif "," in cleaned:
        parts = cleaned.split(",")
        if len(parts[-1]) in (0, 3):
            cleaned = cleaned.replace(",", "")
        else:
            cleaned = cleaned.replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _extract_currency_prefix(value: str) -> str:
    m = re.search(r"(USD|EUR|GBP|INR|AUD|CAD|SGD|JPY|CNY|RUB|CHF|SEK|NOK|DKK|AED|\$|€|£|¥|₹)", value or "", re.IGNORECASE)
    return (m.group(1).upper() if m else "")


def _parse_date(value: str) -> Optional[datetime]:
    if not value:
        return None
    s = str(value).strip()
    formats = (
        "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%d-%m-%Y", "%m-%d-%Y",
        "%Y/%m/%d", "%d.%m.%Y", "%m.%d.%Y", "%d %b %Y", "%d %B %Y",
        "%b %d, %Y", "%B %d, %Y",
    )
    for fmt in formats:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _derive_annual_value(tcv: str, start_date: str, end_date: str, billing_frequency: str) -> str:
    amount = _parse_amount_number(tcv)
    if amount is None or amount <= 0:
        return ""
    currency = _extract_currency_prefix(tcv)
    freq = (billing_frequency or "").strip().lower()

    annual: Optional[float] = None

    if "annual" in freq or "year" in freq:
        annual = amount
    elif freq in ("one-time", "one time", "onetime", "single", "once"):
        start = _parse_date(start_date)
        end = _parse_date(end_date)
        if start and end and end > start:
            years = (end - start).days / 365.25
            rounded_years = max(1, round(years))
            annual = amount / rounded_years
        else:
            annual = amount
    else:
        start = _parse_date(start_date)
        end = _parse_date(end_date)
        if start and end and end > start:
            years = (end - start).days / 365.25
            rounded_years = round(years)
            if rounded_years >= 1 and abs(years - rounded_years) <= 0.25:
                annual = amount / rounded_years
            elif years < 1:
                annual = amount
        elif not start or not end:
            annual = amount

    if annual is None:
        return ""
    formatted = f"{annual:,.0f}"
    return f"{currency} {formatted}".strip() if currency else formatted


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

        # Parent-level dates serve as fallback for line items that lack their own dates.
        parent_start = _first_non_empty(parent, ["Start Date", "start_date", "Order Date", "order_date", "PO Date", "po_date"])
        parent_end = _first_non_empty(parent, ["End Date", "end_date", "Delivery Date", "delivery_date", "Due Date", "due_date"])

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
                if f == "Start Date":
                    v = _first_non_empty(item, ["Start Date", "start_date", "Order Date", "order_date", "PO Date", "po_date"])
                    if not v:
                        v = parent_start
                elif f == "End Date":
                    v = _first_non_empty(item, ["End Date", "end_date", "Delivery Date", "delivery_date", "Due Date", "due_date"])
                    if not v:
                        v = parent_end
                elif f == "Quantity":
                    v = _first_non_empty(item, ["Quantity", "quantity", "Qty", "qty"])
                elif f == "Unit":
                    v = _first_non_empty(item, ["Unit", "unit", "UOM", "uom", "Unit of Measure"])
                elif f == "Annual Value":
                    v = _safe_str(item.get(f))
                    if not v:
                        v = _safe_str(item.get("annual_value"))
                else:
                    v = _safe_str(item.get(f))
                row[f] = v

            if not row.get("Annual Value"):
                row["Annual Value"] = _derive_annual_value(
                    tcv=_safe_str(item.get("TCV")) or _safe_str(item.get("tcv")),
                    start_date=row.get("Start Date", ""),
                    end_date=row.get("End Date", ""),
                    billing_frequency=base.get("Billing Frequency", ""),
                )
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
            if f == "Start Date":
                base[f] = _first_non_empty(fields_data, ["Start Date", "start_date", "Order Date", "order_date", "PO Date", "po_date"])
            else:
                base[f] = _first_non_empty(fields_data, ["End Date", "end_date", "Delivery Date", "delivery_date", "Due Date", "due_date"])

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
