"""
In-memory RAG and SOW field extraction using Google GenAI on Vertex AI.
Uses Gemini 2.5 Flash native multimodal OCR for PDF text extraction (no Tesseract).
Text extraction: PDF bytes sent directly to Gemini; then RAG pipeline (VectorStoreIndex) for field extraction.
"""
import re
import logging
from typing import Any, Dict, List, Optional, Tuple

from llama_index.core import Document, VectorStoreIndex, get_response_synthesizer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI

from config import (
    FIELDS,
    GCP_PROJECT,
    GCP_LOCATION,
    VERTEX_MODEL,
    VERTEX_EMBEDDING_MODEL,
)

logger = logging.getLogger(__name__)

# Gemini inline content limit (50 MB). Larger PDFs are skipped.
MAX_PDF_SIZE_BYTES = 50 * 1024 * 1024

# Phrases the LLM may return when a field is not found; we treat these as empty (for storage and list-problems).
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
    """If the LLM returned an 'empty' phrase, return '' so Excel and list-problems treat it as missing."""
    if not value or not value.strip():
        return ""
    v = value.strip()
    if v.lower() in EMPTY_RESPONSE_PHRASES:
        return ""
    return v


# Common language names (single- or two-word) so we can strip junk from "Original Language" when LLM dumps extra text.
_KNOWN_LANGUAGE_NAMES = frozenset({
    "english", "spanish", "french", "german", "portuguese", "dutch", "italian", "russian",
    "chinese", "simplified chinese", "traditional chinese", "japanese", "korean", "arabic",
    "hindi", "turkish", "polish", "greek", "danish", "swedish", "norwegian", "finnish",
    "czech", "romanian", "hungarian", "hebrew", "thai", "vietnamese", "indonesian",
})


def _normalize_original_language(raw: str) -> str:
    """
    Keep only the language name for the Original Language field.
    LLMs sometimes return multiple lines or paste other fields; we strip junk and keep the language.
    We prefer a known language name; if none found, we keep the last short non-junk line so the cell is never empty when the LLM did respond.
    """
    if not raw or not raw.strip():
        logger.debug("Original language: raw=(empty) -> normalized=(empty)")
        return ""
    raw = raw.strip()
    # Log raw response from agent (truncate if very long) for debugging before/after normalization
    raw_preview = raw if len(raw) <= 500 else raw[:500] + "..."
    logger.info("Original language (raw from agent): %s", raw_preview)

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]

    def is_junk_line(line: str) -> bool:
        """True if this line is clearly not a language (numbers, currency, invoice keywords, etc.)."""
        lower = line.lower()
        if any(c.isdigit() for c in line):
            return True
        if any(x in lower for x in ("invoice", "license", "agreement", "contract", "sow", "document", "rub", "usd", "eur", "pc.", "pc", "quantity", "unit", "sku", "workplace")):
            return True
        if len(line) > 40:
            return True
        return False

    def could_be_language(line: str) -> bool:
        """Short line, no digits, no junk keywords — might be a language name (including non-Latin e.g. Русский)."""
        if not line or len(line) > 30:
            return False
        return not is_junk_line(line)

    candidates = [ln for ln in lines if could_be_language(ln)]
    out = ""

    # 1) Prefer a line that exactly matches a known language name (last such line)
    for ln in reversed(candidates):
        if ln.lower() in _KNOWN_LANGUAGE_NAMES:
            out = _normalize_field_value(ln)
            break
    if not out and candidates:
        # 2) Use last candidate (short non-junk line — often the LLM puts "Russian" or "English" on its own line)
        out = _normalize_field_value(candidates[-1])
    if not out:
        # 3) Search entire raw for a known language name (whole word)
        for lang in sorted(_KNOWN_LANGUAGE_NAMES, key=len, reverse=True):
            if re.search(r"\b" + re.escape(lang) + r"\b", raw, re.IGNORECASE):
                out = _normalize_field_value(lang.title())
                break
    if not out:
        # 4) Last resort: use last line if it's short and not junk (e.g. "Русский" or unexpected format)
        for ln in reversed(lines):
            if could_be_language(ln):
                out = _normalize_field_value(ln)
                break

    logger.info("Original language (after normalization): %s", out or "(empty)")
    return out


# Invoice / scalar fields: strip multi-line LLM dumps to a single tight value (reduce hallucination noise).
_INVOICE_SCALAR_TIGHT_FIELDS = frozenset({
    "Contract ID",
    "Vendor",
    "Customer",
    "Contract Type",
    "Billing Frequency",
    "Pricing Model",
    "Licenses Purchased",
})
_SCALAR_MAX_LEN = {
    "Contract Name": 200,
    "Owner/Contact": 150,
    "Contract ID": 48,
    "Vendor": 300,
    "Customer": 300,
    "Contract Type": 120,
    "Billing Frequency": 80,
    "Pricing Model": 120,
    "Licenses Purchased": 200,
}


def _tight_one_line(raw: str, field: str) -> str:
    """First non-empty line only; strip list markers; cap length."""
    v = _normalize_field_value(raw)
    if not v:
        return ""
    first = ""
    for ln in v.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        first = re.sub(r"^[\-\*\u2022\d\.\)\s]+", "", ln).strip()
        break
    if not first:
        return ""
    max_len = _SCALAR_MAX_LEN.get(field, 200)
    out = first[:max_len].strip()
    logger.debug("Field '%s' tight: %s -> %s", field, (raw[:120] + "..." if len(raw) > 120 else raw), out)
    return out


def _tight_currency(raw: str) -> str:
    """Single currency code or symbol; no amounts or sentences."""
    v = _normalize_field_value(raw)
    if not v:
        return ""
    m = re.search(r"\b([A-Z]{3})\b", v.upper())
    if m:
        return m.group(1)
    for sym in ("$", "€", "£", "¥", "₹"):
        if sym in v:
            return sym
    return _tight_one_line(v, "Currency")[:12]


def _tight_tcv(raw: str) -> str:
    """Total contract value: one line, amount-focused, no explanations."""
    v = _normalize_field_value(raw)
    if not v:
        return ""
    first = v.splitlines()[0].strip() if v.splitlines() else v.strip()
    first = re.sub(r"^[\-\*\u2022\d\.\)\s]+", "", first).strip()
    return first[:120].strip()


def _tight_products_modules(raw: str) -> str:
    """Allow multiple product lines but cap total length (no full-document dumps)."""
    v = (raw or "").strip()
    if not v:
        return ""
    lines = [ln.strip() for ln in v.splitlines() if ln.strip()]
    if not lines:
        lines = [v]
    joined = "; ".join(lines[:20])
    return joined[:2000]


# Lazy-init LLM and embedding (Vertex via GOOGLE_APPLICATION_CREDENTIALS)
_llm: Any = None
_embed_model: Any = None

_VERTEXAI_CONFIG = {"project": GCP_PROJECT, "location": GCP_LOCATION}


def _get_llm():
    global _llm
    if _llm is None:
        _llm = GoogleGenAI(
            model=VERTEX_MODEL,
            vertexai_config=_VERTEXAI_CONFIG,
            context_window=200_000,
            max_tokens=8192,  # Enough for full OCR transcription without cutoff
        )
    return _llm


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = GoogleGenAIEmbedding(
            model_name=VERTEX_EMBEDDING_MODEL,
            vertexai_config=_VERTEXAI_CONFIG,
        )
    return _embed_model


def _get_genai_client():
    """Google GenAI client for Vertex (used for PDF-inline OCR)."""
    from google import genai
    from google.genai.types import HttpOptions
    return genai.Client(
        vertexai=True,
        project=GCP_PROJECT,
        location=GCP_LOCATION,
        http_options=HttpOptions(api_version="v1"),
    )


_OCR_PROMPT = """You are an expert OCR and document analysis engine.
We need to process all files: first produce text in English, then that text will be used for structured field extraction.

**Important:** At the very beginning of your response, on the first line, write the detected original language of the document in this exact format:
  DETECTED_LANGUAGE: <language name>
Examples: DETECTED_LANGUAGE: English   or   DETECTED_LANGUAGE: Russian   or   DETECTED_LANGUAGE: Spanish
Use the language name in English (e.g. Russian, not русский). Then leave a blank line and output the rest of the content.

- If the document is NOT in English: translate it into English while maintaining the layout, tables, and headers. Output the full translated text after the DETECTED_LANGUAGE line.
- If the document is already in English: perform a standard OCR transcription only.

In all cases: extract all visible text (including handwritten notes or scanned images); maintain layout, tables, and headers. Your output must always be in English (after the DETECTED_LANGUAGE line) so we can process every file the same way."""

# Regex to extract DETECTED_LANGUAGE from the first line of OCR output (case-insensitive).
_DETECTED_LANGUAGE_RE = re.compile(r"^\s*DETECTED_LANGUAGE:\s*(.+?)(?:\s*$)", re.IGNORECASE | re.MULTILINE)


def _parse_ocr_response(full_text: str) -> Tuple[str, str]:
    """
    Parse OCR response: extract DETECTED_LANGUAGE from the start and return (body_text, detected_language).
    If no tag is found, return (full_text, "").
    """
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
    """
    Send PDF bytes directly to Gemini for native multimodal OCR. No Tesseract or local conversion.
    Returns (extracted_text_without_language_tag, detected_original_language).
    On failure or if over 50 MB, returns ("", "").
    """
    if len(pdf_bytes) > MAX_PDF_SIZE_BYTES:
        logger.error("PDF %s exceeds 50 MB limit (%s MB). Skipping.", file_name, len(pdf_bytes) / (1024 * 1024))
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
    """
    Extract text from PDF using Gemini native OCR.
    Returns (text_without_language_tag, detected_original_language).
    """
    if len(pdf_bytes) > MAX_PDF_SIZE_BYTES:
        return ("", "")
    return extract_text_from_scanned_pdf(pdf_bytes, file_name or "PDF")


def pdf_diagnose_empty(pdf_bytes: bytes, file_name: str = "") -> str:
    """Diagnose why a PDF might yield no text (size, corrupted, or OCR returned empty)."""
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


def build_index_from_text(text: str) -> VectorStoreIndex:
    """Build an ephemeral in-memory VectorStoreIndex from document text."""
    if not text or not text.strip():
        raise ValueError("Empty document text")
    doc = Document(text=text)
    parser = SentenceSplitter(chunk_size=2048, chunk_overlap=200)
    nodes = parser.get_nodes_from_documents([doc])
    embed_model = _get_embed_model()
    index = VectorStoreIndex(nodes, embed_model=embed_model)
    return index


# System context injected into every field query.
_SYSTEM_CONTEXT = (
    "You are a financial analyst reviewing contract and invoice documents. "
    "First determine whether the document is a 'License Invoice' (licence, subscription, vendor software invoice) "
    "or an 'SOW Document' (Statement of Work, professional services). "
    "For License Invoices, extract contract ID, vendor, customer, contract type, billing frequency, "
    "products/modules, currency, total contract value (TCV), pricing model, licenses purchased, and dates. "
    "For SOW Documents, extract contract name, commercial value, currency, owner/contact, and dates. "
    "If a field does not apply to this document type, respond with an empty string. "
    "Never invent values not supported by the document text. "
    "Never respond with phrases like 'N/A', 'Not found', 'Not applicable' — use an empty string instead. "
    "For every answer: respond with ONLY the field value — one line when possible, no bullet lists, no extra fields."
)

# Per-field extraction prompts with specific instructions for each field.
_FIELD_PROMPTS: Dict[str, str] = {
    "Document Type": (
        "Determine the document type. Respond with exactly one of these two values:\n"
        "'License Invoice' — if this is a software licence, subscription, or vendor invoice for products/SKUs.\n"
        "'SOW Document' — if this is a Statement of Work, professional services contract, or agreement."
    ),
    "Original Language": (
        "What is the original language of this document? "
        "Always give the language name (e.g. English, Russian, Spanish, French, German). "
        "If the document was translated to English, state the language it was written in. "
        "Put the language name clearly — ideally on its own line or at the end of your response."
    ),
    "Contract Name": (
        "For SOW Documents only: extract the formal contract or Statement of Work title. "
        "For License Invoices leave empty. Respond with the title only, one line."
    ),
    "Commercial Value": (
        "For SOW Documents only: extract the total commercial or contract value (amount). "
        "For License Invoices leave empty (use TCV on invoices). "
        "Respond with the amount only (one line), no explanation."
    ),
    "Owner/Contact": (
        "For SOW Documents only: extract the primary owner or contact name. "
        "For License Invoices leave empty. One line, name only."
    ),
    "Contract ID": (
        "For License Invoices only: extract the contract or agreement ID/reference (e.g. 3E-2015-01, KS-2011-005). "
        "For SOW Documents leave empty. Respond with the ID only, one line, no labels."
    ),
    "Vendor": (
        "For License Invoices only: extract the vendor / supplier / licensor legal name. "
        "For SOW Documents leave empty. One line only."
    ),
    "Customer": (
        "For License Invoices only: extract the customer / buyer / licensee legal name. "
        "For SOW Documents leave empty. One line only."
    ),
    "Contract Type": (
        "For License Invoices only: extract the contract type "
        "(e.g. Addendum / Subscription, Perpetual, Service / PO). "
        "For SOW Documents leave empty. One short phrase, one line."
    ),
    "Billing Frequency": (
        "For License Invoices only: extract billing or payment frequency "
        "(e.g. Annual, One-time, Project-based). "
        "For SOW Documents leave empty. One line only."
    ),
    "Products / Modules": (
        "For License Invoices only: list software products, modules, or SKUs covered (comma- or semicolon-separated). "
        "Use only text explicitly supported by the document. "
        "For SOW Documents leave empty. Do not repeat vendor, customer, or amounts."
    ),
    "Currency": (
        "Extract the currency for the main contract or invoice amounts. "
        "Respond with a 3-letter code (USD, EUR, RUB) or a single symbol ($, €). One token only."
    ),
    "TCV": (
        "For License Invoices only: Total Contract Value — the total monetary value for this contract/invoice. "
        "Respond with the amount in one line (e.g. $130,600 or 167,000). No words or explanations. "
        "For SOW Documents leave empty."
    ),
    "Pricing Model": (
        "For License Invoices only: extract the pricing model "
        "(e.g. Fixed, Hybrid (Fixed + User)). "
        "For SOW Documents leave empty. One short phrase, one line."
    ),
    "Licenses Purchased": (
        "For License Invoices only: summarize what was purchased in licence terms "
        "(e.g. '10 Users (WebInsight)', '16 Seats', '15 Seats'). "
        "If not stated, leave empty. Do not paste unrelated contract text. One line if possible."
    ),
    "Start Date": (
        "Extract the contract or licence start date. Respond with the date in its original format, one line."
    ),
    "End Date": (
        "Extract the contract or licence end date. Respond with the date in its original format, one line."
    ),
}


def extract_fields_from_index(
    index: VectorStoreIndex,
    folder_name: str,
    file_name: str,
    original_language_from_ocr: Optional[str] = None,
) -> Dict[str, str]:
    """
    Query the in-memory index for each field using targeted prompts.
    Original Language is taken from the OCR step when provided (original_language_from_ocr); otherwise not queried.
    Returns dict with Folder, Filename + all FIELDS (empty string for non-applicable fields).
    """
    result = {"Folder": folder_name, "Filename": file_name}
    # Use language detected during OCR; no RAG query for Original Language when we have it
    if original_language_from_ocr is not None:
        result["Original Language"] = _normalize_field_value(original_language_from_ocr.strip())
    else:
        result["Original Language"] = ""

    llm = _get_llm()
    response_synthesizer = get_response_synthesizer(llm=llm)
    query_engine = index.as_query_engine(
        llm=llm,
        response_synthesizer=response_synthesizer,
        similarity_top_k=5,
        response_mode="compact",
    )

    for field in FIELDS:
        if field == "Original Language":
            continue
        try:
            field_instruction = _FIELD_PROMPTS.get(
                field,
                f"Extract only the value of '{field}' from the document. Respond with just the value.",
            )
            prompt = f"{_SYSTEM_CONTEXT}\n\n{field_instruction}"
            response = query_engine.query(prompt)
            raw = (response.response or "").strip()
            if field in _INVOICE_SCALAR_TIGHT_FIELDS:
                result[field] = _tight_one_line(raw, field)
            elif field in ("Contract Name", "Owner/Contact"):
                result[field] = _tight_one_line(raw, field)
            elif field == "Commercial Value":
                result[field] = _tight_tcv(raw)
            elif field == "Currency":
                result[field] = _tight_currency(raw)
            elif field == "TCV":
                result[field] = _tight_tcv(raw)
            elif field == "Products / Modules":
                result[field] = _tight_products_modules(raw)
            else:
                result[field] = _normalize_field_value(raw)
        except Exception as e:
            logger.warning("Field '%s' extraction failed: %s", field, e)
            result[field] = ""

    doc_type = result.get("Document Type", "")
    logger.info("  [fields] %s: Document Type = '%s', Original Language = '%s'", file_name, doc_type or "(unknown)", result.get("Original Language", ""))
    return result


def process_pdf_bytes(
    pdf_bytes: bytes,
    folder_name: str,
    file_name: str,
) -> Dict[str, str]:
    """
    Full pipeline: 50 MB safety check -> Gemini native OCR (PDF bytes) -> in-memory index -> extract FIELDS.
    Returns one row dict (Folder, Filename, ...FIELDS). Skips files over 50 MB with _debug_note.
    """
    if len(pdf_bytes) > MAX_PDF_SIZE_BYTES:
        size_mb = len(pdf_bytes) / (1024 * 1024)
        logger.error("Skipping %s: PDF too large (%.1f MB > 50 MB limit).", file_name, size_mb)
        result = {"Folder": folder_name, "Filename": file_name, **_dict_from_fields("")}
        result["_debug_note"] = f"PDF too large ({size_mb:.1f} MB > 50 MB). Skipped."
        return result

    text, detected_language = pdf_bytes_to_text(pdf_bytes, file_name=file_name)
    logger.info("  [text] %s: %s chars from Gemini OCR, detected language: %s", file_name, len(text), detected_language or "(none)")
    if not text.strip():
        reason = pdf_diagnose_empty(pdf_bytes, file_name)
        logger.warning("  [text] %s: Gemini OCR returned EMPTY — %s", file_name, reason)
        result = {"Folder": folder_name, "Filename": file_name, **_dict_from_fields("")}
        result["_debug_note"] = reason
        return result

    index = build_index_from_text(text)
    row = extract_fields_from_index(index, folder_name, file_name, original_language_from_ocr=detected_language or None)
    empty_fields = [f for f in FIELDS if not _normalize_field_value(str(row.get(f) or ""))]
    if empty_fields:
        logger.warning("  [fields] %s: empty response for: %s", file_name, ", ".join(empty_fields))
    return row


def _dict_from_fields(value: str) -> Dict[str, str]:
    return {f: value for f in FIELDS}


def debug_pdf_bytes(pdf_bytes: bytes, file_name: str = "unknown.pdf") -> Dict[str, Any]:
    """Debug a single PDF: Gemini OCR -> index -> fields; return everything for inspection."""
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
        index = build_index_from_text(text)
        fields = extract_fields_from_index(index, "", file_name, original_language_from_ocr=detected_language or None)
        result["fields"] = {f: fields.get(f, "") for f in FIELDS}
        empty = [f for f in FIELDS if not (fields.get(f) or "").strip()]
        result["diagnosis"] = f"Fields with empty response: {', '.join(empty)}." if empty else "All fields extracted successfully."
    except Exception as e:
        result["fields"] = _dict_from_fields("")
        result["diagnosis"] = f"Extraction failed: {type(e).__name__}: {e}"
    return result


# Run unit tests: python ai_processor.py
if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main(["-v", "tests/test_ai_processor.py"]))
