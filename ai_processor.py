"""
In-memory RAG and SOW field extraction using Google GenAI on Vertex AI.
Uses Gemini 2.5 Flash native multimodal OCR for PDF text extraction (no Tesseract).
Text extraction: PDF bytes sent directly to Gemini; then RAG pipeline (VectorStoreIndex) for field extraction.
"""
import logging
from typing import Any, Dict, List

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

- If the document is NOT in English: translate it into English while maintaining the layout, tables, and headers. Output the full translated text.
- If the document is already in English: perform a standard OCR transcription only.

In all cases: extract all visible text (including handwritten notes or scanned images); maintain layout, tables, and headers. Your output must always be in English so we can process every file the same way."""


def extract_text_from_scanned_pdf(pdf_bytes: bytes, file_name: str) -> str:
    """
    Send PDF bytes directly to Gemini for native multimodal OCR. No Tesseract or local conversion.
    Returns extracted text; empty string on failure or if over 50 MB (caller should check size first).
    """
    if len(pdf_bytes) > MAX_PDF_SIZE_BYTES:
        logger.error("PDF %s exceeds 50 MB limit (%s MB). Skipping.", file_name, len(pdf_bytes) / (1024 * 1024))
        return ""
    try:
        from google.genai.types import Part
    except ImportError:
        logger.error("google.genai not available for PDF OCR")
        return ""
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
        return (text or "").strip()
    except Exception as e:
        logger.error("Gemini OCR failed for %s: %s", file_name, e)
        return ""


def pdf_bytes_to_text(pdf_bytes: bytes, file_name: str = "") -> str:
    """
    Extract text from PDF using Gemini 2.5 Flash native OCR (no PyMuPDF text, no Tesseract).
    PDF is sent directly to the model for full visual reading including scanned pages.
    """
    if len(pdf_bytes) > MAX_PDF_SIZE_BYTES:
        return ""
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
    "First determine whether the document is a 'License Invoice' (software licences, subscriptions, SKUs, "
    "vendor invoices for products) or an 'SOW Document' (Statement of Work, professional services contract). "
    "For License Invoices, extract SKU/product descriptions, quantities, units of measure, invoice values, "
    "date ranges, and calculate the annual cost when possible. "
    "For SOW Documents, extract the contract name, start/end dates, commercial value, and owner/contact. "
    "If a field does not apply to this document type, respond with an empty string. "
    "Never respond with phrases like 'N/A', 'Not found', 'Not applicable' — use an empty string instead."
)

# Per-field extraction prompts with specific instructions for each field.
_FIELD_PROMPTS: Dict[str, str] = {
    "Document Type": (
        "Determine the document type. Respond with exactly one of these two values:\n"
        "'License Invoice' — if this is a software licence, subscription, or vendor invoice for products/SKUs.\n"
        "'SOW Document' — if this is a Statement of Work, professional services contract, or agreement."
    ),
    "Contract Name": (
        "Extract the contract name or agreement title. "
        "For a License Invoice this is the software product or vendor name. "
        "For an SOW Document this is the formal contract or Statement of Work title. "
        "Respond with just the name."
    ),
    "Licences Acquired": (
        "Extract the description of the licences being acquired — typically the SKU description or product name "
        "from the vendor (e.g. 'Microsoft 365 E3', 'Oracle Database Enterprise Edition'). "
        "This applies to License Invoices. For SOW Documents leave empty. "
        "Respond with the description only."
    ),
    "Quantity": (
        "Extract the number of units included in the licence. "
        "This is a numeric value such as 100, 500, or 1000 users/seats/processors. "
        "This applies to License Invoices. For SOW Documents leave empty. "
        "Respond with the number only."
    ),
    "Unit": (
        "Identify the unit of measure for the quantity. "
        "Common values: user, seat, transaction, processor, core, storage (GB/TB), device. "
        "This applies to License Invoices. For SOW Documents leave empty. "
        "Respond with the unit type only."
    ),
    "Invoice Value": (
        "Extract the total invoice value — the total monetary cost of the licences on this invoice. "
        "This applies to License Invoices. For SOW Documents use Commercial Value instead. "
        "Respond with the amount and currency (e.g. USD 10,000 or $10,000)."
    ),
    "Annual Cost": (
        "Derive or extract the annual cost of the licences. "
        "If an annual cost is explicitly stated, use that value. "
        "If not, calculate it: divide the total invoice value by the number of years in the date range "
        "(e.g. if the invoice is $30,000 for 3 years, the annual cost is $10,000/year). "
        "This applies to License Invoices. For SOW Documents leave empty. "
        "Respond with the annual amount and currency."
    ),
    "Start Date": (
        "Extract the start date of the contract, licence period, or service period. "
        "Respond with the date in its original format."
    ),
    "End Date": (
        "Extract the end date of the contract, licence period, or service period. "
        "Respond with the date in its original format."
    ),
    "Commercial Value": (
        "Extract the total commercial value or contract value. "
        "This is the total monetary amount of the SOW or services agreement. "
        "This applies to SOW Documents. For License Invoices use Invoice Value instead. "
        "Respond with the amount and currency."
    ),
    "Owner/Contact": (
        "Extract the name of the owner, primary contact, or responsible person for this contract. "
        "This typically appears as 'Prepared by', 'Account Manager', 'Contact', or 'Owner'. "
        "Respond with the name only."
    ),
}


def extract_fields_from_index(
    index: VectorStoreIndex,
    folder_name: str,
    file_name: str,
) -> Dict[str, str]:
    """
    Query the in-memory index for each field using targeted prompts.
    Document-type-aware: identifies the type first, then extracts type-specific fields.
    Returns dict with Folder, Filename + all FIELDS (empty string for non-applicable fields).
    """
    result = {"Folder": folder_name, "Filename": file_name}
    llm = _get_llm()
    response_synthesizer = get_response_synthesizer(llm=llm)
    query_engine = index.as_query_engine(
        llm=llm,
        response_synthesizer=response_synthesizer,
        similarity_top_k=20,
        response_mode="compact",
    )

    for field in FIELDS:
        try:
            field_instruction = _FIELD_PROMPTS.get(
                field,
                f"Extract only the value of '{field}' from the document. Respond with just the value.",
            )
            prompt = f"{_SYSTEM_CONTEXT}\n\n{field_instruction}"
            response = query_engine.query(prompt)
            raw = (response.response or "").strip()
            result[field] = _normalize_field_value(raw)
        except Exception as e:
            logger.warning("Field '%s' extraction failed: %s", field, e)
            result[field] = ""

    doc_type = result.get("Document Type", "")
    logger.info("  [fields] %s: Document Type = '%s'", file_name, doc_type or "(unknown)")
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

    text = pdf_bytes_to_text(pdf_bytes, file_name=file_name)
    logger.info("  [text] %s: %s chars from Gemini OCR", file_name, len(text))
    if not text.strip():
        reason = pdf_diagnose_empty(pdf_bytes, file_name)
        logger.warning("  [text] %s: Gemini OCR returned EMPTY — %s", file_name, reason)
        result = {"Folder": folder_name, "Filename": file_name, **_dict_from_fields("")}
        result["_debug_note"] = reason
        return result

    index = build_index_from_text(text)
    row = extract_fields_from_index(index, folder_name, file_name)
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
    text = pdf_bytes_to_text(pdf_bytes, file_name=file_name)
    result["text_length"] = len(text)
    result["text_preview"] = text[:2000] if text else "(empty)"
    result["text_empty"] = not bool(text.strip())
    if not text.strip():
        result["fields"] = _dict_from_fields("")
        result["diagnosis"] = "Gemini OCR returned no text. " + pdf_diagnose_empty(pdf_bytes, file_name)
        return result
    try:
        index = build_index_from_text(text)
        fields = extract_fields_from_index(index, "", file_name)
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
