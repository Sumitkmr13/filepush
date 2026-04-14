# Production-ready RAG SOW Extractor: Vertex AI + SharePoint
# Python 3.12-slim
FROM python:3.12-slim

WORKDIR /app

# Install Python dependencies (PyMuPDF for diagnostics; Gemini does PDF OCR)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code (single-shot extraction in root ai_processor.py; VDI may ship this file only)
COPY config.py state_manager.py sharepoint_utils.py ai_processor.py data_utils.py gcs_utils.py secret_loader.py main.py ./

# Production Cloud Run: copy a completed .env from the build context (file must exist next to Dockerfile).
# Omit GOOGLE_APPLICATION_CREDENTIALS in that file on Cloud Run; use the Cloud Run service account for GCP APIs.
COPY .env .env

# Persistent data: state + Excel output (mount as volume)
RUN mkdir -p /app/data

ENV DATA_DIR=/app/data
ENV PYTHONUNBUFFERED=1

# Cloud Run sets PORT (often 8080). Default 8000 for local/docker-compose.
EXPOSE 8080

# JSON CMD does not expand env; use shell form so ${PORT:-8000} works.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
