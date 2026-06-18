# Project Delivery & Handover Guide

**Project:** SOW & Invoice Extraction Intelligence (`gem-rag`)  
**Purpose:** Production web app that reads PDFs from SharePoint/OneDrive (per logged-in user), extracts structured fields via **Vertex AI (Gemini)**, and writes cumulative Excel outputs with **Smart Resume** and per-user isolation.

This document is for engineering teams who will maintain, extend, or redeploy the system.

---

## 1. Executive summary

| Item | Detail |
|------|--------|
| **Runtime** | Python 3.12, FastAPI + Uvicorn |
| **AI** | Google Vertex AI (Gemini) — native PDF OCR + JSON field extraction |
| **Document source** | Microsoft SharePoint / OneDrive via **delegated OAuth** (user signs in) |
| **Outputs** | `contract_metrics.xlsx` (SOW/contracts), `license_metrics.xlsx` (licences/invoices) |
| **Auth model** | Per-user Microsoft login; app acts on behalf of each user |
| **Deployment target** | Google Cloud Run (recommended) or Docker Compose locally |
| **Persistence** | Local `data/users/<user_id>/` + optional **GCS** sync for Cloud Run |

**Related docs (read these for depth):**

| Document | Audience |
|----------|----------|
| [USER_GUIDE.md](USER_GUIDE.md) | End users |
| [CLOUD_RUN_DEPLOY.md](CLOUD_RUN_DEPLOY.md) | DevOps / deploy |
| [CLIENT_OAUTH_AND_USER_ISOLATION_EXPLAINER.md](CLIENT_OAUTH_AND_USER_ISOLATION_EXPLAINER.md) | Client / security |
| [CLIENT_REQUIREMENTS.md](../CLIENT_REQUIREMENTS.md) | Credentials checklist |
| [tools/README.md](../tools/README.md) | SharePoint URL diagnostics |
| [README.md](../README.md) | Developer quick start |

---

## 2. Architecture

```text
┌─────────────┐     OAuth2      ┌──────────────────┐
│   Browser   │◄──────────────►│  Microsoft Entra │
│  (UI in     │                 │  (delegated)     │
│   main.py)  │                 └────────┬─────────┘
└──────┬──────┘                          │
       │ HTTPS                           │ Files.Read.All, Sites.Read.All
       ▼                                 ▼
┌──────────────────────────────────────────────────────┐
│  FastAPI (main.py)                                    │
│  • Session + in-memory token store (auth_utils.py)   │
│  • Per-user context, extraction jobs, downloads       │
└───┬──────────────────────┬───────────────────────────┘
    │                      │
    ▼                      ▼
┌───────────────┐   ┌──────────────────┐
│ sharepoint_   │   │ ai_processor.py   │
│ utils.py      │   │ (Gemini / Vertex) │
│ Graph API     │   └──────────────────┘
└───────────────┘
    │                      │
    ▼                      ▼
┌───────────────┐   ┌──────────────────┐
│ PDF bytes     │   │ Extracted rows    │
│ (in memory)   │   │ → data_utils dedup│
└───────────────┘   └────────┬─────────┘
                               ▼
                    ┌──────────────────────┐
                    │ Excel + state JSON    │
                    │ data/users/<id>/      │
                    │ optional GCS upload   │
                    └──────────────────────┘
```

### Design principles

1. **PDFs never touch disk** — streamed from Graph → memory → Gemini → discarded.
2. **Delegated access only for user data** — SharePoint reads use the logged-in user's token.
3. **Per-user isolation** — state, Excel, and GCS prefixes are scoped by `user_id`.
4. **Smart Resume** — processed SharePoint item IDs + eTag/lastModified skip unchanged files.
5. **Cumulative Excel** — each run appends/updates; dedup keeps latest contract versions.

---

## 3. Repository layout

```text
gem-rag/
├── main.py                 # FastAPI app, UI (embedded HTML), extraction orchestration
├── config.py               # Env vars, field lists, paths
├── auth_utils.py           # Microsoft OAuth, session tokens, scopes
├── sharepoint_utils.py     # Graph API: list, download, verify paths
├── ai_processor.py         # Gemini PDF extraction, prompts, routing hints
├── data_utils.py           # Excel cleanup, contract/revision deduplication
├── state_manager.py        # extraction_state.json Smart Resume
├── user_storage.py         # Per-user paths (data/users/<id>/)
├── gcs_utils.py            # Optional GCS upload/download
├── secret_loader.py        # Optional GCP Secret Manager for SharePoint secrets
├── check_connections.py    # Pre-flight connectivity CLI
├── Dockerfile              # Production image (copies .env at build — see deploy doc)
├── docker-compose.yml      # Local Docker with mounted GCP key
├── requirements.txt
├── tests/                  # pytest suite (mocked external services)
├── tools/                  # SharePoint diagnostic scripts
└── docs/                   # User, deploy, and handover documentation
```

---

## 4. Module reference (what to change for common tasks)

| If you need to… | Edit |
|-----------------|------|
| Add/remove Excel columns | `config.py` → `SOW_FIELDS`, `INVOICE_FIELDS`, `FIELDS`; see **§14 How to add a new property** |
| Change Gemini model / region | `.env` → `VERTEX_MODEL`, `GCP_LOCATION`, `GCP_PROJECT` |
| Change document routing (SOW vs licence) | `main.py` → `_route_doc_type()`; extraction prompt in `ai_processor.py` |
| Change SharePoint OAuth scopes | `auth_utils.py` → `AUTH_SCOPES` or `GRAPH_DELEGATED_SCOPES` env |
| Fix Copy-link / shared folder resolution | `main.py` → `_resolve_share_link_via_graph`, `_auto_resolve_sharepoint_context_from_url` |
| Change PDF listing (recursive, search) | `sharepoint_utils.py` → `list_pdf_items_streaming`, `_list_pdfs_in_drive_iter` |
| Change dedup rules (contract versions) | `data_utils.py` → `dedupe_by_contract_id`, `dedupe_keep_latest_revision` |
| Change skip/reprocess logic | `state_manager.py` → `classify_item`, `should_process_item` |
| Change per-user file locations | `user_storage.py` |
| Add API endpoint | `main.py` (use `Depends(get_current_user)` for protected routes) |
| Change UI | `main.py` → `DASHBOARD_HTML` block (~line 1976+) |
| GCS paths / sync | `main.py` → `_sync_persistent_data_from_gcs`, `_upload_state_to_gcs`; `gcs_utils.py` |

---

## 5. End-to-end data flow

### 5.1 Login

1. `GET /auth/login` → redirect to Microsoft.
2. `GET /auth/callback` → exchange code, store tokens in `_SESSION_TOKEN_STORE` (in-memory), small `sid` in session cookie.
3. Scopes default: `User.Read`, `Files.Read.All`, `Sites.Read.All` (see `auth_utils.py`).

### 5.2 Save SharePoint context

1. User pastes **Copy link** (not browser URL) → `POST /sharepoint/context`.
2. App resolves via Graph `/shares` → `drive_id`, `item_id`, `drive_path`.
3. Lightweight validation: lists **first page** of children only (fast; does not scan all subfolders).
4. Saves `context.json` under `data/users/<user_id>/`.

### 5.3 Extraction

1. `POST /extract-sow/start` → background thread runs `_run_extraction`.
2. **Producer thread:** `list_pdf_items_streaming` recursively finds all PDFs under saved root.
3. **Consumer:** for each PDF, download bytes → `process_pdf_bytes` → append rows.
4. **Smart Resume:** skip if same SharePoint item id + unchanged eTag/lastModified.
5. On Excel save: merge → `dedupe_keep_latest_revision` → `dedupe_by_contract_id` → `remove_duplicate_entries`.
6. Upload Excel + state to GCS if `GCS_OUTPUT_BUCKET` set.

### 5.4 Document type routing

- LLM returns `Document Type` string.
- `_route_doc_type()`: contains `invoice` or `licen` → `license_metrics.xlsx`; else → `contract_metrics.xlsx`.

---

## 6. Deduplication behaviour

Applied on every Excel write (`main.py` → `_write_excel`):

| Step | Function | What it does |
|------|----------|--------------|
| 1 | Replace by filename | Reprocessed file replaces all existing rows with same **Filename** |
| 2 | `dedupe_keep_latest_revision` | Same filename base with `Rev 1`, `v2`, etc. → keep highest revision |
| 3 | `dedupe_by_contract_id` | Related Contract IDs (`1123`, `1123-1`, `1123-2`) → keep latest by Start Date / suffix |
| 4 | Same filename, multiple folders | Same name + contract family in different URLs → keep one copy |
| 5 | `remove_duplicate_entries` | Exact duplicate rows |

**State:** `extraction_state.json` tracks SharePoint **item IDs**, not contract IDs. Failed downloads are **not** marked processed → retried on next run.

---

## 7. API surface (authenticated unless noted)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Dashboard UI (redirects to login if needed) |
| GET | `/auth/login`, `/auth/callback`, `/auth/logout` | OAuth |
| GET | `/auth/me` | Current user |
| GET/POST | `/sharepoint/context` | Get/save folder context |
| GET | `/sharepoint/browse` | List folder children (saved context) |
| POST | `/extract-sow/start` | Background extraction (resume) |
| POST | `/extract-sow/reprocess-all` | Ignore state, reprocess all PDFs |
| POST | `/extract-sow/stop` | Stop after current file |
| GET | `/extract-sow/status` | Progress, `last_error` |
| GET | `/extract-sow/scan` | Count new/changed/up-to-date PDFs |
| GET | `/download`, `/download/{file_name}` | Download user Excel |
| GET | `/download/filtered/{file_name}` | Date-filtered download |
| GET | `/state`, `/state/by-type` | Processed file metadata |
| GET | `/api/ui-version` | Public UI version marker |

**Debug** (require `DEBUG_AUTH=1`): `/debug/sharepoint/test-url`, `/debug/token-info`, `/debug/sharepoint/list-pdfs`, `/extract-sow/start-app` (app-only token + `.env` paths).

Interactive API docs: `/docs` when server is running.

---

## 8. Configuration reference

Copy [`.env.example`](../.env.example) to `.env`. Key variables:

| Variable | Required | Notes |
|----------|----------|-------|
| `SESSION_SECRET` | Production | Long random string for session signing |
| `SHAREPOINT_CLIENT_ID/SECRET/TENANT_ID` | Yes | Entra app registration |
| `GOOGLE_APPLICATION_CREDENTIALS` | Local/Docker | Path to GCP SA JSON; **omit on Cloud Run** |
| `GCP_PROJECT` | Cloud Run | When not using JSON key |
| `GCP_LOCATION` | Yes | e.g. `us-central1` |
| `VERTEX_MODEL` | Yes | e.g. `gemini-2.5-flash` |
| `GCS_OUTPUT_BUCKET` | Cloud Run | Strongly recommended (ephemeral disk) |
| `GRAPH_DELEGATED_SCOPES` | Optional | Override OAuth scopes |
| `DEBUG_AUTH` | Optional | `1` enables debug endpoints |
| `DATA_DIR` | Optional | Default `./data` |
| `SHAREPOINT_SITE_URL` etc. | Optional | Used for app-only debug extraction only |

**Entra permissions (delegated):** `Files.Read.All`, `Sites.Read.All` (+ admin consent). Required for shared folders on work/school accounts.

**`GCP_CREDENTIALS_PATH`:** Docker Compose **host** mount path only — not used by the app at runtime or during `docker build`.

---

## 9. Local development

```bash
pip install -r requirements.txt
copy .env.example .env   # fill secrets
mkdir secrets && place gcp-key.json
python check_connections.py    # all checks
uvicorn main:app --reload --port 8000
```

**Tests:**

```bash
pytest tests/ -v
```

Tests mock SharePoint/Gemini; no live credentials required.

**Docker Compose:**

```bash
docker compose up --build
```

---

## 10. Deployment (summary)

Full steps: [CLOUD_RUN_DEPLOY.md](CLOUD_RUN_DEPLOY.md).

1. Build image → push to Artifact Registry.
2. Deploy Cloud Run with runtime service account (Vertex AI User + Storage Object Admin).
3. Set env vars + Secret Manager for SharePoint secret.
4. Set `GCS_OUTPUT_BUCKET`.
5. Configure Entra redirect URI: `https://<service-url>/auth/callback`.
6. Do **not** bake `GOOGLE_APPLICATION_CREDENTIALS` into Cloud Run env.

---

## 11. Storage layout

### Per user (local)

```text
data/users/<safe_user_id>/
├── context.json              # site_url, drive_id, drive_path, item_id
├── extraction_state.json     # processed_ids, processed_meta, dedup_log
├── contract_metrics.xlsx
└── license_metrics.xlsx
```

### GCS (when configured)

```text
gs://<bucket>/users/<safe_user_id>/contract_metrics.xlsx
gs://<bucket>/users/<safe_user_id>/license_metrics.xlsx
gs://<bucket>/users/<safe_user_id>/extraction_state.json
```

On extraction start, app **downloads** from GCS if blobs exist; uploads on run completion/stop.

---

## 12. Troubleshooting

| Symptom | Likely cause | Action |
|---------|--------------|--------|
| Save context fails on shared folder | Missing `Files.Read.All` consent | IT admin consent + user re-login |
| `401 Unauthorized` mid-extraction | OAuth token expired (~1h) | Sign out/in; run extraction again (failed files retry) |
| `403` on save | User lacks SharePoint access to folder | Owner shares folder with user |
| GCS upload fails | Bucket unset, wrong SA role, or bad credentials | `python check_connections.py` → GCS check |
| Data lost after Cloud Run restart | No `GCS_OUTPUT_BUCKET` | Set bucket + sync |
| 0 PDFs found | Wrong folder / Search API empty | Check context; recursive fallback should run |
| Wrong Excel (licence vs SOW) | LLM `Document Type` misclassification | Tune prompt in `ai_processor.py` |
| Copy link not resolving | `/shares` failed | Use Copy link; verify scopes; see `tools/test_sharepoint_url.py` |

**Diagnostics:**

```bash
python check_connections.py
python tools/test_sharepoint_url.py "<paste-url>"   # with user token in env
```

With `DEBUG_AUTH=1`: `GET /debug/token-info`, `POST /debug/sharepoint/test-url`.

---

## 13. Known limitations

1. **Token refresh in background jobs** — extraction captures the access token at start; runs longer than ~1 hour may hit Graph `401` on later files. Mitigation: re-login and resume.
2. **In-memory token store** — server restart invalidates sessions; users must sign in again. Cloud Run scale-to-zero causes same.
3. **Save context** does not count all PDFs — only validates the root folder is reachable.
4. **App-only credentials** in `.env` are for diagnostics / optional debug endpoint — not used for normal user extraction.
5. **Monitor loop** disabled in delegated per-user mode (`EXTRACTION_MONITOR_INTERVAL_MINUTES` ignored for auto global scan).

---

## 14. How to add a new property (extracted field)

This is the most common customization. A **property** is one column in the Excel output (e.g. `Payment Terms`, `Contract ID`, `TCV`).

### 14.1 Decide where the field belongs

| Document type | Excel file | Config list to edit |
|---------------|------------|---------------------|
| SOW / contract only | `contract_metrics.xlsx` | `SOW_FIELDS` in `config.py` |
| Licence / invoice — same on every line | `license_metrics.xlsx` | `INVOICE_PARENT_FIELDS` |
| Licence / invoice — varies per table row | `license_metrics.xlsx` | `INVOICE_LINE_FIELDS` |
| Both document types | Both Excels | Add to both `SOW_FIELDS` and `INVOICE_PARENT_FIELDS` (or line fields as needed) |

Also add the name to the master list **`FIELDS`** in `config.py`. That list is used when building empty error rows and validation; it should include every field the LLM may return.

**Column order in Excel** = `Filename`, `SharePoint URL`, then the fields in `SOW_FIELDS` or `INVOICE_FIELDS` (parent fields first, then line fields for invoices).

---

### 14.2 Step-by-step checklist

#### Step 1 — `config.py`

```python
# Example: add "Renewal Date" to SOW contracts only

FIELDS = [
    ...
    "Renewal Date",   # add here
]

SOW_FIELDS = [
    ...
    "Renewal Date",   # add here (position = column order in Excel)
]
```

For invoices, use `INVOICE_PARENT_FIELDS` and/or `INVOICE_LINE_FIELDS` instead of (or in addition to) `SOW_FIELDS`.

#### Step 2 — `ai_processor.py` — extraction prompt

The prompt template `_EXTRACTION_PROMPT` builds JSON field lists from config:

- `{sow_fields}` ← `SOW_FIELDS` (minus `Original Language`)
- `{parent_fields}` ← `INVOICE_PARENT_FIELDS`
- `{line_fields}` ← `INVOICE_LINE_FIELDS`

So adding a name to those config lists **automatically includes it in the JSON schema** sent to Gemini.

You must also add a **human-readable definition** in the prompt under the right section so the model knows what to extract. Find:

- **SOW:** section `Field definitions for SOW:` (~line 293)
- **Invoice parent:** `Field definitions for invoices — PARENT` (~line 202)
- **Invoice line:** `Field definitions for invoices — LINE ITEMS` (~line 246)

Example (SOW):

```text
  "Renewal Date": date the contract renews or auto-renews, if explicitly stated; null otherwise.
```

Keep definitions precise: what label to look for, format (copy as printed), and when to use `null`.

#### Step 3 — `ai_processor.py` — parsing (usually automatic)

`_parse_extraction_response()` maps JSON → row dicts:

- **SOW:** loops `for f in SOW_FIELDS` and reads `fields_data.get(f)`
- **Invoice parent:** loops `INVOICE_PARENT_FIELDS` from `data["parent"]`
- **Invoice line:** loops `INVOICE_LINE_FIELDS` from each `line_items[]` entry

**No code change needed** if the JSON key matches the Excel column name exactly.

**Custom mapping needed** only if:

- The model might return alternate JSON keys (see `Quantity` / `Unit` aliases in the invoice loop).
- You need post-processing (see `Payment Terms` fallback, `Annual Value` derivation).

Add a branch in the `for f in INVOICE_LINE_FIELDS:` loop or SOW section similar to existing special cases.

#### Step 4 — `main.py` — Excel formatting (optional)

| Field type | Where to wire |
|------------|----------------|
| **Money** (`TCV`, `Commercial Value`, …) | Add column name to `_AMOUNT_COLS` in `main.py` → `clean_amount()` runs on save |
| **Date** (`Start Date`, `End Date`, …) | `_write_excel` already runs `clean_date()` on `Start Date` and `End Date`. For a **new** date column, add it to the same loop in `_write_excel` |
| **Plain text** | No change |

#### Step 5 — `data_utils.py` (optional)

- **Dedup key:** If the new field should affect duplicate detection, update `remove_duplicate_entries` field list in `_write_excel` (currently `output_fields + ["Filename"]`).
- **Contract dedup:** Uses `Contract ID` and dates only — new fields do not affect `dedupe_by_contract_id` unless you change that function.

#### Step 6 — Filtered download (optional)

`GET /download/filtered/{file_name}` filters by **Start Date** and **End Date** only. To filter on a new date column, extend the handler in `main.py` (~`/download/filtered`).

#### Step 7 — UI (optional)

The dashboard does not list columns explicitly; downloads use generated Excel files. No UI change unless you add a new file type or filter control in `DASHBOARD_HTML`.

#### Step 8 — Tests

| Test file | What to add |
|-----------|-------------|
| `tests/test_ai_processor.py` | Mock JSON parsing with the new field; prompt not unit-tested directly |
| `tests/test_data_utils.py` | Only if you add date/amount cleaning rules for the new column |
| `tests/test_main.py` | Only if API/Excel write behaviour changes |

Run: `pytest tests/ -v`

#### Step 9 — Reprocess existing PDFs

New columns appear only for **newly processed** files (or after **Reprocess all**). Existing Excel rows keep old columns until those files are extracted again. Smart Resume skips unchanged PDFs — use **Process New Files Now** for new docs only, or **reprocess-all** to refill every PDF.

---

### 14.3 Worked example — add `Renewal Date` to SOW Excel

1. **`config.py`**
   - Add `"Renewal Date"` to `FIELDS` and `SOW_FIELDS` (e.g. after `End Date`).

2. **`ai_processor.py`** — in `Field definitions for SOW:` add:
   ```text
   "Renewal Date": renewal or auto-renewal date if explicitly stated; null otherwise.
   ```

3. **No parser change** — `_parse_extraction_response` already copies all `SOW_FIELDS` from `fields`.

4. **`main.py`** — if it is a date column, extend the `clean_date` loop in `_write_excel`:
   ```python
   for date_col, position in (("Start Date", "start"), ("End Date", "end"), ("Renewal Date", "start")):
   ```

5. **Test** with one PDF via debug: `GET /debug/pdf/inspect?item_id=...` (with `DEBUG_AUTH=1`).

6. **Deploy** and run extraction on a sample folder.

---

### 14.4 Worked example — add `Discount %` to invoice line items

1. **`config.py`** — add to `FIELDS` and `INVOICE_LINE_FIELDS`.

2. **`ai_processor.py`** — add definition under `LINE ITEMS`:
   ```text
   "Discount %": percentage discount for this line if shown in the table; null otherwise.
   ```

3. **Parser** — automatic via `INVOICE_LINE_FIELDS` loop unless you need aliases.

4. **`main.py`** — do **not** add to `_AMOUNT_COLS` (percent, not currency).

5. **Test** with a multi-line invoice PDF.

---

### 14.5 Fields you should NOT add via config alone

| Case | What to do |
|------|------------|
| New **document type** (third Excel) | New field list in `config.py`, extend `_route_doc_type`, `_run_extraction`, `user_storage.py`, UI downloads — see §14.6 |
| **Routing** field (like `Document Type`) | Stays in `FIELDS` for LLM; usually **excluded** from `SOW_FIELDS` / `INVOICE_FIELDS` so it does not appear as an Excel column |
| **Internal only** (e.g. `Original Language`) | In `SOW_FIELDS` for extraction but can be omitted from client-facing exports if you change `_write_excel` column list |
| **Computed** field (e.g. `Annual Value`) | Add prompt hint + implement derivation in `_parse_extraction_response` (see `_derive_annual_value`) |

---

### 14.6 Other extensions (summary)

#### Support a new document type / third Excel

1. Extend `_route_doc_type` in `main.py` and extraction prompt in `ai_processor.py`.
2. Add a new field list and filename in `config.py`.
3. Add path in `user_storage.py` and wire `_write_excel` in `_run_extraction`.
4. Update UI download section in `DASHBOARD_HTML`.

#### Change SharePoint listing performance

- `sharepoint_utils.py`: Graph Search API is tried first; recursive walk is fallback (reliable for OneDrive).
- Tune `GRAPH_REQUEST_TIMEOUT`, `GRAPH_RETRY_*` constants.

#### Harden production

- Remove or gate `DEBUG_AUTH` endpoints.
- Use GCP Secret Manager (`secret_loader.py`).
- Set `CORS_ALLOWED_ORIGINS` to your domain.
- Enable Cloud Run min instances if cold starts are an issue.

---

## 15. Testing strategy

| Area | File |
|------|------|
| Config / paths | `tests/test_config.py` |
| Dedup / dates | `tests/test_data_utils.py` |
| Smart Resume | `tests/test_state_manager.py` |
| SharePoint mocks | `tests/test_sharepoint_utils.py` |
| API endpoints | `tests/test_main.py` |
| AI processor | `tests/test_ai_processor.py` |

Run before release: `pytest tests/ -v` and `python check_connections.py` on target environment.

---

## 16. Ownership & handover checklist

- [ ] Entra app registration documented (client ID, redirect URIs, API permissions).
- [ ] GCP project, Cloud Run service URL, Artifact Registry repo documented.
- [ ] GCS bucket name and IAM bindings documented.
- [ ] `SESSION_SECRET` rotation process defined.
- [ ] `.env` / Secret Manager values stored in team vault (not in git).
- [ ] End-user guide shared: [USER_GUIDE.md](USER_GUIDE.md).
- [ ] Deploy runbook shared: [CLOUD_RUN_DEPLOY.md](CLOUD_RUN_DEPLOY.md).
- [ ] Support contact for IT (Graph consent, SharePoint sharing).

---

## 17. Version history (delivery baseline)

| Area | Baseline behaviour |
|------|-------------------|
| Auth | Delegated OAuth, per-user isolation |
| SharePoint | Copy link via `/shares`, shared folders via `item_id` + `Files.Read.All` |
| Extraction | Gemini single-shot, dual Excel outputs |
| Dedup | Contract ID families, filename revisions, folder copy collapse |
| Deploy | Docker + Cloud Run, optional GCS persistence |

*Update this section when your team ships significant changes.*

---

*Document generated for project handover. For questions about a specific module, start with the table in §4 and the related source file.*
