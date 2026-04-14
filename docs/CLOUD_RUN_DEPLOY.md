# Deploy the SOW / Invoice extractor to Google Cloud Run

This guide is written in plain language so you can forward it to teammates. It matches how this repository is built (`Dockerfile`, `config.py`, `.env.example`).

**In one sentence:** You package the app in a **Docker image**, store it in **Artifact Registry**, then run it on **Cloud Run** with **secrets** and **settings** supplied by Google—not baked into the image.

---

## Important ideas (read once)

1. **Docker image** = packaged app + dependencies. It must **not** contain passwords, API secrets, or a committed `.env` file.
2. **Artifact Registry** = Google’s place to store Docker images.
3. **Cloud Run** = runs your container and gives you an HTTPS URL.
4. **On Cloud Run, GCP access usually does not use `gcp-key.json` inside the container.** You attach a **service account** to the Cloud Run service; Google provides credentials automatically. That is the recommended approach.
5. **SharePoint client secret** (and similar) should live in **Secret Manager**, not in the image.
6. **Cloud Run’s filesystem is ephemeral.** Set **`GCS_OUTPUT_BUCKET`** so Excel outputs and synced state can live in Cloud Storage across restarts (as this app supports).
7. **Cloud Run sets the `PORT` environment variable** (commonly `8080`). The Docker image listens on **`$PORT`**, defaulting to `8000` when unset (local Compose).
8. **On startup**, if `GCS_OUTPUT_BUCKET` is set, the app **downloads** `extraction_state.json` and the two Excel files from that bucket when they exist, so a **new container** can resume Smart Resume and serve `/download` without losing prior work.

---

## Part A — One-time Google Cloud setup

### A1. Prerequisites

- A **Google Cloud project** and its **Project ID** (e.g. `my-company-rag`).
- **Billing** enabled.
- Permission to use **Artifact Registry**, **Cloud Run**, **Secret Manager**, and **IAM** (or ask an admin).

### A2. Tools on the build machine (e.g. VDI or laptop)

1. **Google Cloud SDK** (`gcloud`):

   ```bash
   gcloud --version
   ```

2. **Docker** (Docker Desktop or Docker Engine):

   ```bash
   docker --version
   ```

3. Sign in and set the project:

   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

### A3. Enable APIs

```bash
gcloud config set project YOUR_PROJECT_ID

gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com \
  aiplatform.googleapis.com \
  storage.googleapis.com
```

### A4. Create an Artifact Registry repository

Pick a **region** (example: `us-central1`):

```bash
gcloud artifacts repositories create gem-rag-docker \
  --repository-format=docker \
  --location=us-central1 \
  --description="Docker images for gem-rag"
```

Your image URL will look like:

`us-central1-docker.pkg.dev/YOUR_PROJECT_ID/gem-rag-docker/gem-rag:latest`

### A5. Create the runtime service account (Cloud Run identity)

```bash
gcloud iam service-accounts create gem-rag-runtime \
  --display-name="Gem RAG Cloud Run runtime"
```

The email will look like:

`gem-rag-runtime@YOUR_PROJECT_ID.iam.gserviceaccount.com`

### A6. Grant IAM roles to the runtime service account

Typical needs for this app: **Vertex AI** and **Cloud Storage** (if using `GCS_OUTPUT_BUCKET`).

Example using project-wide roles (your org may prefer narrower, bucket-only roles):

```bash
SA="gem-rag-runtime@YOUR_PROJECT_ID.iam.gserviceaccount.com"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:${SA}" \
  --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:${SA}" \
  --role="roles/storage.objectAdmin"
```

### A7. Store the SharePoint client secret in Secret Manager

Do not put this value in the Docker image or in public env files.

```bash
echo -n "PASTE_THE_REAL_CLIENT_SECRET_HERE" | gcloud secrets create sharepoint-client-secret --data-file=-
```

Allow the **runtime** service account to read it:

```bash
SA="gem-rag-runtime@YOUR_PROJECT_ID.iam.gserviceaccount.com"

gcloud secrets add-iam-policy-binding sharepoint-client-secret \
  --member="serviceAccount:${SA}" \
  --role="roles/secretmanager.secretAccessor"
```

Create additional secrets the same way if you prefer not to use plain env vars for other sensitive values.

### A8. (Recommended) GCS bucket for outputs

Bucket names must be globally unique:

```bash
gsutil mb -l us-central1 gs://YOUR_UNIQUE_BUCKET_NAME
```

Ensure the runtime service account can read/write this bucket (via project role above or bucket-level IAM).

---

## Part B — Build the Docker image and push to Artifact Registry

### B1. Project files

From the repository root you should have `Dockerfile`, `requirements.txt`, and the application files listed in the `Dockerfile`.

### B2. Build the image

```bash
docker build -t gem-rag:latest .
```

No Docker Hub login is required for a local build.

### B2a. `.env` for the same image you will deploy to Cloud Run

The `Dockerfile` copies **`.env`** into the image (`COPY .env .env`). For Cloud Run you should **not** bake a path to a JSON key file.

1. In **`.env`** (repo root, next to the `Dockerfile`), **comment out** the line that points at the key file, for example:
   - `# GOOGLE_APPLICATION_CREDENTIALS=./secrets/gcp-key.json`
2. Set **`GCP_PROJECT=your-project-id`** (required when no key file is used).
3. Fill in SharePoint, **`GCS_OUTPUT_BUCKET`**, models, etc., following [`.env.example`](../.env.example).

Rebuild after any change to `.env`.

### B2b. Test that image locally on the VDI (before Cloud Run)

Cloud Run will use the **runtime service account** for GCP. Your PC does not have that, so for a **local** test only, **mount** your `gcp-key.json` and set the env var for this one `docker run` (this does not change the image; it overrides/adds env at start):

```powershell
docker run --rm -p 8000:8000 `
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/secrets/gcp-key.json `
  -v "D:\path\to\your\gcp-key.json:/app/secrets/gcp-key.json:ro" `
  gem-rag:latest
```

Use your real key path. Then open **http://localhost:8000/** or **http://localhost:8000/docs**.

Stop the container with **Ctrl+C** (or run without `--rm` and `docker stop`).

When you deploy the **same** image to Cloud Run, **do not** mount the key; attach the **Cloud Run service account** with Vertex + GCS roles (see Part C). The baked `.env` must already have **`GCP_PROJECT`** and no `GOOGLE_APPLICATION_CREDENTIALS` line.

### B3. Configure Docker for Artifact Registry (one-time per machine)

```bash
gcloud auth configure-docker us-central1-docker.pkg.dev
```

### B4. Tag the image

```bash
docker tag gem-rag:latest \
  us-central1-docker.pkg.dev/YOUR_PROJECT_ID/gem-rag-docker/gem-rag:latest
```

### B5. Push the image

```bash
docker push us-central1-docker.pkg.dev/YOUR_PROJECT_ID/gem-rag-docker/gem-rag:latest
```

Your Google user (or a **deployer** service account) needs permission to push to Artifact Registry.

---

## Part C — Deploy to Cloud Run

### C1. Where configuration lives

| Kind of value | Where to put it |
|---------------|-----------------|
| Passwords / client secrets | **Secret Manager**, exposed to the container as env vars |
| Non-secret settings (URLs, tenant ID, model names, bucket name) | Cloud Run **environment variables** |
| GCP access for Vertex + GCS | **Cloud Run service account** (runtime SA)—no JSON key inside the image |

### C2. Environment variables to set (non-secrets)

Align with [`.env.example`](../.env.example) and [`config.py`](../config.py):

| Variable | Notes |
|----------|--------|
| `GCP_PROJECT` | **Set explicitly** on Cloud Run (project is not read from a key file if you do not mount one). |
| `GCP_LOCATION` | e.g. `us-central1` |
| `VERTEX_MODEL` | e.g. `gemini-2.5-flash` |
| `VERTEX_EMBEDDING_MODEL` | e.g. `text-embedding-004` |
| `SHAREPOINT_CLIENT_ID` | MSAL app ID |
| `SHAREPOINT_TENANT_ID` | Azure AD tenant |
| `SHAREPOINT_SITE_URL` | Site URL, no trailing slash |
| `SHAREPOINT_DRIVE_PATH` | Folder path under the library (if used) |
| `SHAREPOINT_DRIVE_ID` | Only if you use a non-default document library |
| `GCS_OUTPUT_BUCKET` | **Required for production on Cloud Run** — without it, disk is wiped when instances restart and progress/download endpoints lose prior Excel/state. |
| `GCS_PDF_STORAGE_PREFIX` | Optional; default `pdfs` |
| `EXTRACTION_MONITOR_INTERVAL_MINUTES` | `0` to disable periodic runs |
| `DATA_DIR` | Use `/app/data` (matches the Docker image) |

**On Cloud Run with an attached service account, do not set `GOOGLE_APPLICATION_CREDENTIALS`** unless your team explicitly uses a mounted secret as a key file (uncommon; prefer the service account attachment).

**Do not set `PORT` yourself on Cloud Run** unless you have a reason—Google injects it. The container must listen on that port (the provided `Dockerfile` does).

**Scaling note:** Long-running extraction uses in-memory/thread state. For predictable behavior, many teams set **maximum instances to `1`** for this workload until you redesign for a queue/worker model.

### C3. Secret → environment variable

Map Secret Manager secret `sharepoint-client-secret` to env var `SHAREPOINT_CLIENT_SECRET` (see deploy command below or use the Cloud Console **Variables & secrets** UI).

### C4. Example `gcloud run deploy`

Replace placeholders. Adjust `--allow-unauthenticated` for your security requirements.

```bash
gcloud run deploy gem-rag \
  --image us-central1-docker.pkg.dev/YOUR_PROJECT_ID/gem-rag-docker/gem-rag:latest \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --service-account gem-rag-runtime@YOUR_PROJECT_ID.iam.gserviceaccount.com \
  --set-env-vars "\
GCP_PROJECT=YOUR_PROJECT_ID,\
GCP_LOCATION=us-central1,\
VERTEX_MODEL=gemini-2.5-flash,\
VERTEX_EMBEDDING_MODEL=text-embedding-004,\
SHAREPOINT_CLIENT_ID=YOUR_SP_CLIENT_ID,\
SHAREPOINT_TENANT_ID=YOUR_TENANT_ID,\
SHAREPOINT_SITE_URL=https://yourtenant.sharepoint.com/sites/YourSite,\
SHAREPOINT_DRIVE_PATH=Shared Documents/YourFolder,\
DATA_DIR=/app/data,\
GCS_OUTPUT_BUCKET=YOUR_BUCKET_NAME,\
EXTRACTION_MONITOR_INTERVAL_MINUTES=0" \
  --set-secrets "SHAREPOINT_CLIENT_SECRET=sharepoint-client-secret:latest"
```

If a flag differs for your `gcloud` version, use the Cloud Console: **Cloud Run → your service → Edit & deploy new revision → Variables & secrets**.

### C5. After deploy

Open the service URL:

- `/` — dashboard UI  
- `/docs` — OpenAPI (FastAPI)  

---

## Part D — When you still use `gcp-key.json`

| Situation | Need `gcp-key.json`? |
|-----------|----------------------|
| **Cloud Run runtime** (recommended setup) | **No** — use the Cloud Run **service account**. |
| **Local machine / VDI** to run `gcloud`, `docker push`, or test with Docker Compose | Often **yes**, or use `gcloud auth application-default login`. That credential is for **build/deploy/dev**, not inside the Cloud Run image. |

---

## Part E — Deployment checklist

- [ ] GCP project + billing ready  
- [ ] APIs enabled (Run, Artifact Registry, Secret Manager, Vertex, Storage)  
- [ ] Artifact Registry repository created  
- [ ] Runtime service account created; Vertex (+ GCS) permissions granted  
- [ ] SharePoint secret in Secret Manager; runtime SA has **Secret Accessor**  
- [ ] (Recommended) GCS bucket created and writable by runtime SA  
- [ ] `docker build` succeeds  
- [ ] `gcloud auth configure-docker` done for the registry host  
- [ ] `docker push` succeeds  
- [ ] Cloud Run service deployed with **service account**, **env vars**, and **secret mapping**  
- [ ] Service URL tested (`/health` or UI)  

---

## Related files in this repo

- [`Dockerfile`](../Dockerfile) — image build  
- [`docker-compose.yml`](../docker-compose.yml) — local Docker Compose (not used by Cloud Run)  
- [`.env.example`](../.env.example) — variable names (copy to local `.env`; do not commit secrets)  
- [`config.py`](../config.py) — how env vars are read  
