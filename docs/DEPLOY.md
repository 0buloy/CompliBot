# Deploying CompliBot (near-zero cost)

The API is a single stateless container. The index is baked into the image at build time, so
there's no database, volume or vector store to pay for. The only real running cost is LLM tokens.

| Option | Cost at demo traffic | Notes |
|---|---|---|
| **Google Cloud Run, `asia-southeast1` (Singapore)** ← recommended | Typically within the monthly free tier; scales to zero | Singapore region = data-residency talking point. Secrets in Secret Manager. |
| Render free web service | Free | Sleeps after inactivity (slow first request); ~512 MB RAM, so set `COMPLIBOT_PII_ENGINE=regex`. |
| Streamlit Community Cloud (UI only) | Free | Hosts `ui/app.py` from your GitHub repo and calls the API via `COMPLIBOT_API_URL`. |

Free-tier limits and regional prices change; check the current Cloud Run pricing page before you
deploy, and **set a billing budget alert** (Billing → Budgets & alerts, e.g. USD 5) on day one.

## 0. Build and run locally

```bash
docker build -t complibot .
docker run -p 8000:8000 --env-file .env complibot
curl localhost:8000/health
```

The image installs Presidio with the spaCy `en_core_web_sm` model and indexes whatever is in
`data/raw/` at build time. Put the regulatory PDFs there before building.

## 1. Google Cloud Run (Singapore)

One-time setup:

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com \
    secretmanager.googleapis.com
```

Store secrets in Secret Manager, never in env files or the image:

```bash
printf '%s' "sk-ant-..." | gcloud secrets create anthropic-api-key --data-file=-
printf '%s' "$(openssl rand -hex 24)" | gcloud secrets create complibot-internal-keys --data-file=-
PROJECT_NUMBER=$(gcloud projects describe "$(gcloud config get-value project)" --format='value(projectNumber)')
for s in anthropic-api-key complibot-internal-keys; do
  gcloud secrets add-iam-policy-binding "$s" \
    --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
    --role=roles/secretmanager.secretAccessor
done
```

Deploy. Cloud Build builds the Dockerfile, so no local Docker is needed:

```bash
gcloud run deploy complibot \
  --source . \
  --region asia-southeast1 \
  --allow-unauthenticated \
  --memory 1Gi --cpu 1 \
  --min-instances 0 --max-instances 2 --concurrency 10 \
  --set-env-vars COMPLIBOT_LLM_PROVIDER=anthropic,ANTHROPIC_MODEL=claude-opus-5-5,COMPLIBOT_AUDIT_LOG=/tmp/audit.jsonl \
  --set-secrets ANTHROPIC_API_KEY=anthropic-api-key:latest,COMPLIBOT_INTERNAL_API_KEYS=complibot-internal-keys:latest
```

Cost and safety knobs:

- `--min-instances 0` scales to zero, so you pay nothing while idle. The first request after idle
  takes a few seconds because Presidio and spaCy load at startup.
- `--max-instances 2` caps runaway scaling (and runaway LLM spend).
- LLM spend: set a **monthly spend limit** on the API key's workspace in the Anthropic Console. For a
  public demo you can point `ANTHROPIC_MODEL` at a cheaper model; run the eval first to see what you
  give up. The public role only sees `data/raw/public`.
- Optional tracing: add `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` as secrets (the Langfuse
  Cloud hobby tier is free).
- Cloud Run writes stdout to Cloud Logging. `/tmp/audit.jsonl` lives in memory and is lost when
  the instance stops. For a durable audit trail, log audit records to stdout as JSON or ship them
  to a bucket.

Smoke test:

```bash
URL=$(gcloud run services describe complibot --region asia-southeast1 --format='value(status.url)')
curl -s "$URL/health"
curl -s -X POST "$URL/ask" -H 'Content-Type: application/json' \
     -d '{"question": "When must an organisation notify the PDPC of a data breach?"}'
```

### Optional: deploy from GitHub Actions (CD)

Use Workload Identity Federation (no long-lived JSON keys in GitHub). After setting up a WIF pool
and a deployer service account with `roles/run.admin`, `roles/cloudbuild.builds.editor`,
`roles/artifactregistry.writer` and `roles/iam.serviceAccountUser`, add this job to
`.github/workflows/ci.yml`:

```yaml
  deploy:
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    needs: [eval, docker]
    runs-on: ubuntu-latest
    permissions: {contents: read, id-token: write}
    steps:
      - uses: actions/checkout@v4
      - uses: google-github-actions/auth@v2
        with:
          workload_identity_provider: ${{ vars.GCP_WIF_PROVIDER }}
          service_account: ${{ vars.GCP_DEPLOY_SA }}
      - uses: google-github-actions/deploy-cloudrun@v2
        with:
          service: complibot
          region: asia-southeast1
          source: .
```

The eval job runs first, so a change that breaks retrieval, refusals or citations never reaches
production.

## 2. Streamlit UI on Streamlit Community Cloud

1. Push the repo to GitHub, then on share.streamlit.io create an app with main file `ui/app.py`.
2. In the app's **Secrets**, set `COMPLIBOT_API_URL = "https://<your-cloud-run-url>"`.
3. Visitors use the public role. Internal access needs a key you hand out, entered in the sidebar.

Without `COMPLIBOT_API_URL` the UI loads the index in-process, which is handy for a local demo:
`streamlit run ui/app.py`.

## 3. Render (alternative)

New → Web Service → from your repo → Runtime: Docker. Add env vars `COMPLIBOT_LLM_PROVIDER`,
`ANTHROPIC_API_KEY` (as a secret), `COMPLIBOT_INTERNAL_API_KEYS` and `COMPLIBOT_PII_ENGINE=regex` (fits
the free tier's memory). Render injects `$PORT`, which the image already honours.

## Production checklist (beyond a portfolio demo)

- [ ] Real identity (OIDC/SSO) mapping users to roles instead of static API keys.
- [ ] Per-key rate limiting (e.g. at an API gateway) and request size limits.
- [ ] Durable, tamper-evident audit log storage with a retention policy.
- [ ] Confirm the LLM provider's data retention and processing location against your PDPA transfer
      limitation obligations. PII is redacted before the call, but questions can still be sensitive.
- [ ] Re-run `complibot eval` on the real golden set before each model or prompt change.
- [ ] Slim the image (multi-stage build, CPU-only wheels) if cold start matters.
