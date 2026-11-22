# CompliBot API image.  Build:  docker build -t complibot .
#                       Run:    docker run -p 8000:8000 --env-file .env complibot
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000

WORKDIR /app

# Dependencies first (layer cache): only re-installed when pyproject.toml changes.
COPY pyproject.toml README.md ./
COPY src/complibot/__init__.py src/complibot/__init__.py
RUN pip install ".[anthropic,openai,agent,guardrails,tracing]" \
 && python -m spacy download en_core_web_sm

COPY src ./src
RUN pip install --no-deps .

# Documents are baked in and indexed at build time, so the container starts instantly
# and needs no writable volume. (Real PDFs: put them in data/raw before building.)
COPY data/raw ./data/raw
RUN complibot ingest

RUN useradd --create-home --uid 10001 app && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", 8000)}/health', timeout=4)"

# $PORT lets the same image run on Cloud Run / Render / Fly, which inject their own port.
CMD ["sh", "-c", "uvicorn complibot.api:app --host 0.0.0.0 --port ${PORT} --workers 1"]
