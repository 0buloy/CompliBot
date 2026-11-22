"""FastAPI service.

    GET  /health   -> liveness + index stats
    POST /ask      -> AnswerResult (answer, citations, status, usage, latency)
    POST /agent    -> AgentResult (tool-calling agent: search, compare, checklist JSON)
    POST /search   -> retrieved clauses only (no LLM call; handy for debugging)

The caller's role is derived server-side from the X-API-Key header, never from
the request body: a client cannot simply claim to be "internal".
Run:  uvicorn complibot.api:app --reload
"""

from __future__ import annotations

import hmac
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .agent import Agent, AgentResult
from .answer import CompliBot
from .config import Settings
from .schemas import AnswerResult

log = logging.getLogger("complibot.api")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class SearchHit(BaseModel):
    chunk_id: str
    doc_title: str
    section_id: str
    section_title: str
    page_start: int
    access_level: str
    score: float
    text: str


def role_for_key(api_key: str | None, settings: Settings) -> str:
    if api_key:
        for key in settings.internal_api_keys:
            if hmac.compare_digest(api_key, key):  # constant-time comparison
                return "internal"
    return "public"


def create_app(bot: CompliBot | None = None, settings: Settings | None = None) -> FastAPI:
    """App factory. Tests inject a ready-made bot; production loads from env."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings or (bot.settings if bot else Settings.from_env())
        app.state.bot = bot
        app.state.load_error = None
        if app.state.bot is None:
            try:
                app.state.bot = CompliBot.from_settings(app.state.settings)
            except (FileNotFoundError, ValueError) as e:  # start anyway; report via /health
                app.state.load_error = str(e)
                log.error("CompliBot failed to load: %s", e)
        yield

    app = FastAPI(title="CompliBot", version="0.1.0", lifespan=lifespan,
                  description="Cited answers over Singapore regulatory documents.")

    @app.middleware("http")
    async def request_id(request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
        request.state.request_id = rid
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        # Log the details server-side; never send stack traces (or input text) to the client.
        rid = getattr(request.state, "request_id", "-")
        log.exception("unhandled error request_id=%s", rid)
        return JSONResponse(status_code=500, content={"detail": "Internal error", "request_id": rid})

    def get_bot(request: Request) -> CompliBot:
        if request.app.state.bot is None:
            raise HTTPException(status_code=503, detail=f"Index not loaded: {request.app.state.load_error}")
        return request.app.state.bot

    def get_role(request: Request, x_api_key: str | None = Header(default=None)) -> str:
        return role_for_key(x_api_key, request.app.state.settings)

    @app.get("/health")
    def health(request: Request) -> dict:
        b = request.app.state.bot
        return {
            "status": "ok" if b else "degraded",
            "chunks": len(b.retriever.chunks) if b else 0,
            "llm": f"{b.llm.name}:{b.llm.model}" if b else None,
            "error": request.app.state.load_error,
        }

    @app.post("/ask", response_model=AnswerResult)
    def ask(body: AskRequest, bot: CompliBot = Depends(get_bot), role: str = Depends(get_role)) -> AnswerResult:
        result = bot.ask(body.question, role=role, top_k=body.top_k)
        log.info("ask role=%s status=%s latency_ms=%.0f", role, result.status, result.latency_ms)
        return result

    @app.post("/agent", response_model=AgentResult)
    def agent(body: AskRequest, bot: CompliBot = Depends(get_bot), role: str = Depends(get_role)) -> AgentResult:
        result = Agent.from_bot(bot).run(body.question, role=role)
        log.info("agent role=%s status=%s steps=%d latency_ms=%.0f", role, result.status, len(result.steps),
                 result.latency_ms)
        return result

    @app.post("/search", response_model=list[SearchHit])
    def search(body: AskRequest, bot: CompliBot = Depends(get_bot), role: str = Depends(get_role)) -> list[SearchHit]:
        check = bot.guardrails.check_input(body.question)  # redact before searching/logging
        hits = bot.retriever.search(check.text, role=role, top_k=body.top_k)
        return [
            SearchHit(chunk_id=h.chunk.chunk_id, doc_title=h.chunk.doc_title, section_id=h.chunk.section_id,
                      section_title=h.chunk.section_title, page_start=h.chunk.page_start,
                      access_level=h.chunk.access_level, score=round(h.score, 5), text=h.chunk.text)
            for h in hits
        ]

    return app


app = create_app()
