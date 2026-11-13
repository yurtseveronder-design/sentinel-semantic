"""FastAPI service exposing the semantic router and metadata extractor.

Environment:
  SEMANTIC_API_KEY        if set, /api/v1/* requires a matching X-API-Key header
  SEMANTIC_CORS_ORIGINS   comma-separated allowed browser origins (default: none)
  SEMANTIC_ENGINE         auto | keyword | embedding (see router.py)
"""

import hmac
import logging
import os
import time
import uuid
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sentinel_semantic import __version__
from sentinel_semantic.metadata import ExtractedFilters, MetadataExtractor
from sentinel_semantic.router import RouteResult, SemanticRouter

log = logging.getLogger("sentinel_semantic")
MAX_PROMPT_CHARS = 8000


class PromptRequest(BaseModel):
    prompt: str = Field(..., max_length=MAX_PROMPT_CHARS, description="The natural-language user prompt.")


class AnalyzeResponse(BaseModel):
    route: RouteResult
    filters: ExtractedFilters
    vector_db_filter: dict
    engine: str


def create_app() -> FastAPI:
    api_key = os.getenv("SEMANTIC_API_KEY")
    if not api_key:
        log.warning("SEMANTIC_API_KEY is not set: the API is unauthenticated.")

    app = FastAPI(
        title="Sentinel Semantic Layer",
        description="Semantic intent router and metadata extractor for AI-security and RAG pipelines.",
        version=__version__,
    )

    origins = [o.strip() for o in os.getenv("SEMANTIC_CORS_ORIGINS", "").split(",") if o.strip()]
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "X-API-Key", "X-Correlation-ID"],
        )

    @app.middleware("http")
    async def tracing(request: Request, call_next):
        correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = correlation_id
        response.headers["X-Process-Time"] = f"{time.perf_counter() - started:.4f}s"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    router = SemanticRouter()
    extractor = MetadataExtractor()

    def require_key(x_api_key: Optional[str] = Header(None)) -> None:
        if api_key and not (x_api_key and hmac.compare_digest(x_api_key.encode(), api_key.encode())):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing API key.")

    @app.get("/health", summary="Health check")
    def health() -> dict:
        return {"status": "ok", "service": "sentinel-semantic", "version": __version__, "engine": router.engine}

    @app.post("/api/v1/semantic/route", summary="Classify intent and pick a guardrail profile",
              dependencies=[Depends(require_key)])
    def route(payload: PromptRequest) -> dict:
        return router.route(payload.prompt).model_dump()

    @app.post("/api/v1/semantic/filter", summary="Extract metadata and RAG filters",
              dependencies=[Depends(require_key)])
    def filters(payload: PromptRequest) -> dict:
        return extractor.extract(payload.prompt).model_dump()

    @app.post("/api/v1/semantic/analyze", summary="Route and extract filters in one call",
              response_model=AnalyzeResponse, dependencies=[Depends(require_key)])
    def analyze(payload: PromptRequest) -> AnalyzeResponse:
        extracted = extractor.extract(payload.prompt)
        return AnalyzeResponse(
            route=router.route(payload.prompt),
            filters=extracted,
            vector_db_filter=extractor.to_vector_db_filter(extracted),
            engine=router.engine,
        )

    return app


app = create_app()
