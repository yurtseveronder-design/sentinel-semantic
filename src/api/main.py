import time
import uuid
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any

from semantic.router import SemanticRouter
from semantic.metadata import MetadataExtractor

app = FastAPI(
    title="Sentinel Semantic Layer",
    description="Enterprise-grade semantic router and metadata extraction gateway for AI and RAG applications.",
    version="1.0.0"
)

# 1. CORS Middleware for sentinel-ui integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production domains as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Lightweight Correlation ID & Request Timer Middleware
@app.middleware("http")
async def add_security_and_tracing_headers(request: Request, call_next):
    correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    start_time = time.perf_counter()
    
    # Process request through FastAPI stack
    response = await call_next(request)
    
    process_time = time.perf_counter() - start_time
    response.headers["X-Correlation-ID"] = correlation_id
    response.headers["X-Process-Time"] = f"{process_time:.4f}s"
    return response

# Initialize Core Semantic Engines
router = SemanticRouter()
metadata_extractor = MetadataExtractor()

class PromptRequest(BaseModel):
    prompt: str = Field(..., description="The incoming natural language user prompt.")

@app.get("/health", summary="Health Check")
async def health_check():
    return {"status": "healthy", "service": "sentinel-semantic"}

@app.post("/api/v1/semantic/route", summary="Route Semantic Intent & Guardrails")
async def route_semantic_intent(payload: PromptRequest):
    """
    Analyzes the prompt using vector similarity against YAML anchors,
    detects multi-intents, and assigns security/PII guardrail profiles.
    """
    result = router.route(payload.prompt)
    return result.model_dump()

@app.post("/api/v1/semantic/filter", summary="Extract Metadata and RAG Filters")
async def extract_metadata(payload: PromptRequest):
    """
    Extracts structured database filters and cleans queries for RAG vector retrieval.
    """
    filters = metadata_extractor.extract(payload.prompt)
    return filters