from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from semantic.router import SemanticRouter, RouteResult
from semantic.metadata import MetadataExtractor, ExtractedFilters

app = FastAPI(
    title="Sentinel Semantic Layer",
    description="API contract for intent routing and metadata filtering.",
    version="1.0.0"
)

router = SemanticRouter()
extractor = MetadataExtractor()

# --- Request & Response Contracts ---

class SemanticRouteRequest(BaseModel):
    prompt: str = Field(..., description="The raw user prompt from the UI.")

class MetadataFilterRequest(BaseModel):
    prompt: str = Field(..., description="The user prompt to extract filters from.")

# --- Endpoints ---

@app.post("/api/v1/semantic/route", response_model=RouteResult, status_code=status.HTTP_200_OK)
async def route_intent(payload: SemanticRouteRequest):
    """
    Analyzes user prompt intent to guide ai-sentinel security policies.
    """
    try:
        return router.route(payload.prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/semantic/filter", status_code=status.HTTP_200_OK)
async def extract_metadata(payload: MetadataFilterRequest):
    """
    Extracts structured database filters and cleans queries for vector RAG search.
    """
    try:
        extracted: ExtractedFilters = extractor.extract(payload.prompt)
        db_filters = extractor.to_vector_db_filter(extracted)
        
        return {
            "extracted_filters": db_filters,
            "semantic_search_query": extracted.cleaned_query
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    return {"status": "healthy", "service": "sentinel-semantic"}