# sentinel-semantic
Semantic router and metadata extraction gateway for AI security and RAG applications.
[ sentinel-ui / Client App ]
                    │
                    ▼ (HTTP Request via CORS)
      ┌──────────────────────────────────────────┐
      │        sentinel-semantic Gateway         │
      │                                          │
      │  ┌────────────────────────────────────┐  │
      │  │          FastAPI Endpoints         │  │
      │  └──────────────────┬─────────────────┘  │
      │                     │                    │
      │          ┌──────────┴──────────┐         │
      │          ▼                     ▼         │
      │  ┌──────────────┐     ┌──────────────┐   │
      │  │ Semantic     │     │ Metadata     │   │
      │  │ Router       │     │ Extractor    │   │
      │  └──────┬───────┘     └──────┬───────┘   │
      │         │                    │           │
      │         ▼                    ▼           │
      │  (YAML Anchors)     (Cleaned Query &     │
      │  (Cosine Similarity)  DB Filters)        │
      └──────────────────────────────────────────┘
                │                      │
                ▼                      ▼
      [ Guardrail Profiles ]   [ Vector Database Search ]
      (e.g., strict_financial, (RAG Retrieval in UI)
       maximum_security_block)
