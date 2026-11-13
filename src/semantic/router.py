import os
import re
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

try:
    from sentence_transformers import SentenceTransformer, util
    HAS_TRANSFORMERS = True
except (ImportError, OSError):
    HAS_TRANSFORMERS = False

class RouteResult(BaseModel):
    primary_intent: str = Field(..., description="The primary classified intent of the user prompt.")
    all_intents: List[str] = Field(..., description="Multi-intent detection results matching the threshold.")
    confidence_scores: Dict[str, float] = Field(..., description="Cosine similarity confidence scores per intent.")
    guardrail_profile: str = Field(..., description="The targeted security profile required by ai-sentinel.")
    requires_rag: bool = Field(..., description="Whether the query requires vector database retrieval.")
    sensitivity_tags: List[str] = Field(..., description="Detected data sensitivity or risk markers.")

class SemanticRouter:
    """
    Enterprise-grade semantic router loading decoupled YAML anchors,
    supporting strict model loading, confidence thresholds, and PII tagging.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        if not config_path:
            # Look for config/intents.yaml relative to project root
            base_dir = Path(__file__).resolve().parent.parent.parent
            config_path = base_dir / "config" / "intents.yaml"
            
        self.config = self._load_config(config_path)
        self.intent_anchors = {k: v["anchors"] for k, v in self.config.get("intents", {}).items()}
        self.confidence_threshold = self.config.get("confidence_threshold", 0.60)
        
        # Uppercase alias required by test suite
        self.CONFIDENCE_THRESHOLD = self.confidence_threshold
        
        # Strict Model Loading Check via Environment Variable
        strict_loading = os.getenv("STRICT_MODEL_LOADING", "false").lower() == "true"
        model_name = os.getenv("SEMANTIC_MODEL_NAME", "all-MiniLM-L6-v2")
        
        self.has_model = False
        if HAS_TRANSFORMERS:
            try:
                self.model = SentenceTransformer(model_name)
                self._precompute_anchors()
                self.has_model = True
            except Exception as e:
                if strict_loading:
                    raise RuntimeError(f"Strict model loading enabled, but failed to load SentenceTransformer: {e}")
                self.has_model = False

    def _load_config(self, path: Path) -> dict:
        """Loads intent anchors and configuration from YAML."""
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        # Fallback default if YAML is missing
        return {
            "confidence_threshold": 0.60,
            "intents": {
                "general_chat": {"anchors": ["Hello", "How are you?"]}
            }
        }

    def _precompute_anchors(self):
        """Precomputes vector embeddings for anchor intent sentences."""
        self.anchor_embeddings = {}
        for intent, sentences in self.intent_anchors.items():
            self.anchor_embeddings[intent] = self.model.encode(sentences, convert_to_tensor=True)

    def route(self, prompt: str) -> RouteResult:
        prompt_lower = prompt.lower()
        confidence_scores: Dict[str, float] = {}
        detected_intents: List[str] = []
        sensitivity_tags: List[str] = []

        # 1. Entity & Sensitivity Tagging (PII / Security Scans)
        if any(term in prompt_lower for term in ["password", "credit card", "ssn", "secret token", "api key"]):
            sensitivity_tags.append("SENSITIVE_DATA_EXPOSURE")
        if re.search(r'\b\d{3}-\d{2}-\d{4}\b', prompt) or re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', prompt):
            sensitivity_tags.append("PII_DETECTED")

        # 2. Embedding Similarity or Fallback Heuristics
        if self.has_model:
            prompt_embedding = self.model.encode(prompt, convert_to_tensor=True)
            for intent, anchor_tensor in self.anchor_embeddings.items():
                sims = util.cos_sim(prompt_embedding, anchor_tensor)
                max_sim = float(sims.max().item())
                confidence_scores[intent] = round(max_sim, 3)
        else:
            keywords = {
                "financial_query": ["budget", "cost", "revenue", "expense", "bütçe", "maliyet"],
                "hr_policy": ["leave", "vacation", "benefit", "izin", "tatil"],
                "technical_query": ["code", "error", "pytest", "api", "hata"],
                "potential_jailbreak": ["ignore previous instructions", "bypass", "yoksay"]
            }
            for intent, kws in keywords.items():
                score = 0.85 if any(kw in prompt_lower for kw in kws) else 0.20
                confidence_scores[intent] = score

        # 3. Multi-Intent Detection & Threshold Filtering
        for intent, score in confidence_scores.items():
            if score >= self.confidence_threshold:
                detected_intents.append(intent)

        # 4. Jailbreak Security Override
        if "potential_jailbreak" in detected_intents or confidence_scores.get("potential_jailbreak", 0) > 0.70:
            sensitivity_tags.append("SECURITY_BYPASS_ATTEMPT")
            return RouteResult(
                primary_intent="potential_jailbreak",
                all_intents=["potential_jailbreak"],
                confidence_scores=confidence_scores,
                guardrail_profile="maximum_security_block",
                requires_rag=False,
                sensitivity_tags=list(set(sensitivity_tags))
            )

        if detected_intents:
            detected_intents.sort(key=lambda x: confidence_scores[x], reverse=True)
            primary_intent = detected_intents[0]
        else:
            primary_intent = "general_chat"
            detected_intents = ["general_chat"]

        guardrail_profile = "strict_financial" if primary_intent == "financial_query" else "standard"
        if "SENSITIVE_DATA_EXPOSURE" in sensitivity_tags or "PII_DETECTED" in sensitivity_tags:
            guardrail_profile = "strict_pii_redaction"

        requires_rag = any(intent in ["financial_query", "hr_policy", "technical_query"] for intent in detected_intents)

        return RouteResult(
            primary_intent=primary_intent,
            all_intents=detected_intents,
            confidence_scores=confidence_scores,
            guardrail_profile=guardrail_profile,
            requires_rag=requires_rag,
            sensitivity_tags=list(set(sensitivity_tags))
        )