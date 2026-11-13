"""Intent routing: embedding similarity when available, deterministic keywords otherwise."""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field

try:  # optional heavy dependency: pip install "sentinel-semantic[embeddings]"
    from sentence_transformers import SentenceTransformer, util

    HAS_TRANSFORMERS = True
except (ImportError, OSError):
    HAS_TRANSFORMERS = False

DEFAULT_CONFIG = Path(__file__).resolve().parent / "intents.yaml"
RAG_INTENTS = {"financial_query", "hr_policy", "technical_query"}
SENSITIVE_TERMS = ("password", "credit card", "ssn", "secret token", "api key")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")


class RouteResult(BaseModel):
    primary_intent: str = Field(..., description="The primary classified intent of the prompt.")
    all_intents: List[str] = Field(..., description="Every intent at or above the confidence threshold.")
    confidence_scores: Dict[str, float] = Field(..., description="Confidence per intent (0-1).")
    guardrail_profile: str = Field(..., description="Security profile the downstream gateway should apply.")
    requires_rag: bool = Field(..., description="Whether the query needs vector-database retrieval.")
    sensitivity_tags: List[str] = Field(..., description="Detected data-sensitivity or risk markers.")


class SemanticRouter:
    """Classifies prompts into intents and picks a guardrail profile.

    engine: "auto" (default; embeddings if installed and loadable, else keywords),
            "keyword" (always deterministic), or "embedding" (fail if unavailable).
    Also configurable with SEMANTIC_ENGINE; SEMANTIC_INTENTS_PATH overrides the YAML.
    """

    def __init__(self, config_path: Optional[str] = None, engine: Optional[str] = None):
        path = Path(config_path or os.getenv("SEMANTIC_INTENTS_PATH") or DEFAULT_CONFIG)
        self.config = self._load_config(path)
        intents = self.config.get("intents", {})
        self.intent_anchors = {k: v.get("anchors", []) for k, v in intents.items()}
        self.intent_keywords = {k: [w.lower() for w in v.get("keywords", [])] for k, v in intents.items()}
        self.jailbreak_patterns = [re.compile(p, re.IGNORECASE) for p in self.config.get("jailbreak_patterns", [])]
        self.confidence_threshold = float(self.config.get("confidence_threshold", 0.60))
        self.CONFIDENCE_THRESHOLD = self.confidence_threshold  # kept for backward compatibility

        requested = (engine or os.getenv("SEMANTIC_ENGINE", "auto")).lower()
        if requested not in {"auto", "keyword", "embedding"}:
            raise ValueError(f"Unknown engine '{requested}'. Use auto, keyword or embedding.")
        strict = requested == "embedding" or os.getenv("STRICT_MODEL_LOADING", "false").lower() == "true"

        self.has_model = False
        if requested != "keyword" and HAS_TRANSFORMERS:
            try:
                self.model = SentenceTransformer(os.getenv("SEMANTIC_MODEL_NAME", "all-MiniLM-L6-v2"))
                self._precompute_anchors()
                self.has_model = True
            except Exception as exc:
                if strict:
                    raise RuntimeError(f"Could not load the embedding model: {exc}") from exc
        elif requested == "embedding":
            raise RuntimeError("engine=embedding requested but sentence-transformers is not installed")

    @property
    def engine(self) -> str:
        return "embedding" if self.has_model else "keyword"

    @staticmethod
    def _load_config(path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Intent configuration not found: {path}")
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if not isinstance(data, dict) or "intents" not in data:
            raise ValueError(f"Invalid intent configuration (missing 'intents'): {path}")
        return data

    def _precompute_anchors(self) -> None:
        self.anchor_embeddings = {
            intent: self.model.encode(sentences, convert_to_tensor=True)
            for intent, sentences in self.intent_anchors.items()
        }

    def _scores(self, prompt: str) -> Dict[str, float]:
        if self.has_model:
            embedding = self.model.encode(prompt, convert_to_tensor=True)
            return {
                intent: round(float(util.cos_sim(embedding, tensor).max().item()), 3)
                for intent, tensor in self.anchor_embeddings.items()
            }
        lowered = prompt.lower()
        return {
            intent: (0.85 if any(word in lowered for word in words) else 0.20)
            for intent, words in self.intent_keywords.items()
        }

    def route(self, prompt: str) -> RouteResult:
        lowered = prompt.lower()
        tags: List[str] = []
        if any(term in lowered for term in SENSITIVE_TERMS):
            tags.append("SENSITIVE_DATA_EXPOSURE")
        if SSN_RE.search(prompt) or EMAIL_RE.search(prompt):
            tags.append("PII_DETECTED")

        scores = self._scores(prompt)
        detected = [intent for intent, score in scores.items() if score >= self.confidence_threshold]

        pattern_hit = any(p.search(prompt) for p in self.jailbreak_patterns)
        if pattern_hit or "potential_jailbreak" in detected:
            tags.append("SECURITY_BYPASS_ATTEMPT")
            scores["potential_jailbreak"] = max(scores.get("potential_jailbreak", 0.0), 0.95 if pattern_hit else 0.0)
            return RouteResult(
                primary_intent="potential_jailbreak",
                all_intents=["potential_jailbreak"],
                confidence_scores=scores,
                guardrail_profile="maximum_security_block",
                requires_rag=False,
                sensitivity_tags=sorted(set(tags)),
            )

        if detected:
            detected.sort(key=lambda name: scores[name], reverse=True)
        else:
            detected = ["general_chat"]
        primary = detected[0]

        profile = "strict_financial" if primary == "financial_query" else "standard"
        if "SENSITIVE_DATA_EXPOSURE" in tags or "PII_DETECTED" in tags:
            profile = "strict_pii_redaction"

        return RouteResult(
            primary_intent=primary,
            all_intents=detected,
            confidence_scores=scores,
            guardrail_profile=profile,
            requires_rag=any(intent in RAG_INTENTS for intent in detected),
            sensitivity_tags=sorted(set(tags)),
        )
