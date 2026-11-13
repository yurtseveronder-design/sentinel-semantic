from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class RouteResult(BaseModel):
    intent: str = Field(..., description="The classified intent of the user prompt.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence score.")
    guardrail_profile: str = Field(..., description="The security profile required by ai-sentinel.")
    requires_rag: bool = Field(..., description="Whether the query requires vector database retrieval.")

class SemanticRouter:
    """
    Lightweight intent router and classifier for the sentinel-semantic service.
    Classifies prompts to guide ai-sentinel security policies and sentinel-ui RAG routing.
    """
    
    INTENT_KEYWORDS = {
        "financial_query": ["budget", "cost", "revenue", "expense", "financial", "salary", "invoice"],
        "hr_policy": ["leave", "vacation", "benefit", "hr", "policy", "employee", "onboarding"],
        "technical_query": ["code", "error", "bug", "pytest", "api", "backend", "deploy", "github"],
        "potential_jailbreak": ["ignore previous instructions", "system prompt", "bypass", "jailbreak", "override"]
    }

    def route(self, prompt: str) -> RouteResult:
        prompt_lower = prompt.lower()
        
        # 1. Check for security bypasses / jailbreaks first (Highest priority for ai-sentinel)
        for keyword in self.INTENT_KEYWORDS["potential_jailbreak"]:
            if keyword in prompt_lower:
                return RouteResult(
                    intent="potential_jailbreak",
                    confidence=0.99,
                    guardrail_profile="maximum_security_block",
                    requires_rag=False
                )

        # 2. Check specialized domain intents
        for intent, keywords in self.INTENT_KEYWORDS.items():
            if intent == "potential_jailbreak":
                continue
            if any(kw in prompt_lower for kw in keywords):
                guardrail = "strict_financial" if intent == "financial_query" else "standard"
                requires_rag = intent in ["financial_query", "hr_policy", "technical_query"]
                return RouteResult(
                    intent=intent,
                    confidence=0.88,
                    guardrail_profile=guardrail,
                    requires_rag=requires_rag
                )

        # 3. Default fallback intent
        return RouteResult(
            intent="general_chat",
            confidence=0.75,
            guardrail_profile="standard",
            requires_rag=False
        )