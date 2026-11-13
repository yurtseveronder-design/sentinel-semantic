import re
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class ExtractedFilters(BaseModel):
    department: Optional[str] = Field(default=None, description="Target department (e.g., finance, hr, engineering).")
    access_level: Optional[str] = Field(default=None, description="Security clearance level (e.g., public, internal, confidential).")
    year: Optional[int] = Field(default=None, description="Extracted year reference.")
    tags: List[str] = Field(default_factory=list, description="Additional topical tags.")
    cleaned_query: str = Field(..., description="The original prompt with metadata keywords stripped for better embedding search.")

class MetadataExtractor:
    """
    Extracts structured database filters and cleans queries from natural language prompts.
    Designed to be plug-and-play with vector databases (Chroma, Qdrant, Pinecone).
    """
    
    DEPARTMENTS = ["finance", "hr", "engineering", "legal", "sales", "marketing"]
    ACCESS_LEVELS = ["public", "internal", "confidential", "restricted"]

    def __init__(self):
        self._dept_pattern = re.compile(r'\b(' + '|'.join(self.DEPARTMENTS) + r')\b', re.IGNORECASE)
        self._access_pattern = re.compile(r'\b(' + '|'.join(self.ACCESS_LEVELS) + r')\b', re.IGNORECASE)
        self._year_pattern = re.compile(r'\b(20\d{2})\b')

    def extract(self, prompt: str) -> ExtractedFilters:
        dept_match = self._dept_pattern.search(prompt)
        department = dept_match.group(1).lower() if dept_match else None

        access_match = self._access_pattern.search(prompt)
        access_level = access_match.group(1).lower() if access_match else "internal"

        year_match = self._year_pattern.search(prompt)
        year = int(year_match.group(1)) if year_match else None

        cleaned_query = prompt
        if dept_match:
            cleaned_query = cleaned_query.replace(dept_match.group(0), "")
        if access_match:
            cleaned_query = cleaned_query.replace(access_match.group(0), "")
        
        cleaned_query = re.sub(r'\s+', ' ', cleaned_query).strip()

        return ExtractedFilters(
            department=department,
            access_level=access_level,
            year=year,
            tags=[department] if department else [],
            cleaned_query=cleaned_query if cleaned_query else prompt
        )

    def to_vector_db_filter(self, filters: ExtractedFilters) -> Dict[str, Any]:
        db_filter: Dict[str, Any] = {}
        if filters.department:
            db_filter["department"] = filters.department
        if filters.access_level:
            db_filter["access_level"] = filters.access_level
        if filters.year:
            db_filter["year"] = filters.year
        return db_filter