from typing import Literal
from pydantic import BaseModel, Field


class CandidateSelectionSchema(BaseModel):
    """
    Schema for selecting the correct match from multiple candidates.
    """
    selected_candidate: int = Field(ge=0, description="Index of matching candidate (1-based), or 0 for no match")