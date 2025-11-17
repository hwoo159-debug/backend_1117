from pydantic import BaseModel, Field
from typing import Optional, List

class AskRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=20)

class AskResponse(BaseModel):
    answer: str
    sources: list[dict]

class IngestItem(BaseModel):
    title: str
    content: str
    metadata: Optional[dict] = None

class IngestBatch(BaseModel):
    items: List[IngestItem]
