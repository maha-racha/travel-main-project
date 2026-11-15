from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class QueryRequest(BaseModel):
    query: str
    user_id: Optional[str] = "default_user"
    trip_id: Optional[int] = None
    top_k: Optional[int] = 5
    max_tokens: Optional[int] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]


class ChatMessage(BaseModel):
    type: str
    content: str
    metadata: Optional[Dict[str, Any]] = None
