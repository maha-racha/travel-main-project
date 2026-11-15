from pydantic import BaseModel
from typing import Dict, Any
from datetime import datetime


class Document(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any]
    source: str
    created_at: datetime = datetime.now()
