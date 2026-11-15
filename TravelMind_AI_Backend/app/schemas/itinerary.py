from pydantic import BaseModel
from typing import List, Dict, Any
from .trip import TripResponse


class FullItineraryResponse(BaseModel):
    trip: TripResponse
    days: List[Dict[str, Any]]
