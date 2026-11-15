from pydantic import BaseModel
from typing import Optional


class AddDayRequest(BaseModel):
    trip_id: int
    day_number: int
    date: Optional[str] = None
    description: Optional[str] = None


class DayResponse(BaseModel):
    id: int
    trip_id: int
    day_number: int
    date: Optional[str] = None
    description: Optional[str] = None
