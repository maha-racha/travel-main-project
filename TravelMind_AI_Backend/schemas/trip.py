from pydantic import BaseModel
from typing import Dict, Any, Optional


class CreateTripRequest(BaseModel):
    user_id: str
    title: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    budget: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None


class UpdateTripRequest(BaseModel):
    title: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    budget: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None


class TripResponse(BaseModel):
    id: int
    user_id: str
    title: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    budget: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None
    created_at: str
