from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime


class Document(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any]
    source: str
    created_at: datetime = datetime.now()


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


class AddItemRequest(BaseModel):
    day_id: int
    name: str
    activity_type: Optional[str] = None
    description: Optional[str] = None
    location: Optional[str] = None
    time: Optional[str] = None
    booking_info: Optional[str] = None
    order_index: Optional[int] = 0


class UpdateItemRequest(BaseModel):
    activity_type: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    location: Optional[str] = None
    time: Optional[str] = None
    booking_info: Optional[str] = None
    order_index: Optional[int] = None


class ItemResponse(BaseModel):
    id: int
    day_id: int
    activity_type: Optional[str] = None
    name: str
    description: Optional[str] = None
    location: Optional[str] = None
    time: Optional[str] = None
    booking_info: Optional[str] = None
    order_index: int


class FullItineraryResponse(BaseModel):
    trip: TripResponse
    days: List[Dict[str, Any]]