from pydantic import BaseModel
from typing import Optional


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
