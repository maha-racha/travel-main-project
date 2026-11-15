from fastapi import APIRouter, HTTPException
from app.models.schemas import (
    CreateTripRequest, UpdateTripRequest, TripResponse,
    AddDayRequest, DayResponse,
    AddItemRequest, UpdateItemRequest, ItemResponse,
    FullItineraryResponse
)
from app.services.itinerary_service import ItineraryService
from typing import List

router = APIRouter()


def setup_itinerary_routes(itinerary_service: ItineraryService):
    @router.post("/trips", response_model=TripResponse)
    async def create_trip(request: CreateTripRequest):
        try:
            trip_id = itinerary_service.create_trip(
                user_id=request.user_id,
                title=request.title,
                start_date=request.start_date,
                end_date=request.end_date,
                budget=request.budget,
                preferences=request.preferences
            )
            trip = itinerary_service.get_trip(trip_id)
            return TripResponse(**trip)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/trips/{trip_id}", response_model=TripResponse)
    async def get_trip(trip_id: int):
        try:
            trip = itinerary_service.get_trip(trip_id)
            if not trip:
                raise HTTPException(status_code=404, detail="Trip not found")
            return TripResponse(**trip)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/users/{user_id}/trips", response_model=List[TripResponse])
    async def get_user_trips(user_id: str):
        try:
            trips = itinerary_service.get_user_trips(user_id)
            return [TripResponse(**trip) for trip in trips]
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/users/{user_id}/current-trip", response_model=TripResponse)
    async def get_current_trip(user_id: str):
        try:
            trip = itinerary_service.get_current_trip(user_id)
            if not trip:
                raise HTTPException(status_code=404, detail="No trips found for user")
            return TripResponse(**trip)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.put("/trips/{trip_id}", response_model=TripResponse)
    async def update_trip(trip_id: int, request: UpdateTripRequest):
        try:
            update_data = {k: v for k, v in request.model_dump().items() if v is not None}
            if not update_data:
                raise HTTPException(status_code=400, detail="No fields to update")

            success = itinerary_service.update_trip(trip_id, **update_data)
            if not success:
                raise HTTPException(status_code=404, detail="Trip not found")

            trip = itinerary_service.get_trip(trip_id)
            return TripResponse(**trip)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.delete("/trips/{trip_id}")
    async def delete_trip(trip_id: int):
        try:
            success = itinerary_service.delete_trip(trip_id)
            if not success:
                raise HTTPException(status_code=404, detail="Trip not found")
            return {"message": "Trip deleted successfully"}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/days", response_model=DayResponse)
    async def add_day(request: AddDayRequest):
        try:
            day_id = itinerary_service.add_day(
                trip_id=request.trip_id,
                day_number=request.day_number,
                date=request.date,
                description=request.description
            )
            day = {
                "id": day_id,
                "trip_id": request.trip_id,
                "day_number": request.day_number,
                "date": request.date,
                "description": request.description
            }
            return DayResponse(**day)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/trips/{trip_id}/days", response_model=List[DayResponse])
    async def get_trip_days(trip_id: int):
        try:
            days = itinerary_service.get_trip_days(trip_id)
            return [DayResponse(**day) for day in days]
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.delete("/days/{day_id}")
    async def delete_day(day_id: int):
        try:
            success = itinerary_service.delete_day(day_id)
            if not success:
                raise HTTPException(status_code=404, detail="Day not found")
            return {"message": "Day deleted successfully"}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/items", response_model=ItemResponse)
    async def add_item(request: AddItemRequest):
        try:
            item_id = itinerary_service.add_item(
                day_id=request.day_id,
                name=request.name,
                activity_type=request.activity_type,
                description=request.description,
                location=request.location,
                time=request.time,
                booking_info=request.booking_info,
                order_index=request.order_index
            )
            item = {
                "id": item_id,
                "day_id": request.day_id,
                "activity_type": request.activity_type,
                "name": request.name,
                "description": request.description,
                "location": request.location,
                "time": request.time,
                "booking_info": request.booking_info,
                "order_index": request.order_index
            }
            return ItemResponse(**item)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/days/{day_id}/items", response_model=List[ItemResponse])
    async def get_day_items(day_id: int):
        try:
            items = itinerary_service.get_day_items(day_id)
            return [ItemResponse(**item) for item in items]
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.put("/items/{item_id}", response_model=ItemResponse)
    async def update_item(item_id: int, request: UpdateItemRequest):
        try:
            update_data = {k: v for k, v in request.model_dump().items() if v is not None}
            if not update_data:
                raise HTTPException(status_code=400, detail="No fields to update")

            success = itinerary_service.update_item(item_id, **update_data)
            if not success:
                raise HTTPException(status_code=404, detail="Item not found")

            conn = itinerary_service.db.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, day_id, activity_type, name, description, location, time, booking_info, order_index
                FROM itinerary_items WHERE id = ?
            """, (item_id,))
            row = cursor.fetchone()
            conn.close()

            if not row:
                raise HTTPException(status_code=404, detail="Item not found")

            item = {
                "id": row[0],
                "day_id": row[1],
                "activity_type": row[2],
                "name": row[3],
                "description": row[4],
                "location": row[5],
                "time": row[6],
                "booking_info": row[7],
                "order_index": row[8]
            }
            return ItemResponse(**item)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.delete("/items/{item_id}")
    async def delete_item(item_id: int):
        try:
            success = itinerary_service.delete_item(item_id)
            if not success:
                raise HTTPException(status_code=404, detail="Item not found")
            return {"message": "Item deleted successfully"}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/trips/{trip_id}/full")
    async def get_full_itinerary(trip_id: int):
        try:
            itinerary = itinerary_service.get_full_itinerary(trip_id)
            if not itinerary:
                raise HTTPException(status_code=404, detail="Trip not found")
            return itinerary
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/trips/{trip_id}/summary")
    async def get_itinerary_summary(trip_id: int):
        try:
            summary = itinerary_service.generate_itinerary_summary(trip_id)
            if not summary:
                raise HTTPException(status_code=404, detail="Trip not found")
            return {"trip_id": trip_id, "summary": summary}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/trips/{trip_id}/context")
    async def get_itinerary_context(trip_id: int):
        try:
            itinerary = itinerary_service.get_full_itinerary(trip_id)
            if not itinerary:
                raise HTTPException(status_code=404, detail="Trip not found")
            
            context_parts = []
            
            trip_title = itinerary.get("title", "Trip")
            num_days = len(itinerary.get("days", []))
            context_parts.append(f"User trip: {trip_title} ({num_days} days)")

            budget = itinerary.get("budget")
            preferences = itinerary.get("preferences")
            if budget or preferences:
                pref_str = ""
                if budget:
                    pref_str += f"Budget: {budget}"
                if preferences:
                    if isinstance(preferences, dict):
                        pref_list = [f"{k}: {v}" for k, v in preferences.items()]
                        pref_str += ", " + ", ".join(pref_list) if pref_str else ", ".join(pref_list)
                    elif isinstance(preferences, str):
                        pref_str += f", {preferences}" if pref_str else preferences
                context_parts.append(pref_str)

            for day in itinerary.get("days", []):
                day_num = day.get("day_number", 1)
                items = day.get("items", [])

                if items:
                    items_str = ", ".join([
                        f"{item['name']}" + (f" at {item['time']}" if item.get('time') else "")
                        for item in items[:4]
                    ])
                    context_parts.append(f"Day {day_num}: {items_str}")
                elif day.get("description"):
                    context_parts.append(f"Day {day_num}: {day['description']}")

            context = "\n".join(context_parts)
            
            if len(context) > 1200:
                context = context[:1197] + "..."

            return {"trip_id": trip_id, "context": context}
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return router