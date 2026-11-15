import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from app.models.models import Trip, ItineraryDay, ItineraryItem
from .database_manager import DatabaseManager

from app.core.logger import get_logger

logger = get_logger(__name__)


class ItineraryService:
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def create_trip(self, user_id: str, title: str, start_date: Optional[str] = None,
                    end_date: Optional[str] = None, budget: Optional[str] = None,
                    preferences: Optional[Dict[str, Any]] = None) -> int:
        
        session = self.db.get_session()
        try:
            trip = Trip(
                user_id=user_id,
                title=title,
                start_date=start_date,
                end_date=end_date,
                budget=budget,
                preferences=json.dumps(preferences) if preferences else None
            )
            session.add(trip)
            session.commit()
            session.refresh(trip)  # Get the generated ID
            
            logger.info(f"Created trip {trip.id} for user {user_id}")
            return trip.id
        finally:
            session.close()

    def get_trip(self, trip_id: int) -> Optional[Dict[str, Any]]:
        
        session = self.db.get_session()
        try:
            trip = session.query(Trip).filter_by(id=trip_id).first()
            
            if not trip:
                return None
            
            return {
                "id": trip.id,
                "user_id": trip.user_id,
                "title": trip.title,
                "start_date": trip.start_date,
                "end_date": trip.end_date,
                "budget": trip.budget,
                "preferences": json.loads(trip.preferences) if trip.preferences else None,
                "created_at": trip.created_at.isoformat() if trip.created_at else None
            }
        finally:
            session.close()

    def get_user_trips(self, user_id: str) -> List[Dict[str, Any]]:
        
        session = self.db.get_session()
        try:
            trips_query = session.query(Trip).filter_by(
                user_id=user_id
            ).order_by(Trip.created_at.desc()).all()
            
            trips = []
            for trip in trips_query:
                trips.append({
                    "id": trip.id,
                    "user_id": trip.user_id,
                    "title": trip.title,
                    "start_date": trip.start_date,
                    "end_date": trip.end_date,
                    "budget": trip.budget,
                    "preferences": json.loads(trip.preferences) if trip.preferences else None,
                    "created_at": trip.created_at.isoformat() if trip.created_at else None
                })
            
            return trips
        finally:
            session.close()

    def get_current_trip(self, user_id: str) -> Optional[Dict[str, Any]]:
        trips = self.get_user_trips(user_id)
        return trips[0] if trips else None

    def update_trip(self, trip_id: int, **kwargs) -> bool:
        
        allowed_fields = ["title", "start_date", "end_date", "budget", "preferences"]
        
        session = self.db.get_session()
        try:
            trip = session.query(Trip).filter_by(id=trip_id).first()
            
            if not trip:
                return False
            
            updated = False
            for field, value in kwargs.items():
                if field in allowed_fields:
                    if field == "preferences" and isinstance(value, dict):
                        value = json.dumps(value)
                    setattr(trip, field, value)
                    updated = True
            
            if updated:
                trip.updated_at = datetime.utcnow()
                session.commit()
            
            return updated
        finally:
            session.close()

    def delete_trip(self, trip_id: int) -> bool:
        
        session = self.db.get_session()
        try:
            trip = session.query(Trip).filter_by(id=trip_id).first()
            
            if not trip:
                return False
            
            session.delete(trip)
            session.commit()
            return True
        finally:
            session.close()

    def add_day(self, trip_id: int, day_number: int, date: Optional[str] = None,
                description: Optional[str] = None) -> int:
        
        session = self.db.get_session()
        try:
            day = ItineraryDay(
                trip_id=trip_id,
                day_number=day_number,
                date=date,
                description=description
            )
            session.add(day)
            session.commit()
            session.refresh(day)
            
            return day.id
        finally:
            session.close()

    def get_trip_days(self, trip_id: int) -> List[Dict[str, Any]]:
        
        session = self.db.get_session()
        try:
            days_query = session.query(ItineraryDay).filter_by(
                trip_id=trip_id
            ).order_by(ItineraryDay.day_number).all()
            
            days = []
            for day in days_query:
                days.append({
                    "id": day.id,
                    "trip_id": day.trip_id,
                    "day_number": day.day_number,
                    "date": day.date,
                    "description": day.description
                })
            
            return days
        finally:
            session.close()

    def add_item(self, day_id: int, name: str, activity_type: Optional[str] = None,
                 description: Optional[str] = None, location: Optional[str] = None,
                 time: Optional[str] = None, booking_info: Optional[str] = None,
                 order_index: int = 0) -> int:
        
        session = self.db.get_session()
        try:
            item = ItineraryItem(
                day_id=day_id,
                name=name,
                activity_type=activity_type,
                description=description,
                location=location,
                time=time,
                booking_info=booking_info,
                order_index=order_index
            )
            session.add(item)
            session.commit()
            session.refresh(item)
            
            return item.id
        finally:
            session.close()

    def get_day_items(self, day_id: int) -> List[Dict[str, Any]]:
        
        session = self.db.get_session()
        try:
            items_query = session.query(ItineraryItem).filter_by(
                day_id=day_id
            ).order_by(ItineraryItem.order_index, ItineraryItem.time).all()
            
            items = []
            for item in items_query:
                items.append({
                    "id": item.id,
                    "day_id": item.day_id,
                    "activity_type": item.activity_type,
                    "name": item.name,
                    "description": item.description,
                    "location": item.location,
                    "time": item.time,
                    "booking_info": item.booking_info,
                    "order_index": item.order_index
                })
            
            return items
        finally:
            session.close()

    def get_full_itinerary(self, trip_id: int) -> Dict[str, Any]:
        session = self.db.get_session()
        try:
            # Get trip
            trip = session.query(Trip).filter_by(id=trip_id).first()
            if not trip:
                return None
            
            trip_dict = {
                "id": trip.id,
                "user_id": trip.user_id,
                "title": trip.title,
                "start_date": trip.start_date,
                "end_date": trip.end_date,
                "budget": trip.budget,
                "preferences": json.loads(trip.preferences) if trip.preferences else None,
                "created_at": trip.created_at.isoformat() if trip.created_at else None
            }
            
            # Get days
            days_query = session.query(ItineraryDay).filter_by(
                trip_id=trip_id
            ).order_by(ItineraryDay.day_number).all()
            
            days = []
            for day in days_query:
                day_dict = {
                    "id": day.id,
                    "trip_id": day.trip_id,
                    "day_number": day.day_number,
                    "date": day.date,
                    "description": day.description
                }
                
                # Get items for this day
                items_query = session.query(ItineraryItem).filter_by(
                    day_id=day.id
                ).order_by(ItineraryItem.order_index, ItineraryItem.time).all()
                
                items = []
                for item in items_query:
                    items.append({
                        "id": item.id,
                        "day_id": item.day_id,
                        "activity_type": item.activity_type,
                        "name": item.name,
                        "description": item.description,
                        "location": item.location,
                        "time": item.time,
                        "booking_info": item.booking_info,
                        "order_index": item.order_index
                    })
                
                day_dict["items"] = items
                days.append(day_dict)
            
            trip_dict["days"] = days
            return trip_dict
        finally:
            session.close()

    def generate_itinerary_summary(self, trip_id: int) -> str:
        itinerary = self.get_full_itinerary(trip_id)
        if not itinerary:
            return ""

        summary_parts = []

        trip_title = itinerary.get("title", "Trip")
        num_days = len(itinerary.get("days", []))
        summary_parts.append(f"User trip: {trip_title} ({num_days} days)")

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
            summary_parts.append(pref_str)

        for day in itinerary.get("days", []):
            day_num = day.get("day_number", 1)
            items = day.get("items", [])

            if items:
                items_str = ", ".join([
                    f"{item['name']}" + (f" at {item['time']}" if item.get('time') else "")
                    for item in items[:3]
                ])
                summary_parts.append(f"Day {day_num}: {items_str}")
            elif day.get("description"):
                summary_parts.append(f"Day {day_num}: {day['description']}")

        summary = "\n".join(summary_parts)

        if len(summary) > 800:
            summary = summary[:797] + "..."

        return summary

    def update_item(self, item_id: int, **kwargs) -> bool:
        
        allowed_fields = ["activity_type", "name", "description", "location", "time", "booking_info", "order_index"]
        
        session = self.db.get_session()
        try:
            item = session.query(ItineraryItem).filter_by(id=item_id).first()
            
            if not item:
                return False
            
            updated = False
            for field, value in kwargs.items():
                if field in allowed_fields:
                    setattr(item, field, value)
                    updated = True
            
            if updated:
                session.commit()
            
            return updated
        finally:
            session.close()

    def delete_item(self, item_id: int) -> bool:
        
        session = self.db.get_session()
        try:
            item = session.query(ItineraryItem).filter_by(id=item_id).first()
            
            if not item:
                return False
            
            session.delete(item)
            session.commit()
            return True
        finally:
            session.close()

    def delete_day(self, day_id: int) -> bool:
        
        session = self.db.get_session()
        try:
            day = session.query(ItineraryDay).filter_by(id=day_id).first()
            
            if not day:
                return False
            
            session.delete(day)
            session.commit()
            return True
        finally:
            session.close()