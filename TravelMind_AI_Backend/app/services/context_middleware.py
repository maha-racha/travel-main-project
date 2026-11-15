import logging
from typing import Optional, Tuple
from .external_memory_service import ExternalMemoryService
from .itinerary_service import ItineraryService

from app.core.logger import get_logger

logger = get_logger(__name__)


class ContextMiddleware:
    """Middleware to inject memory and itinerary into every OpenAI request"""
    
    def __init__(self, external_memory_service: ExternalMemoryService, itinerary_service: ItineraryService):
        self.external_memory_service = external_memory_service
        self.itinerary_service = itinerary_service
    
    def get_context_summaries(self, user_id: str, trip_id: Optional[int] = None) -> Tuple[Optional[str], Optional[str]]:
        """Fetch both memory and itinerary summaries for injection into prompt"""
        
        memory_summary = None
        itinerary_summary = None
        
        # Fetch memory summary
        try:
            memory_summary = self.external_memory_service.get_memory_summary(user_id)
            if memory_summary:
                logger.info(f"Retrieved memory summary for user {user_id}: {len(memory_summary)} chars")
        except Exception as e:
            logger.error(f"Error fetching memory summary: {str(e)}")
        
        # Fetch itinerary summary  
        try:
            current_trip = None
            if trip_id:
                current_trip = self.itinerary_service.get_trip(trip_id)
            else:
                current_trip = self.itinerary_service.get_current_trip(user_id)
            
            if current_trip:
                itinerary_summary = self._build_itinerary_context(current_trip['id'])
                if itinerary_summary:
                    logger.info(f"Retrieved itinerary summary for user {user_id}: {len(itinerary_summary)} chars")
        except Exception as e:
            logger.error(f"Error fetching itinerary summary: {str(e)}")
        
        return memory_summary, itinerary_summary
    
    def _build_itinerary_context(self, trip_id: int) -> Optional[str]:
        """Build compact itinerary context (100-300 tokens)"""
        
        try:
            itinerary = self.itinerary_service.get_full_itinerary(trip_id)
            if not itinerary:
                return None
            
            context_parts = []
            
            # Trip header
            trip_title = itinerary.get("title", "Trip")
            num_days = len(itinerary.get("days", []))
            context_parts.append(f"User trip: {trip_title} ({num_days} days)")
            
            # Budget and preferences
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
            
            # Daily activities
            for day in itinerary.get("days", []):
                day_num = day.get("day_number", 1)
                items = day.get("items", [])
                
                if items:
                    items_str = ", ".join([
                        f"{item['name']}" + (f" at {item['time']}" if item.get('time') else "")
                        for item in items[:4]  # Limit to 4 items per day for token efficiency
                    ])
                    context_parts.append(f"Day {day_num}: {items_str}")
                elif day.get("description"):
                    context_parts.append(f"Day {day_num}: {day['description']}")
            
            context = "\n".join(context_parts)
            
            # Token limit: ~300 tokens = ~1200 chars
            if len(context) > 1200:
                context = context[:1197] + "..."
            
            return context
            
        except Exception as e:
            logger.error(f"Error building itinerary context: {str(e)}")
            return None
    
    def ensure_context_injection(self, user_id: str, trip_id: Optional[int] = None) -> Tuple[Optional[str], Optional[str]]:
        """MANDATORY: Ensure every request has memory and itinerary context"""
        
        memory_summary, itinerary_summary = self.get_context_summaries(user_id, trip_id)
        
        # Log context injection for monitoring
        logger.info(f"Context injection for user {user_id}: Memory={'Yes' if memory_summary else 'No'}, Itinerary={'Yes' if itinerary_summary else 'No'}")
        
        return memory_summary, itinerary_summary