from __future__ import annotations
import logging
import aiohttp
from typing import List, Optional, Any
from langchain_core.tools import tool, StructuredTool
from langchain_core.messages import HumanMessage
from .itinerary_service import ItineraryService 
from .external_memory_service import ExternalMemoryService 
from app.core.logger import get_logger

logger = get_logger(__name__)

# --- Standalone Tool Definitions ---

@tool
async def get_weather(city: str) -> str:
    """
    Get the current weather for a specific city.
    Use this when the user asks about weather, temperature, or if it's raining in a specific location.
    """
    try:
        async with aiohttp.ClientSession() as session:
            # 1. Get coordinates
            geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
            async with session.get(geo_url) as resp:
                if resp.status != 200:
                    return f"Could not find location '{city}'."
                data = await resp.json()
                
                if not data.get("results"):
                    return f"Location '{city}' not found."
                
                lat = data["results"][0]["latitude"]
                lon = data["results"][0]["longitude"]
                location_name = f"{data['results'][0]['name']}, {data['results'][0].get('country', '')}"
            
            # 2. Get weather
            weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,weather_code&wind_speed_unit=ms"
            async with session.get(weather_url) as resp:
                if resp.status != 200:
                    return "Could not retrieve weather data."
                w_data = await resp.json()
                
                current = w_data.get("current", {})
                temp = current.get("temperature_2m", "N/A")
                code = current.get("weather_code", 0)
                
                # Simple code mapping (WMO Weather interpretation codes)
                condition = "Unknown"
                if code == 0: condition = "Clear sky"
                elif code in [1, 2, 3]: condition = "Mainly clear, partly cloudy, and overcast"
                elif code in [45, 48]: condition = "Fog"
                elif code in [51, 53, 55]: condition = "Drizzle"
                elif code in [61, 63, 65]: condition = "Rain"
                elif code in [71, 73, 75]: condition = "Snow fall"
                elif code in [80, 81, 82]: condition = "Rain showers"
                elif code in [95, 96, 99]: condition = "Thunderstorm"
                
                return f"Weather in {location_name}: {condition}, Temperature: {temp}°C"
    except Exception as e:
        logger.error(f"Error getting weather: {e}")
        return f"Error getting weather: {str(e)}"

def get_functional_tools(
    user_id: str,
    trip_id: Optional[int],
    itinerary_service: Optional[ItineraryService] = None,
    external_memory_service: Optional[ExternalMemoryService] = None,
) -> List[StructuredTool]:
    """
    Returns a list of functional tools (actions) bound to the specific user/trip context.
    Does NOT include retrieval tools (vector search) as those are handled by graph nodes.
    """
    
    @tool
    async def create_new_trip(title: str, start_date: str = "", end_date: str = "", budget: str = "") -> str:
        """
        Create a new trip for the user. Use this when the user wants to start planning a new journey.
        """
        if not itinerary_service:
            return "Itinerary service is not available."
            
        try:
            trip_id_new = itinerary_service.create_trip(
                user_id=user_id,
                title=title,
                start_date=start_date if start_date else None,
                end_date=end_date if end_date else None,
                budget=budget if budget else None
            )
            return f"Successfully created new trip '{title}' (ID: {trip_id_new}). You can now add items to it."
        except Exception as e:
            logger.error(f"Error creating trip: {e}")
            return f"Error creating trip: {str(e)}"

    @tool
    async def add_to_itinerary(name: str, type: str, description: str = "", location: str = "", time: str = "") -> str:
        """
        Add an item to the user's current trip itinerary.
        Use this when the user explicitly asks to add something or agrees to a suggestion.
        types: 'restaurant', 'attraction', 'activity', 'hotel', 'transport'.
        """
        if not itinerary_service:
            return "Itinerary service is not available."
            
        try:
            # Resolve trip
            current_trip = None
            if trip_id:
                current_trip = itinerary_service.get_trip(trip_id)
            else:
                current_trip = itinerary_service.get_current_trip(user_id)
            
            if not current_trip:
                return "No active trip found for user."

            trip_days = itinerary_service.get_trip_days(current_trip['id'])
            if not trip_days:
                day_id = itinerary_service.add_day(current_trip['id'], 1, None, "Generated day")
            else:
                day_id = trip_days[0]['id']

            itinerary_service.add_item(
                day_id=day_id,
                name=name,
                activity_type=type,
                description=description,
                location=location,
                time=time,
                order_index=0 # Will be appended
            )
            return f"Successfully added '{name}' to the itinerary."
        except Exception as e:
            logger.error(f"Error adding to itinerary: {e}")
            return f"Error adding to itinerary: {str(e)}"

    @tool
    async def clear_trip_data() -> str:
        """
        Clear all trip data for the user to start planning fresh.
        Use this when the user says "start over", "clear trip", "new plan", or wants to delete their current itinerary.
        """
        if not itinerary_service:
            return "Itinerary service is not available."
            
        try:
            # Resolve trip to delete
            trip_to_delete = None
            if trip_id:
                trip_to_delete = itinerary_service.get_trip(trip_id)
            else:
                trip_to_delete = itinerary_service.get_current_trip(user_id)
            
            if not trip_to_delete:
                return "No active trip found to clear."

            success = itinerary_service.delete_trip(trip_to_delete['id'])
            if success:
                return "Successfully cleared trip data. You can now start planning a new trip."
            else:
                return "Failed to clear trip data."
        except Exception as e:
            logger.error(f"Error clearing trip data: {e}")
            return f"Error clearing trip data: {str(e)}"

    @tool
    async def save_user_city(city: str) -> str:
        """
        Save the city where the user is currently located. 
        Use this when the user mentions their current valid city.
        """
        if not external_memory_service:
            return "External memory service is not available."
        
        try:
            if not city or city.lower() in ["unknown", "none", "null"]:
                return f"Invalid city value: {city}. Not saved."
            
            external_memory_service.update_user_city(user_id, city)
            return f"Successfully saved current city as {city}."
        except Exception as e:
            logger.error(f"Error saving user city: {e}")
            return f"Error saving user city: {str(e)}"

    @tool
    async def get_user_info_tool() -> str:
        """
        Retrieve the user's saved information, specifically their current city, preferences (likes/dislikes), amenities, and budget.
        Use this to personalize recommendations or when you need to know where the user is.
        """
        if not external_memory_service:
            return "External memory service is not available."
            
        try:
            info = external_memory_service.get_user_info(user_id)
            city = info.get("current_city") or "Unknown"
            preferences = info.get("preferences") or "None"
            language = info.get("language") or "Unknown"
            amenities = info.get("amenities") or []
            amenities_str = ", ".join(amenities) if amenities else "None"
            budget = info.get("budget") or "Not set"
            return f"Current City: {city}\nPreferences: {preferences}\nLanguage: {language}\nAmenities: {amenities_str}\nBudget: {budget}"
        except Exception as e:
            logger.error(f"Error getting user info: {e}")
            return f"Error getting user info: {str(e)}"

    @tool
    async def update_user_preferences(preference: str) -> str:
        """
        Save a new user preference (what they like or dislike). 
        Use this when the user explicitly states a preference (e.g., 'I love sushi', 'I hate hiking').
        """
        if not external_memory_service:
            return "External memory service is not available."
            
        try:
            external_memory_service.update_user_preferences(user_id, preference)
            return f"Successfully updated user preferences: {preference}"
        except Exception as e:
            logger.error(f"Error updating user preferences: {e}")
            return f"Error updating user preferences: {str(e)}"

    @tool
    async def save_user_language(language: str) -> str:
        """
        Save the detected language of the user.
        Use this when the language node determines the user's language.
        """
        if not external_memory_service:
            return "External memory service is not available."
        
        try:
            if not language or language.lower() in ["unknown", "none", "null"]:
                return f"Invalid language value: {language}. Not saved."

            external_memory_service.update_user_language(user_id, language)
            return f"Successfully saved user language: {language}"
        except Exception as e:
            logger.error(f"Error saving user language: {e}")
            return f"Error saving user language: {str(e)}"

    @tool
    async def save_user_amenities(amenities: List[str]) -> str:
        """
        Save the user's amenities preferences.
        Use this when amenities are extracted from an accommodation query to remember for future searches.
        """
        if not external_memory_service:
            return "External memory service is not available."
        
        try:
            external_memory_service.update_user_amenities(user_id, amenities)
            amenities_str = ", ".join(amenities) if amenities else "none"
            return f"Successfully saved user amenities: {amenities_str}"
        except Exception as e:
            logger.error(f"Error saving user amenities: {e}")
            return f"Error saving user amenities: {str(e)}"

    @tool
    async def save_user_budget(budget: str) -> str:
        """
        Save the user's budget preference for accommodations.
        Use this when budget/price range is extracted from an accommodation query.
        Budget should be in format "min-max" (e.g., "100-200").
        """
        if not external_memory_service:
            return "External memory service is not available."
        
        try:
            external_memory_service.update_user_budget(user_id, budget)
            return f"Successfully saved user budget: {budget}"
        except Exception as e:
            logger.error(f"Error saving user budget: {e}")
            return f"Error saving user budget: {str(e)}"

    return [
        get_weather,
        create_new_trip,
        add_to_itinerary,
        clear_trip_data,
        save_user_city,
        get_user_info_tool,
        update_user_preferences,
        save_user_language,
        save_user_amenities,
        save_user_budget
    ]
