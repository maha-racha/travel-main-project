import aiohttp
import logging
from typing import List, Dict, Any
from app.core.config import Settings

from app.core.logger import get_logger

logger = get_logger(__name__)


class DataFetcher:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_url = settings.vlad_ai_base_url
        self.api_key = settings.vlad_ai_api_key
        self.endpoints = [
            "accommodations",
            "restaurants",
            "attractions",
            "helpful-things",
            "festivals"
        ]

    async def fetch_endpoint(self, session: aiohttp.ClientSession, endpoint: str) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/{endpoint}"
        headers = {"X-API-Key": self.api_key}

        try:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Successfully fetched {endpoint}: {len(data) if isinstance(data, list) else 1} items")
                    return data if isinstance(data, list) else [data]
                else:
                    logger.error(f"Failed to fetch {endpoint}: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error fetching {endpoint}: {str(e)}")
            return []

    async def fetch_all_data(self) -> Dict[str, List[Dict[str, Any]]]:
        async with aiohttp.ClientSession() as session:
            results = {}
            for endpoint in self.endpoints:
                data = await self.fetch_endpoint(session, endpoint)
                results[endpoint] = data
            return results

    def _extract_semantic_sections(self, description: str) -> Dict[str, str]:
        """
        Extract semantic sections from accommodation descriptions.
        Splits description into: Location, Price & Rental, Rooms, Amenities.
        Returns dict with section_name: section_text
        """
        if not description or len(description.strip()) < 50:
            return {}
        
        # Section detection patterns (case-insensitive)
        section_patterns = {
            "location": [
                "location and access", "area and access", "location:", 
                "situated in", "located in", "access to the property",
                "conveniently located", "strategically located", "benefiting from"
            ],
            "price": [
                "price and rental", "rates:", "rental type:", 
                "prices are", "budget", "accommodation price",
                "price offer", "nightly rate", "to receive an exact"
            ],
            "rooms": [
                "rooms and comfort", "bedrooms", "accommodation capacity",
                "the unit offers", "room types:", "capacity of up to",
                "each room", "matrimonial room", "double room", "triple room"
            ],
            "amenities": [
                "key facilities", "key amenities", "amenities include",
                "features include", "facilities are designed", 
                "outdoor area", "indoor", "guests benefit from",
                "the guesthouse provides", "at the destination"
            ]
        }
        
        # Lowercase description for pattern matching
        desc_lower = description.lower()
        
        # Find section boundaries
        section_starts = {}
        
        for section_name, patterns in section_patterns.items():
            for pattern in patterns:
                idx = desc_lower.find(pattern)
                if idx != -1:
                    # Find the start of the sentence (backtrack to previous period or start)
                    sentence_start = idx
                    while sentence_start > 0 and description[sentence_start - 1] not in '.!?\n':
                        sentence_start -= 1
                    
                    # Skip whitespace
                    while sentence_start < len(description) and description[sentence_start] in ' \n':
                        sentence_start += 1
                    
                    section_starts[section_name] = sentence_start
                    break  # Found a pattern for this section, move to next
        
        # If no sections detected, return empty
        if not section_starts:
            logger.debug("No semantic sections detected in description")
            return {}
        
        # Sort sections by position in text
        sorted_sections = sorted(section_starts.items(), key=lambda x: x[1])
        
        # Extract section text based on boundaries
        sections = {}
        for i, (section_name, start_pos) in enumerate(sorted_sections):
            # End position is either start of next section or end of description
            if i + 1 < len(sorted_sections):
                end_pos = sorted_sections[i + 1][1]
            else:
                end_pos = len(description)
            
            section_text = description[start_pos:end_pos].strip()
            
            # Only include substantial sections (at least 10 words)
            if len(section_text.split()) >= 10:
                sections[section_name] = section_text
        
        logger.debug(f"Extracted {len(sections)} semantic sections: {list(sections.keys())}")
        return sections

    def construct_chunks(self, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Construct structured chunks for the item.
        Implements 'Conditional Exploding' for rich nested lists,
        otherwise aggregates content into a main context-rich chunk.
        """
        chunks = []
        
        # --- 1. Main Chunk Construction ---
        parts = []
        name = item.get("name") or item.get("title", "Unknown")
        city_obj = item.get("city", {})
        city_name = city_obj.get("name", "") if isinstance(city_obj, dict) else ""
        category_obj = item.get("category", {})
        category_name = category_obj.get("name", "") if isinstance(category_obj, dict) else ""
        category_type = category_obj.get("type", "") if isinstance(category_obj, dict) else ""
        
        # Context Header (Template based)
        context_header = f"{name}"
        if category_name:
            context_header += f", a {category_name}"
        if city_name:
            context_header += f" located in {city_name}"
        context_header += "."
        
        parts.append(context_header)
        
        # Description
        description = item.get("description", "")
        if description:
            parts.append(description)
            
        # Address & Contact
        address = item.get("address", "")
        if address:
            parts.append(f"Address: {address}")
        
        # Flatten simple lists (Amenities, Features, etc.)
        # Accommodations
        if "amenities" in item and isinstance(item["amenities"], list):
            amenities = [str(x) for x in item["amenities"] if x]
            if amenities:
                parts.append(f"Amenities: {', '.join(amenities)}")
                
        # Restaurants
        if "cuisines" in item and isinstance(item["cuisines"], list):
            cuisines = [str(x) for x in item["cuisines"] if x]
            if cuisines:
                parts.append(f"Cuisine: {', '.join(cuisines)}")
        
        if "dietaryOptions" in item and isinstance(item["dietaryOptions"], list):
            diet = [str(x) for x in item["dietaryOptions"] if x]
            if diet:
                parts.append(f"Dietary Options: {', '.join(diet)}")
                
        # Attractions
        if "travelStyles" in item and isinstance(item["travelStyles"], list):
            styles = [str(x) for x in item["travelStyles"] if x]
            if styles:
                parts.append(f"Ideal for: {', '.join(styles)}")
                
        main_text = " ".join(parts)
        
        # --- SEMANTIC CHUNKING ---
        # Currently optimized for Accommodations, but extensible
        if category_type == "accommodations":
            # Extract sections
            sections = self._extract_semantic_sections(description)
            
            # Add section chunks
            for section_name, section_text in sections.items():
                chunks.append({
                    "chunk_id": section_name,
                    "text": f"{context_header}\n\n{section_text}",
                    "metadata": {
                        "type": "section",
                        "section_type": section_name,
                        "parent_id": item.get("id"),
                        "chunk_type": section_name 
                    }
                })
            
        # --- FULL CHUNK (Standard for ALL categories) ---
        # This acts as fallback for accommodations and primary chunk for others
        chunks.append({
            "chunk_id": "full",
            "text": main_text,
            "metadata": {
                "type": "full", 
                "chunk_type": "full",
                "parent_id": item.get("id")
            }
        })
        
        # --- 2. Conditional Exploding (Sub-chunks) ---
        # Scan for lists of rich objects (e.g., nested sub-items like 'rooms' or specific 'reviews' if they existed)
        # Based on data analysis, current structure is mostly flat. 
        # But we implement generic logic for future proofing or if specific fields appear.
        
        # Example: if 'extra_details' was a list of dicts with 'description'
        potential_nested_fields = ["sub_items", "specific_variations"] # Add actual fields if discovered
        
        for field in potential_nested_fields:
            if field in item and isinstance(item[field], list):
                for idx, sub_item in enumerate(item[field]):
                    if isinstance(sub_item, dict) and ("description" in sub_item or "text" in sub_item):
                        sub_text = sub_item.get("description") or sub_item.get("text") or ""
                        if sub_text:
                            # Prepend context!
                            full_sub_text = f"{context_header} [{field}]: {sub_text}"
                            chunks.append({
                                "chunk_id": f"{field}_{idx}",
                                "text": full_sub_text,
                                "metadata": {"type": "sub_item", "parent_field": field}
                            })

        return chunks

    def construct_searchable_text(self, item: Dict[str, Any]) -> str:
        """
        Legacy method kept for compatibility, now wraps construct_chunks.
        Returns the text of the main chunk.
        """
        chunks = self.construct_chunks(item)
        for chunk in chunks:
            if chunk["chunk_id"] == "main":
                return chunk["text"]
        return ""
    
    def normalize_item(self, item: Dict[str, Any], category: str) -> Dict[str, Any]:
        """
        Normalize item structure and types for Qdrant payload.
        Ensures strict schema conformity (Integers, Booleans, etc.).
        """
        # Common fields normalization
        normalized = {
            "success": True, # Payload doesn't strictly need this wrapper but we follow schema
            "message": "Normalized data", 
            "timestamp": item.get("timestamp") or "", 
             # Flattening pure data for the payload itself might be better for filtering, 
             # but user schema asked for "data" wrapper? 
             # Actually user schema showed top level structure. 
             # For Qdrant payload, we usually flatten. 
             # LET'S FLATTEN for Qdrant payload efficiency, but keep schema keys.
             # Wait, the user prompt said: "we these 5 main categories with this structure... [ { ... data: { ... } } ]"
             # But for Qdrant PAYLOAD, we need direct access to fields for filtering like `category.type`.
             # So I will return the FLATTENED object that corresponds to the inner item, 
             # but formatted perfectly.
        }
        
        # Helper to parse price
        def parse_price(p):
            if not p: return None
            try:
                # Extract numbers if string "100 RON"
                import re
                nums = re.findall(r"[-+]?\d*\.\d+|\d+", str(p))
                return float(nums[0]) if nums else None
            except: 
                return None

        # Helper to parse time to minutes
        def parse_time_minutes(t_str):
            if not t_str: return None
            try:
                # Format HH:MM or HH:MM:SS
                parts_t = str(t_str).split(":")
                h = int(parts_t[0])
                m = int(parts_t[1]) if len(parts_t) > 1 else 0
                return h * 60 + m
            except:
                return None

        # Helper to parse date to timestamp
        def parse_date_ts(d_str):
             if not d_str: return None
             try:
                 from datetime import datetime
                 # Handle various ISO formats if needed, assuming YYYY-MM-DD
                 dt = datetime.fromisoformat(str(d_str).replace("Z", "+00:00"))
                 return int(dt.timestamp())
             except:
                 return None

        # Base object - copy known fields with safe defaults
        base = {
            "name": str(item.get("name") or item.get("title") or ""), # 'title' for helpful-things
            "description": str(item.get("description") or ""),
            "address": str(item.get("address") or ""),
            "website": str(item.get("website")) if item.get("website") else None,
            "phone": str(item.get("phone") or ""),
            "isActive": bool(item.get("isActive", True)),
        }

        # City normalization
        city_raw = item.get("city")
        if isinstance(city_raw, dict):
            base["city"] = {"name": str(city_raw.get("name", "")), "country": str(city_raw.get("country", ""))}
        else:
            base["city"] = {"name": "", "country": ""}

        # Category normalization
        cat_raw = item.get("category")
        if isinstance(cat_raw, dict):
             base["category"] = {"name": str(cat_raw.get("name", "")), "type": category}
        else:
             # Fallback if category invalid
             base["category"] = {"name": category.capitalize(), "type": category}


        # ID (preserve it)
        base["id"] = item.get("id") or item.get("_id")

        # Specific Logic
        if category == "accommodations":
            base.update({
                "priceLevel": str(item.get("priceLevel") or ""),
                "pricePerNight": str(item.get("pricePerNight")) if item.get("pricePerNight") else None,
                "pricePerNight_val": parse_price(item.get("pricePerNight")), # INDEXED
                "amenities": [str(x) for x in item.get("amenities", []) if x],
                "allowsFamilies": bool(item.get("allowsFamilies", False)),
                "maxGuests": int(item.get("maxGuests") or 0),
                "minimumNightsStay": int(item.get("minimumNightsStay") or 1),
                "isPetFriendly": bool(item.get("isPetFriendly", False)),
                "isWheelchairAccessible": bool(item.get("isWheelchairAccessible", False)),
            })
        
        elif category == "restaurants":
            base.update({
                "priceLevel": str(item.get("priceLevel") or ""),
                "features": [str(x) for x in item.get("features", []) if x],
                "dietaryOptions": [str(x) for x in item.get("dietaryOptions", []) if x],
                "mealsServed": [str(x) for x in item.get("mealsServed", []) if x],
                "cuisines": [str(x) for x in item.get("cuisines", []) if x],
                "allowsFamilies": bool(item.get("allowsFamilies", False)),
                "maxGuests": int(item.get("maxGuests") or 0),
                "isPetFriendly": bool(item.get("isPetFriendly", False)),
                "isWheelchairAccessible": bool(item.get("isWheelchairAccessible", False)),
                "opensAt": str(item.get("opensAt")) if item.get("opensAt") else None,
                "closesAt": str(item.get("closesAt")) if item.get("closesAt") else None,
                "opensAt_minutes": parse_time_minutes(item.get("opensAt")), # INDEXED
                "closesAt_minutes": parse_time_minutes(item.get("closesAt")) # INDEXED
            })

        elif category == "attractions":
            base.update({
                "isPetFriendly": bool(item.get("isPetFriendly", False)),
                "isWheelchairAccessible": bool(item.get("isWheelchairAccessible", False)),
                "travelStyles": [str(x) for x in item.get("travelStyles", []) if x],
                "opensAt": str(item.get("opensAt")) if item.get("opensAt") else None,
                "closesAt": str(item.get("closesAt")) if item.get("closesAt") else None,
                "opensAt_minutes": parse_time_minutes(item.get("opensAt")), # INDEXED
                "closesAt_minutes": parse_time_minutes(item.get("closesAt")), # INDEXED
                "reopens": bool(item.get("reopens", False)),
                "requiresReservation": bool(item.get("requiresReservation", False))
            })

        elif category == "festivals":
            base.update({
                "startDate": str(item.get("startDate", "")),
                "endDate": str(item.get("endDate", "")),
                "startDate_ts": parse_date_ts(item.get("startDate")), # INDEXED
                "endDate_ts": parse_date_ts(item.get("endDate")), # INDEXED
                "startTime": str(item.get("startTime")) if item.get("startTime") else None,
                "endTime": str(item.get("endTime")) if item.get("endTime") else None,
                "startTime_minutes": parse_time_minutes(item.get("startTime")),
                "ticketRequired": bool(item.get("ticketRequired", False)),
            })
            
        elif category == "helpful-things":
            base.update({
                "title": str(item.get("title") or item.get("name") or ""),
                "contactInfo": str(item.get("contactInfo") or ""),
                "images": [str(x) for x in item.get("images", []) if x],
                "externalLinks": [str(x) for x in item.get("externalLinks", []) if x],
            })
            # Remove 'name' redundancy if title exists
            if "name" in base and base["name"] == base["title"]:
                 pass # keep for consistency

        return base

    # keep method for backward compatibility if needed, or alias it
    def format_document_content(self, item: Dict[str, Any], source: str) -> str:
        return self.construct_searchable_text(item)
