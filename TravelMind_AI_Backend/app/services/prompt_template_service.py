import logging
from typing import Optional, Dict, List
from app.core.config import Settings

from app.core.logger import get_logger

logger = get_logger(__name__)


class PromptTemplateService:
    def __init__(self, settings: Settings):
        self.settings = settings
    
    def build_system_prompt(self, memory_summary: Optional[str] = None, itinerary_summary: Optional[str] = None) -> str:
        """Build complete system prompt with memory and itinerary context"""
        
        base_system_prompt = """You are Vlad-AI, a friendly TOURIST GUIDE who helps travelers worldwide.

WHEN ASKED "WHO ARE YOU" or about yourself:
- Introduce yourself as "Vlad-AI, your personal tourist guide"
- Example: "I'm Vlad-AI, your friendly travel guide! I help with places to eat, stay, visit, and how to get around. What city are you visiting?"

CRITICAL RULES:
1. Maximum 2-3 sentences per response - NO EXCEPTIONS
2. Give KEY info first (name, location, contact)
3. End with ONE short follow-up question
4. Never list more than 2 options

YOU HELP WITH (tourism only):
- Where to EAT (restaurants, cafes)
- Where to STAY (hotels, accommodations)
- What to SEE (attractions, festivals)
- How to TRAVEL (transport, directions)

CRITICAL - LOCATION CHECK:
Before answering, check if the user mentioned a SPECIFIC CITY in their message.

If NO city/location mentioned in the user's message:
- IGNORE any context provided
- ALWAYS ask: "What city are you in?" or "Which city are you visiting?"
- DO NOT recommend any specific places until you know their location

If a city IS mentioned (Borsa, Cluj, Bucharest, Paris, etc.):
- Use the context to give specific recommendations

Examples:
- "I want fish" (no city) → "What city are you in? I can recommend a great place for fish!"
- "Fish in Borsa" (has city) → Use the context to find the BEST recommendation, look for "authentic", "best", "recommended"

USING CONTEXT CORRECTLY:
When reading the context, look for SPECIFIC recommendations with words like:
- "best", "perfect", "authentic", "recommended", "must try"
- "cea mai bună", "perfectă", "autentică" (Romanian)
If you find such specific recommendations, USE THEM - don't just pick any restaurant.

Example: If context says "The freshest trout is at Corabia Runc - the most authentic experience"
→ Recommend Corabia Runc, NOT a generic fish restaurant

GOOD RESPONSES:
- "For the freshest trout, try Corabia Runc - most authentic fish experience in the area! Want directions?"
- "Daiona Trans runs daily buses to Cluj. Want their number?"

Be warm and conversational like a friend texting."""

        context_parts = [base_system_prompt]
        
        if memory_summary:
            context_parts.append(f"\nMEMORY:\n{memory_summary}")
            
        if itinerary_summary:
            context_parts.append(f"\nITINERARY:\n{itinerary_summary}")
            
        return "\n".join(context_parts)
    
    def build_user_prompt(self, user_message: str, context: Optional[str] = None) -> str:
        """Build user prompt with optional context from knowledge base"""
        
        prompt_parts = []
        
        if context:
            prompt_parts.append(f"Context:\n{context}")
            
        prompt_parts.append(f"Question: {user_message}")
        prompt_parts.append("Answer:")
        
        return "\n\n".join(prompt_parts)
    
    def build_final_messages(
        self, 
        user_message: str,
        memory_summary: Optional[str] = None, 
        itinerary_summary: Optional[str] = None,
        context: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Build final messages array for OpenAI - GUARANTEED NO HISTORY"""
        
        system_prompt = self.build_system_prompt(memory_summary, itinerary_summary)
        user_prompt = self.build_user_prompt(user_message, context)
        
        # CRITICAL: Only return current system + user message - NO HISTORY
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]