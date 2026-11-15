import logging
import openai
from typing import Dict, Any, Optional
from app.core.config import Settings

logger = logging.getLogger(__name__)


class WebSearchService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = openai.OpenAI(api_key=settings.openai_api_key)

    def search_and_respond(self, query: str, context: str = None, max_tokens: Optional[int] = None) -> Dict[str, Any]:
        try:
            if context:
                enhanced_query = f"""You are a TOURIST GUIDE. Using local information, answer this travel question:

Context: {context}

Question: {query}

CRITICAL RULES:
- Maximum 2-3 sentences only
- Give KEY info first (name, location)
- End with ONE short follow-up question
- ONLY tourism answers: restaurants, hotels, attractions, transport
- NO cooking tips or recipes - redirect to restaurants instead
- Sound like a friend texting"""
            else:
                enhanced_query = f"""You are a TOURIST GUIDE. Answer this travel question briefly:

{query}

CRITICAL RULES:
- Maximum 2-3 sentences only
- Give KEY info first
- End with ONE short follow-up question
- ONLY tourism answers: restaurants, hotels, attractions, transport
- NO cooking tips - redirect to restaurants instead
- Sound like a friend texting"""

            effective_max_tokens = max_tokens or self.settings.openai_max_tokens

            response = self.client.chat.completions.create(
                model=self.settings.openai_chat_model,
                messages=[
                    {"role": "system", "content": "You are a knowledgeable travel assistant with access to current information."},
                    {"role": "user", "content": enhanced_query}
                ],
                temperature=self.settings.openai_temperature,
                max_tokens=effective_max_tokens
            )

            output_text = response.choices[0].message.content or ""
            
            usage = response.usage

            logger.info(
                  f"TOKENS | total={usage.total_tokens} | "
                  f"prompt={usage.prompt_tokens} | "
                  f"completion={usage.completion_tokens} | "
                  f"cached={usage.prompt_tokens_details.cached_tokens if usage.prompt_tokens_details else 0}"
            )

            logger.info(f"Web search for '{query}' completed")

            return {
                "text": output_text,
                "citations": [],
                "success": True
            }

        except Exception as e:
            logger.error(f"Error performing web search: {str(e)}", exc_info=True)
            return {
                "text": "",
                "citations": [],
                "success": False,
                "error": str(e)
            }