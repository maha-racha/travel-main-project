import openai
import logging
import re
from typing import Literal
from app.core.config import Settings

from app.core.logger import get_logger

logger = get_logger(__name__)


class IntentClassifier:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = openai.OpenAI(api_key=settings.openai_api_key)

        # Use regex patterns with word boundaries to avoid false matches like "know" matching "now"
        self.web_search_patterns = [
            r'\bweather\b', r'\btemperature\b', r'\bforecast\b', r'\btoday\b', r'\btomorrow\b',
            r'\bright now\b', r'\bcurrent\b', r'\blatest news\b', r'\bhappening now\b',
            r'\bevent today\b', r'\bevent tomorrow\b', r'\blive\b', r'\bcurrently\b', r'\bat the moment\b'
        ]

        self.hybrid_keywords = [
            'how to get', 'how can i get', 'how do i get', 'how to arrive',
            'how to reach', 'direction', 'route', 'way to', 'path to',
            'transport to', 'travel to', 'go to', 'getting to'
        ]

        self.knowledge_base_keywords = [
            'where can i', 'where to', 'best', 'recommend', 'suggest', 'suggestion',
            'restaurant', 'hotel', 'accommodation', 'eat', 'stay',
            'visit', 'see', 'attraction', 'place', 'festival', 'event',
            'good pizza', 'good fish', 'good food', 'nice', 'top',
            # Month names - for festival/event queries
            'january', 'february', 'march', 'april', 'may', 'june',
            'july', 'august', 'september', 'october', 'november', 'december'
        ]

        self.out_of_scope_keywords = [
            'code', 'programming', 'python', 'math', 'calculate',
            'formula', 'equation', 'algorithm', 'function'
            # Note: Cooking queries are handled by the system prompt which redirects to restaurants
        ]

    async def classify_intent(self, query: str) -> Literal["knowledge_base", "web_search", "out_of_scope", "hybrid"]:
        query_lower = query.lower().strip()

        logger.info(f"Classifying query: {query}")

        if any(keyword in query_lower for keyword in self.out_of_scope_keywords):
            logger.info(f"Rule-based: out_of_scope (found non-tourism keyword)")
            return "out_of_scope"

        has_web_keywords = any(re.search(pattern, query_lower) for pattern in self.web_search_patterns)
        has_hybrid_keywords = any(keyword in query_lower for keyword in self.hybrid_keywords)
        has_kb_keywords = any(keyword in query_lower for keyword in self.knowledge_base_keywords)

        logger.info(f"Keyword analysis - web: {has_web_keywords}, hybrid: {has_hybrid_keywords}, kb: {has_kb_keywords}")

        if has_hybrid_keywords:
            logger.info(f"Rule-based: hybrid (found direction/transport keywords)")
            return "hybrid"

        # Prioritize knowledge_base when both match - local data is more reliable
        if has_kb_keywords:
            logger.info(f"Rule-based: knowledge_base (found location/recommendation keywords)")
            return "knowledge_base"

        if has_web_keywords:
            logger.info(f"Rule-based: web_search (found real-time keywords)")
            return "web_search"

        tourism_patterns = [
            r'\b(restaurant|hotel|accommodation|attraction|place|eat|stay|visit)\b',
            r'\b(where|what|which)\b.*\b(in|at|near)\b',
            r'\b(best|good|nice|top)\b.*\b(in|at|near)\b'
        ]

        for pattern in tourism_patterns:
            if re.search(pattern, query_lower):
                logger.info(f"Pattern matched: knowledge_base")
                return "knowledge_base"

        try:
            system_prompt = """Classify this query into ONE category:
- knowledge_base: Questions about specific places, restaurants, hotels, attractions, recommendations
- web_search: Real-time information like weather, current events, news
- hybrid: Questions about directions, how to get somewhere, transportation
- out_of_scope: Non-tourism questions

Respond with ONLY the category name."""

            response = self.client.chat.completions.create(
                model=self.settings.openai_chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.0,
                max_tokens=10
            )

            intent = response.choices[0].message.content.strip().lower()

            logger.info(f"LLM classification: {intent}")

            if "out_of_scope" in intent or "out-of-scope" in intent:
                return "out_of_scope"
            elif "hybrid" in intent:
                return "hybrid"
            elif "web_search" in intent or "web-search" in intent:
                return "web_search"
            else:
                return "knowledge_base"

        except Exception as e:
            logger.error(f"Error in LLM classification: {str(e)}, defaulting to knowledge_base")
            return "knowledge_base"
