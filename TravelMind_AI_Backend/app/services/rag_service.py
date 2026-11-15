import openai
import logging
from typing import List, Dict, Any, AsyncGenerator, Optional
from datetime import datetime
from app.core.config import Settings
from .embedding_service import EmbeddingService
from .vector_store import VectorStore
from .web_search_service import WebSearchService
from .intent_classifier import IntentClassifier
from .external_memory_service import ExternalMemoryService
from .prompt_template_service import PromptTemplateService
from .context_middleware import ContextMiddleware
from app.models.schemas import Document

from app.core.logger import get_logger

logger = get_logger(__name__)


class RAGService:
    def __init__(
        self,
        settings: Settings,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        web_search_service: Optional[WebSearchService] = None,
        intent_classifier: Optional[IntentClassifier] = None,
        itinerary_service=None,
        external_memory_service: Optional[ExternalMemoryService] = None
    ):
        self.settings = settings
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.web_search_service = web_search_service or WebSearchService(settings)
        self.intent_classifier = intent_classifier or IntentClassifier(settings)
        self.itinerary_service = itinerary_service
        self.external_memory_service = external_memory_service
        self.client = openai.OpenAI(api_key=settings.openai_api_key)
        
        # Initialize new services
        self.prompt_template_service = PromptTemplateService(settings)
        self.context_middleware = ContextMiddleware(external_memory_service, itinerary_service) if external_memory_service and itinerary_service else None

    async def query(self, query_text: str, user_id: str = "default_user", trip_id: Optional[int] = None, top_k: int = 5, max_tokens: Optional[int] = None) -> tuple[str, List[Dict[str, Any]]]:
        # MIDDLEWARE: Inject memory and itinerary context
        memory_summary, itinerary_summary = None, None
        if self.context_middleware:
            memory_summary, itinerary_summary = self.context_middleware.ensure_context_injection(user_id, trip_id)
        
        if self.external_memory_service:
            self.external_memory_service.add_user_message(user_id, query_text, max_tokens)

        query_embedding = await self.embedding_service.get_embedding(query_text)
        results = self.vector_store.search(query_embedding, top_k)

        if not results:
            return "I don't have enough information to answer your question.", []

        context = self._build_context(results)
        sources = self._format_sources(results)

        effective_max_tokens = max_tokens or self.settings.openai_max_tokens

        # Build final messages using template service - NO HISTORY GUARANTEE
        messages = self.prompt_template_service.build_final_messages(
            user_message=query_text,
            memory_summary=memory_summary,
            itinerary_summary=itinerary_summary,
            context=context
        )

        response = self.client.chat.completions.create(
            model=self.settings.openai_chat_model,
            messages=messages,  # ONLY current system + user message
            temperature=self.settings.openai_temperature,
            max_tokens=effective_max_tokens
        )

        answer = response.choices[0].message.content

        if self.external_memory_service:
            self.external_memory_service.add_assistant_message(user_id, answer, max_tokens)

        await self._analyze_and_update_itinerary(user_id, trip_id, query_text, answer)

        return answer, sources

    async def query_stream(self, query_text: str, user_id: str = "default_user", trip_id: Optional[int] = None, top_k: int = 5, max_tokens: Optional[int] = None) -> AsyncGenerator[Dict[str, Any], None]:
        try:
            if self.external_memory_service:
                self.external_memory_service.add_user_message(user_id, query_text, max_tokens)

            intent = await self.intent_classifier.classify_intent(query_text)

            yield {
                "type": "intent",
                "content": intent
            }

            if intent == "out_of_scope":
                out_of_scope_msg = "I'm a specialized tourism assistant. I can help you with:\n\n• Accommodations and hotels\n• Restaurants and dining\n• Tourist attractions and activities\n• Festivals and events\n• Travel tips and helpful information\n• Weather conditions\n\nPlease ask me something related to tourism or travel!"

                if self.external_memory_service:
                    self.external_memory_service.add_assistant_message(user_id, out_of_scope_msg, max_tokens)

                yield {
                    "type": "answer",
                    "content": out_of_scope_msg,
                    "done": True
                }
                return

            query_embedding = await self.embedding_service.get_embedding(query_text)
            kb_results = self.vector_store.search(query_embedding, top_k)
            has_kb_data = len(kb_results) > 0 and kb_results[0][1] < 1.5

            logger.info(f"Knowledge base search: {len(kb_results)} results, best score: {kb_results[0][1] if kb_results else 'N/A'}")

            full_response = ""
            if intent == "web_search":
                if has_kb_data:
                    logger.info("Web search + KB data found, using hybrid approach")
                    async for chunk in self._handle_hybrid_stream(query_text, kb_results, user_id, trip_id, max_tokens):
                        yield chunk
                        if chunk.get("type") == "answer" and chunk.get("content"):
                            full_response += chunk["content"]
                else:
                    logger.info("Web search only (no KB data)")
                    async for chunk in self._handle_web_search_stream(query_text, user_id, trip_id, max_tokens):
                        yield chunk
                        if chunk.get("type") == "answer" and chunk.get("content"):
                            full_response += chunk["content"]
            elif intent == "hybrid":
                if has_kb_data:
                    logger.info("Hybrid query with KB data")
                    async for chunk in self._handle_hybrid_stream(query_text, kb_results, user_id, trip_id, max_tokens):
                        yield chunk
                        if chunk.get("type") == "answer" and chunk.get("content"):
                            full_response += chunk["content"]
                else:
                    logger.info("Hybrid query without KB data, using web search")
                    async for chunk in self._handle_web_search_stream(query_text, user_id, trip_id, max_tokens):
                        yield chunk
                        if chunk.get("type") == "answer" and chunk.get("content"):
                            full_response += chunk["content"]
            else:
                if has_kb_data:
                    logger.info("Knowledge base query with data")
                    async for chunk in self._handle_knowledge_base_stream(query_text, kb_results, user_id, trip_id, max_tokens):
                        yield chunk
                        if chunk.get("type") == "answer" and chunk.get("content"):
                            full_response += chunk["content"]
                else:
                    logger.info("No KB data found, trying web search")
                    async for chunk in self._handle_web_search_stream(query_text, user_id, trip_id, max_tokens):
                        yield chunk
                        if chunk.get("type") == "answer" and chunk.get("content"):
                            full_response += chunk["content"]

            if self.external_memory_service and full_response:
                self.external_memory_service.add_assistant_message(user_id, full_response, max_tokens)

        except Exception as e:
            logger.error(f"Error in query_stream: {str(e)}")
            yield {
                "type": "error",
                "content": f"An error occurred: {str(e)}",
                "done": True
            }

    async def _handle_knowledge_base_stream(self, query_text: str, results: List[tuple[Document, float]], user_id: str = "default_user", trip_id: Optional[int] = None, max_tokens: Optional[int] = None) -> AsyncGenerator[Dict[str, Any], None]:
        if not results:
            yield {
                "type": "answer",
                "content": "Hmm, I don't have specific information about that at the moment. Could you try asking about accommodations, restaurants, attractions, or festivals? I'd love to help! 😊",
                "done": True
            }
            return

        # MIDDLEWARE: Inject memory and itinerary context
        memory_summary, itinerary_summary = None, None
        if self.context_middleware:
            memory_summary, itinerary_summary = self.context_middleware.ensure_context_injection(user_id, trip_id)

        context = self._build_context(results)
        effective_max_tokens = max_tokens or self.settings.openai_max_tokens

        # Build final messages using template service - NO HISTORY GUARANTEE
        messages = self.prompt_template_service.build_final_messages(
            user_message=query_text,
            memory_summary=memory_summary,
            itinerary_summary=itinerary_summary,
            context=context
        )

        stream = self.client.chat.completions.create(
            model=self.settings.openai_chat_model,
            messages=messages,  # ONLY current system + user message
            temperature=self.settings.openai_temperature,
            max_tokens=effective_max_tokens,
            stream=True
        )

        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                yield {
                    "type": "answer",
                    "content": content,
                    "done": False
                }

        yield {
            "type": "answer",
            "content": "",
            "done": True
        }

        if full_response:
            await self._analyze_and_update_itinerary(user_id, trip_id, query_text, full_response)

    async def _handle_hybrid_stream(self, query_text: str, results: List[tuple[Document, float]], user_id: str = "default_user", trip_id: Optional[int] = None, max_tokens: Optional[int] = None) -> AsyncGenerator[Dict[str, Any], None]:
        context = ""
        if results:
            context = self._build_context(results)

        search_result = self.web_search_service.search_and_respond(query_text, context, max_tokens)

        if not search_result["success"]:
            if results:
                async for chunk in self._handle_knowledge_base_stream(query_text, results, user_id, trip_id, max_tokens):
                    yield chunk
            else:
                yield {
                    "type": "answer",
                    "content": "I couldn't find specific information about that right now. Could you try asking differently? 🔍",
                    "done": True
                }
            return

        response_text = search_result["text"]

        if not response_text:
            if results:
                async for chunk in self._handle_knowledge_base_stream(query_text, results, user_id, trip_id, max_tokens):
                    yield chunk
            else:
                yield {
                    "type": "answer",
                    "content": "I couldn't find specific information about that right now. Could you try asking differently? 🔍",
                    "done": True
                }
            return

        for char in response_text:
            yield {
                "type": "answer",
                "content": char,
                "done": False
            }

        yield {
            "type": "answer",
            "content": "",
            "done": True
        }

        await self._analyze_and_update_itinerary(user_id, trip_id, query_text, response_text)

    async def _handle_web_search_stream(self, query_text: str, user_id: str = "default_user", trip_id: Optional[int] = None, max_tokens: Optional[int] = None) -> AsyncGenerator[Dict[str, Any], None]:
        search_result = self.web_search_service.search_and_respond(query_text, None, max_tokens)

        if not search_result["success"]:
            yield {
                "type": "answer",
                "content": "I couldn't find current information about that right now. Could you try rephrasing your question? 🔍",
                "done": True
            }
            return

        response_text = search_result["text"]

        if not response_text:
            yield {
                "type": "answer",
                "content": "I couldn't find current information about that right now. Could you try rephrasing your question? 🔍",
                "done": True
            }
            return

        for char in response_text:
            yield {
                "type": "answer",
                "content": char,
                "done": False
            }

        yield {
            "type": "answer",
            "content": "",
            "done": True
        }

        await self._analyze_and_update_itinerary(user_id, trip_id, query_text, response_text)

    async def _analyze_and_update_itinerary(self, user_id: str, trip_id: Optional[int], query: str, response: str):
        if not self.itinerary_service:
            return

        try:
            current_trip = None
            if trip_id:
                current_trip = self.itinerary_service.get_trip(trip_id)
            else:
                current_trip = self.itinerary_service.get_current_trip(user_id)

            if not current_trip:
                return

            analysis_prompt = f"""Analyze this conversation to extract any new activities or itinerary updates:

User Query: {query}
Assistant Response: {response}

Extract any specific activities, restaurants, attractions, or places mentioned that could be added to an itinerary. Return a JSON array of activities in this format:
[
  {{
    "name": "Activity name",
    "type": "restaurant|attraction|activity|hotel|transport",
    "time": "HH:MM or null",
    "location": "address or area",
    "description": "brief description"
  }}
]

If no specific activities are mentioned that should be added to an itinerary, return an empty array: []

Only extract concrete suggestions, not general categories or areas."""

            analysis_response = self.client.chat.completions.create(
                model=self.settings.openai_chat_model,
                messages=[
                    {"role": "system", "content": "You extract specific activities from travel conversations and return them as JSON."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )

            import json
            activities_text = analysis_response.choices[0].message.content.strip()
            
            if activities_text.startswith('```json'):
                activities_text = activities_text[7:-3]
            elif activities_text.startswith('```'):
                activities_text = activities_text[3:-3]
            
            activities = json.loads(activities_text)

            if not activities:
                return

            trip_days = self.itinerary_service.get_trip_days(current_trip['id'])
            if not trip_days:
                day_id = self.itinerary_service.add_day(current_trip['id'], 1, None, "Generated day")
            else:
                day_id = trip_days[0]['id']

            for activity in activities:
                self.itinerary_service.add_item(
                    day_id=day_id,
                    name=activity.get('name', ''),
                    activity_type=activity.get('type'),
                    description=activity.get('description'),
                    location=activity.get('location'),
                    time=activity.get('time'),
                    order_index=0
                )

            logger.info(f"Added {len(activities)} activities to itinerary for user {user_id}")

        except Exception as e:
            logger.error(f"Error analyzing itinerary updates: {str(e)}")

    def _build_context(self, results: List[tuple[Document, float]]) -> str:
        context_parts = []
        for doc, score in results:
            context_parts.append(f"[Source: {doc.source}]\n{doc.content}\n")
        return "\n---\n".join(context_parts)

    def _build_prompt(self, query: str, context: str) -> str:
        return f"""Based on the following context, please answer the question.

Context:
{context}

Question: {query}

Answer:"""

    def _format_sources(self, results: List[tuple[Document, float]]) -> List[Dict[str, Any]]:
        sources = []
        for doc, score in results:
            sources.append({
                "id": doc.id,
                "source": doc.source,
                "score": score,
                "metadata": doc.metadata
            })
        return sources
        context_parts = []
        for doc, score in results:
            context_parts.append(f"[Source: {doc.source}]\n{doc.content}\n")
        return "\n---\n".join(context_parts)

    def _build_prompt(self, query: str, context: str) -> str:
        return f"""Based on the following context, please answer the question.

Context:
{context}

Question: {query}

Answer:"""

    def _format_sources(self, results: List[tuple[Document, float]]) -> List[Dict[str, Any]]:
        sources = []
        for doc, score in results:
            sources.append({
                "id": doc.id,
                "source": doc.source,
                "score": score,
                "metadata": doc.metadata
            })
        return sources