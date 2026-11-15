from __future__ import annotations
import logging
import json
from typing import List, Dict, Any, AsyncGenerator, Optional, Literal, TypedDict, Annotated
from datetime import datetime
import operator
import asyncio

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_community.cache import GPTCache as LangChainGPTCache
from langchain_core.globals import set_llm_cache
from langchain_core.callbacks import BaseCallbackHandler
from gptcache import Cache
from gptcache.processor.pre import get_prompt

from langgraph.graph import StateGraph, END

from app.core.config import Settings
from .embedding_service import EmbeddingService
from .vector_store import VectorStore
from .web_search_service import WebSearchService
from .tools import get_functional_tools, get_weather
from .external_memory_service import ExternalMemoryService
from .context_middleware import ContextMiddleware
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range, MatchAny
from gptcache.manager import get_data_manager, CacheBase, VectorBase
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation


from app.core.logger import get_logger

logger = get_logger(__name__)

class TokenUsageLogger(BaseCallbackHandler):
    def __init__(self, node_name: str):
        self.node_name = node_name

    def on_llm_end(self, response, **kwargs):
        usage = None
        
        # 1. Try generic llm_output (OpenAI/Legacy)
        if response.llm_output and 'token_usage' in response.llm_output:
            usage = response.llm_output['token_usage']
            
        # 2. Try usage_metadata from the first generation (Newer LangChain)
        if not usage and response.generations:
            try:
                # generations is List[List[Generation]]
                first_gen = response.generations[0][0]
                if hasattr(first_gen, 'message') and hasattr(first_gen.message, 'usage_metadata'):
                    usage = first_gen.message.usage_metadata
            except Exception:
                pass

        if usage:
            # Handle different key names (OpenAI vs LangChain Standard)
            total = usage.get('total_tokens', 0)
            prompt = usage.get('prompt_tokens') or usage.get('input_tokens', 0)
            completion = usage.get('completion_tokens') or usage.get('output_tokens', 0)
            
            logger.info(f"💰 [{self.node_name}] Tokens: {total} (Prompt: {prompt}, Completion: {completion})")
        else:
            # Only log warning if strictly debugging, otherwise it might be too noisy for normal flows
            # But the user asked for logs, so let's keep a debug/warning entry if missing
            logger.debug(f"[{self.node_name}] No token usage found in response.")

# --- State Definition ---
class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """
    question: str
    user_id: str
    trip_id: Optional[int]
    intent: str
    filters: Dict[str, Any]
    documents: List[str]
    generation: str
    user_language: str
    translated_question: Optional[str]
    max_tokens: Optional[int]
    keywords: Optional[str]
    recommended_items: List[str]
    price_clarification_needed: Optional[bool]

# --- Output Schemas ---
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    intent: Literal["accommodations", "restaurants", "attractions", "helpful-things", "festivals", "weather", "trip_planning", "user_context", "price_inquiry", "out_of_scope"] = Field(
        ...,
        description="The specific intent of the user query."
    )
    location: Optional[str] = Field(
        default=None,
        description="The specific city or location mentioned in the query. Extract if present (e.g. 'Brasov', 'Cluj')."
    )
    response: Optional[str] = Field(
        default=None,
        description="A friendly response to the user. POPULATE ONLY IF intent is 'out_of_scope'. It MUST clearly state that I can't help with that and I only help with tourism and travels in Romania."
    )

class GeneratorOutput(BaseModel):
    """Final response generation with recommended items."""
    response: str = Field(description="The friendly response to the user.")
    recommended_items: List[str] = Field(
        default=[],
        description="List of exact names of the items (hotels, restaurants, festivals) that were explicitly recommended in the response."
    )

class Amenities(BaseModel):
    """Extract all amenities mentioned in the query."""
    amenities: List[str] = Field(
        description="List of ALL amenities, facilities, or features mentioned in the query. Normalize to UPPERCASE. Do not restrict to a specific list. Example: ['TERRACE', 'HOT TUB', 'POOL', 'BALCONY', 'VIEW']."
    )
    request_type: Literal["filtering", "fetch_all"] = Field(
        default="filtering",
        description="Type of request. 'filtering' if user wants to find places with specific amenities (default). 'fetch_all' if user asks 'what amenities does Hotel X have?' or 'list all facilities'."
    )
    target_accommodation: Optional[str] = Field(
        default=None,
        description="Name of the specific accommodation if request_type is 'fetch_all'. Example: 'Hotel Aro Palace'."
    )

class FestivalQuery(BaseModel):
    """Refine festival queries with specific details."""
    festival_type: Optional[str] = Field(
        default=None,
        description="Type of festival (e.g., 'music', 'food', 'market', 'traditional', 'art')."
    )
    date_context: Optional[str] = Field(
        default=None,
        description="Timeframe mentioned in the query (e.g., 'this weekend', 'August', 'next summer', '2026-06-22')."
    )
    user_preferences: Optional[str] = Field(
        default=None,
        description="Inferred user preferences based on conversation history (e.g., 'likes jazz', 'prefers outdoor events')."
    )
    clarification_question: Optional[str] = Field(
        default=None,
        description="If the user's query is too vague (e.g. 'are there festivals?' without date or type), provide a polite question asking for more details (e.g. 'What kind of festivals do you like?')."
    )

class FestivalSearchParams(BaseModel):
    """Parameters for festival search including date range and query."""
    search_query: str = Field(description="Optimized search query for vector database (English keywords).")
    start_date_ts: Optional[int] = Field(description="Start timestamp (Unix epoch) of the search window. None if no specific date.")
    end_date_ts: Optional[int] = Field(description="End timestamp (Unix epoch) of the search window. None if no specific date.")


class LanguageDetermination(BaseModel):
    """Determine the language of the user's query."""
    language_code: Literal["en", "ro", "ru", "es", "fr", "de", "it", "pt", "pl", "uk", "hu"] = Field(
        ...,
        description="The 2-letter language code of the user's query."
    )

class Translation(BaseModel):
    """Translate the user's query to English."""
    translated_text: str = Field(
        ...,
        description="The English translation of the user's query."
    )

class CityExtraction(BaseModel):
    """Extract city from history."""
    city: Optional[str] = Field(description="The city mentioned in the conversation context.")
    clarification_question: Optional[str] = Field(description="Question to ask user if city is missing.")

class KeywordExtraction(BaseModel):
    """Extract strict keywords from query and history."""
    keywords: Optional[str] = Field(
        description="Space-separated list of strict keywords to filter by. Include specific features (e.g. 'playground', 'breakfast included') or specific item names from history."
    )

class PriceExtraction(BaseModel):
    """Extract price/budget information from accommodation queries."""
    has_price_mention: bool = Field(
        description="Whether the query mentions price, budget, cost, expensive, cheap, or similar terms."
    )
    min_price: Optional[int] = Field(
        default=None,
        description="Minimum price in the range (in local currency). Extract if explicitly mentioned."
    )
    max_price: Optional[int] = Field(
        default=None,
        description="Maximum price in the range (in local currency). Extract if explicitly mentioned."
    )
    clarification_question: Optional[str] = Field(
        default=None,
        description="If price is mentioned but no specific range is given (e.g., 'expensive', 'cheap'), ask for specific budget in the user's language."
    )


class SaveRecommendations(BaseModel):
    """Save the exact names of recommended items."""
    recommended_items: List[str] = Field(
        default=[],
        description="List of exact names of the items (hotels, restaurants, festivals) that were explicitly recommended in the response."
    )


class AgenticRAGService:
    def __init__(
        self,
        settings: Settings,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        web_search_service: Optional[WebSearchService] = None,
        intent_classifier=None,
        itinerary_service=None,
        external_memory_service: Optional[ExternalMemoryService] = None
    ):
        self.settings = settings
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.web_search_service = web_search_service
        self.itinerary_service = itinerary_service
        self.external_memory_service = external_memory_service
        self.context_middleware = ContextMiddleware(external_memory_service, itinerary_service) if external_memory_service and itinerary_service else None

        # Enable Semantic Caching with GPTCache + Redis
        # Uses Redis for scalable, multi-instance caching with automatic TTL
        # Similarity threshold: 0.85 (85% similar = cache hit)
        def init_gptcache(cache_obj: Cache):
            """Initialize GPTCache with Redis backend for semantic similarity matching"""
            # Redis connection for cache storage
            redis_host = "redis"  # Docker service name
            redis_port = 6379
            
            # Cache base using Redis
            cache_base = CacheBase(
                name='redis',
                host=redis_host,
                port=redis_port,
                maxmemory='512mb',
                policy='allkeys-lru',
                ttl=3600  # Cache entries expire after 1 hour
            )
            
            # Vector base for embedding similarity search (in-memory FAISS)
            vector_base = VectorBase(
                name='faiss',
                dimension=1536,  # OpenAI text-embedding-ada-002 dimension
            )
            
            # Data manager combines cache and vector storage
            data_manager = get_data_manager(cache_base, vector_base)
            
            # Initialize cache with embedding-based similarity
            cache_obj.init(
                pre_embedding_func=get_prompt,
                embedding_func="openai",  # Use OpenAI for embeddings
                data_manager=data_manager,
                similarity_evaluation=SearchDistanceEvaluation(
                    # Distance threshold: 1 - 0.85 = 0.15 (lower distance = more similar)
                    distance_threshold=0.15
                ),
            )
        
        # Set up LangChain to use GPTCache with Redis
        set_llm_cache(LangChainGPTCache(init_gptcache))
        logger.info("Semantic caching enabled with GPTCache + Redis (similarity: 0.85, TTL: 1h)")

        # Initialize LLM
        # Note: OpenAI's prompt caching works automatically for supported models (gpt-4o, gpt-4o-mini, etc.)
        # on prompts >1024 tokens - no explicit configuration needed
        self.llm = ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.openai_chat_model,
            temperature=0, # Low temperature for routing/extraction consistency
            streaming=True,
            max_tokens=settings.openai_max_tokens,
            stream_usage=True, # Enable usage stats for streaming
        )
        logger.info("ChatOpenAI initialized with Redis-backed semantic caching")
        
        # Tools setup (Functional Tools only)
        # We need a way to bind these to the graph state context, 
        # but for the `tool_node`, we can just instantiate them with the current context 
        # or bind them dynamically in the node.
        # For efficiency, we'll keep the list factory handy.
        
        # Build the graph
        self.app = self._build_graph()

    def _build_graph(self):
        """Builds the LangGraph workflow."""
        workflow = StateGraph(GraphState)

        # Add Nodes
        workflow.add_node("language_determination", self.language_determination_node)
        workflow.add_node("translation", self.translation_node)
        workflow.add_node("classifier", self.classifier_node)
        workflow.add_node("out_of_scope_translation", self.out_of_scope_translation_node)
        workflow.add_node("festival_filter", self.festival_filter_node)
        workflow.add_node("festival_retriever", self.festival_retriever_node)
        workflow.add_node("accommodation_retriever", self.accommodation_retriever_node)
        workflow.add_node("generic_retriever", self.generic_retriever_node)
        workflow.add_node("city_determination", self.city_determination_node)
        workflow.add_node("amenities_extractor", self.amenities_extractor_node)
        workflow.add_node("price_extractor", self.price_extractor_node)
        workflow.add_node("budget_clarification", self.budget_clarification_node) # NEW
        workflow.add_node("tool_node", self.tool_node)
        workflow.add_node("price_retriever", self.price_retriever_node)
        workflow.add_node("fallback_search", self.fallback_search_node)
        workflow.add_node("generator", self.generator_node)
        workflow.add_node("keyword_extractor", self.keyword_extractor_node)


        # Add Edges
        workflow.set_entry_point("language_determination")

        workflow.add_conditional_edges(
            "language_determination",
            self.route_after_language_determination,
            {
                "translation": "translation",
                "classifier": "classifier"
            }
        )

        workflow.add_edge("translation", "classifier")
        
        workflow.add_conditional_edges(
            "classifier",
            self.route_to_retriever,
            {
                "city_determination": "city_determination",
                "out_of_scope_translation": "out_of_scope_translation",
                "price_retriever": "price_retriever",
                "tool_node": "tool_node", 
                "end": END
            }
        )

        # City determination now goes to keyword_extractor for substantive intents
        workflow.add_conditional_edges(
            "city_determination",
            self.route_after_city_determination,
            {
                "keyword_extractor": "keyword_extractor",
                "tool_node": "tool_node",
                "end": END
            }
        )
        
        # Keyword Extractor distributes to specific nodes
        workflow.add_conditional_edges(
            "keyword_extractor",
            self.route_after_keyword_extractor,
            {
                "amenities_extractor": "amenities_extractor",
                "festival_filter": "festival_filter",
                "generic_retriever": "generic_retriever"
            }
        )

        workflow.add_conditional_edges(
            "festival_filter",
            self.route_after_festival_filter,
            {
                "festival_retriever": "festival_retriever",
                "end": END
            }
        )

        # Check results before generator, fall back if empty
        workflow.add_conditional_edges(
            "accommodation_retriever",
            self.route_check_retriever_results,
            {
                "generator": "generator",
                "fallback_search": "fallback_search"
            }
        )
        workflow.add_conditional_edges(
            "generic_retriever",
            self.route_check_retriever_results,
            {
                 "generator": "generator",
                 "fallback_search": "fallback_search"
            }
        )
        workflow.add_conditional_edges(
            "festival_retriever",
            self.route_check_retriever_results,
             {
                 "generator": "generator",
                 "fallback_search": "fallback_search"
            }
        )
        workflow.add_conditional_edges(
            "price_retriever",
            self.route_after_price_retriever,
             {
                 "generator": "generator",
                 "end": END,
                 "fallback_search": "fallback_search"
            }
        )

        workflow.add_conditional_edges(
            "amenities_extractor",
            self.route_after_amenities_extractor,
            {
                 "price_extractor": "price_extractor",
                 "price_retriever": "price_retriever",
                 "out_of_scope_translation": "out_of_scope_translation"
            }
        )
        
        workflow.add_conditional_edges(
            "price_extractor",
            self.route_after_price_extractor,
            {
                 "budget_clarification": "budget_clarification",
                 "accommodation_retriever": "accommodation_retriever",
                 "out_of_scope_translation": "out_of_scope_translation"
            }
        )

        workflow.add_edge("budget_clarification", "out_of_scope_translation")
        workflow.add_edge("tool_node", "generator")
        workflow.add_edge("fallback_search", "generator")
        workflow.add_edge("out_of_scope_translation", END)
        workflow.add_edge("generator", END)

        return workflow.compile()

    def _format_doc_with_metadata(self, doc) -> str:
        """
        Format a document with relevant metadata appended.
        Includes priceLevel for accommodations and restaurants.
        """
        if not hasattr(doc, 'metadata') or not doc.metadata:
             return f"[Source: {doc.source}]\n{doc.content}"

        metadata_lines = []
        
        # 1. Title (Priority)
        title = doc.metadata.get('title')
        if title:
            metadata_lines.append(f"Title: {title}")
        
        name = doc.metadata.get('name')
        if name:
            metadata_lines.append(f"Name: {name}")

        # 2. Iterate other metadata
        ignored_keys = {'name', 'title', 'source', 'parent_id', 'chunk_id', 'relevance_score'}
        
        for k, v in doc.metadata.items():
            if k in ignored_keys or v is None or v == "":
                continue
                
            # Special handling for Price Per Night
            if k == 'pricePerNight':
                if str(v) != "0":
                    val_str = str(v).strip()
                    # If no currency symbol/code found, assume RON
                    if not any(curr in val_str.lower() for curr in ["ron", "lei", "eur", "usd", "€", "$", "£"]):
                        val_str += " RON"
                    metadata_lines.append(f"Price Per Night: {val_str}")
                continue
            
            # Default formatting
            metadata_lines.append(f"{k.capitalize()}: {v}")
            
        metadata_block = "\n".join(metadata_lines)
        return f"[Source: {doc.source}]\n{metadata_block}\n\n{doc.content}"

    def route_after_language_determination(self, state: GraphState):
        """
        Routes based on the user's language.
        """
        user_lang = state.get("user_language")
        if user_lang == "en":
            return "classifier"
        return "translation"

    # --- Nodes ---

    async def language_determination_node(self, state: GraphState):
        """
        Determines the user's language and saves it to profile if not present.
        """
        question = state["question"]
        user_id = state.get("user_id")
        logger.info(f"---LANGUAGE DETERMINATION NODE--- query: {question}")
        
        # 1. Check if we already have the language in state (passed from memory check in previous steps if we had one, but we don't yet)
        # OR Check tool/memory
        
        search_tools = get_functional_tools(user_id=user_id, trip_id=state.get("trip_id"), external_memory_service=self.external_memory_service)
        get_info_tool = next((t for t in search_tools if t.name == "get_user_info_tool"), None)
        save_lang_tool = next((t for t in search_tools if t.name == "save_user_language"), None)
        
        detected_lang = None
        
        # Attempt to load from profile
        if get_info_tool:
            try:
                info_text = await get_info_tool.ainvoke({})
                if "Language: " in info_text:
                    parts = info_text.split("Language: ")[1].split("\n")[0].strip()
                    if parts and parts.lower() not in ["unknown", "none", "null", "null?"]:
                        detected_lang = parts
                        logger.info(f"Loaded language from profile: {detected_lang}")
            except Exception: pass

        if not detected_lang:
            logger.info("Language unknown. Detecting via LLM.")
            structured_llm = self.llm.with_structured_output(LanguageDetermination)
            system_prompt = """You are a language detector.
Determine the language of the user's input.
Map to one of these codes:
- en (English)
- ro (Romanian)
- ru (Russian)
- es (Spanish)
- fr (French)
- de (German)
- it (Italian)
- pt (Portuguese)
- pl (Polish)
- uk (Ukrainian)
- hu (Hungarian)

If uncertain, default to 'en'.
"""
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=question)]
            res = await structured_llm.ainvoke(messages, config={"callbacks": [TokenUsageLogger("language_determination")]})
            detected_lang = res.language_code
            logger.info(f"Detected language: {detected_lang}")
            
            # Save it
            if save_lang_tool:
                try:
                    await save_lang_tool.ainvoke({"language": detected_lang})
                except Exception: pass
        
        return {"user_language": detected_lang}

    async def translation_node(self, state: GraphState):
        """
        Translates non-English queries to English.
        """
        question = state["question"]
        source_lang = state["user_language"]
        logger.info(f"---TRANSLATION NODE--- from {source_lang}")
        
        user_id = state.get("user_id")
        return {"translated_question": await self._translate_to_english(question, source_lang, user_id)}

    # CORRECTED FIX for agentic_rag_service.py
# Replace the _translate_to_english method (around line 570-589) with this:

    async def _translate_to_english(self, text: str, source_lang: str, user_id: str) -> str:
        """
        Translate text from source_lang to English.
        
        CRITICAL: This is for back-translating ASSISTANT responses to save in memory.
        
        Args:
            text: The assistant's response text in source_lang
            source_lang: Source language code (e.g., 'ro', 'fr')
            user_id: User ID for context
            
        Returns:
            Translated text in English (NEVER returns None)
        """
        from langchain_core.messages import HumanMessage, SystemMessage
        
        # CRITICAL: Validate input
        if not text or not text.strip():
            logger.error("_translate_to_english: input text is None or empty")
            return ""  # Return empty string instead of None
        
        # Build clear system prompt
        system_prompt = f"""You are a professional translator.

    CRITICAL TASK: Translate the following assistant response from {source_lang} to English.

    RULES:
    1. The text is a COMPLETE assistant response (bot's message to user)
    2. Translate the ENTIRE text - do not skip any parts
    3. If text contains mixed languages, translate ALL of it to English
    4. Preserve the meaning, tone, and structure
    5. Return ONLY the translated text - no explanations, no preamble
    6. Do NOT return the user's query - translate the ASSISTANT'S response

    Example:
    Input ({source_lang}): "Restaurant Edo's oferă o atmosferă elegantă. Vrei să afli mai multe?"
    Output (English): "Restaurant Edo's offers an elegant atmosphere. Would you like to learn more?"

    Now translate this assistant response to English:"""

        try:
            structured_llm = self.llm.with_structured_output(Translation)
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=text)
            ]
            
            res = await structured_llm.ainvoke(
                messages, 
                config={"callbacks": [TokenUsageLogger("translation")]}
            )
            
            # CRITICAL: Handle None response from LLM
            if not res or not hasattr(res, 'translated_text'):
                logger.error("Translation response has no translated_text attribute")
                return text  # Fallback to original
            
            translated = res.translated_text
            
            # CRITICAL: Validate result is not None or empty
            if not translated or not isinstance(translated, str):
                logger.error(f"Translation returned invalid type: {type(translated)}")
                return text  # Fallback to original
            
            translated = translated.strip()
            
            if not translated:
                logger.error("Translation returned empty string after strip")
                return text  # Fallback to original
            
            # Check if translation is too short (possible failure)
            if len(translated) < len(text) / 3:
                logger.warning(
                    f"Translation suspiciously short ({len(translated)} chars vs {len(text)} chars). "
                    f"Using anyway."
                )
            
            # Check if translation is identical to input (possible failure)
            if translated.lower() == text.lower():
                logger.warning("Translation identical to input")
                # Still return it - might actually be identical
            
            logger.info(f"Back-translated ({source_lang}->en): '{text[:100]}...' -> '{translated[:100]}...'")
            return translated
            
        except Exception as e:
            logger.error(f"Translation failed with exception: {e}")
            logger.error(f"Returning original text as fallback")
            return text  # ALWAYS return something, never None



    async def _translate_from_english(self, text: str, target_lang: str) -> str:
        """
        Translate text FROM English TO target language.
        Used when generator fails to respond in correct language.
        
        Args:
            text: English text to translate
            target_lang: Target language code (e.g., 'ro', 'fr')
            
        Returns:
            Translated text in target language
        """
        from langchain_core.messages import HumanMessage, SystemMessage
        
        lang_names = {
            'ro': 'Romanian',
            'ru': 'Russian', 
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'pl': 'Polish',
            'uk': 'Ukrainian',
            'hu': 'Hungarian'
        }
        
        target_lang_name = lang_names.get(target_lang, target_lang)
        
        system_prompt = f"""You are a professional translator.

TASK: Translate the following text FROM English TO {target_lang_name}.

CRITICAL RULES:
1. Translate the ENTIRE text to {target_lang_name}
2. Preserve meaning, tone, and structure  
3. Keep the casual, friendly tone
4. Return ONLY the {target_lang_name} translation
5. NO explanations, NO preambles
6. Make sure EVERY word is in {target_lang_name}

TEXT TO TRANSLATE:"""

        try:
            structured_llm = self.llm.with_structured_output(Translation)
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=text)
            ]
            
            res = await structured_llm.ainvoke(
                messages,
                config={"callbacks": [TokenUsageLogger("translation")]}
            )
            
            translated = res.translated_text.strip()
            
            logger.info(f"Translated (en->{target_lang}): '{text[:80]}...' -> '{translated[:80]}...'")
            return translated
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return text  # Fallback to original

            
            # Check if translator returned user query instead of translation
            if len(translated) < len(text) / 3:
                logger.warning(
                    f"Translation suspiciously short ({len(translated)} chars vs {len(text)} chars). "
                    f"Possible translation failure. Returning original."
                )
                return text
            
            # Check if translation is identical to input (possible failure)
            if translated.lower() == text.lower():
                logger.warning("Translation identical to input - possible failure")
                return text
            
            logger.info(f"Back-translated ({source_lang}->en): '{text[:100]}...' -> '{translated[:100]}...'")
            return translated
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            # Fallback: return original text
            return text

    async def out_of_scope_translation_node(self, state: GraphState):
        """
        Translates pre-generated out_of_scope responses to the user's language.
        """
        existing_response = state.get("generation", "")
        user_lang_code = state.get("user_language", "en")
        
        logger.info(f"---OUT OF SCOPE TRANSLATION NODE--- to {user_lang_code}")
        
        # If already in English or no response, return as-is
        if user_lang_code == "en" or not existing_response:
            logger.info("Response already in English or empty, returning as-is")
            return {"generation": existing_response}
        
        # Language mapping
        lang_map = {
            "en": "English",
            "ro": "Romanian (Română)",
            "ru": "Russian (Русский)",
            "es": "Spanish (Español)",
            "fr": "French (Français)",
            "de": "German (Deutsch)",
            "it": "Italian (Italiano)",
            "pt": "Portuguese (Português)",
            "pl": "Polish (Polski)",
            "uk": "Ukrainian (Українська)",
            "hu": "Hungarian (Magyar)"
        }
        target_lang_name = lang_map.get(user_lang_code, "English")
        
        # Translate the response
        system_prompt = f"""You are a translator.
Translate the following message to {target_lang_name}.
Maintain the friendly, helpful tone and keep the same meaning.
Do not add or remove any information, just translate accurately.
"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=existing_response)
        ]
        
        response = await self.llm.ainvoke(messages, config={"callbacks": [TokenUsageLogger("out_of_scope_translation")]})
        translated_response = response.content
        
        logger.info(f"Translated out_of_scope response to {target_lang_name}")
        
        return {"generation": translated_response}

    async def classifier_node(self, state: GraphState):
        """
        Determines the intent and extracts amenities (if applicable).
        """
        question = state.get("translated_question") or state["question"]
        user_id = state.get("user_id")
        trip_id = state.get("trip_id")
        max_tokens = state.get("max_tokens")
        
        logger.info(f"---CLASSIFIER NODE--- query: {question}")

        # Save user message to memory (saving the TRANSLATED English version or original if already English)
        # We do this here to ensure we save the translated version.
        if self.external_memory_service:
             self.external_memory_service.add_user_message(user_id, question, max_tokens, trip_id)

        # Get recent messages from memory for context
        history_messages = []
        if self.external_memory_service and user_id:
            recent_msgs = self.external_memory_service.get_recent_messages(user_id, limit=10)
            if recent_msgs:
                for msg in recent_msgs:
                    if msg["role"] == "user":
                        history_messages.append(HumanMessage(content=msg["content"]))
                    else:
                        history_messages.append(AIMessage(content=msg["content"]))

        structured_llm_router = self.llm.with_structured_output(RouteQuery)
        
        system_prompt = f"""You are an expert router for a tourism chatbot.
Classify the user's query into one of 6 intents:
1. accommodations: Searching for hotels, cabins, or places to stay.
2. restaurants: Searching for food, dining, or cafes.
3. attractions: Searching for sightseeing, landmarks, activities, or places to hang out/spend leisure time.
4. helpful-things: Transport, navigation, how to get somewhere, bus schedules, taxi contacts, general tips, shopping, buying things (groceries, pharmacy, souvenirs, clothes).
5. festivals: Searching for events or celebrations.
6. weather: Questions about weather, temperature, rain, forecast.
7. trip_planning: Creating trips, adding items to itinerary, clearing trip data.
8. user_context: Saving current location/city, updating preferences, checking profile.
9. price_inquiry: Questions about price, cost, rates, "how much", or "expensive".
10. out_of_scope: Strictly for greetings or questions CLEARLY unrelated to travel/tourism (e.g., math, coding, politics). If the user asks for a place or activity, it is NOT out_of_scope.

CRITICAL DIRECTION:
- Use the provided conversation history to understand context if the current query is vague (e.g., "tell me more about it", "how much is it").
- Any query asking "how to get to [place]", "directions to [place]", "bus to [place]", or "taxi to [place]" 
    MUST be classified as 'helpful-things' even if the place name sounds like an accommodation or attraction.
- If the user asks about "weather there", "rain there", "temperature there" or similar contextual queries, YOU MUST CHECK HISTORY. 
    If a city was mentioned previously, classify as 'weather'.
- General price questions like "How much does it cost?", "What are the rates?", "Is it expensive?" MUST be classified as 'price_inquiry', even if the item isn't specified (it might be in history).
- Do NOT classify weather, transport, location, or price questions as 'out_of_scope', even if vague. Use specific intents.
- If the intent is 'out_of_scope', you MUST provide a friendly, helpful response in the 'response' field. Keep it concise (max 2 sentences).
- For 'accommodations', extract any specific amenities mentioned.
- Queries like "what amenities does [place] have?" or "list all facilities" MUST be classified as 'accommodations'.

Examples:
- "Find me a hotel with a pool" -> intent: accommodations
- "Is there a good place to hang out with friends?" -> intent: attractions
- "Where can I buy a traditional shirt?" -> intent: helpful-things
- "I need a pharmacy" -> intent: helpful-things
- "How can I get to Ioana Guesthouse?" -> intent: helpful-things
- "Directions to the monastery" -> intent: helpful-things
- "Are there any parties tonight?" -> intent: festivals
- "Is it raining in Cluj?" -> intent: weather, location: "Cluj-Napoca"
- "Create a trip to Brasov" -> intent: trip_planning, location: "Brasov"
- "I am in Bucharest now" -> intent: user_context, location: "Bucharest"
- "How much does it cost?" -> intent: price_inquiry
- "What are the prices?" -> intent: price_inquiry
- "Hello there!" -> intent: out_of_scope, response: "Hello! I'm Vlad-AI, your Romanian travel buddy. I can help you find accommodations, restaurants, and sights in Romania! What's on your mind?"
- "What's the weather like in Paris?" -> intent: out_of_scope, response: "I can't really help with that! I only specialize in tourism and travel within Romania. Feel free to ask about any Romanian destination!"

CONTEXT RESOLUTION RULES:
- If the user provides a specific location (e.g. just a city name) in response to a clarification question, look back at the conversation history.
- IDENTIFY the MAIN UNRESOLVED request. If the user previously asked for accommodation but was stuck on "which city", and their latest message provides that city, RESUME the 'accommodations' intent.
- Do not be distracted by minor intermediate questions (like "where is the market?" or "how far is the center?") if the main goal (accommodation) was clearly established and pending.
        """
        
        messages = [SystemMessage(content=system_prompt)] + history_messages + [HumanMessage(content=question)]
        prompt = ChatPromptTemplate.from_messages(messages)

        router = prompt | structured_llm_router
        result = await router.ainvoke({"question": question}, config={"callbacks": [TokenUsageLogger("classifier")]})

        # amenities removed from RouteQuery, handled in separate node
        logger.info(f"Classifier result: intent={result.intent}")

        output = {
            "intent": result.intent,
            "filters": {} 
        }

        if result.location:
            output["filters"]["location"] = result.location

        if result.intent == "out_of_scope" and result.response:
            output["generation"] = result.response

        return output

    async def city_determination_node(self, state: GraphState):
        """
        Ensures a city/location is present using tools and history.
        Acts as a Context Enrichment layer.
        """
        question = state.get("translated_question") or state["question"]
        filters = state.get("filters", {})
        location = filters.get("location")
        user_id = state.get("user_id")
        intent = state.get("intent")
        
        logger.info(f"---CITY DETERMINATION/CONTEXT--- Intent: {intent}, Incoming Loc: {location}")

        # Helper to run async tools manually
        search_tools = get_functional_tools(user_id=user_id, trip_id=state.get("trip_id"), external_memory_service=self.external_memory_service)
        save_city_tool = next((t for t in search_tools if t.name == "save_user_city"), None)
        get_info_tool = next((t for t in search_tools if t.name == "get_user_info_tool"), None)

        # 1. IF location is explicitly provided/extracted -> SAVE IT
        if location:
            logger.info(f"Location explicitly provided: {location}. Saving to profile.")
            if save_city_tool:
                try:
                    await save_city_tool.ainvoke({"city": location})
                    logger.info(f"User city saved to database: {location}")
                except Exception as e:
                    logger.error(f"Failed to save city: {e}")
            return {} # Propagate existing state

        # 2. IF location missing -> LOAD IT
        logger.info("Location missing. Attempting to retrieve from user profile.")
        saved_city = None
        if get_info_tool:
            try:
                # The tool returns a string like "Current City: ... \n Preferences: ..."
                info_text = await get_info_tool.ainvoke({})
                # Simple parsing (robust enough for this tool's output)
                if "Current City: " in info_text:
                    parts = info_text.split("Current City: ")[1].split("\n")[0].strip()
                    if parts and parts.lower() not in ["unknown", "none", "null"]:
                        saved_city = parts
            except Exception as e:
                logger.error(f"Failed to get user info: {e}")

        if saved_city:
            logger.info(f"Retrieved saved city: {saved_city}")
            return {"filters": {"location": saved_city}}

        # 3. IF STILL MISSING -> Check Intent
        # Only BLOCK/ASK if intent strictly requires it (accommodations)
        # For others (restaurants, etc.), we can search broadly or try to infer, 
        # but sticking to the request "determine city... if not provided asks".
        
        if intent in ["accommodations", "festivals", "restaurants", "attractions", "helpful-things"]:
            pass # Proceed to history check
        else:
            # For other intents, if we don't have a city, we just proceed.
            # (e.g. "Festivals" might be general Romania, or "Weather" allows tool to handle it)
            logger.info(f"City not found, but intent '{intent}' does not strictly enforce blocking. Proceeding.")
            return {}

        # 4. History Check (Last Resort for Accommodations)
        # Get recent messages from memory for context
        history_messages = []
        if self.external_memory_service and user_id:
            recent_msgs = self.external_memory_service.get_recent_messages(user_id, limit=6)
            if recent_msgs:
                for msg in recent_msgs:
                    if msg["role"] == "user":
                        history_messages.append(HumanMessage(content=msg["content"]))
                    else:
                        history_messages.append(AIMessage(content=msg["content"]))

        structured_llm = self.llm.with_structured_output(CityExtraction)
        
        # Get user's language for generating response in correct language
        user_lang_code = state.get("user_language", "en")
        lang_map = {
            "en": "English",
            "ro": "Romanian (Română)",
            "ru": "Russian (Русский)",
            "es": "Spanish (Español)",
            "fr": "French (Français)",
            "de": "German (Deutsch)",
            "it": "Italian (Italiano)",
            "pt": "Portuguese (Português)",
            "pl": "Polish (Polski)",
            "uk": "Ukrainian (Українська)",
            "hu": "Hungarian (Magyar)"
        }
        target_lang_name = lang_map.get(user_lang_code, "English")
        
        system_prompt = f"""You are a helpful assistant. The user is looking for accommodation. 
We need a city to search in. 
Check the conversation history. Is a city mentioned implicitly or explicitly that applies to this request?
If yes, extract it. 
If no, generate a polite question asking for the needed city.

IMPORTANT: If you generate a clarification question, it MUST be in {target_lang_name}.
"""
        
        messages = [SystemMessage(content=system_prompt)] + history_messages + [HumanMessage(content=question)]
        prompt = ChatPromptTemplate.from_messages(messages)
        
        chain = prompt | structured_llm
        result = await chain.ainvoke({"question": question}, config={"callbacks": [TokenUsageLogger("city_determination")]})
        
        if result.city:
             logger.info(f"City determined from history: {result.city}")
             # Save it too!
             if save_city_tool:
                 try:
                    await save_city_tool.ainvoke({"city": result.city})
                 except Exception: pass
             return {
                 "filters": {"location": result.city}
             }
        
        if result.clarification_question:
             logger.info(f"City not found, asking user: {result.clarification_question}")
             return {"generation": result.clarification_question}
        
        # Fallback question in user's language
        fallback_questions = {
            "en": "Could you please specify which city you are interested in?",
            "ro": "Ai putea să specifici în ce oraș ești interesant?",
            "ru": "Не могли бы вы указать, какой город вас интересует?",
            "es": "¿Podrías especificar en qué ciudad estás interesado?",
            "fr": "Pourriez-vous préciser quelle ville vous intéresse?",
            "de": "Könnten Sie bitte angeben, an welcher Stadt Sie interessiert sind?",
            "it": "Potresti specificare quale città ti interessa?",
            "pt": "Você poderia especificar em qual cidade está interessado?",
            "pl": "Czy mógłbyś określić, które miasto Cię interesuje?",
            "uk": "Чи не могли б ви вказати, яке місто вас цікавить?",
            "hu": "Meg tudnád mondani, melyik város érdekel?"
        }
        fallback_question = fallback_questions.get(user_lang_code, fallback_questions["en"])
        return {"generation": fallback_question}



    async def keyword_extractor_node(self, state: GraphState):
        """
        Extracts strict keywords from query and history to use as text filters.
        Handles: quoted terms, implicit item references ("that hotel"), and unstructured features ("playground").
        """
        question = state.get("translated_question") or state["question"]
        user_id = state.get("user_id")
        logger.info(f"---KEYWORD EXTRACTOR--- query: {question}")
        
        # Get history context
        history_messages = []
        if self.external_memory_service and user_id:
            recent_msgs = self.external_memory_service.get_recent_messages(user_id, limit=6)
            if recent_msgs:
                for msg in recent_msgs:
                    if msg["role"] == "user":
                        history_messages.append(HumanMessage(content=msg["content"]))
                    else:
                        history_messages.append(AIMessage(content=msg["content"]))
        
        structured_llm = self.llm.with_structured_output(KeywordExtraction)
        
        system_prompt = """You are an expert search optimizer.
Extract specific, significant nouns or adjectives that should be present in the search result text.

RULES:
1. Extract terms that imply a STRICT requirement.
2. DO NOT extract the following amenities (they are handled by a separate filter):
   - POOL, SWIMMING POOL, HOT TUB, JACUZZI
   - SAUNA, SPA, WELLNESS
   - WIFI, INTERNET
   - PARKING, GARAGE
   - RESTAURANT, DINING, FOOD
   - TERRACE, BALCONY, VIEW
   - BBQ, GRILL
   - KITCHEN, KITCHENETTE
   - FAMILY ROOMS
   - PET FRIENDLY
   - AIR CONDITIONING, AC
   - BREAKFAST
   - ACCESSIBLE, WHEELCHAIR
   - RECEPTION

3. DO NOT extract temporal/date information (they are handled separately):
   - DATE, DATES, DAY, DAYS
   - MONTH, MONTHS (January, February, etc.)
   - SEASON, SEASONS (summer, winter, etc.)
   - YEAR, WEEKEND, WEEKDAY
   - TODAY, TOMORROW, TONIGHT
   - TIME-RELATED terms (morning, evening, etc.)

4. DO NOT extract place names or city names (they are handled by location filters):
   - CITY, CITIES, TOWN, VILLAGE
   - Romanian cities (Bucharest, Cluj, Brasov, Sibiu, Timisoara, Iasi, Constanta, etc.)
   - Regions or counties (Transylvania, Maramures, Moldavia, etc.)
   - General location terms (downtown, center, mountain, seaside, etc.)

5. If the user refers to a specific item from history (e.g., "that hotel", "it", "the place we discussed"), EXTRACT ITS NAME.
5. If words are in quotes (e.g. "infinity pool"), ALWAYS extract them.
6. Output a space-separated string.
        """
        
        # Include history
        messages = [SystemMessage(content=system_prompt)] + history_messages + [HumanMessage(content=question)]
        prompt = ChatPromptTemplate.from_messages(messages)
        
        chain = prompt | structured_llm
        result = await chain.ainvoke({"question": question}, config={"callbacks": [TokenUsageLogger("keyword_extractor")]})
        
        if result.keywords:
            # Check for null-like strings which LLM might output
            cleaned_keywords = result.keywords.strip()
            if cleaned_keywords.lower() in ["none", "null", "no", "nan", "undefined"]:
                 logger.info(f"Extracted strict keywords detected as null-like ('{cleaned_keywords}'), ignoring.")
                 return {"keywords": None}
                 
            logger.info(f"Extracted strict keywords: '{result.keywords}'")
            return {"keywords": result.keywords}
        
        return {"keywords": None}

    async def amenities_extractor_node(self, state: GraphState):
        """
        Extracts ALL amenities mentioned in the query without strict filtering.
        Also handles requests for "all amenities" of a specific place.
        """
        question = state.get("translated_question") or state["question"]
        user_id = state.get("user_id")
        logger.info(f"---AMENITIES EXTRACTOR--- query: {question}")
        
        # Get recent messages from memory for context
        history_messages = []
        if self.external_memory_service and user_id:
            recent_msgs = self.external_memory_service.get_recent_messages(user_id, limit=6)
            if recent_msgs:
                for msg in recent_msgs:
                    if msg["role"] == "user":
                        history_messages.append(HumanMessage(content=msg["content"]))
                    else:
                        history_messages.append(AIMessage(content=msg["content"]))
        
        structured_llm = self.llm.with_structured_output(Amenities)
        
        system_prompt = """You are an expert at extracting travel amenities.

Task 1: FILTERING (Default)
Extract EVERY amenity mentioned in the user's query or CONTEXT, but ONLY if it matches the valid list below.

VALID AMENITIES (Strict Map):
- "POOL" (swimming pool, swim)
- "HOT TUB" (jacuzzi, ciubar)
- "SAUNA / SPA" (sauna, spa, wellness)
- "WIFI" (internet)
- "PARKING" (garage, car)
- "RESTAURANT" (food, dining)
- "TERRACE" (balcony, view)
- "BBQ" (grill, barbeque)
- "KITCHEN" (kitchenette, cooking)
- "FAMILY ROOMS" (large rooms, family)
- "PET FRIENDLY" (dogs, cats, pets)
- "AIR CONDITIONING" (AC, cool)
- "BREAKFAST POSSIBILTY" (breakfast available, serves breakfast)
- "DISABLE PERSON" (wheelchair, accessible)
- "RECEPTION 24H"

Task 2: FETCH ALL
If the user specifically asks for the amenities or facilities of a specific accommodation (e.g., "What amenities does [Hotel Name] have?"), set request_type='fetch_all' and extract 'target_accommodation'. 
IMPORTANT: Do NOT trigger this for general requests like "Tell me more about [Hotel Name]" or "Give me details about [Hotel Name]". Only trigger if 'amenities' or 'facilities' are explicitly requested.

INSTRUCTIONS:
- CRITICAL: If the user asks for 'breakfast included', 'free breakfast', or 'breakfast in the price', YOU MUST extract 'BREAKFAST POSSIBILTY'.
- Output a list of strings from the VALID AMENITIES list above.
- Normalize input terms to the exact UPPERCASE string from the list.
- TRANSLATE to English to match the amenities list if the query is in another language.
- Example: "I want a place with a jacuzzi and internet" -> ["HOT TUB", "WIFI"]
- Example: "piscină și grătar" -> ["POOL", "BBQ"]
- Example: "What amenities does Hotel Alpin have?" -> request_type='fetch_all', target_accommodation='Hotel Alpin', amenities=[]
- IF NO match found, return an empty list.
"""
        
        # Include history in messages
        messages = [SystemMessage(content=system_prompt)] + history_messages + [HumanMessage(content=question)]
        prompt = ChatPromptTemplate.from_messages(messages)
        
        chain = prompt | structured_llm
        result = await chain.ainvoke({"question": question}, config={"callbacks": [TokenUsageLogger("amenities_extractor")]})
        
        # Handle Fetch All Request
        if result.request_type == "fetch_all" and result.target_accommodation:
            logger.info(f"Fetch All Amenities for: {result.target_accommodation}")
             
            # Search for the accommodation
            query_embedding = await self.embedding_service.get_embedding(result.target_accommodation)
            sparse_embedding = self.embedding_service.get_sparse_embedding(result.target_accommodation)
             
            search_filters = Filter(must=[FieldCondition(key="category.type", match=MatchValue(value="accommodations"))])
             
            results = self.vector_store.search(
                query_dense_embedding=query_embedding,
                query_sparse_embedding=sparse_embedding,
                top_k=1,
                filters=search_filters
            )
             
            if results:
                doc, _ = results[0]
                meta = doc.metadata or {}
                amenities_list = meta.get("amenities", [])
                place_name = meta.get("title", result.target_accommodation)
                 
                if amenities_list:
                    # Join them nicely
                    # Amenities in DB might be snake_case or uppercase, convert to readable
                    readable_amenities = [a.replace("_", " ").title() for a in amenities_list]
                    amenities_str = ", ".join(readable_amenities)
                    response = f"The amenities at {place_name} include: {amenities_str}."
                else:
                    response = f"I found {place_name}, but I don't have a specific list of amenities for it in my database."
            else:
                response = f"I couldn't find an accommodation named '{result.target_accommodation}' in my database."
                
            return {"generation": response}

        amenities = result.amenities if result.amenities else []
        logger.info(f"Extracted amenities from query: {amenities}")
        
        # Get tools to save/retrieve amenities
        search_tools = get_functional_tools(user_id=user_id, trip_id=state.get("trip_id"), external_memory_service=self.external_memory_service)
        save_amenities_tool = next((t for t in search_tools if t.name == "save_user_amenities"), None)
        
        # If amenities were extracted, save them (replacing any old ones)
        if amenities:
            logger.info(f"Amenities specified in query: {amenities}. Saving to user profile.")
            if save_amenities_tool:
                try:
                    await save_amenities_tool.ainvoke({"amenities": amenities})
                except Exception as e:
                    logger.error(f"Failed to save amenities: {e}")
        else:
            # No amenities in current query - try to load saved ones
            logger.info("No amenities in query. Checking for saved amenities.")
            if self.external_memory_service:
                try:
                    saved_amenities = self.external_memory_service.get_user_amenities(user_id)
                    if saved_amenities:
                        logger.info(f"Using saved amenities from user profile: {saved_amenities}")
                        amenities = saved_amenities
                    else:
                        logger.info("No saved amenities found.")
                except Exception as e:
                    logger.error(f"Failed to load saved amenities: {e}")
        
        return {
            "filters": {"amenities": amenities}
        }

    def route_after_amenities_extractor(self, state: GraphState):
        """
        Routes after amenities extractor based on intent.
        - If direct response generated (fetch_all), go to translation
        - If intent is "accommodations", continue to price_extractor (full accommodation pipeline)
        - If intent is "price_inquiry", go to price_retriever (direct price lookup)
        """
        # First check if we generated a direct response (e.g., fetch_all amenities)
        if state.get("generation"):
            return "out_of_scope_translation"
        
        # Route based on intent
        intent = state.get("intent")
        if intent == "accommodations":
            return "price_extractor"
        elif intent == "price_inquiry":
            return "price_retriever"
        
        # Default fallback to price_extractor
        return "price_extractor"

    async def price_extractor_node(self, state: GraphState):
        """
        Extracts price/budget information from accommodation queries.
        Asks for clarification if price is mentioned but no specific range is given.
        Stores budget in user_info for future queries.
        """
        question = state.get("translated_question") or state["question"]
        user_id = state.get("user_id")
        logger.info(f"---PRICE EXTRACTOR--- query: {question}")
        
        # Get history context
        history_messages = []
        if self.external_memory_service and user_id:
            recent_msgs = self.external_memory_service.get_recent_messages(user_id, limit=6)
            if recent_msgs:
                for msg in recent_msgs:
                    if msg["role"] == "user":
                        history_messages.append(HumanMessage(content=msg["content"]))
                    else:
                        history_messages.append(AIMessage(content=msg["content"]))
        
        structured_llm = self.llm.with_structured_output(PriceExtraction)
        
        # Get user's language for generating response in correct language
        user_lang_code = state.get("user_language", "en")
        lang_map = {
            "en": "English",
            "ro": "Romanian (Română)",
            "ru": "Russian (Русский)",
            "es": "Spanish (Español)",
            "fr": "French (Français)",
            "de": "German (Deutsch)",
            "it": "Italian (Italiano)",
            "pt": "Portuguese (Português)",
            "pl": "Polish (Polski)",
            "uk": "Ukrainian (Українська)",
            "hu": "Hungarian (Magyar)"
        }
        target_lang_name = lang_map.get(user_lang_code, "English")
        
        system_prompt = f"""You are an expert at extracting price and budget information from accommodation queries.

Analyze the user's query and conversation history to extract budget information.

RULES:
1. Set has_price_mention=True if the query mentions: price, cost, budget, expensive, cheap, affordable, or any monetary amounts.
2. If specific price range is mentioned (e.g., "100-200 lei", "under 150", "maximum 300"):
   - Extract min_price and max_price
   - Leave clarification_question as None
3. If price is mentioned without specific amounts (e.g., "expensive hotel", "cheap place", "affordable accommodation"):
   - Set has_price_mention=True
   - Leave min_price and max_price as None
   - Generate a friendly clarification_question asking "What is your specific budget per night (in lei)?" in {target_lang_name}
4. If no price mention at all, set has_price_mention=False and leave everything else as None.

IMPORTANT: If you generate a clarification question, it MUST be in {target_lang_name} and should explicitly ask for a NUMERIC budget range or maximum price.

Examples:
- "expensive hotels in Cluj" → has_price_mention=True, clarification_question="What is your specific budget for accommodation per night (in lei)?"
- "hotels 150-250 lei" → has_price_mention=True, min_price=150, max_price=250
- "under 200 lei" → has_price_mention=True, min_price=None, max_price=200
- "hotels in Brasov" → has_price_mention=False

"""
        
        messages = [SystemMessage(content=system_prompt)] + history_messages + [HumanMessage(content=question)]
        prompt = ChatPromptTemplate.from_messages(messages)
        
        chain = prompt | structured_llm
        result = await chain.ainvoke({"question": question}, config={"callbacks": [TokenUsageLogger("price_extractor")]})
        
        # HEURISTIC OVERRIDE: Check for explicit price keywords
        price_keywords = ["price", "cost", "budget", "expensive", "cheap", "affordable", "economy", "luxury", "value", "deal", "lei", "ron", "euro", "eur", "usd", "$", "€"]
        keyword_detected = any(kw in question.lower() for kw in price_keywords)
        
        effective_has_price = result.has_price_mention or keyword_detected
        logger.info(f"Price extraction: LLM={result.has_price_mention}, Keyword={keyword_detected} -> Effective={effective_has_price}")

        # If no price mention, proceed without price filtering
        if not effective_has_price:
            logger.info("No price mention detected. Proceeding without price filter.")
            return {"filters": {}}
        
        # Get tools to save/retrieve budget
        search_tools = get_functional_tools(user_id=user_id, trip_id=state.get("trip_id"), external_memory_service=self.external_memory_service)
        save_budget_tool = next((t for t in search_tools if t.name == "save_user_budget"), None)
        
        # If specific price range was extracted, save it and use it
        if result.min_price is not None or result.max_price is not None:
            min_p = result.min_price or 0
            max_p = result.max_price or 999999
            budget_str = f"{min_p}-{max_p}"
            
            logger.info(f"Budget extracted from query: {budget_str}. Saving to user profile.")
            if save_budget_tool:
                try:
                    await save_budget_tool.ainvoke({"budget": budget_str})
                except Exception as e:
                    logger.error(f"Failed to save budget: {e}")
            
            return {
                "filters": {
                    "min_price": min_p,
                    "max_price": max_p
                }
            }
        
        # Price mentioned but no specific range - check if user has saved budget
        if self.external_memory_service and user_id:
            saved_budget = self.external_memory_service.get_user_budget(user_id)
            if saved_budget:
                logger.info(f"Using saved budget from user profile: {saved_budget}")
                try:
                    # Parse saved budget (format: "min-max")
                    parts = saved_budget.split("-")
                    if len(parts) == 2:
                        min_p = int(parts[0])
                        max_p = int(parts[1])
                        return {
                            "filters": {
                                "min_price": min_p,
                                "max_price": max_p
                            }
                        }
                except Exception as e:
                    logger.error(f"Error parsing saved budget: {e}")
        
        # No specific range and no saved budget -> flag for clarification node
        logger.info("Price mentioned but no specific budget found. Flagging for clarification.")
        return {"price_clarification_needed": True}

    def budget_clarification_node(self, state: GraphState):
        """
        Generates a specific budget clarification question in the user's language.
        Used when price intent is detected but no specific range is found.
        """
        user_lang_code = state.get("user_language", "en")
        
        fallback_questions = {
            "en": "To help me find the best options for you, could you tell me your budget for accommodation per night (in lei)?",
            "ro": "Pentru a te ajuta să găsesc cele mai bune opțiuni, ai putea să-mi spui care este bugetul tău pentru cazare pe noapte (în lei)?",
            "ru": "Чтобы помочь мне найти лучшие варианты для вас, не могли бы вы назвать свой бюджет на проживание за ночь (в леях)?",
            "es": "Para ayudarte a encontrar las mejores opciones, ¿podrías indicarme tu presupuesto para el alojamiento por noche (en lei)?",
            "fr": "Pour m'aider à trouver les meilleures options pour vous, pourriez-vous me donner votre budget d'hébergement par nuit (en lei) ?",
            "de": "Damit ich die besten Optionen für Sie finden kann, könnten Sie mir bitte Ihr Budget pro Nacht für die Unterkunft (in Lei) nennen?",
            "it": "Per aiutarmi a trovare le migliori opzioni per te, potresti dirmi qual è il tuo budget per l'alloggio per notte (in lei)?",
            "pt": "Para me ajudar a encontrar as melhores opções para você, poderia me dizer qual é o seu orçamento para acomodação por noite (em lei)?",
            "pl": "Aby pomóc mi znaleźć najlepsze opcje dla Ciebie, czy mógłbyś podać swój budżet na zakwaterowanie za noc (w lei)?",
            "uk": "Щоб допомогти мені знайти найкращі варіанти для вас, чи могли б ви повідомити свій бюджет на проживання за ніч (у леях)?",
            "hu": "Hogy segítsek megtalálni a legjobb lehetőségeket, meg tudnád mondani, mennyi a költségkereted éjszakánkénti szállásra (lejben)?"
        }
        
        question = fallback_questions.get(user_lang_code, fallback_questions["en"])
        logger.info(f"Generating budget clarification question: {question}")
        
        return {"generation": question}

    def route_after_price_extractor(self, state: GraphState):
        """
        Routes after price extractor.
        If clarification needed, go to budget_clarification node.
        Otherwise proceed to accommodation retriever.
        """
        if state.get("price_clarification_needed"):
            return "budget_clarification"
        if state.get("generation"):
             return "out_of_scope_translation"
        return "accommodation_retriever"

    async def festival_filter_node(self, state: GraphState):
        """
        Extracts festival preferences and asks for clarification if needed.
        """
        question = state.get("translated_question") or state["question"]
        user_id = state.get("user_id")
        logger.info(f"---FESTIVAL FILTER NODE--- query: {question}")

        # Get history context
        # Get history context
        history_messages = []
        if self.external_memory_service and user_id:
            recent_msgs = self.external_memory_service.get_recent_messages(user_id, limit=10)
            if recent_msgs:
                for msg in recent_msgs:
                    if msg["role"] == "user":
                        history_messages.append(HumanMessage(content=msg["content"]))
                    else:
                        history_messages.append(AIMessage(content=msg["content"]))

        structured_llm = self.llm.with_structured_output(FestivalQuery)
        
        system_prompt = """You are an expert travel assistant specializing in festivals.
Analyze the user's query and conversation history to extract festival preferences.

Task:
1. Extract 'festival_type' (e.g., music, food, art) and 'date_context' (e.g., specific dates, months, seasons).
3. If the user is vague (e.g., "are there festivals?", "what's happening?", "I don't know, surprise me"), DO NOT ask for clarification.
    - Instead, set 'date_context' to "upcoming" (or the specific month if mentioned).
    - Leave 'festival_type' as None or "general" to find the best matches.
4. ONLY populate 'clarification_question' if the input is completely nonsensical.
    - GOAL: Show results immediately rather than asking 20 questions.
"""
        messages = [SystemMessage(content=system_prompt)] + history_messages + [HumanMessage(content=question)]
        prompt = ChatPromptTemplate.from_messages(messages)
        
        chain = prompt | structured_llm
        result = await chain.ainvoke({"question": question}, config={"callbacks": [TokenUsageLogger("festival_filter")]})

        
        logger.info(f"Festival Filter result: type={result.festival_type}, date={result.date_context}, clarify={result.clarification_question}")

        if result.clarification_question:
            return {"generation": result.clarification_question}
        
        return {
            "filters": {
                "festival_type": result.festival_type,
                "date_context": result.date_context,
                "user_preferences": result.user_preferences
            }
        }

    async def accommodation_retriever_node(self, state: GraphState):
        """
        Retriever strictly for accommodations with amenity filtering.
        """
        question = state.get("translated_question") or state["question"]
        filters = state["filters"]
        amenities = filters.get("amenities", [])
        
        logger.info(f"---ACCOMMODATION RETRIEVER--- amenities: {amenities}")

        # Query Broadening for Navigation/Transport (Safety Fallback)
        search_query = question
        
        system_prompt = """You are an expert at optimizing search queries for a vector database.
The user is asking a question related to 'accommodations'.

CRITICAL DIRECTION:
- Output a BROAD single-line string of keywords, synonyms, and related travel terms IN ENGLISH.
- DO NOT use bullet points or newlines.
- TRANSLATE the query to English if it is in another language.
- Translate any Romanian terms into their English equivalents.
- Include broader region names if a specific town is mentioned.
- Focus on vibe, location, and property type.
- Example: "Guesthouse Borsa Maramures mountain view traditional cabin rustic nature stay"
- If the user asks about prices, include price-related keywords to retrieve pricing information.
- ONLY IF the user explicitly asks for 'breakfast included' or 'free breakfast', then append: "breakfast included", "free breakfast", "rates include breakfast".
- Example: "Guesthouse Borsa Maramures mountain view traditional cabin rustic nature stay"

If the user is asking for DIRECTIONS/transport:
- Include: bus schedule, taxi, transport, route.
"""

        # Get recent messages from memory for context
        history_messages = []
        user_id = state.get("user_id")
        if self.external_memory_service and user_id:
            recent_msgs = self.external_memory_service.get_recent_messages(user_id, limit=6)
            if recent_msgs:
                for msg in recent_msgs:
                    if msg["role"] == "user":
                        history_messages.append(HumanMessage(content=msg["content"]))
                    else:
                        history_messages.append(AIMessage(content=msg["content"]))

        broaden_prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt)] + history_messages + [("human", "{question}")]
        )
        chain = broaden_prompt | self.llm.with_config({"tags": ["retrieval_query_gen"], "run_name": "retrieval_query_gen"})
        response = await chain.ainvoke({"question": question}, config={"callbacks": [TokenUsageLogger("accommodation_broadening")]})
        search_query = response.content
        logger.info(f"Accommodation query broadened/detailed: '{search_query}'")

        # 1. Embed query (using detailed query)
        query_embedding = await self.embedding_service.get_embedding(search_query)
        sparse_embedding = self.embedding_service.get_sparse_embedding(search_query)

        # 2. Build Strict Filters
        # FIXED: Schema uses "category.type". Value matches the endpoint name "accommodations" (plural) set by SyncService.
        must_conditions = [
             FieldCondition(key="category.type", match=MatchValue(value="accommodations"))
        ]
        must_not_conditions = []

        # Filter out seen items
        if self.external_memory_service and user_id:
            seen_items = self.external_memory_service.get_seen_items(user_id)
            if seen_items:
                logger.info(f"Filtering out {len(seen_items)} seen items")
                must_not_conditions.append(
                    FieldCondition(key="title", match=MatchAny(any=seen_items))
                )

        # Add Location Filter if present
        location = filters.get("location")
        if location:
             # DIACRITIC FIX: Normalize common variations for Borșa
            if location.lower() in ["borsa", "borșa"]:
                location = "Borșa"
             
            logger.info(f"Applying strict location filter: {location}")
            must_conditions.append(
                FieldCondition(key="city.name", match=MatchValue(value=location))
            )

        if amenities:
            for amenity in amenities:
                # Assuming amenities are stored in a list field named "amenities"
                must_conditions.append(
                    FieldCondition(key="amenities", match=MatchValue(value=amenity))
                )

        # Add Price Filter if present
        min_price = filters.get("min_price")
        max_price = filters.get("max_price")
        if min_price is not None or max_price is not None:
            logger.info(f"Applying price filter: min={min_price}, max={max_price}")
            # Build range condition for price field
            range_kwargs = {}
            if min_price is not None:
                range_kwargs["gte"] = min_price
            if max_price is not None:
                range_kwargs["lte"] = max_price
            
            must_conditions.append(
                FieldCondition(key="price", range=Range(**range_kwargs))
            )

        search_filters = Filter(must=must_conditions, must_not=must_not_conditions if must_not_conditions else None)

        logger.info(f"Accommodation filters: {search_filters}")

        # 3. Search
        # Extract strict keywords from state (populated by keyword_extractor_node)
        kw_filter = state.get("keywords")
        if kw_filter:
            logger.info(f"Applying strict keyword filter: '{kw_filter}'")

        results = self.vector_store.search(
            query_dense_embedding=query_embedding,
            query_sparse_embedding=sparse_embedding,
            top_k=25, # Fetch more to allow grouping/diversity
            filters=search_filters,
            keyword_filter=kw_filter
        )
        
        logger.info(f"Accommodation search found {len(results)} raw chunks.")

        # Group by parent_id to ensure diversity
        grouped_docs = {}
        unique_results = []
        
        for doc, score in results:
            # Use parent_id if available (chunks), otherwise doc.id
            parent_id = doc.metadata.get("parent_id") or doc.id
            
            if parent_id not in grouped_docs:
                grouped_docs[parent_id] = []
                unique_results.append((doc, score, parent_id))
            
            # Avoid exact duplicate content
            if doc.content not in [d.content for d in grouped_docs[parent_id]]:
                grouped_docs[parent_id].append(doc)
        
        # 4. Format outputs (Top 5 Unique Accommodations)
        docs = []
        for base_doc, score, pid in unique_results[:5]:
            # Merge content from all relevant chunks
            all_chunks = grouped_docs[pid]
            
            # Join with separator
            merged_content = "\n\n---\n\n".join([d.content for d in all_chunks])
            
            # Update content of the representative doc
            base_doc.content = merged_content
            
            logger.info(f"Accommodation result ({pid}): {len(all_chunks)} chunks merged.")
            docs.append(self._format_doc_with_metadata(base_doc))

        return {"documents": docs}


    # ... inside AgenticRAGService class ...

    async def festival_retriever_node(self, state: GraphState):
        """
        Retriever for festivals with date and type analysis.
        """
        question = state.get("translated_question") or state["question"]
        filters = state.get("filters", {})
        
        festival_type = filters.get("festival_type")
        date_context = filters.get("date_context")
        user_preferences = filters.get("user_preferences")
        
        logger.info(f"---FESTIVAL RETRIEVER--- type: {festival_type}, date: {date_context}")

        current_date_str = datetime.now().strftime("%Y-%m-%d")
        
        # LLM Reasoning for Query + Dates
        system_prompt = f"""You are a search expert for a festival database.
Current Date: {current_date_str}

Task:
1. Generate an optimized English search query based on the user's question, festival type, and preferences.
    - Include specific keywords like "music", "food", "art" if applicable.
    - If the user asks about prices or ticket costs, include price-related keywords.
    - User Preferences: {user_preferences}
    
2. Calculate the Unix Timestamp range (start_date_ts, end_date_ts) based on '{date_context}'.
    - If a month/season is mentioned (e.g., "August"), ALWAYS calculate the full range (1st-31st).
    - If vague or uncertain, set start_date_ts = Current Time and end_date_ts = None (Search All Future).
    - DO NOT return empty timestamps if a time hint exists.
    - We have 'startDate' and 'endDate' in the DB, so providing a broad window is safe.
        """
        
        # Get recent messages from memory for context
        history_messages = []
        user_id = state.get("user_id")
        if self.external_memory_service and user_id:
            recent_msgs = self.external_memory_service.get_recent_messages(user_id, limit=6)
            if recent_msgs:
                for msg in recent_msgs:
                    if msg["role"] == "user":
                        history_messages.append(HumanMessage(content=msg["content"]))
                    else:
                        history_messages.append(AIMessage(content=msg["content"]))

        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt)] + history_messages + [("human", "{question}")]
        )
        
        llm_with_tools = self.llm.with_structured_output(FestivalSearchParams)
        chain = prompt | llm_with_tools
        
        search_params = await chain.ainvoke({"question": question}, config={"callbacks": [TokenUsageLogger("festival_retriever")]})
        logger.info(f"Festival Search Params: {search_params}")

        # 1. Embed Query
        query_embedding = await self.embedding_service.get_embedding(search_params.search_query)
        sparse_embedding = self.embedding_service.get_sparse_embedding(search_params.search_query)

        # 2. Build Filters
        must_conditions = [
            FieldCondition(key="category.type", match=MatchValue(value="festivals"))
        ]
        
        # Date Filter (Range)
        # We want festivals that OVERLAP with the window or start within it.
        # Simple logic: Festival Start Date <= Window End AND Festival End Date >= Window Start
        # But Qdrant Range is usually simpler: "startDate_ts" >= Window Start (if we just want upcoming in that window)
        
        if search_params.start_date_ts:
             # Filter: endDate_ts >= search_window_start (Overlap logic: ends after window starts)
             must_conditions.append(
                 FieldCondition(key="endDate_ts", range=Range(gte=search_params.start_date_ts))
             )
        
        if search_params.end_date_ts:
             # Filter: startDate_ts <= search_window_end (Overlap logic: starts before window ends)
             must_conditions.append(
                 FieldCondition(key="startDate_ts", range=Range(lte=search_params.end_date_ts))
             )

        must_not_conditions = []
        if self.external_memory_service and user_id:
            seen_items = self.external_memory_service.get_seen_items(user_id)
            if seen_items:
                must_not_conditions.append(
                    FieldCondition(key="title", match=MatchAny(any=seen_items))
                )

        search_filters = Filter(must=must_conditions, must_not=must_not_conditions if must_not_conditions else None)
        logger.info(f"Festival filters: {search_filters}")

        # 3. Search
        kw_filter = state.get("keywords")
        
        results = self.vector_store.search(
            query_dense_embedding=query_embedding,
            query_sparse_embedding=sparse_embedding,
            top_k=5,
            filters=search_filters,
            keyword_filter=kw_filter
        )

        # 4. Format with metadata
        docs = []
        for doc, _ in results:
             docs.append(self._format_doc_with_metadata(doc))

        return {"documents": docs}

    async def generic_retriever_node(self, state: GraphState):
        """
        Retriever for other categories with dynamic filtering.
        """
        question = state.get("translated_question") or state["question"]
        intent = state["intent"]
        user_id = state.get("user_id")
        
        logger.info(f"---GENERIC RETRIEVER--- intent: {intent}")

        # Intent matches endpoint name exactly (plural), so we use it directly.
        mapped_type = intent
        logger.info(f"Using intent '{intent}' as category.type")

        # Get recent messages from memory for context
        history_messages = []
        if self.external_memory_service and user_id:
            recent_msgs = self.external_memory_service.get_recent_messages(user_id, limit=6)
            if recent_msgs:
                for msg in recent_msgs:
                    if msg["role"] == "user":
                        history_messages.append(HumanMessage(content=msg["content"]))
                    else:
                        history_messages.append(AIMessage(content=msg["content"]))

        # Query Broadening using LLM
        # We use the LLM to rewrite the query to be more effective for vector retrieval
        # specifically for the target intent.
        
        system_prompt = f"""You are an expert at optimizing search queries for a vector database.
The user is asking a question related to '{intent}'.

CRITICAL DIRECTION:
- Output ONLY a list of keywords, synonyms, and related travel terms relevant to '{intent}' IN ENGLISH.
- TRANSLATE the query to English if it is in another language.
- Translate any Romanian terms into their English equivalents.
- Do NOT use first-person language ("I want", "I am looking for").
- Do NOT add new constraints (e.g., don't add "vegan" unless the user asked for it).
- USE CONTEXT: If the query refers to "there", "that city", or implies a location from previous messages, INCLUDE that location in the keywords.
- If the user asks about prices or costs, include price-related keywords to retrieve pricing information.
- Example: "Authentic restaurants Borsa traditional food local cuisine"

If the intent is 'helpful-things' or asking about transport:
- Include: bus, taxi, schedule, route, transport.
- If the user wants to BUY something, include related terms (shop, store, market, price, where to buy, souvenirs, groceries).
        """
        
        messages = [SystemMessage(content=system_prompt)] + history_messages + [HumanMessage(content=question)]
        broaden_prompt = ChatPromptTemplate.from_messages(messages)
        
        chain = broaden_prompt | self.llm.with_config({"tags": ["retrieval_query_gen"], "run_name": "retrieval_query_gen"})
        response = await chain.ainvoke({"question": question}, config={"callbacks": [TokenUsageLogger("generic_broadening")]})
        search_query = response.content
        logger.info(f"Original query: '{question}' -> Broadened query: '{search_query}'")

        # 1. Embed query (using augmented query)
        query_embedding = await self.embedding_service.get_embedding(search_query)
        sparse_embedding = self.embedding_service.get_sparse_embedding(search_query)

        # 2. Dynamic Filter
        # FIXED: Schema uses "category.type"
        # 2. Dynamic Filter
        # FIXED: Schema uses "category.type"
        must_conditions = [
            FieldCondition(key="category.type", match=MatchValue(value=mapped_type))
        ]
        
        must_not_conditions = []
        if self.external_memory_service and user_id:
            seen_items = self.external_memory_service.get_seen_items(user_id)
            if seen_items:
                must_not_conditions.append(
                    FieldCondition(key="title", match=MatchAny(any=seen_items))
                )
        
        search_filters = Filter(
            must=must_conditions,
            must_not=must_not_conditions if must_not_conditions else None
        )
        logger.info(f"Generic filters: {search_filters}")

        # 3. Search
        kw_filter = state.get("keywords")
        
        results = self.vector_store.search(
            query_dense_embedding=query_embedding,
            query_sparse_embedding=sparse_embedding,
            top_k=10, # Increased from 5 to ensure transport docs are found
            filters=search_filters,
            keyword_filter=kw_filter
        )
        
        logger.info(f"Generic search found {len(results)} results")

        # 4. Format outputs with metadata
        docs = []
        for doc, _ in results:
             content_snippet = (doc.content[:100] + '...') if len(doc.content) > 100 else doc.content
             logger.info(f"Retrieved doc: {content_snippet}")
             docs.append(self._format_doc_with_metadata(doc))

        return {"documents": docs}

    async def fallback_search_node(self, state: GraphState):
        """
        Broad fallback search when specific retrievers fail.
        Searches without category constraints.
        """
        question = state.get("translated_question") or state["question"]
        user_id = state.get("user_id")
        logger.info(f"---FALLBACK SEARCH NODE--- query: {question}")

        # Get recent messages from memory for context
        history_messages = []
        if self.external_memory_service and user_id:
             recent_msgs = self.external_memory_service.get_recent_messages(user_id, limit=4)
             if recent_msgs:
                 for msg in recent_msgs:
                     if msg["role"] == "user":
                         history_messages.append(HumanMessage(content=msg["content"]))
                     else:
                         history_messages.append(AIMessage(content=msg["content"]))

        system_prompt = """You are an expert at broad search optimization.
The specific search failed. We need to cast a wider net.

Task:
- Generate a broad, general search query based on the user's intent.
- Focus on the core topic (place, activity, or concept).
- Remove specific constraints that might be too strict.
- Example: "Hotels in Cluj with pool" -> "Accommodation Cluj pool"
"""

        messages = [SystemMessage(content=system_prompt)] + history_messages + [HumanMessage(content=question)]
        broaden_prompt = ChatPromptTemplate.from_messages(messages)
        
        chain = broaden_prompt | self.llm.with_config({"tags": ["fallback_query_gen"], "run_name": "fallback_query_gen"})
        response = await chain.ainvoke({"question": question}, config={"callbacks": [TokenUsageLogger("fallback_broadening")]})
        search_query = response.content
        logger.info(f"Fallback query: '{search_query}'")

        # 1. Embed query
        query_embedding = await self.embedding_service.get_embedding(search_query)
        sparse_embedding = self.embedding_service.get_sparse_embedding(search_query)

        # 2. Search WITHOUT category filters
        kw_filter = state.get("keywords")
        
        must_not_conditions = []
        if self.external_memory_service and user_id:
            seen_items = self.external_memory_service.get_seen_items(user_id)
            if seen_items:
                must_not_conditions.append(
                    FieldCondition(key="title", match=MatchAny(any=seen_items))
                )

        search_filters = None
        if must_not_conditions:
            search_filters = Filter(must_not=must_not_conditions)

        results = self.vector_store.search(
            query_dense_embedding=query_embedding,
            query_sparse_embedding=sparse_embedding,
            top_k=5, 
            filters=search_filters,  # Fallback search + exclusion
            keyword_filter=kw_filter
        )
        
        logger.info(f"Fallback search found {len(results)} results")

        # 3. Format outputs with metadata
        docs = []
        for doc, _ in results:
             content_snippet = (doc.content[:100] + '...') if len(doc.content) > 100 else doc.content
             logger.info(f"Fallback retrieved doc: {content_snippet}")
             docs.append(self._format_doc_with_metadata(doc))

        return {"documents": docs}

    async def tool_node(self, state: GraphState):
        """
        Executes functional tools (Weather, Itinerary, User Info).
        """
        question = state.get("translated_question") or state["question"]
        intent = state["intent"]
        user_id = state.get("user_id")
        trip_id = state.get("trip_id")
        location = state["filters"].get("location") if state.get("filters") else None
        
        logger.info(f"---TOOL NODE--- intent: {intent}, location: {location}")

        # 1. Get functional tools
        tools = get_functional_tools(
            user_id=user_id,
            trip_id=trip_id,
            itinerary_service=self.itinerary_service,
            external_memory_service=self.external_memory_service
        )
        
        # 2. Bind tools to LLM
        llm_with_tools = self.llm.bind_tools(tools)
        
        # 3. Augment prompt with location context if needed
        # (The tool might need explicit location if not in query, but LLM usually handles it)
        content = question
        if location and "weather" in intent:
             content += f" (Location context: {location})"

        # 4. Get recent messages from memory for context resolution (e.g. "weather there")
        history = []
        if self.external_memory_service and user_id:
            recent_msgs = self.external_memory_service.get_recent_messages(user_id, limit=6)
            for msg in recent_msgs:
                if msg["role"] == "user":
                    history.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    history.append(AIMessage(content=msg["content"]))

        messages = [SystemMessage(content="You are a helpful assistant. Use the provided tools to answer the user's request. Rely on conversation history to resolve implicit references (e.g., 'there', 'that city').")] + history + [HumanMessage(content=content)]

        # 5. Invoke
        # We invoke the LLM w/ tools. It should decide which tool to call.
        response = await llm_with_tools.ainvoke(messages)
        
        tool_output = "No action taken."
        
        # 6. Execute Tool Call (Manual Execution for simplicity in this node)
        # In a full ReAct loop LangGraph manages this, but here we just want single execution.
        if response.tool_calls:
            tool_call = response.tool_calls[0] # Take the first one
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            logger.info(f"Tool selected: {tool_name}, args: {tool_args}")
            
            # Find the tool implementation
            selected_tool = next((t for t in tools if t.name == tool_name), None)
            
            if selected_tool:
                # Await if async
                try:
                    # LangChain tools can be sync or async. 
                    # structured tools from @tool are callables.
                    if selected_tool.coroutine:
                        tool_result = await selected_tool.ainvoke(tool_args)
                    else:
                        tool_result = selected_tool.invoke(tool_args)
                    
                    tool_output = str(tool_result)
                except Exception as e:
                     logger.error(f"Tool execution failed: {e}")
                     tool_output = f"Error executing tool {tool_name}: {str(e)}"
            else:
                 tool_output = f"Tool {tool_name} not found."
        else:
             tool_output = response.content # Fallback if no tool was called (e.g. LLM just chitchatted)

        # Return as "documents" so generator can use it
        return {"documents": [f"Tool Output ({intent}): {tool_output}"]}

    # --- Nodes ---

    async def generator_node(self, state: GraphState):
        """
        Generates the final response to the user.
        """
        question = state.get("translated_question") or state["question"]
        user_id = state.get("user_id")
        documents = state.get("documents", [])
        
        logger.info(f"---GENERATOR NODE--- query: {question}")
        
        context_text = "\n\n".join(documents) if documents else "No specific documents found."
        
        current_date = datetime.now().strftime("%Y-%m-%d")

        # Get context summaries from middleware
        memory_summary, itinerary_summary = "", ""
        if self.context_middleware and user_id:
            memory_summary, itinerary_summary = self.context_middleware.ensure_context_injection(user_id, state.get("trip_id"))


        user_lang_code = state.get("user_language", "en")
        lang_map = {
            "en": "English",
            "ro": "Romanian (Română)",
            "ru": "Russian (Русский)",
            "es": "Spanish (Español)",
            "fr": "French (Français)",
            "de": "German (Deutsch)",
            "it": "Italian (Italiano)",
            "pt": "Portuguese (Português)",
            "pl": "Polish (Polski)",
            "uk": "Ukrainian (Українська)",
            "hu": "Hungarian (Magyar)"
        }
        target_lang_name = lang_map.get(user_lang_code, "English")

        system_prompt = f"""
### IDENTITY & ROLE
- **Name:** Vlad-AI
- **Role:** A friendly, savvy local friend helping travelers in Romania.
- **Tone:** Casual, warm, and concise (like a helpful text message).
- **Current Date:** {current_date}

### LANGUAGE & COMMUNICATION (CRITICAL - READ CAREFULLY)
- **Target Language:** {target_lang_name} ({user_lang_code})
- **CRITICAL RULE:** Your response MUST be 100% in {target_lang_name}
- **STRICT ENFORCEMENT:** Do NOT use English unless the target language is English
- **DOUBLE CHECK:** Before sending your response, verify EVERY sentence is in {target_lang_name}
- **EXAMPLE:** If target is Romanian (ro), respond ONLY in Romanian (like: "Pensiunea Ioana oferă...")
- **EXAMPLE:** If target is English (en), respond ONLY in English (like: "Pensiunea Ioana offers...")
- **NO MIXING:** Do NOT mix {target_lang_name} with English or other languages

### CORE OBJECTIVE
Synthesize the provided **Database Search Results** into a helpful, concise answer.

### CRITICAL RULES:
1. **DATABASE IS TRUTH:** You have **NO** outside knowledge. Base your answer **EXCLUSIVELY** on the provided search results.
2. **NO ROBOTIC LISTS:** Do NOT use numbered lists or bullet points. Write naturally like a friend texting.
3. **RECOMMEND ONE:** Focus on the SINGLE best recommendation.
4. **MAX 2 TOTAL:** If the request is broad, you can mention a second option. NEVER more than 2 places in one message.
5. **COMPLETENESS:** Include specific constraints (e.g., "Cash only", "Closed Mondays") if present in the data.

### PRIORITIZATION RULES
- If the user asks for "breakfast included", prioritize accommodations that EXPLICITLY state "breakfast included in the price" or "free breakfast" in their description over those that just offer breakfast or are "Bed & Breakfast" type but don't explicitly state inclusion.
- If multiple options exist, recommend the one with the most explicit confirmation of the user's specific constraint.

### STYLE & FORMAT
- **Ending:** Always end with ONE short follow-up question (e.g., "Do you need help with anything else?").

### CONTEXT
- **User Memory:** {memory_summary}
- **Current Itinerary:** {itinerary_summary}
- **Database Search Results:**
{context_text}

### HANDLING MISSING RESULTS
- If **Database Search Results** says "No specific documents found...":
    - This likely means the user has seen all available options for their specific request.
    - You **MUST** texturally say: "I don't have any other recommendations for that." (or similar meaning in {target_lang_name}).
    - Do NOT make up places.
    - Ask if they want to broaden their search (e.g. "Want to try a different city or category?").

### OUTPUT INSTRUCTION
- Keep the response SHORT and conversational, like a text to a friend.
"""

        # Get recent messages from memory
        history = []
        if self.external_memory_service and user_id:
            recent_msgs = self.external_memory_service.get_recent_messages(user_id, limit=6)
            for msg in recent_msgs:
                if msg["role"] == "user":
                    history.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    history.append(AIMessage(content=msg["content"]))

        # If history is somehow empty (it shouldn't be as we just added the user message), 
        # fallback to the current question
        if not history:
             history = [HumanMessage(content=question)]
        else:
            # Check if the last message in history is the same as the current question
            # to avoid duplication, or if we should just trust the history.
            # Usually, the history will contain the latest message because it's added in query_stream.
            pass

        messages = [SystemMessage(content=system_prompt)] + history
        
        # Use "final_generator" tag to enable streaming in query_stream
        # We NO LONGER bind tools here to save tokens. Recommendation extraction is done in background.
        response = await self.llm.ainvoke(
            messages, 
            config={
                "tags": ["final_generator"],
                "callbacks": [TokenUsageLogger("generator")]
            }
        )
        response_text = response.content.strip()
        
        # CRITICAL: Force-translate if generator responded in wrong language
        if user_lang_code != "en":
            # Check if response is in English (when it shouldn't be)
            english_indicators = [
                ' the ', ' and ', ' is ', ' are ', ' was ', ' were ',
                ' offers ', ' provides ', ' located ', ' available ',
                ' has ', ' have ', ' with ', ' for ', ' at '
            ]
            
            response_lower = f' {response_text.lower()} '
            english_count = sum(1 for indicator in english_indicators if indicator in response_lower)
            
            # If we detect 4+ English indicators, generator failed to follow instructions
            if english_count >= 4:
                logger.warning(
                    f"⚠️ Generator responded in English instead of {user_lang_code}. "
                    f"Detected {english_count} English indicators. Force-translating..."
                )
                
                try:
                    # Force translate from English to target language
                    response_text = await self._translate_from_english(
                        response_text, 
                        user_lang_code
                    )
                    
                    logger.info(f"✅ Successfully force-translated response to {user_lang_code}")
                    
                except Exception as e:
                    logger.error(f"❌ Force-translation failed: {e}")
                    # Keep the English response as fallback
        
        return {"generation": response_text}

    # --- Conditional Edges ---

    def route_to_retriever(self, state: GraphState):
        """
        Routes to the city determination (context enrichment) node for most intents.
        """
        intent = state["intent"]
        if intent == "out_of_scope":
            return "out_of_scope_translation"  # Route to translation node
        elif intent == "price_inquiry":
            return "price_retriever"
        
        # All substantive intents go through context enrichment first
        return "city_determination"

    def route_after_city_determination(self, state: GraphState):
        """
        Routes after city determination.
        """
        if state.get("generation"):
            return "end"
            
        intent = state["intent"]
        if intent == "accommodations":
            return "amenities_extractor"
        elif intent == "festivals":
            return "festival_filter"
    def route_after_city_determination(self, state: GraphState):
        """
        Routes to keyword_extractor for substantive intents.
        """
        if state.get("generation"):
            return "end"
            
        intent = state["intent"]
        if intent in ["accommodations", "festivals", "restaurants", "attractions", "helpful-things"]:
            return "keyword_extractor"
        elif intent in ["weather", "trip_planning", "user_context"]:
            return "tool_node"
        else:
             # Should practically not happen if classifier works well, but safety:
            return "end"
            
    def route_after_keyword_extractor(self, state: GraphState):
        """
        Routes after keywords have been extracted.
        """
        intent = state["intent"]
        if intent == "accommodations":
            return "amenities_extractor"
        elif intent == "festivals":
            return "festival_filter"
        else:
            return "generic_retriever"

    async def route_check_retriever_results(self, state: GraphState) -> Literal["generator", "fallback_search"]:
        """
        Check if documents were found. If not, route to fallback search.
        """
        documents = state.get("documents", [])
        if not documents:
            logger.info("Specific retriever returned no results. Routing to fallback_search.")
            return "fallback_search"
        return "generator"

    def route_after_festival_filter(self, state: GraphState):
        """
        Routes after festival filter. If generation is present (clarification), go to END.
        """
        if state.get("generation"):
            return "end"
        return "festival_retriever"

    def route_after_price_retriever(self, state: GraphState):
        """
        Routes after price retriever. 
        If we generated a direct response, go to END.
        Otherwise (no results), go to fallback search.
        """
        if state.get("generation"):
            return "end"
        if state.get("documents"):
            return "generator"
        return "fallback_search"

    async def price_retriever_node(self, state: GraphState):
        """
        Retriever specifically for price inquiries.
        """
        question = state.get("translated_question") or state["question"]
        user_id = state.get("user_id")
        logger.info(f"---PRICE RETRIEVER--- query: {question}")
        
        # 1. Identify the target item (explicit in query or implicit in history)
        history_messages = []
        if self.external_memory_service and user_id:
            recent_msgs = self.external_memory_service.get_recent_messages(user_id, limit=6)
            if recent_msgs:
                for msg in recent_msgs:
                    if msg["role"] == "user":
                        history_messages.append(HumanMessage(content=msg["content"]))
                    else:
                        history_messages.append(AIMessage(content=msg["content"]))

        # Get last seen/recommended item for context
        last_seen_context = "None"
        if self.external_memory_service and user_id:
            seen_items = self.external_memory_service.get_seen_items(user_id)
            if seen_items:
                last_seen_context = seen_items[-1]

        system_prompt = f"""You are an expert at identifying the specific item a user is asking about.
The user is asking about PRICE/COST. 
NOTE: prices are usually "flexible" or "upon request". 
Your goal is to find the EXACT ITEM NAME so we can retrieve its policy/description.

Context:
- Last Recommended/Seen Item: {last_seen_context}

Logic:
1. If the item is named (e.g., "price of Hotel Alpin"), output "Hotel Alpin".
2. If implicit (e.g., "how much is it?"), use the LAST SEEN item ('{last_seen_context}') if available.
3. If not in last seen, check conversation history.
4. If the user asks for "cheap places", output "economic accommodations".

Output Format: Just the search query string (best identifier for the item).
"""
        messages = [SystemMessage(content=system_prompt)] + history_messages + [HumanMessage(content=question)]
        response = await self.llm.ainvoke(messages)
        target_item_query = response.content.strip()
        
        logger.info(f"Price target identified: '{target_item_query}'")
        
        # 2. Search for this specific item
        # Generate embeddings first
        query_embedding = await self.embedding_service.get_embedding(target_item_query)
        sparse_embedding = self.embedding_service.get_sparse_embedding(target_item_query)

        # Call the correct search method
        # For price retrieval, we can use the item name itself as a keyword filter if it looks like a proper name
        # But let's stick to quotes or just reliance on dense/sparse for now. 
        # Actually, using the target item as a keyword filter is smart!
        
        results = self.vector_store.search(
            query_dense_embedding=query_embedding,
            query_sparse_embedding=sparse_embedding,
            top_k=3,
            keyword_filter=target_item_query # Try to force match the item name in content
        )
        
        if not results:
             logger.info("No price information found.")
             return {"documents": []}

        formatted_docs = []
        for doc, score in results:
            # Append specific instruction to look for price
            formatted = f"[Item Found (Score: {score:.2f})]\n{doc.content}"
            
            # Extract price info from metadata
            meta = doc.metadata or {}
            price_info_parts = []
            
            p_level = meta.get("priceLevel")
            if p_level:
                price_info_parts.append(f"Price Level: {p_level}")

            p_range = meta.get("priceRange")
            if p_range and str(p_range) != "0":
                price_info_parts.append(f"Price Range: {p_range}")
            
            p_night = meta.get("pricePerNight")
            if p_night and str(p_night) != "0":
                val_str = str(p_night).strip()
                # If no currency symbol/code found, assume RON
                if not any(curr in val_str.lower() for curr in ["ron", "lei", "eur", "usd", "€", "$", "£"]):
                    val_str += " RON"
                price_info_parts.append(f"Price Per Night: {val_str}")

            ticket_req = meta.get("ticketRequired")
            if ticket_req is not None:
                if ticket_req is False:
                     price_info_parts.append("Price: Free entry")
                else:
                     price_info_parts.append("Price: Ticket required")
                
            if price_info_parts:
                formatted += f"\nPRICE INFO: {', '.join(price_info_parts)}"
                formatted += f"\nCONTEXT FOR GENERATOR: The information labeled 'Price Level', 'Price Range', or 'Price' IS the price information. Answer based on these details. below, do NOT guess. Say 'prices are flexible', 'it is a moderate option', or 'it is free' as applicable."
            else:
                formatted += f"\nPRICE INFO: Specific prices are not listed as they are flexible. \nCONTEXT FOR GENERATOR: Explain to the user that prices vary available information is limited. \nSTRICT CONSTRAINT: DO NOT INVENT SPECIFIC NUMBERS. Suggest contacting the host."
                
            formatted_docs.append(formatted)
            
        # 3. Generate Answer Directly
        context_text = "\n\n".join(formatted_docs)
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        user_lang_code = state.get("user_language", "en")
        lang_map = {
            "en": "English",
            "ro": "Romanian (Română)",
            "ru": "Russian (Русский)",
            "es": "Spanish (Español)",
            "fr": "French (Français)",
            "de": "German (Deutsch)",
            "it": "Italian (Italiano)",
            "pt": "Portuguese (Português)",
            "pl": "Polish (Polski)",
            "uk": "Ukrainian (Українська)",
            "hu": "Hungarian (Magyar)"
        }
        target_lang_name = lang_map.get(user_lang_code, "English")

        gen_system_prompt = f"""
### IDENTITY & ROLE
- **Name:** Vlad-AI
- **Role:** A friendly, savvy local friend helping travelers in Romania.
- **Tone:** Casual, warm, and concise (like a helpful text message).
- **Current Date:** {current_date}

### LANGUAGE & COMMUNICATION (CRITICAL)
- **Target Language:** {target_lang_name} ({user_lang_code})
- You **MUST** respond in **{target_lang_name}**.
- If the user spoke {target_lang_name}, reply in {target_lang_name}.

### CORE OBJECTIVE
Answer the user's specific price/cost question based on the **Database Search Results**.

### CRITICAL RULES:
1. **DATABASE IS TRUTH:** You have **NO** outside knowledge. Base your answer **EXCLUSIVELY** on the provided search results.
2. **NO ROBOTIC LISTS:** Do NOT use numbered lists or bullet points. Write naturally like a friend texting.
3. **BE DIRECT:** If the price is in the docs, say it. If it says "flexible" or "contact host", say that.
4. **DON'T INVENT:** If no price info is present, say you don't have the exact pricing but can help find contact details or other info.

### STYLE & FORMAT
- **Ending:** Always end with ONE short follow-up question (e.g., "Do you need help with anything else?").

### CONTEXT
- **Database Search Results:**
{context_text}
"""
        messages = [SystemMessage(content=gen_system_prompt)] + history_messages + [HumanMessage(content=question)]

        # Apply max_tokens specifically for this generation if provided in state
        llm_to_use = self.llm

        llm_to_use = llm_to_use.bind(max_tokens=780)

        response = await llm_to_use.with_config({"tags": ["final_generator"], "run_name": "final_generator"}).ainvoke(messages)
        
        return {"generation": response.content, "documents": formatted_docs}


    async def query_stream(self, query_text: str, user_id: str, trip_id: Optional[int] = None, top_k: int = 5, max_tokens: Optional[int] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Main entry point for the RAG service. Streams the response using astream_events.
        """
        # Save user message moved to classifier_node to support translation

        inputs = {
            "question": query_text,
            "user_id": user_id,
            "trip_id": trip_id,
            "max_tokens": max_tokens
        }
        
        final_answer = ""
        detected_user_lang = "en"
        
        try:
            async for event in self.app.astream_events(inputs, version="v2"):
                kind = event.get("event")
                
                # We specifically want tokens from the 'generator' node's LLM call
                if kind == "on_chat_model_stream":
                    # Only yield if this is the final generator (not the query broadening step)
                    tags = event.get("tags", [])
                    if "final_generator" in tags:
                        content = event["data"]["chunk"].content
                        if content:
                            final_answer += content
                            yield {
                                "type": "answer",
                                "content": content,
                                "done": False
                            }
                
                # Process node outputs if streaming didn't happen (e.g. out_of_scope)
                elif kind == "on_chain_end" and event.get("name") == "LangGraph":
                    output = event.get("data", {}).get("output", {})
                    # Capture detected language from the final state
                    if output and "user_language" in output:
                        detected_user_lang = output["user_language"]

                    if not final_answer and output.get("generation"):
                        final_answer = output["generation"]
                        yield {
                            "type": "answer",
                            "content": final_answer,
                            "done": False
                        }
                    logger.info("Finished LangGraph execution")

        except Exception as e:
            logger.error(f"Error in graph execution: {e}")
            yield {
                "type": "error",
                "content": str(e),
                "done": True
            }
            return

        yield {
            "type": "answer",
            "content": "",
            "done": True
        }

        if self.external_memory_service and final_answer:
            # Run back-translation and saving in background
            asyncio.create_task(self._save_assistant_message_async(
                user_id,
                final_answer,
                detected_user_lang,
                max_tokens,
                trip_id
            ))
            
            # Start background recommendation extraction
            asyncio.create_task(self._extract_and_save_recommendations(user_id, final_answer))

    async def _save_assistant_message_async(self, user_id: str, content: str, detected_lang: str, max_tokens: Optional[int], trip_id: Optional[int]):
        """
        Background task to handle back-translation and saving of assistant messages.
        ROBUST: Handles None/empty content gracefully.
        """
        
        # CRITICAL: Validate input content
        if not content or not content.strip():
            logger.error(f"Cannot save assistant message: content is None or empty")
            logger.error(f"User: {user_id}, Lang: {detected_lang}")
            return  # EXIT EARLY
        
        message_to_save = content  # Default: use original content
        
        # Attempt back-translation if needed
        if detected_lang and detected_lang != "en":
            try:
                logger.info(f"Back-translating response from {detected_lang} to English for memory...")
                translated = await self._translate_to_english(content, detected_lang, user_id)
                
                # CRITICAL: Validate translation result
                if translated and isinstance(translated, str) and translated.strip():
                    message_to_save = translated
                    logger.info(f"✅ Back-translation successful: '{translated[:80]}...'")
                else:
                    logger.warning(
                        f"⚠️ Back-translation returned invalid result: {type(translated)}. "
                        f"Using original content."
                    )
                    message_to_save = content
                    
            except Exception as e:
                logger.error(f"Failed to back-translate assistant response: {e}")
                message_to_save = content  # Fallback to original
        
        # FINAL VALIDATION before database save
        if not message_to_save or not isinstance(message_to_save, str) or not message_to_save.strip():
            logger.error(
                f"Cannot save to database: message_to_save is invalid. "
                f"Type: {type(message_to_save)}, Value: {message_to_save}"
            )
            logger.error(f"Original content: '{content[:100] if content else 'None'}...'")
            return  # EXIT - prevents DB constraint error
        
        # Save to database
        try:
            self.external_memory_service.add_assistant_message(
                user_id, 
                message_to_save, 
                max_tokens, 
                trip_id
            )
            logger.info("✅ Assistant message saved to memory (background)")
        except Exception as e:
            logger.error(f"❌ Failed to save assistant message to database: {e}")
            logger.error(f"   User: {user_id}")
            logger.error(f"   Content length: {len(message_to_save) if message_to_save else 0}")
            logger.error(f"   Content preview: '{message_to_save[:100] if message_to_save else 'None'}...'")


    async def _extract_and_save_recommendations(self, user_id: str, response_text: str):
        """
        Background task to extract and save recommended items from the final response.
        Uses a lightweight LLM call to parse the text.
        """
        try:
            logger.info("Starting background recommendation extraction...")
            
            system_prompt = """You are an expert at extracting recommended places from text.
Task: Extract the exact names of hotels, restaurants, festivals, or attractions that were explicitly recommended in the text.
Rules:
- Only extract items that are positively recommended.
- Return a JSON object with a single key "recommended_items" containing a list of strings.
- If nothing is recommended, return {"recommended_items": []}.
- Do NOT include general terms like "hotels", "restaurants".
- Example Output: {"recommended_items": ["Hotel Aro Palace", "Bistro de l'Arte"]}
"""
            
            human_message = f"Text to analyze:\n{response_text}"
            
            # Use structured output for reliability
            structured_llm = self.llm.with_structured_output(SaveRecommendations)
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_message)
            ]
            
            # Use a smaller/faster model config if possible, but reusing self.llm is fine
            # We tag it to avoid confusion in logs
            chain = structured_llm.with_config({"tags": ["background_extraction"], "run_name": "background_extraction"})
            
            result = await chain.ainvoke(messages)
            
            if result and result.recommended_items:
                logger.info(f"Background extraction found items: {result.recommended_items}")
                if self.external_memory_service:
                    self.external_memory_service.add_seen_items(user_id, result.recommended_items)
            else:
                logger.info("No recommended items found in background extraction.")
                
        except Exception as e:
            logger.error(f"Error in background recommendation extraction: {e}")
