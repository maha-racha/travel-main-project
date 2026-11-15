import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.core.config import get_settings
from app.services import (
    DataFetcher,
    EmbeddingService,
    VectorStore,
    RAGService,
    SyncService,
    SyncScheduler,
    WebSearchService,
)

from app.services.agentic_rag_service import AgenticRAGService
from app.services.text_processor import TextProcessor
from app.services.intent_classifier import IntentClassifier

from app.services.translation_service import TranslationService
from app.core.limiter import limiter
from app.core.gpt_cache import init_gptcache
from app.services.itinerary_service import ItineraryService
from app.services.external_memory_service import ExternalMemoryService
from app.services.database_manager import DatabaseManager
from app.models.models import Trip, ItineraryDay, ItineraryItem, ConversationMemory, ConversationMessage, SyncMetadata, ChatMemory
from app.api import setup_routes, setup_websocket_routes
from app.api.itinerary_routes import setup_itinerary_routes

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from app.core.logger import get_logger

logger = get_logger(__name__)

settings = get_settings()

# Initialize database manager with session factory
db_manager = DatabaseManager(database_url=settings.database_url)
db_manager.create_tables()

embedding_service = EmbeddingService(settings)
vector_store = VectorStore(settings, embedding_service.get_embedding_dimension())
data_fetcher = DataFetcher(settings)
text_processor = TextProcessor(settings)
web_search_service = WebSearchService(settings)
intent_classifier = IntentClassifier(settings)

itinerary_service = ItineraryService(db_manager)
external_memory_service = ExternalMemoryService(settings, db_manager)

rag_service = RAGService(
    settings,
    vector_store,
    embedding_service,
    web_search_service,
    intent_classifier,
    itinerary_service,
    external_memory_service
)

agentic_rag_service = AgenticRAGService(
    settings,
    vector_store,
    embedding_service,
    web_search_service,
    intent_classifier,
    itinerary_service,
    external_memory_service
)

translation_service = TranslationService(settings)
sync_service = SyncService(settings, data_fetcher, text_processor, translation_service, embedding_service, vector_store, db_manager)
scheduler = SyncScheduler(settings, sync_service)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting application...")
    init_gptcache()
    await scheduler.start()
    yield
    logger.info("Shutting down application...")
    await scheduler.stop()


app = FastAPI(
    title="VLAD AI Chatbot API",
    description="RAG-based chatbot with Qdrant vector database and external memory system",
    version="2.0.0",
    lifespan=lifespan
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_router = setup_routes(rag_service, sync_service, external_memory_service)
app.include_router(api_router, prefix="/api")

itinerary_router = setup_itinerary_routes(itinerary_service)
app.include_router(itinerary_router, prefix="/api/itinerary", tags=["Itinerary"])

websocket_router = setup_websocket_routes(rag_service, agentic_rag_service)
app.include_router(websocket_router)


@app.get("/")
async def read_root():
    """Serve the test client HTML page at root route"""
    html_path = Path(__file__).parent.parent / "test_client.html"
    return FileResponse(html_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True
    )