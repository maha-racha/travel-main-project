from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # API Keys
    openai_api_key: str
    vlad_ai_api_key: str
    vlad_ai_base_url: str = "https://api.vladai.eu/api/chat-bot"

    # Sync Settings
    sync_interval_hours: int = 12

    # Qdrant LOCAL configuration (for development)
    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333
    qdrant_grpc_port: int = 6334
    qdrant_collection_name: str = "vlad_ai_cleaned"
    
    # Qdrant CLOUD configuration (for production)
    qdrant_cloud_url: str | None = None
    qdrant_api_key: str | None = None
    
    database_url: str = "sqlite:///tourism_chatbot.db"

    # Redis Cloud configuration (for conversation memory)
    redis_url: str = "redis://localhost:6379"  # Default local
    # For Redis Cloud: redis://default:password@host:port
    redis_enabled: bool = True  # Set to False to use in-memory fallback

    # Text Processing
    chunk_size: int = 512
    chunk_overlap: int = 50

    # OpenAI Models
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4.1-mini"
    openai_temperature: float = 0.7
    openai_max_tokens: int = 500
    
    # Features
    enable_prompt_caching: bool = True
    enable_itinerary_analysis: bool = True
    enable_memory_summaries: bool = True
    enable_translation: bool = True
    
    # Token Limits
    max_context_tokens: int = 3000  
    max_history_tokens: int = 500  
    max_memory_tokens: int = 300
    
    # Security & Rate Limiting
    rate_limit_requests: int = 10
    rate_limit_window_seconds: int = 60
    max_output_tokens: int = 300
    max_output_chars: int = 1000
    max_context_chars: int = 2000
    max_query_length: int = 500
    max_top_k: int = 7
    max_websocket_connections: int = 100

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()