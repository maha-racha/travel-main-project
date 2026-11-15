import logging
from datetime import datetime
from typing import Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from app.core.config import Settings
import uuid

logger = logging.getLogger(__name__)


class QueryLoggerService:
    """Service for logging user queries to Qdrant for analytics"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.collection_name = "user_queries"
        
        # Connect to Qdrant Cloud OR local based on settings
        if settings.qdrant_cloud_url and settings.qdrant_api_key:
            # CLOUD MODE
            logger.info(f"[QUERY_LOGGER] Connecting to Qdrant Cloud")
            self.client = QdrantClient(
                url=settings.qdrant_cloud_url,
                api_key=settings.qdrant_api_key,
                timeout=30
            )
        else:
            # LOCAL MODE
            logger.info(f"[QUERY_LOGGER] Connecting to local Qdrant: {settings.qdrant_host}:{settings.qdrant_port}")
            self.client = QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                grpc_port=settings.qdrant_grpc_port,
                prefer_grpc=True
            )
        
        self._init_collection()
    
    def _init_collection(self):
        """Initialize the user_queries collection if it doesn't exist"""
        try:
            collections = [c.name for c in self.client.get_collections().collections]
            
            if self.collection_name not in collections:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=1,
                        distance=Distance.DOT
                    )
                )
                logger.info(f"[QUERY_LOGGER] Created collection '{self.collection_name}'")
            else:
                logger.info(f"[QUERY_LOGGER] Collection '{self.collection_name}' already exists")
        except Exception as e:
            logger.error(f"[QUERY_LOGGER] Error initializing collection: {str(e)}")
    
    def log_query(
        self,
        user_id: str,
        query: str,
        response: str,
        retrieved_docs: int,
        language: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log a user query to Qdrant"""
        try:
            point_id = str(uuid.uuid4())
            
            payload = {
                "user_id": user_id,
                "query": query,
                "response": response,
                "retrieved_docs": retrieved_docs,
                "language": language,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {}
            }
            
            vector = [1.0]
            
            point = PointStruct(
                id=point_id,
                vector=vector,
                payload=payload
            )
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            logger.debug(f"[QUERY_LOGGER] Logged query from user {user_id}")
            
        except Exception as e:
            logger.error(f"[QUERY_LOGGER] Failed to store query: {str(e)}")
    
    def get_user_history(self, user_id: str, limit: int = 10) -> list:
        """Retrieve query history for a user"""
        try:
            results = []
            offset = None
            
            while len(results) < limit:
                page, next_offset = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter={
                        "must": [
                            {
                                "key": "user_id",
                                "match": {"value": user_id}
                            }
                        ]
                    },
                    limit=limit,
                    offset=offset,
                    with_payload=True
                )
                
                if not page:
                    break
                
                for point in page:
                    results.append(point.payload)
                    if len(results) >= limit:
                        break
                
                if not next_offset or len(results) >= limit:
                    break
                    
                offset = next_offset
            
            results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            
            return results[:limit]
            
        except Exception as e:
            logger.error(f"[QUERY_LOGGER] Error retrieving user history: {str(e)}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get query logging statistics"""
        try:
            collection_info = self.client.get_collection(collection_name=self.collection_name)
            
            return {
                "total_queries": collection_info.points_count,
                "collection_name": self.collection_name
            }
        except Exception as e:
            logger.error(f"[QUERY_LOGGER] Error getting stats: {str(e)}")
            return {
                "total_queries": 0,
                "collection_name": self.collection_name
            }