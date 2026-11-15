import redis
import json
import logging
from typing import List, Dict, Optional
from datetime import datetime
from collections import deque

logger = logging.getLogger(__name__)


class UnifiedMemoryService:
    """
    Unified memory service combining Redis (fast context) + SQLite (persistence).
    
    Redis: Short-term conversation context (5 turns, 1 hour TTL)
    SQLite: Long-term persistence (query logs, summaries)
    """
    
    def __init__(self, redis_url: str, db_manager=None):
        """
        Initialize memory service.
        
        Args:
            redis_url: Redis connection URL (e.g., redis://default:password@host:port)
            db_manager: Optional DatabaseManager for persistence
        """
        self.redis_url = redis_url
        self.db = db_manager
        
        # Connect to Redis
        try:
            self.redis_client = redis.from_url(
                redis_url,
                decode_responses=True,
                socket_connect_timeout=10,
                socket_timeout=10,
                ssl_cert_reqs=None
            )
            # Test connection
            self.redis_client.ping()
            logger.info("[MEMORY] ✅ Connected to Redis Cloud")
            self.use_redis = True
        except Exception as e:
            logger.error(f"[MEMORY] ❌ Redis connection failed: {e}")
            logger.warning("[MEMORY] Falling back to in-memory storage")
            self.use_redis = False
            # Fallback to in-memory
            self.fallback_store = {}
        
        # Configuration
        self.max_turns = 5  # Keep last 5 turns (10 messages)
        self.ttl_seconds = 1800  # 30 mins expiry
        
    def add_turn(
        self, 
        user_id: str, 
        user_message: str, 
        assistant_message: str,
        entities: Optional[List[str]] = None
    ):
        """
        Add a conversation turn to Redis.
        
        Stores in Redis with automatic expiry.
        Optionally logs to SQLite for persistence.
        """
        try:
            key = f"chat:{user_id}"
            
            # Get existing messages
            messages = self._get_messages(user_id)
            
            # Add new turn
            messages.append({
                "role": "user",
                "content": user_message,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            messages.append({
                "role": "assistant",
                "content": assistant_message,
                "timestamp": datetime.utcnow().isoformat(),
                "entities": entities or []
            })
            
            # Keep only last N turns
            messages = messages[-(self.max_turns * 2):]
            
            # Store in Redis with TTL
            if self.use_redis:
                self.redis_client.setex(
                    key,
                    self.ttl_seconds,
                    json.dumps(messages)
                )
            else:
                self.fallback_store[key] = messages
            
            logger.info(
                f"[MEMORY] Added turn for {user_id}. "
                f"Total messages: {len(messages)}"
            )
            
            # Optionally persist to SQLite
            if self.db:
                self._persist_to_db(user_id, user_message, assistant_message)
            
        except Exception as e:
            logger.error(f"[MEMORY] Error adding turn: {e}")
    
    def get_conversation_history(
        self, 
        user_id: str, 
        last_n_turns: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """
        Get conversation history from Redis.
        
        Args:
            user_id: User identifier
            last_n_turns: Number of turns to retrieve (None = all)
            
        Returns:
            List of message dicts with role, content, timestamp
        """
        try:
            messages = self._get_messages(user_id)
            
            if last_n_turns:
                messages = messages[-(last_n_turns * 2):]
            
            return messages
            
        except Exception as e:
            logger.error(f"[MEMORY] Error getting history: {e}")
            return []
    
    def _get_messages(self, user_id: str) -> List[Dict]:
        """Get messages from Redis or fallback storage"""
        key = f"chat:{user_id}"
        
        try:
            if self.use_redis:
                data = self.redis_client.get(key)
                if data:
                    return json.loads(data)
            else:
                return self.fallback_store.get(key, [])
        except Exception as e:
            logger.error(f"[MEMORY] Error retrieving messages: {e}")
        
        return []
    
    def clear_conversation(self, user_id: str):
        """Clear conversation history from Redis"""
        key = f"chat:{user_id}"
        
        try:
            if self.use_redis:
                self.redis_client.delete(key)
            else:
                if key in self.fallback_store:
                    del self.fallback_store[key]
            
            logger.info(f"[MEMORY] Cleared conversation for {user_id}")
            
        except Exception as e:
            logger.error(f"[MEMORY] Error clearing conversation: {e}")
    
    def get_stats(self) -> Dict:
        """Get memory system statistics"""
        try:
            if self.use_redis:
                # Count active conversations
                keys = self.redis_client.keys("chat:*")
                active_users = len(keys)
                
                # Get Redis info
                info = self.redis_client.info("memory")
                
                return {
                    "storage": "redis",
                    "active_users": active_users,
                    "memory_used_mb": round(info.get("used_memory", 0) / 1024 / 1024, 2),
                    "ttl_seconds": self.ttl_seconds,
                    "max_turns": self.max_turns
                }
            else:
                return {
                    "storage": "in-memory-fallback",
                    "active_users": len(self.fallback_store),
                    "max_turns": self.max_turns
                }
                
        except Exception as e:
            logger.error(f"[MEMORY] Error getting stats: {e}")
            return {"error": str(e)}
    
    def _persist_to_db(self, user_id: str, user_msg: str, assistant_msg: str):
        """
        Persist conversation to SQLite for long-term storage.
        This allows recovery after restarts.
        """
        if not self.db:
            return
        
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            # Store user message
            cursor.execute("""
                INSERT INTO conversation_messages (user_id, trip_id, role, content)
                VALUES (?, NULL, ?, ?)
            """, (user_id, "user", user_msg))
            
            # Store assistant message
            cursor.execute("""
                INSERT INTO conversation_messages (user_id, trip_id, role, content)
                VALUES (?, NULL, ?, ?)
            """, (user_id, "assistant", assistant_msg))
            
            conn.commit()
            conn.close()
            
            logger.debug(f"[MEMORY] Persisted turn to SQLite for {user_id}")
            
        except Exception as e:
            logger.error(f"[MEMORY] Error persisting to DB: {e}")
    
    def load_from_db(self, user_id: str, limit: int = 5) -> List[Dict]:
        """
        Load recent conversation from SQLite.
        Used for restoring context after restart.
        """
        if not self.db:
            return []
        
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT role, content, created_at 
                FROM conversation_messages
                WHERE user_id = ? AND trip_id IS NULL
                ORDER BY created_at DESC
                LIMIT ?
            """, (user_id, limit * 2))  # Get N turns (2N messages)
            
            rows = cursor.fetchall()
            conn.close()
            
            # Convert to message format (reverse to chronological order)
            messages = []
            for row in reversed(rows):
                messages.append({
                    "role": row[0],
                    "content": row[1],
                    "timestamp": row[2]
                })
            
            logger.info(f"[MEMORY] Loaded {len(messages)} messages from SQLite for {user_id}")
            return messages
            
        except Exception as e:
            logger.error(f"[MEMORY] Error loading from DB: {e}")
            return []
    
    def restore_to_redis(self, user_id: str):
        """
        Restore conversation from SQLite to Redis.
        Called when user reconnects after server restart.
        """
        # Load from DB
        messages = self.load_from_db(user_id, limit=self.max_turns)
        
        if not messages:
            return
        
        # Store in Redis
        key = f"chat:{user_id}"
        
        try:
            if self.use_redis:
                self.redis_client.setex(
                    key,
                    self.ttl_seconds,
                    json.dumps(messages)
                )
            else:
                self.fallback_store[key] = messages
            
            logger.info(f"[MEMORY] Restored {len(messages)} messages to Redis for {user_id}")
            
        except Exception as e:
            logger.error(f"[MEMORY] Error restoring to Redis: {e}")