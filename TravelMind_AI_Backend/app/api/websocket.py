import json
import logging
import uuid
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.rag_service import RAGService
from app.services.agentic_rag_service import AgenticRAGService
from typing import Optional
from app.core.limiter import TokenBucket

from app.core.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")


manager = ConnectionManager()


def setup_websocket_routes(rag_service: RAGService, agentic_service: Optional[AgenticRAGService] = None):
    @router.websocket("/ws/chat")
    async def websocket_endpoint(websocket: WebSocket):
        await manager.connect(websocket)
        
        # Generate a universally unique user ID for this connection
        user_id = str(uuid.uuid4())
        logger.info(f"New WebSocket connection with user_id: {user_id}")
        
        # Rate limiter: 10 messages per minute with capacity of 10
        limiter = TokenBucket(rate=10, capacity=10)

        try:
            while True:
                data = await websocket.receive_text()

                if not await limiter.acquire():
                    await websocket.send_json({
                        "type": "error",
                        "content": "Rate limit exceeded. Please wait a moment.",
                        "done": True
                    })
                    continue

                try:
                    message = json.loads(data)
                    query = message.get("query", "")
                    trip_id = message.get("trip_id")
                    top_k = message.get("top_k", 5)
                    max_tokens = message.get("max_tokens")

                    if not query:
                        await websocket.send_json({
                            "type": "error",
                            "content": "Query is required",
                            "done": True
                        })
                        continue

                    use_agent = message.get("use_agent", True) # Default to agentic if available
                    
                    service_to_use = agentic_service
                    
                    async for chunk in service_to_use.query_stream(query, user_id, trip_id, top_k, max_tokens):
                        await websocket.send_json(chunk)

                except json.JSONDecodeError:
                    await websocket.send_json({
                        "type": "error",
                        "content": "Invalid JSON format",
                        "done": True
                    })
                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}")
                    await websocket.send_json({
                        "type": "error",
                        "content": f"Error processing request: {str(e)}",
                        "done": True
                    })

        except WebSocketDisconnect:
            manager.disconnect(websocket)
        except Exception as e:
            logger.error(f"WebSocket error: {str(e)}")
            manager.disconnect(websocket)

    return router