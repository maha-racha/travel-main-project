from fastapi import APIRouter, HTTPException
from app.schemas import QueryRequest, QueryResponse, UpdateLanguageRequest, UpdateLanguageResponse
from app.services.rag_service import RAGService
from app.services.sync_service import SyncService
from app.core.limiter import limiter
from fastapi import Request

router = APIRouter()


def setup_routes(rag_service: RAGService, sync_service: SyncService, external_memory_service=None):
    @router.post("/query", response_model=QueryResponse)
    @limiter.limit("10/minute")
    async def query_endpoint(request: QueryRequest, req: Request):
        try:
            answer, sources = await rag_service.query(
                request.query,
                user_id=request.user_id,
                trip_id=request.trip_id,
                top_k=request.top_k,
                max_tokens=request.max_tokens
            )
            return QueryResponse(answer=answer, sources=sources)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/sync")
    @limiter.limit("1/minute")
    async def trigger_sync(request: Request):
        """
        Trigger incremental sync - only embeds new/changed documents.
        This is cost-efficient and should be used for regular syncs.
        """
        try:
            result = await sync_service.sync_data()
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/sync/full")
    @limiter.limit("1/minute")
    async def trigger_full_sync(request: Request):
        """
        Trigger full sync - clears and re-embeds ALL documents.
        Use this only when you need to rebuild the entire knowledge base.
        Warning: This is more expensive in terms of OpenAI API costs.
        """
        try:
            result = await sync_service.force_full_sync()
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/sync/status")
    @limiter.limit("60/minute")
    async def get_sync_status(request: Request):
        try:
            return sync_service.get_sync_status()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/sync/info")
    @limiter.limit("60/minute")
    async def get_sync_info(request: Request):
        """Get detailed sync information including whether sync is needed"""
        try:
            last_sync = sync_service.get_last_sync_time()
            stats = sync_service.vector_store.get_stats()

            return {
                "last_sync": last_sync.isoformat() if last_sync else None,
                "has_data": stats.get("total_documents", 0) > 0,
                "total_documents": stats.get("total_documents", 0),
                "should_sync": sync_service.should_sync(log_if_not_needed=False),
                "sync_interval_hours": sync_service.settings.sync_interval_hours,
                "vector_store_stats": stats
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/status")
    @limiter.limit("60/minute")
    async def get_status(request: Request):
        try:
            return sync_service.get_sync_status()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/health")
    @limiter.limit("60/minute")
    async def health_check(request: Request):
        return {"status": "healthy"}

    @router.post("/user/language", response_model=UpdateLanguageResponse)
    @limiter.limit("60/minute")
    async def update_user_language(language_request: UpdateLanguageRequest, request: Request):
        """Update user's language preference"""
        try:
            # Validate language code
            supported_languages = ["en", "ro", "ru", "es", "fr", "de", "it", "pt", "pl", "uk", "hu"]
            if language_request.language not in supported_languages:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Unsupported language code. Supported: {', '.join(supported_languages)}"
                )
            
            # Update language in database
            if external_memory_service:
                external_memory_service.update_user_language(language_request.user_id, language_request.language)
                return UpdateLanguageResponse(
                    success=True,
                    message=f"Language updated to {language_request.language}",
                    language=language_request.language
                )
            else:
                raise HTTPException(status_code=500, detail="External memory service not available")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return router
