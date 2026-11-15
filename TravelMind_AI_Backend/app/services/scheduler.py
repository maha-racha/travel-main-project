import logging
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from app.core.config import Settings
from app.services.sync_service import SyncService

logger = logging.getLogger(__name__)


class SyncScheduler:
    def __init__(self, settings: Settings, sync_service: SyncService):
        self.settings = settings
        self.sync_service = sync_service
        self.scheduler = AsyncIOScheduler()

    async def start(self):
        logger.info("Starting sync scheduler...")

        # Check collection health
        stats = self.sync_service.vector_store.get_stats()
        total_docs = stats.get("total_documents", 0)
        
        # FIXED: Only force full sync if collection is EMPTY (0 docs)
        # NOT just because it's < 227 (which triggers on every restart)
        if total_docs == 0:
            logger.info(f"Collection is empty, forcing initial full sync...")
            asyncio.create_task(self._force_initial_sync())
        # If collection has data, do incremental sync (only changed docs)
        elif self.sync_service.should_sync():
            logger.info(f"Collection has {total_docs} documents, checking for updates (incremental sync)...")
            asyncio.create_task(self._initial_sync())
        else:
            logger.info(f"Knowledge base has {total_docs} documents and is up-to-date, skipping sync")

        # Schedule periodic sync checks
        self.scheduler.add_job(
            self._sync_with_check,
            trigger=IntervalTrigger(hours=self.settings.sync_interval_hours),
            id="data_sync_job",
            name="Periodic data synchronization check",
            replace_existing=True
        )

        self.scheduler.start()
        logger.info(f"Scheduler started. Will check for sync every {self.settings.sync_interval_hours} hours")

    async def _force_initial_sync(self):
        """Force full sync for empty collection"""
        try:
            logger.info("Starting FORCE FULL sync (empty collection)...")
            result = await self.sync_service.force_full_sync()
            
            if result.get("success"):
                logger.info("✅ Force sync completed successfully")
                logger.info(f"Sync type: {result.get('sync_type', 'unknown')}")
                logger.info(f"Documents: {result.get('total_documents', 0)}")
                logger.info(f"Embeddings: {result.get('total_embeddings', 0)}")
                logger.info(f"Duration: {result.get('duration_seconds', 0):.2f}s")
                logger.info(f"Breakdown: {result.get('breakdown', {})}")
            else:
                logger.error(f"❌ Force sync failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"Force sync failed with exception: {str(e)}")

    async def _initial_sync(self):
        """Perform incremental sync on startup if needed"""
        try:
            logger.info("Starting incremental sync check...")
            result = await self.sync_service.sync_data()
            
            if result.get("success"):
                sync_type = result.get('sync_type', 'unknown')
                logger.info(f"✅ Sync completed successfully (type: {sync_type})")
                logger.info(f"Documents: {result.get('total_documents', 0)}")
                logger.info(f"Embeddings: {result.get('total_embeddings', 0)}")
                logger.info(f"Duration: {result.get('duration_seconds', 0):.2f}s")
                
                if sync_type == 'incremental':
                    logger.info(f"Changes: +{result.get('added', 0)} updated:{result.get('updated', 0)} deleted:{result.get('deleted', 0)} skipped:{result.get('skipped', 0)}")
                else:
                    logger.info(f"Breakdown: {result.get('breakdown', {})}")
            else:
                logger.error(f"❌ Sync failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"Sync failed with exception: {str(e)}")

    async def _sync_with_check(self):
        """Periodic incremental sync - only embeds changed documents"""
        try:
            logger.info("Periodic sync check triggered")
            
            # Double-check if sync is needed (time-based)
            if self.sync_service.should_sync():
                logger.info("Time threshold reached, starting periodic incremental sync...")
                result = await self.sync_service.sync_data()
                
                if result.get("success"):
                    sync_type = result.get('sync_type', 'unknown')
                    logger.info(f"✅ Periodic sync completed (type: {sync_type})")
                    logger.info(f"Documents: {result.get('total_documents', 0)}")
                    logger.info(f"Embeddings: {result.get('total_embeddings', 0)}")
                    logger.info(f"Duration: {result.get('duration_seconds', 0):.2f}s")
                    
                    if sync_type == 'incremental':
                        logger.info(f"Changes: +{result.get('added', 0)} updated:{result.get('updated', 0)} deleted:{result.get('deleted', 0)} skipped:{result.get('skipped', 0)}")
                else:
                    logger.error(f"❌ Periodic sync failed: {result.get('error', 'Unknown error')}")
            else:
                logger.info("Periodic check: Sync not needed yet")
                
        except Exception as e:
            logger.error(f"Periodic sync failed with exception: {str(e)}")

    async def stop(self):
        """Stop the scheduler"""
        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("Scheduler stopped")

    async def trigger_manual_sync(self):
        """Manually trigger an incremental sync (ignores time threshold)"""
        logger.info("Manual incremental sync triggered")
        try:
            result = await self.sync_service.sync_data()
            logger.info("Manual sync completed")
            return result
        except Exception as e:
            logger.error(f"Manual sync failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": None
            }

    def get_sync_info(self):
        """Get information about sync status"""
        last_sync = self.sync_service.get_last_sync_time()
        stats = self.sync_service.vector_store.get_stats()
        
        return {
            "last_sync": last_sync.isoformat() if last_sync else None,
            "has_data": stats.get("total_documents", 0) > 0,
            "total_documents": stats.get("total_documents", 0),
            "should_sync": self.sync_service.should_sync(),
            "sync_interval_hours": self.settings.sync_interval_hours
        }