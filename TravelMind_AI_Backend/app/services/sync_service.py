import logging
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List, Set
from app.core.config import Settings
from .text_processor import TextProcessor
from .data_fetcher import DataFetcher
from .translation_service import TranslationService
from .embedding_service import EmbeddingService
from .vector_store import VectorStore
from .database_manager import DatabaseManager
from app.models.models import SyncMetadata
from app.models import Document


from app.core.logger import get_logger

logger = get_logger(__name__)


class SyncService:
    def __init__(
        self,
        settings: Settings,
        data_fetcher: DataFetcher,
        text_processor: TextProcessor,
        translation_service: TranslationService,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        db_manager: DatabaseManager = None
    ):
        self.settings = settings
        self.data_fetcher = data_fetcher
        self.text_processor = text_processor
        self.translation_service = translation_service
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.db_manager = db_manager
        self.last_sync: datetime = None
        self._is_syncing = False

    def get_last_sync_time(self) -> Optional[datetime]:
        if not self.db_manager:
            return None

        session = self.db_manager.get_session()
        try:
            # Query the most recent sync metadata
            last_sync = session.query(SyncMetadata).order_by(
                SyncMetadata.id.desc()
            ).first()
            
            if not last_sync:
                return None
                
            # If the datetime is naive, assume it's UTC
            sync_time = last_sync.last_sync_time
            if sync_time.tzinfo is None:
                sync_time = sync_time.replace(tzinfo=timezone.utc)
                
            return sync_time
        finally:
            session.close()

    def should_sync(self, log_if_not_needed: bool = True) -> bool:
        """
        Check if enough time has passed since last sync (12 hours)
        
        Args:
            log_if_not_needed: If True, log INFO when sync is NOT needed.
                               Set to False to prevent log spam when polling.
        """
        last_sync = self.get_last_sync_time()

        if not last_sync:
            logger.info("No previous sync found, sync required")
            return True

        # Use UTC for current time
        now = datetime.now(timezone.utc)
        time_since_last_sync = now - last_sync
        hours_since_sync = time_since_last_sync.total_seconds() / 3600

        # Only log if requested or if sync IS needed (important info)
        should_sync = hours_since_sync >= self.settings.sync_interval_hours
        
        if should_sync:
            logger.info(f"Last sync was {hours_since_sync:.2f} hours ago (threshold: {self.settings.sync_interval_hours} hours)")
            logger.info("Sync interval exceeded, sync required")
        elif log_if_not_needed:
            logger.info(f"Last sync was {hours_since_sync:.2f} hours ago (threshold: {self.settings.sync_interval_hours} hours)")
            logger.info(f"Sync not needed, {self.settings.sync_interval_hours - hours_since_sync:.2f} hours remaining")

        return should_sync

    def save_sync_metadata(self, documents: int, chunks: int, duration: float,
                           added: int = 0, updated: int = 0, deleted: int = 0, skipped: int = 0):
        if not self.db_manager:
            return

        session = self.db_manager.get_session()
        try:
            # Create new sync metadata record
            sync_metadata = SyncMetadata(
                last_sync_time=datetime.now(timezone.utc),
                documents_synced=documents,
                chunks_synced=chunks,
                duration_seconds=duration,
                added=added,
                updated=updated,
                deleted=deleted,
                skipped=skipped
            )
            
            session.add(sync_metadata)
            session.commit()
        finally:
            session.close()

    def has_existing_data(self) -> bool:
        """Check if knowledge base has data AND if it's recent (within 12 hours)"""
        stats = self.vector_store.get_stats()
        has_documents = stats.get("total_documents", 0) > 0

        if not has_documents:
            logger.info("No documents in knowledge base")
            return False

        # Check if last sync is recent
        last_sync = self.get_last_sync_time()
        if not last_sync:
            logger.info("No sync metadata found, treating as fresh installation")
            return False

        # Use UTC
        now = datetime.now(timezone.utc)
        time_since_last_sync = now - last_sync
        hours_since_sync = time_since_last_sync.total_seconds() / 3600

        is_recent = hours_since_sync < self.settings.sync_interval_hours

        if is_recent:
            logger.info(f"Knowledge base is recent ({hours_since_sync:.2f} hours old), skipping sync")
        else:
            logger.info(f"Knowledge base is stale ({hours_since_sync:.2f} hours old), sync needed")

        return is_recent

    async def sync_data(self, force_full: bool = False) -> Dict[str, Any]:
        """
        Incremental sync - only embeds changed/new documents.

        Args:
            force_full: If True, performs full sync (clears and re-embeds everything)
        """
        logger.info("Starting data synchronization...")
        start_time = datetime.now(timezone.utc)

        try:
            # Fetch all data from API
            all_data = await self.data_fetcher.fetch_all_data()

            # Get existing document hashes from Qdrant
            existing_hashes = self.vector_store.get_all_document_hashes()
            existing_doc_ids = set(existing_hashes.keys())

            logger.info(f"Found {len(existing_doc_ids)} existing documents in knowledge base")

            # If no existing data or force_full, do full sync
            if not existing_hashes or force_full:
                logger.info("Performing full sync (no existing data or forced)")
                return await self._full_sync(all_data)

            # Check if sync is already in progress
            if self._is_syncing:
                logger.warning("Sync already in progress, skipping incremental sync request")
                return {
                    "success": False,
                    "error": "Sync in progress",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }

            self._is_syncing = True
            try:

                # Incremental sync
                total_documents = 0
                total_chunks = 0
                added_count = 0
                updated_count = 0
                skipped_count = 0
                
                # Semaphore to limit concurrent translations/embeddings
                sem = asyncio.Semaphore(5) # Reduced concurrency due to heavier processing

                new_doc_ids: Set[str] = set()

                for source, items in all_data.items():
                    # Extract actual items list if wrapped in structure (Logic from original file)
                    actual_items = items
                    if isinstance(items, list) and len(items) == 1 and isinstance(items[0], dict):
                         # Check if it's the wrapper structure
                         first_item = items[0]
                         if "data" in first_item and isinstance(first_item["data"], dict):
                             # Try to find the list inside data
                             # It usually matches the source name, e.g. "accommodations"
                             inner_data = first_item["data"]
                             if source in inner_data and isinstance(inner_data[source], list):
                                 actual_items = inner_data[source]
                                 logger.info(f"Unwrapped {len(actual_items)} items for {source} from nested structure")
                             # Fallback: check values() for the first list found
                             else:
                                 for v in inner_data.values():
                                     if isinstance(v, list):
                                         actual_items = v
                                         logger.info(f"Unwrapped {len(actual_items)} items for {source} from nested structure (fallback)")
                                         break

                    if not actual_items:
                        continue

                    documents_to_embed = []
                    dense_embeddings_to_store = []
                    sparse_embeddings_to_store = []
                    hashes_to_store = []

                    # Process items concurrently
                    logger.info(f"Starting processing of {len(actual_items)} items for {source}...")
                    
                    async def process_item(item):
                        async with sem:
                            # Normalize item
                            normalized_item = self.data_fetcher.normalize_item(item, source)
                            
                            # Construct structured chunks (Exploding + Templates)
                            structured_chunks = self.data_fetcher.construct_chunks(normalized_item)
                            
                            processed_chunks = []
                            for chunk_obj in structured_chunks:
                                chunk_text = chunk_obj["text"]
                                # Translate content to English if needed (Assuming source can be non-English)
                                # Note: translation service handles empty/checks internally
                                translated_text = await self.translation_service.translate(chunk_text)
                                
                                # Chunking logic for VERY long texts (Token limit)
                                # Only if translated text exceeds limit. text_processor.chunk_text splits by token limit.
                                # If split, we effectively create sub-chunks.
                                text_splits = self.text_processor.chunk_text(translated_text)
                                
                                for i, split_text in enumerate(text_splits):
                                    # Generate IDs: source_itemId_chunkId[_splitIndex]
                                    # Use original item index for stability if needed, but here we trust item ID
                                    chunk_suffix = chunk_obj["chunk_id"]
                                    if len(text_splits) > 1:
                                        chunk_suffix += f"_{i}"
                                    
                                    processed_chunks.append({
                                        "text": split_text,
                                        "suffix": chunk_suffix,
                                        "metadata": chunk_obj.get("metadata", {})
                                    })
                            
                            return normalized_item, processed_chunks

                    # Process items logic
                    async def process_item_with_index(index, item):
                         res, chunks = await process_item(item)
                         return index, res, chunks

                    tasks_with_index = [process_item_with_index(i, item) for i, item in enumerate(actual_items)]
                    
                    processed_count = 0
                    total_items = len(actual_items)
                    
                    for future in asyncio.as_completed(tasks_with_index):
                        idx, item, chunks = await future
                        
                        processed_count += 1
                        if processed_count % 10 == 0 or processed_count == total_items:
                            logger.info(f"Processed {processed_count}/{total_items} items for {source}")

                        # Loop through processed chunks (Logical + Token splits)
                        for chunk_data in chunks:
                            chunk_text = chunk_data["text"]
                            chunk_suffix = chunk_data["suffix"]
                            
                            # Stable ID Generation
                            obj_id = item.get('id') or item.get('_id') or (total_documents + idx)
                            doc_id = f"{source}_{obj_id}_{chunk_suffix}"
                            new_doc_ids.add(doc_id)

                            # Compute content hash
                            content_hash = self.vector_store.compute_content_hash(chunk_text)

                            # Check if document exists and has same hash
                            existing_hash = existing_hashes.get(doc_id)

                            if existing_hash == content_hash:
                                skipped_count += 1
                                continue

                            # Document is new or changed
                            if existing_hash:
                                updated_count += 1
                                logger.debug(f"Document {doc_id} changed, will re-embed")
                            else:
                                added_count += 1
                                logger.debug(f"Document {doc_id} is new, will embed")

                            # IMPORTANT: We generate embeddings in BATCHES later to optimize OpenAI/FastEmbed calls?
                            # But here we are iterating. 
                            # If we wait for batch, we need to store the text and generate later.
                            # Current logic accumulates `texts_to_embed`.
                            
                            # For Sparse, we run locally (FastEmbed). It's fast on CPU but better batched?
                            # FastEmbed runs properly on single Strings too.
                            # Let's generate Sparse HERE? Or batch?
                            # `vector_store.add_documents` takes lists.
                            # So we will accumulate.
                            
                            document = Document(
                                id=doc_id,
                                content=chunk_text,
                                metadata={**item, **chunk_data["metadata"]}, # Merge metadata
                                source=source,
                                created_at=datetime.now(timezone.utc)
                            )
                            documents_to_embed.append(document)
                            hashes_to_store.append(content_hash)
                            
                    # Process batches
                    if documents_to_embed:
                        batch_size = 50
                        batch_count = (len(documents_to_embed) + batch_size - 1) // batch_size

                        for i in range(0, len(documents_to_embed), batch_size):
                            batch_docs = documents_to_embed[i:i + batch_size]
                            batch_hashes = hashes_to_store[i:i + batch_size]
                            batch_texts = [d.content for d in batch_docs]
                            current_batch = i // batch_size + 1

                            logger.info(f"Processing batch {current_batch}/{batch_count} for {source} ({len(batch_texts)} items)")

                            try:
                                # 1. Generate Dense Embeddings (OpenAI)
                                dense_embeddings = await self.embedding_service.get_embeddings_batch(batch_texts)
                                
                                # 2. Generate Sparse Embeddings (SPLADE)
                                # Iterate because fastembed helper in service takes single text (or we update it?)
                                # Service method `get_sparse_embedding` takes str.
                                # Let's loop. fastembed is fast.
                                sparse_embeddings = []
                                for text in batch_texts:
                                    sparse_vec = self.embedding_service.get_sparse_embedding(text)
                                    sparse_embeddings.append(sparse_vec)

                                # 3. Upsert with Hybrid Vectors
                                self.vector_store.add_documents(
                                    documents=batch_docs, 
                                    dense_embeddings=dense_embeddings, 
                                    sparse_embeddings=sparse_embeddings, 
                                    content_hashes=batch_hashes
                                )
                                total_chunks += len(batch_docs)

                                if current_batch < batch_count:
                                    await asyncio.sleep(1) # Gentle sleep

                            except Exception as e:
                                logger.error(f"Error processing batch {current_batch} for {source}: {str(e)}")
                                await asyncio.sleep(5)
                                raise

                    total_documents += len(items)

                # Delete documents that no longer exist (cleanup)
                deleted_doc_ids = existing_doc_ids - new_doc_ids
                deleted_count = len(deleted_doc_ids)

                if deleted_doc_ids:
                    logger.info(f"Deleting {deleted_count} documents that no longer exist in source")
                    self.vector_store.delete_documents_by_ids(list(deleted_doc_ids))

                self.vector_store.save()
                
                # ... (stats logic remains same)
                self.last_sync = datetime.now(timezone.utc)
                duration = (self.last_sync - start_time).total_seconds()

                self.save_sync_metadata(
                    total_documents, total_chunks, duration,
                    added=added_count, updated=updated_count,
                    deleted=deleted_count, skipped=skipped_count
                )

                stats = {
                    "success": True,
                    "sync_type": "incremental",
                    "timestamp": self.last_sync.isoformat(),
                    "duration_seconds": duration,
                    "total_documents": total_documents,
                    "total_chunks_processed": total_chunks,
                    "added": added_count,
                    "updated": updated_count,
                    "deleted": deleted_count,
                    "skipped": skipped_count,
                    "sources": list(all_data.keys())
                }

                logger.info(
                    f"Incremental sync completed: "
                    f"{added_count} added, {updated_count} updated, {deleted_count} deleted, {skipped_count} skipped "
                    f"in {duration:.2f}s"
                )
                return stats
            
            finally:
                self._is_syncing = False

        except Exception as e:
            logger.error(f"Error during sync: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    async def _full_sync(self, all_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Perform full sync - clear and re-embed everything"""
        
        # Check if sync is already in progress
        if self._is_syncing:
             logger.warning("Sync already in progress, skipping full sync request")
             return {
                 "success": False,
                 "error": "Sync in progress",
                 "timestamp": datetime.now(timezone.utc).isoformat()
             }
        
        self._is_syncing = True
        try:
            start_time = datetime.now(timezone.utc)

            self.vector_store.clear()

            total_documents = 0
            total_chunks = 0

            for source, items in all_data.items():
                if not items:
                    continue

                # Extract actual items list if wrapped (Logic from original file)
                actual_items = items
                if isinstance(items, list) and len(items) == 1 and isinstance(items[0], dict):
                     first_item = items[0]
                     if "data" in first_item and isinstance(first_item["data"], dict):
                         inner_data = first_item["data"]
                         if source in inner_data and isinstance(inner_data[source], list):
                             actual_items = inner_data[source]
                             logger.info(f"Unwrapped {len(actual_items)} items for {source} from nested structure")
                         else:
                             for v in inner_data.values():
                                 if isinstance(v, list):
                                     actual_items = v
                                     logger.info(f"Unwrapped {len(actual_items)} items for {source} from nested structure (fallback)")
                                     break

                if not actual_items:
                    continue

                documents = []
                hashes_to_store = []

                # Process items sequentially to avoid API rate limits
                logger.info(f"Starting sequential processing of {len(actual_items)} items for {source}...")
                
                async def process_item(item):
                    normalized_item = self.data_fetcher.normalize_item(item, source)
                    structured_chunks = self.data_fetcher.construct_chunks(normalized_item)
                    
                    processed_chunks = []
                    for chunk_obj in structured_chunks:
                        chunk_text = chunk_obj["text"]
                        # We do not this anymore bc backend sends english one directly
                        # translated_text = await self.translation_service.translate(chunk_text)
                        text_splits = self.text_processor.chunk_text(chunk_text)
                        
                        for i, split_text in enumerate(text_splits):
                            chunk_suffix = chunk_obj["chunk_id"]
                            if len(text_splits) > 1:
                                chunk_suffix += f"_{i}"
                            
                            processed_chunks.append({
                                "text": split_text,
                                "suffix": chunk_suffix,
                                "metadata": chunk_obj.get("metadata", {})
                            })
                    return normalized_item, processed_chunks

                processed_count = 0
                total_items = len(actual_items)
                
                # Process items sequentially
                for idx, item in enumerate(actual_items):
                    item, chunks = await process_item(item)
                    
                    processed_count += 1
                    if processed_count % 10 == 0 or processed_count == total_items:
                        logger.info(f"Processed {processed_count}/{total_items} items for {source}")

                    for chunk_data in chunks:
                        chunk_text = chunk_data["text"]
                        chunk_suffix = chunk_data["suffix"]
                        
                        obj_id = item.get('id') or item.get('_id') or (total_documents + idx)
                        doc_id = f"{source}_{obj_id}_{chunk_suffix}"
                        content_hash = self.vector_store.compute_content_hash(chunk_text)

                        document = Document(
                            id=doc_id,
                            content=chunk_text,
                            metadata={**item, **chunk_data["metadata"]},
                            source=source,
                            created_at=datetime.now(timezone.utc)
                        )
                        documents.append(document)
                        hashes_to_store.append(content_hash)

                if documents:
                    batch_size = 50
                    batch_count = (len(documents) + batch_size - 1) // batch_size

                    for i in range(0, len(documents), batch_size):
                        batch_docs = documents[i:i + batch_size]
                        batch_hashes = hashes_to_store[i:i + batch_size]
                        batch_texts = [d.content for d in batch_docs]
                        current_batch = i // batch_size + 1

                        logger.info(f"Processing batch {current_batch}/{batch_count} for {source} ({len(batch_texts)} items)")

                        try:
                            # 1. Dense
                            dense_embeddings = await self.embedding_service.get_embeddings_batch(batch_texts)
                            
                            # 2. Sparse (Sync loop)
                            sparse_embeddings = []
                            for text in batch_texts:
                                sparse_vec = self.embedding_service.get_sparse_embedding(text)
                                sparse_embeddings.append(sparse_vec)
                            
                            # 3. Add
                            self.vector_store.add_documents(
                                documents=batch_docs, 
                                dense_embeddings=dense_embeddings, 
                                sparse_embeddings=sparse_embeddings, 
                                content_hashes=batch_hashes
                            )
                            total_chunks += len(batch_docs)

                            if current_batch < batch_count:
                                await asyncio.sleep(1)

                        except Exception as e:
                            logger.error(f"Error processing batch {current_batch} for {source}: {str(e)}")
                            await asyncio.sleep(5)
                            raise

                total_documents += len(items)

            self.vector_store.save()

            # Update duration
            self.last_sync = datetime.now(timezone.utc)
            duration = (self.last_sync - start_time).total_seconds()

            self.save_sync_metadata(
                total_documents, total_chunks, duration,
                added=total_documents,
                updated=0,
                deleted=0,
                skipped=0
            )

            stats = {
                "success": True,
                "sync_type": "full",
                "timestamp": self.last_sync.isoformat(),
                "duration_seconds": duration,
                "total_documents": total_documents,
                "total_chunks": total_chunks,
                "sources": list(all_data.keys())
            }

            logger.info(f"Full sync completed: {total_documents} documents, {total_chunks} chunks in {duration:.2f}s")
            return stats
            
        finally:
            self._is_syncing = False

    async def force_full_sync(self) -> Dict[str, Any]:
        """Force a full sync regardless of existing data"""
        logger.info("Forcing full sync...")
        return await self.sync_data(force_full=True)

    def get_sync_status(self) -> Dict[str, Any]:
        return {
            "last_sync": self.last_sync.isoformat() if self.last_sync else None,
            "vector_store_stats": self.vector_store.get_stats()
        }
