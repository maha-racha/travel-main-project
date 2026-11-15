import logging
import uuid
import hashlib
from typing import List, Tuple, Dict, Any, Optional, Set
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, MatchText, MatchAny, ScrollRequest, PayloadSchemaType, SparseVectorParams, SparseIndexParams, Prefetch, SparseVector

from app.core.config import Settings
from app.schemas import Document
from app.core.logger import get_logger

logger = get_logger(__name__)

class VectorStore:
    def __init__(self, settings: Settings, dimension: int):
        self.settings = settings
        self.dimension = dimension
        self.collection_name = settings.qdrant_collection_name

        # Connect to Qdrant
        # self.client = QdrantClient(
        #     host=settings.qdrant_host,
        #     port=settings.qdrant_port,
        #     grpc_port=settings.qdrant_grpc_port,
        #     prefer_grpc=True
        # )
        
        self.client = QdrantClient(
            url=settings.qdrant_cloud_url,
            api_key=settings.qdrant_api_key,
            timeout=30
        )

        self._initialize_collection()

    @staticmethod
    def compute_content_hash(content: str) -> str:
        """Compute SHA256 hash of content for change detection"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def _initialize_collection(self):
        """Initialize or create Qdrant collection with Hybrid Search support"""
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_exists = any(c.name == self.collection_name for c in collections)

            if not collection_exists:
                # Create new collection with NAMED VECTORS (Dense + Sparse)
                try:
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config={
                            "text-dense": VectorParams(
                                size=self.dimension,
                                distance=Distance.COSINE
                            )
                        },
                        sparse_vectors_config={
                            "text-sparse": SparseVectorParams(
                                index=SparseIndexParams(
                                    on_disk=True
                                )
                            )
                        }
                    )
                    
                    # --- CREATE INDEXES FOR HYBRID SEARCH ---
                    # 1. Exact matches (Keyword)
                    indexes_keyword = [
                        "category.type", "city.name", "priceLevel", "amenities", "title",
                        "features", "cuisines", "dietaryOptions", "mealsServed", 
                        "isActive", "isPetFriendly", "allowsFamilies", 
                        "isWheelchairAccessible", "reopens", "requiresReservation", 
                        "ticketRequired", "chunk_type", "parent_id"
                    ]
                    for field in indexes_keyword:
                        self.client.create_payload_index(
                            collection_name=self.collection_name,
                            field_name=field,
                            field_schema=PayloadSchemaType.KEYWORD
                        )

                    # 2. Ranges (Integer/Float)
                    indexes_integer = [
                        "startDate_ts", "endDate_ts", 
                        "pricePerNight_val", 
                        "opensAt_minutes", "closesAt_minutes",
                        "maxGuests", "minimumNightsStay"
                    ]
                    for field in indexes_integer:
                         self.client.create_payload_index(
                            collection_name=self.collection_name,
                            field_name=field,
                            field_schema=PayloadSchemaType.INTEGER
                        )
                    
                    # 3. Full Text (Optional)
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name="description",
                        field_schema=PayloadSchemaType.TEXT
                    )
                    
                    # 4. Content Full Text (For boolean keyword filtering)
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name="content",
                        field_schema=PayloadSchemaType.TEXT
                    )

                    logger.info(f"Created new Qdrant collection '{self.collection_name}' with Hybrid Search config")
                except Exception as create_error:
                    # Handle race condition - another worker may have created it
                    if "already exists" in str(create_error).lower():
                        logger.info(f"Collection '{self.collection_name}' already exists (created by another worker)")
                    else:
                        raise
            else:
                logger.info(f"Connected to existing Qdrant collection '{self.collection_name}'")

            # --- ENSURE INDEXES EXIST (Idempotent ops) ---
            # We run this always to support schema migrations (like adding 'content' text index)
            try:
                # 4. Content Full Text (For boolean keyword filtering)
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="content",
                    field_schema=PayloadSchemaType.TEXT
                )
                
                # New semantic fields
                for field in ["chunk_type", "parent_id"]:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field,
                        field_schema=PayloadSchemaType.KEYWORD
                    )
                    
                logger.info("Verified/Created 'content' and semantic indexes")
            except Exception as e:
                # Ignore if already exists (Qdrant might throw if index exists, though usually it's idempotent for same config)
                # But to be safe against errors:
                logger.warning(f"Note on index creation: {e}")
                
        except Exception as e:
            logger.error(f"Error initializing collection: {str(e)}")
            raise

    def add_documents(self, documents: List[Document], dense_embeddings: List[List[float]], sparse_embeddings: Optional[List[Dict[str, Any]]] = None, content_hashes: Optional[List[str]] = None):
        """Add documents with their embeddings to Qdrant"""
        if not documents or not dense_embeddings:
            return

        try:
            points = []
            for idx, (doc, dense_emb) in enumerate(zip(documents, dense_embeddings)):
                # Generate UUID from document ID and source for uniqueness
                unique_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc.id}_{doc.source}"))

                # Compute content hash if not provided
                content_hash = content_hashes[idx] if content_hashes else self.compute_content_hash(doc.content)

                # Construct Vector Payload (Named Vectors)
                vector_payload = {
                    "text-dense": dense_emb
                }
                if sparse_embeddings and idx < len(sparse_embeddings):
                    vector_payload["text-sparse"] = SparseVector(
                        indices=sparse_embeddings[idx]["indices"],
                        values=sparse_embeddings[idx]["values"]
                    )

                # Create point with document data as payload
                point = PointStruct(
                    id=unique_id,
                    vector=vector_payload,
                    payload={
                        **doc.metadata,
                        "id": doc.id,
                        "content": doc.content,
                        "source": doc.source,
                        "content_hash": content_hash  # Store hash for incremental sync
                    }
                )
                points.append(point)

            # Batch upsert to Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

            logger.info(f"Added {len(documents)} documents to Qdrant collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Error adding documents to Qdrant: {str(e)}")
            raise

    def get_all_document_hashes(self) -> Dict[str, str]:
        """Get all document IDs and their content hashes from the collection.
        Returns: Dict mapping doc_id to content_hash
        """
        try:
            all_hashes = {}
            offset = None
            batch_size = 100

            while True:
                # Scroll through all points
                results = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )

                points, next_offset = results

                if not points:
                    break

                for point in points:
                    doc_id = point.payload.get("id", "")
                    content_hash = point.payload.get("content_hash", "")
                    if doc_id and content_hash:
                        all_hashes[doc_id] = content_hash

                if next_offset is None:
                    break
                offset = next_offset

            logger.info(f"Retrieved {len(all_hashes)} document hashes from collection")
            return all_hashes

        except Exception as e:
            logger.error(f"Error getting document hashes: {str(e)}")
            return {}

    def get_all_document_ids(self) -> Set[str]:
        """Get all document IDs currently in the collection"""
        try:
            all_ids = set()
            offset = None
            batch_size = 100

            while True:
                results = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=batch_size,
                    offset=offset,
                    with_payload=["id"],
                    with_vectors=False
                )

                points, next_offset = results

                if not points:
                    break

                for point in points:
                    doc_id = point.payload.get("id", "")
                    if doc_id:
                        all_ids.add(doc_id)

                if next_offset is None:
                    break
                offset = next_offset

            return all_ids

        except Exception as e:
            logger.error(f"Error getting document IDs: {str(e)}")
            return set()

    def _merge_filters(self, existing_filters: Optional[Filter], keyword_filter: Optional[str]) -> Optional[Filter]:
        """
        Helper to merge existing category/city filters with a keyword text match filter.
        """
        if not keyword_filter:
            return existing_filters

        # Create the text match condition
        text_condition = FieldCondition(
            key="content",
            match=MatchText(text=keyword_filter)
        )

        if existing_filters is None:
            return Filter(must=[text_condition])
        
        # If existing_filters has 'must', append to it. Otherwise create new 'must'.
        # We need to be careful not to mutate the original object if it's reused.
        # But Filter objects are Pydantic models, checking how to clone/extend safely.
        
        new_must = []
        if existing_filters.must:
            new_must.extend(existing_filters.must)
        
        new_must.append(text_condition)
        
        return Filter(
            must=new_must,
            should=existing_filters.should,
            must_not=existing_filters.must_not
        )

    def delete_documents_by_ids(self, doc_ids: List[str]):
        """Delete documents by their document IDs"""
        if not doc_ids:
            return

        try:
            # Convert doc_ids to Qdrant point IDs
            point_ids = []
            for doc_id in doc_ids:
                # We need to find the actual Qdrant ID for each doc_id
                # Using scroll with filter
                results = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(
                                key="id",
                                match=MatchValue(value=doc_id)
                            )
                        ]
                    ),
                    limit=100,
                    with_payload=False,
                    with_vectors=False
                )

                points, _ = results
                for point in points:
                    point_ids.append(point.id)

            if point_ids:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=point_ids
                )
                logger.info(f"Deleted {len(point_ids)} documents from collection")

        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            raise

    def search(self, query_dense_embedding: List[float], query_sparse_embedding: Optional[Dict[str, Any]] = None, top_k: int = 5, filters: Optional[Filter] = None, keyword_filter: Optional[str] = None) -> List[Tuple[Document, float]]:
        """
        Search for similar vectors in Qdrant using Hybrid Search (Reciprocal Rank Fusion).
        Executes Dense and Sparse searches in parallel and fuses results.
        """
        try:
            # 1. Dense Search
            dense_results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_dense_embedding,
                using="text-dense",
                query_filter=self._merge_filters(filters, keyword_filter),
                limit=top_k * 2,  # Fetch more for fusion
                with_payload=True,
                with_vectors=False
            ).points

            # 2. Sparse Search (if provided)
            sparse_results = []
            if query_sparse_embedding and query_sparse_embedding["indices"]:
                sparse_vec = SparseVector(
                    indices=query_sparse_embedding["indices"],
                    values=query_sparse_embedding["values"]
                )
                sparse_results = self.client.query_points(
                    collection_name=self.collection_name,
                    query=sparse_vec,
                    using="text-sparse",
                    query_filter=self._merge_filters(filters, keyword_filter),
                    limit=top_k * 2,
                    with_payload=True,
                    with_vectors=False
                ).points

            # 3. Reciprocal Rank Fusion (RRF)
            # RRF score = 1 / (k + rank)
            rrf_k = 60
            fused_scores = {}
            doc_map = {}

            # Process Dense Results
            for rank, hit in enumerate(dense_results):
                doc_id = hit.id
                doc_map[doc_id] = hit
                score = 1 / (rrf_k + rank + 1)
                fused_scores[doc_id] = fused_scores.get(doc_id, 0) + score

            # Process Sparse Results
            for rank, hit in enumerate(sparse_results):
                doc_id = hit.id
                doc_map[doc_id] = hit
                score = 1 / (rrf_k + rank + 1)
                fused_scores[doc_id] = fused_scores.get(doc_id, 0) + score

            # Sort by fused score
            sorted_doc_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
            
            # Take top_k
            final_results = []
            for doc_id in sorted_doc_ids[:top_k]:
                hit = doc_map[doc_id]
                score = fused_scores[doc_id]
                
                # Reconstruct Document
                doc = Document(
                    id=hit.payload.get("id", ""),
                    content=hit.payload.get("content", ""),
                    source=hit.payload.get("source", ""),
                    metadata=hit.payload
                )
                
                # Return RRF score
                final_results.append((doc, score))

            return final_results
        except Exception as e:
            logger.error(f"Error searching in Qdrant: {str(e)}")
            return []

    def save(self):
        """Qdrant persists data automatically, no explicit save needed"""
        logger.info("Qdrant automatically persists data")

    def clear(self):
        """Clear all vectors from the collection"""
        try:
            # Delete and recreate collection
            self.client.delete_collection(collection_name=self.collection_name)
            self._initialize_collection()
            logger.info(f"Cleared Qdrant collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            # Use count() method which is more reliable than get_collection()
            count_result = self.client.count(collection_name=self.collection_name)
            points_count = count_result.count if count_result else 0

            return {
                "total_documents": points_count,
                "index_size": points_count,
                "dimension": self.dimension,
                "collection_name": self.collection_name
            }
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {
                "total_documents": 0,
                "index_size": 0,
                "dimension": self.dimension,
                "collection_name": self.collection_name
            }

