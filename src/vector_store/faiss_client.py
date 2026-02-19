# cockatoo_v1/src/vector_store/faiss_client.py

"""
FAISS Vector Store Client for cockatoo_v1.
Provides CRUD operations and search capabilities optimized for large-scale similarity search.
"""

import faiss
import numpy as np
import json
import logging
import pickle
import uuid
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import os

try:
    from ..core.constants import DATABASE_DIR, VECTOR_DB_NAME
except ImportError:
    DATABASE_DIR = Path.home() / ".cockatoo_v1" / "database"
    VECTOR_DB_NAME = "faiss"

logger = logging.getLogger(__name__)


class FAISSClient:
    """FAISS vector store client for document storage and retrieval."""

    REBUILD_THRESHOLD = 0.2

    def __init__(
        self,
        persist_directory: Optional[Path] = None,
        index_name: str = "documents",
        dimension: int = 384,
        metric: str = "cosine"
    ):
        """Initialize FAISS client with persistent storage."""
        if persist_directory is None:
            persist_directory = DATABASE_DIR / VECTOR_DB_NAME

        self.persist_directory = Path(persist_directory)
        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric
        self.default_top_k = 5

        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.index_path = self.persist_directory / f"{index_name}.faiss"
        self.metadata_path = self.persist_directory / f"{index_name}_metadata.pkl"
        self.config_path = self.persist_directory / f"{index_name}_config.json"
        self.deleted_ids_path = self.persist_directory / f"{index_name}_deleted.pkl"

        self.index = None
        self.metadata_store = {}
        self.id_to_index = {}
        self.index_to_id = {}
        self.deleted_ids = set()
        self.deleted_count = 0

        self._initialize_index()
        self._load_metadata()
        self._load_deleted_ids()

        logger.info(f"FAISS client initialized: {self.persist_directory}")

    def _initialize_index(self) -> None:
        """Initialize or load FAISS index."""
        try:
            if self.index_path.exists():
                self.index = faiss.read_index(str(self.index_path))
                logger.info(f"Loaded existing FAISS index from {self.index_path}")

                if self.index.d != self.dimension:
                    logger.warning(f"Index dimension ({self.index.d}) doesn't match configured dimension ({self.dimension})")
                    self.dimension = self.index.d
            else:
                if self.metric == "cosine":
                    self.index = faiss.IndexFlatIP(self.dimension)
                else:
                    self.index = faiss.IndexFlatL2(self.dimension)

                logger.info(f"Created new FAISS index (dimension: {self.dimension}, metric: {self.metric})")
                self._save_config()
        except Exception as error:
            logger.error(f"Failed to initialize FAISS index: {error}")
            raise RuntimeError(f"FAISS initialization failed: {error}")

    def _save_config(self) -> None:
        """Save index configuration."""
        config = {
            "index_name": self.index_name,
            "dimension": self.dimension,
            "metric": self.metric,
            "created_at": datetime.now().isoformat(),
            "created_by": "FAISSClient"
        }
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2)

    def _load_metadata(self) -> None:
        """Load metadata from disk."""
        try:
            if self.metadata_path.exists():
                with open(self.metadata_path, "rb") as f:
                    data = pickle.load(f)
                    self.metadata_store = data.get("metadata_store", {})
                    self.id_to_index = data.get("id_to_index", {})
                    self.index_to_id = data.get("index_to_id", {})

                logger.info(f"Loaded metadata for {len(self.metadata_store)} documents")
        except Exception as error:
            logger.error(f"Failed to load metadata: {error}")
            self.metadata_store = {}
            self.id_to_index = {}
            self.index_to_id = {}

    def _load_deleted_ids(self) -> None:
        """Load soft-deleted IDs from disk."""
        try:
            if self.deleted_ids_path.exists():
                with open(self.deleted_ids_path, "rb") as f:
                    data = pickle.load(f)
                    self.deleted_ids = data.get("deleted_ids", set())
                    self.deleted_count = len(self.deleted_ids)
                    logger.info(f"Loaded {self.deleted_count} soft-deleted document IDs")
        except Exception as error:
            logger.error(f"Failed to load deleted IDs: {error}")
            self.deleted_ids = set()
            self.deleted_count = 0

    def _save_metadata(self) -> None:
        """Save metadata to disk."""
        try:
            data = {
                "metadata_store": self.metadata_store,
                "id_to_index": self.id_to_index,
                "index_to_id": self.index_to_id,
                "updated_at": datetime.now().isoformat()
            }
            with open(self.metadata_path, "wb") as f:
                pickle.dump(data, f)
            logger.debug(f"Saved metadata for {len(self.metadata_store)} documents")
        except Exception as error:
            logger.error(f"Failed to save metadata: {error}")

    def _save_deleted_ids(self) -> None:
        """Save soft-deleted IDs to disk."""
        try:
            data = {
                "deleted_ids": self.deleted_ids,
                "updated_at": datetime.now().isoformat()
            }
            with open(self.deleted_ids_path, "wb") as f:
                pickle.dump(data, f)
            logger.debug(f"Saved {len(self.deleted_ids)} deleted IDs")
        except Exception as error:
            logger.error(f"Failed to save deleted IDs: {error}")

    def _save_index(self) -> None:
        """Save FAISS index to disk."""
        try:
            if self.index is not None:
                faiss.write_index(self.index, str(self.index_path))
                logger.debug(f"Saved FAISS index to {self.index_path}")
        except Exception as error:
            logger.error(f"Failed to save index: {error}")

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors for cosine similarity."""
        if self.metric == "cosine":
            faiss.normalize_L2(vectors)
        return vectors

    def add_documents(
        self,
        texts: List[str],
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add documents to vector store."""
        if not texts:
            return []

        if embeddings is None:
            raise ValueError("FAISS requires embeddings for document addition")

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(texts))]

        if metadatas is None:
            metadatas = [{} for _ in range(len(texts))]

        if len(metadatas) != len(texts):
            raise ValueError(f"Metadatas length ({len(metadatas)}) must match texts length ({len(texts)})")

        if len(embeddings) != len(texts):
            raise ValueError(f"Embeddings length ({len(embeddings)}) must match texts length ({len(texts)})")

        embedding_matrix = np.array(embeddings).astype("float32")

        if embedding_matrix.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension ({embedding_matrix.shape[1]}) doesn't match index dimension ({self.dimension})")

        embedding_matrix = self._normalize_vectors(embedding_matrix)

        timestamp = datetime.now().isoformat()

        try:
            start_idx = self.index.ntotal
            self.index.add(embedding_matrix)

            for i, (doc_id, text, metadata) in enumerate(zip(ids, texts, metadatas)):
                position = start_idx + i
                metadata["added_at"] = metadata.get("added_at", timestamp)
                metadata["doc_length"] = metadata.get("doc_length", len(text))

                self.metadata_store[doc_id] = {
                    "text": text,
                    "metadata": metadata,
                    "embedding": embeddings[i]
                }
                self.id_to_index[doc_id] = position
                self.index_to_id[position] = doc_id

            self._save_metadata()
            self._save_index()

            logger.info(f"Added {len(texts)} documents to FAISS index '{self.index_name}'")
            return ids

        except Exception as error:
            logger.error(f"Failed to add documents: {error}")
            raise RuntimeError(f"Document addition failed: {error}")

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Search for similar documents using text query."""
        return {
            "ids": [],
            "documents": [],
            "metadatas": [],
            "distances": [],
            "count": 0,
            "error": "FAISS requires query embeddings. Use search_with_embeddings() with pre-computed embeddings or set an embedding function."
        }

    def search_with_embeddings(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Search using pre-computed embeddings."""
        if not query_embeddings:
            return self._empty_search_result("No query embeddings provided")

        if self.index.ntotal == 0:
            return self._empty_search_result("Index is empty")

        try:
            query_matrix = np.array(query_embeddings).astype("float32")
            query_matrix = self._normalize_vectors(query_matrix)

            distances, indices = self.index.search(query_matrix, n_results)

            results = self._format_search_results(distances, indices, n_results, where)

            return results

        except Exception as error:
            logger.error(f"Embedding search failed: {error}")
            return self._empty_search_result(str(error))

    def similarity_search(
        self,
        query_embeddings: Union[List[float], List[List[float]]],
        k: Optional[int] = None,
        threshold: float = 0.3
    ) -> List[Tuple[str, float]]:
        """Perform similarity search with score threshold filtering."""
        if k is None:
            k = self.default_top_k

        if threshold < 0 or threshold > 1:
            raise ValueError("Threshold must be between 0 and 1")

        if isinstance(query_embeddings, list) and len(query_embeddings) > 0 and not isinstance(query_embeddings[0], list):
            query_embeddings = [query_embeddings]

        results = self.search_with_embeddings(query_embeddings=query_embeddings, n_results=k)

        if results.get("error"):
            logger.warning(f"Search error: {results['error']}")
            return []

        documents = results.get("documents", [])
        distances = results.get("distances", [])

        filtered_results = []
        for doc, dist in zip(documents, distances):
            similarity = self._distance_to_similarity(dist)
            if similarity >= threshold:
                filtered_results.append((doc, similarity))

        logger.debug(f"Similarity search: {len(filtered_results)} results after threshold {threshold}")
        return filtered_results

    def batch_similarity_search(
        self,
        query_embeddings_list: Union[List[List[float]], List[List[List[float]]]],
        k: int = 5,
        threshold: float = 0.3
    ) -> List[List[Tuple[str, float]]]:
        """Perform batch similarity search for multiple queries."""
        results = []

        try:
            if not query_embeddings_list:
                return results

            if isinstance(query_embeddings_list[0], (list, float)) and not isinstance(query_embeddings_list[0][0], (list, float)) if query_embeddings_list[0] else False:
                result = self.similarity_search(query_embeddings_list, k, threshold)
                results.append(result)
            else:
                for query_emb in query_embeddings_list:
                    try:
                        result = self.similarity_search(query_emb, k, threshold)
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Batch search failed for query: {e}")
                        results.append([])

        except Exception as error:
            logger.error(f"Batch similarity search failed: {error}")

        return results

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve document by ID."""
        try:
            if doc_id in self.deleted_ids:
                logger.debug(f"Document {doc_id} is soft-deleted")
                return None

            if doc_id in self.metadata_store:
                doc_data = self.metadata_store[doc_id]
                return {
                    "id": doc_id,
                    "document": doc_data["text"],
                    "metadata": doc_data["metadata"],
                    "embedding": doc_data.get("embedding")
                }

            logger.debug(f"Document not found: {doc_id}")
            return None

        except Exception as error:
            logger.error(f"Failed to retrieve document {doc_id}: {error}")
            return None

    def update_document(
        self,
        doc_id: str,
        text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None
    ) -> bool:
        """Update existing document in vector store."""
        try:
            if doc_id not in self.metadata_store:
                logger.warning(f"Document {doc_id} not found")
                return False

            if doc_id in self.deleted_ids:
                logger.warning(f"Document {doc_id} is soft-deleted, cannot update")
                return False

            current_data = self.metadata_store[doc_id]

            new_text = text if text is not None else current_data["text"]
            new_metadata = metadata if metadata is not None else current_data["metadata"].copy()
            new_embedding = embedding if embedding is not None else current_data.get("embedding")

            if new_embedding is None:
                logger.error("Embedding is required for FAISS update")
                return False

            new_metadata["updated_at"] = datetime.now().isoformat()

            self.deleted_ids.add(doc_id)
            self.deleted_count += 1

            new_id = f"{doc_id}_updated_{datetime.now().strftime('%Y%m%d%H%M%S')}"

            new_ids = self.add_documents(
                texts=[new_text],
                embeddings=[new_embedding],
                metadatas=[new_metadata],
                ids=[new_id]
            )

            success = len(new_ids) > 0

            if success:
                self.metadata_store[doc_id] = self.metadata_store[new_id].copy()
                self.metadata_store[doc_id]["metadata"]["original_id"] = doc_id
                self.id_to_index[doc_id] = self.id_to_index[new_id]

                del self.metadata_store[new_id]
                del self.id_to_index[new_id]

                self._save_deleted_ids()
                self._save_metadata()

                logger.info(f"Updated document: {doc_id}")

            return success

        except Exception as error:
            logger.error(f"Failed to update document {doc_id}: {error}")
            return False

    def delete_documents(self, doc_ids: List[str]) -> bool:
        """Delete documents from vector store using soft-delete mechanism."""
        if not doc_ids:
            return True

        try:
            existing_ids = [doc_id for doc_id in doc_ids if doc_id in self.metadata_store]

            if not existing_ids:
                logger.warning("No valid document IDs to delete")
                return True

            for doc_id in existing_ids:
                self.deleted_ids.add(doc_id)
                self.deleted_count += 1

            self._save_deleted_ids()

            total_docs = len(self.metadata_store)
            deletion_ratio = self.deleted_count / total_docs if total_docs > 0 else 0

            if deletion_ratio > self.REBUILD_THRESHOLD:
                logger.info(f"Deletion ratio ({deletion_ratio:.2%}) exceeds threshold, rebuilding index")
                self._rebuild_index()

            logger.info(f"Soft-deleted {len(existing_ids)} documents from FAISS index")
            return True

        except Exception as error:
            logger.error(f"Failed to delete documents: {error}")
            return False

    def _rebuild_index(self) -> None:
        """Rebuild FAISS index from non-deleted documents."""
        try:
            active_docs = {
                doc_id: doc_data
                for doc_id, doc_data in self.metadata_store.items()
                if doc_id not in self.deleted_ids
            }

            if not active_docs:
                if self.metric == "cosine":
                    self.index = faiss.IndexFlatIP(self.dimension)
                else:
                    self.index = faiss.IndexFlatL2(self.dimension)

                self.id_to_index = {}
                self.index_to_id = {}
                self.metadata_store = {}
                self.deleted_ids = set()
                self.deleted_count = 0

                logger.info("Rebuilt empty FAISS index")
            else:
                all_embeddings = []
                new_id_to_index = {}
                new_index_to_id = {}
                new_metadata_store = {}

                for idx, (doc_id, doc_data) in enumerate(active_docs.items()):
                    embedding = doc_data.get("embedding")
                    if embedding is not None:
                        all_embeddings.append(embedding)
                        new_id_to_index[doc_id] = idx
                        new_index_to_id[idx] = doc_id
                        new_metadata_store[doc_id] = doc_data

                if all_embeddings:
                    embedding_matrix = np.array(all_embeddings).astype("float32")
                    embedding_matrix = self._normalize_vectors(embedding_matrix)

                    if self.metric == "cosine":
                        self.index = faiss.IndexFlatIP(self.dimension)
                    else:
                        self.index = faiss.IndexFlatL2(self.dimension)

                    self.index.add(embedding_matrix)

                    self.id_to_index = new_id_to_index
                    self.index_to_id = new_index_to_id
                    self.metadata_store = new_metadata_store
                    self.deleted_ids = set()
                    self.deleted_count = 0

                    logger.info(f"Rebuilt FAISS index with {len(all_embeddings)} documents")

            self._save_index()
            self._save_metadata()
            self._save_deleted_ids()

        except Exception as error:
            logger.error(f"Failed to rebuild index: {error}")
            raise

    def force_rebuild_index(self) -> bool:
        """Force immediate index rebuild regardless of deletion ratio."""
        try:
            self._rebuild_index()
            return True
        except Exception as error:
            logger.error(f"Force rebuild failed: {error}")
            return False

    def get_collection_info(self) -> Dict[str, Any]:
        """Retrieve collection information and statistics."""
        try:
            active_count = len([doc_id for doc_id in self.metadata_store if doc_id not in self.deleted_ids])

            config = {}
            if self.config_path.exists():
                with open(self.config_path, "r") as f:
                    config = json.load(f)

            return {
                "name": self.index_name,
                "document_count": active_count,
                "total_documents": len(self.metadata_store),
                "deleted_documents": self.deleted_count,
                "has_documents": active_count > 0,
                "dimension": self.dimension,
                "metric": self.metric,
                "persist_directory": str(self.persist_directory),
                "index_size": self.index.ntotal if self.index else 0,
                "config": config,
                "default_top_k": self.default_top_k
            }

        except Exception as error:
            logger.error(f"Failed to retrieve collection info: {error}")
            return {
                "name": self.index_name,
                "error": str(error),
                "persist_directory": str(self.persist_directory)
            }

    def reset_collection(self) -> bool:
        """Delete and recreate collection."""
        try:
            for file_path in [self.index_path, self.metadata_path, self.config_path, self.deleted_ids_path]:
                if file_path.exists():
                    os.remove(file_path)

            self.metadata_store = {}
            self.id_to_index = {}
            self.index_to_id = {}
            self.deleted_ids = set()
            self.deleted_count = 0

            self._initialize_index()

            logger.info(f"FAISS index '{self.index_name}' has been reset")
            return True

        except Exception as error:
            logger.error(f"Failed to reset collection: {error}")
            return False

    def create_new_collection(
        self,
        name: str,
        dimension: Optional[int] = None,
        metric: str = "cosine"
    ) -> bool:
        """Create new collection in database."""
        try:
            self.index_name = name
            if dimension is not None:
                self.dimension = dimension
            self.metric = metric

            self.index_path = self.persist_directory / f"{name}.faiss"
            self.metadata_path = self.persist_directory / f"{name}_metadata.pkl"
            self.config_path = self.persist_directory / f"{name}_config.json"
            self.deleted_ids_path = self.persist_directory / f"{name}_deleted.pkl"

            self.reset_collection()

            logger.info(f"Created new FAISS index: {name}")
            return True

        except Exception as error:
            logger.error(f"Failed to create collection '{name}': {error}")
            return False

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on vector store."""
        try:
            info = self.get_collection_info()

            is_healthy = self.index is not None
            if is_healthy and self.index.ntotal > 0:
                test_vector = np.random.randn(1, self.dimension).astype("float32")
                test_vector = self._normalize_vectors(test_vector)
                try:
                    self.index.search(test_vector, 1)
                except:
                    is_healthy = False

            return {
                "status": "healthy" if is_healthy else "degraded",
                "collection_name": self.index_name,
                "document_count": info.get("document_count", 0),
                "has_documents": info.get("has_documents", False),
                "dimension": self.dimension,
                "metric": self.metric,
                "index_size": self.index.ntotal if self.index else 0,
                "persist_directory": str(self.persist_directory),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as error:
            return {
                "status": "unhealthy",
                "error": str(error),
                "timestamp": datetime.now().isoformat()
            }

    def set_top_k(self, k: int) -> None:
        """Set default number of results for search operations."""
        if k <= 0:
            raise ValueError("top_k value must be positive")
        self.default_top_k = k
        logger.info(f"Default top_k set to: {k}")

    def list_collections(self) -> List[str]:
        """List all collections in the database."""
        try:
            collections = []
            for file in self.persist_directory.glob("*.faiss"):
                collections.append(file.stem)
            return collections
        except Exception as error:
            logger.error(f"Failed to list collections: {error}")
            return []

    def calculate_similarities(
        self,
        query_embedding: List[float],
        doc_embeddings: List[List[float]]
    ) -> List[float]:
        """Calculate cosine similarities between query and document embeddings."""
        try:
            query_vec = np.array(query_embedding)
            doc_vecs = np.array(doc_embeddings)

            query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
            doc_norms = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-10)

            if self.metric == "cosine":
                similarities = np.dot(doc_norms, query_norm)
            else:
                distances = np.linalg.norm(doc_vecs - query_vec, axis=1)
                similarities = 1.0 / (1.0 + distances)

            return similarities.tolist()

        except Exception as error:
            logger.error(f"Failed to calculate similarities: {error}")
            return []

    def get_embedding_function(self, model_name: str = "all-MiniLM-L6-v2"):
        """Create embedding function using Sentence Transformers."""
        try:
            from sentence_transformers import SentenceTransformer

            class EmbeddingFunction:
                def __init__(self, model_name: str, dimension: int):
                    self.model = SentenceTransformer(model_name)
                    self.model_name = model_name
                    self.dimension = dimension
                    logger.info(f"Loaded embedding model: {model_name}")

                def __call__(self, texts: List[str]) -> List[List[float]]:
                    if not texts:
                        return []
                    try:
                        embeddings = self.model.encode(texts)
                        return embeddings.tolist()
                    except Exception as error:
                        logger.error(f"Failed to generate embeddings: {error}")
                        return []

                def get_model_info(self) -> Dict[str, Any]:
                    return {
                        "model_name": self.model_name,
                        "embedding_dimension": self.dimension,
                        "max_seq_length": self.model.max_seq_length,
                        "device": str(self.model.device)
                    }

            return EmbeddingFunction(model_name, self.dimension)

        except ImportError:
            error_msg = "sentence-transformers not installed. Install with: pip install sentence-transformers"
            logger.error(error_msg)
            raise ImportError(error_msg)

    def _format_search_results(
        self,
        distances: np.ndarray,
        indices: np.ndarray,
        n_results: int,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Format search results into consistent structure."""
        ids = []
        documents = []
        metadatas = []
        filtered_distances = []

        for query_idx in range(distances.shape[0]):
            query_ids = []
            query_docs = []
            query_metadatas = []
            query_distances = []

            for rank in range(min(n_results, distances.shape[1])):
                idx = indices[query_idx][rank]
                dist = distances[query_idx][rank]

                if idx == -1 or idx >= len(self.index_to_id):
                    continue

                doc_id = self.index_to_id.get(idx)
                if doc_id is None:
                    continue

                if doc_id in self.deleted_ids:
                    continue

                doc_data = self.metadata_store.get(doc_id)
                if doc_data is None:
                    continue

                if where and not self._matches_filter(doc_data["metadata"], where):
                    continue

                query_ids.append(doc_id)
                query_docs.append(doc_data["text"])
                query_metadatas.append(doc_data["metadata"])
                query_distances.append(float(dist))

            ids.append(query_ids)
            documents.append(query_docs)
            metadatas.append(query_metadatas)
            filtered_distances.append(query_distances)

        return {
            "ids": ids[0] if ids else [],
            "documents": documents[0] if documents else [],
            "metadatas": metadatas[0] if metadatas else [],
            "distances": filtered_distances[0] if filtered_distances else [],
            "count": len(ids[0]) if ids else 0
        }

    def _matches_filter(self, metadata: Dict[str, Any], where: Dict[str, Any]) -> bool:
        """Check if metadata matches filter conditions."""
        for key, value in where.items():
            if key not in metadata:
                return False
            if metadata[key] != value:
                return False
        return True

    def _empty_search_result(self, error_message: str) -> Dict[str, Any]:
        """Return empty search result with error message."""
        return {
            "ids": [],
            "documents": [],
            "metadatas": [],
            "distances": [],
            "count": 0,
            "error": error_message
        }

    @staticmethod
    def _distance_to_similarity(distance: float) -> float:
        """Convert FAISS distance to similarity score."""
        if distance <= 0:
            return 1.0
        similarity = 1.0 / (1.0 + distance)
        return similarity

    def test_similarity_conversion(self):
        """Test function for similarity conversion."""
        test_distances = [0, 0.5, 1.0, 1.5, 2.0]
        print("\nFAISS Similarity Conversion Test:")
        for d in test_distances:
            sim = self._distance_to_similarity(d)
            print(f"  Distance: {d:.1f} -> Similarity: {sim:.3f}")


_faiss_instance: Optional[FAISSClient] = None


def get_faiss_client(
    persist_directory: Optional[Path] = None,
    index_name: str = "documents",
    dimension: int = 384,
    metric: str = "cosine"
) -> FAISSClient:
    """Get or create singleton FAISSClient instance."""
    global _faiss_instance

    if _faiss_instance is None:
        _faiss_instance = FAISSClient(persist_directory, index_name, dimension, metric)
    elif (persist_directory is not None and
          Path(persist_directory) != _faiss_instance.persist_directory) or \
         index_name != _faiss_instance.index_name:
        logger.warning("Requested different config than existing singleton, reinitializing")
        _faiss_instance = FAISSClient(persist_directory, index_name, dimension, metric)

    return _faiss_instance


def test_faiss_client():
    """Test function for FAISSClient."""
    import time

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    print("=" * 60)
    print("FAISS CLIENT TEST SUITE")
    print("=" * 60)

    try:
        client = FAISSClient(dimension=384)

        print("\n1. HEALTH CHECK")
        print("-" * 40)
        health = client.health_check()
        print(f"Status: {health['status']}")
        print(f"Document Count: {health.get('document_count', 0)}")
        print(f"Dimension: {health.get('dimension')}")
        print(f"Metric: {health.get('metric')}")

        print("\n2. COLLECTION MANAGEMENT")
        print("-" * 40)
        print(f"Index Name: {client.index_name}")
        print(f"Default Top K: {client.default_top_k}")

        collections = client.list_collections()
        print(f"Available Collections: {collections}")

        print("\n3. DOCUMENT OPERATIONS")
        print("-" * 40)

        test_documents = [
            "Artificial intelligence is revolutionizing multiple industries.",
            "Machine learning algorithms require extensive training data.",
            "Natural language processing enables computers to understand human language.",
            "Python is the most popular language for data science.",
            "Vector databases optimize similarity search for AI applications."
        ]

        test_metadata = [
            {"category": "AI", "source": "test_data", "language": "en", "version": 1},
            {"category": "ML", "source": "test_data", "language": "en", "version": 1},
            {"category": "NLP", "source": "test_data", "language": "en", "version": 1},
            {"category": "programming", "source": "test_data", "language": "en", "version": 1},
            {"category": "database", "source": "test_data", "language": "en", "version": 1}
        ]

        import numpy as np
        np.random.seed(42)
        test_embeddings = np.random.randn(len(test_documents), 384).tolist()

        print("Adding test documents...")
        doc_ids = client.add_documents(
            texts=test_documents,
            embeddings=test_embeddings,
            metadatas=test_metadata
        )
        print(f"Added {len(doc_ids)} documents")

        print("\n4. SEARCH OPERATIONS")
        print("-" * 40)

        query_embeddings = np.random.randn(1, 384).tolist()

        print("Searching with embeddings...")
        results = client.search_with_embeddings(
            query_embeddings=query_embeddings,
            n_results=3
        )
        print(f"Found {results['count']} results")

        if results["count"] > 0:
            for i, (doc, metadata, dist) in enumerate(zip(results["documents"], results["metadatas"], results["distances"]), 1):
                similarity = client._distance_to_similarity(dist)
                print(f"  {i}. Category: {metadata.get('category', 'N/A')} (Similarity: {similarity:.4f})")
                print(f"     Document: {doc[:60]}...")

        print("\n5. SIMILARITY SEARCH WITH THRESHOLD")
        print("-" * 40)
        similarity_results = client.similarity_search(query_embeddings[0], k=3, threshold=0.3)
        print(f"Results above threshold 0.3: {len(similarity_results)}")

        for i, (doc, score) in enumerate(similarity_results, 1):
            print(f"  {i}. Similarity: {score:.4f}")
            print(f"     Document: {doc[:70]}...")

        print("\n6. BATCH SEARCH")
        print("-" * 40)
        batch_queries = np.random.randn(3, 384).tolist()

        print("Testing batch search with multiple queries...")
        batch_results = client.batch_similarity_search(batch_queries, k=2, threshold=0.3)

        for i, results in enumerate(batch_results):
            print(f"  Query {i+1}: Found {len(results)} results")
            for j, (doc, score) in enumerate(results[:2]):
                print(f"    {j+1}. Score: {score:.4f} - {doc[:50]}...")

        print("\n7. DOCUMENT RETRIEVAL")
        print("-" * 40)
        if doc_ids:
            sample_id = doc_ids[0]
            document = client.get_document(sample_id)
            if document:
                print(f"Retrieved document ID: {document['id']}")
                print(f"Category: {document['metadata'].get('category', 'N/A')}")
                print(f"Document length: {len(document['document'])} characters")

        print("\n8. COLLECTION INFORMATION")
        print("-" * 40)
        info = client.get_collection_info()
        for key, value in info.items():
            if key not in ["config"]:
                print(f"{key}: {value}")

        print("\n9. SOFT-DELETE TEST")
        print("-" * 40)
        if len(doc_ids) >= 2:
            delete_ids = [doc_ids[0]]
            success = client.delete_documents(delete_ids)
            print(f"Soft-delete successful: {success}")

            results_after_delete = client.search_with_embeddings(query_embeddings, n_results=5)
            deleted_doc_ids = [doc_ids[0]]
            found_deleted = any(doc_id in deleted_doc_ids for doc_id in results_after_delete.get("ids", []))
            print(f"Deleted document in search results: {found_deleted} (should be False)")

            info_after_delete = client.get_collection_info()
            print(f"Active documents: {info_after_delete.get('document_count')}")
            print(f"Deleted documents: {info_after_delete.get('deleted_documents')}")

        print("\n10. FORCE REBUILD TEST")
        print("-" * 40)
        rebuild_success = client.force_rebuild_index()
        print(f"Force rebuild successful: {rebuild_success}")

        info_after_rebuild = client.get_collection_info()
        print(f"Documents after rebuild: {info_after_rebuild.get('document_count')}")
        print(f"Deleted count after rebuild: {info_after_rebuild.get('deleted_documents')}")

        print("\n11. UPDATE OPERATION")
        print("-" * 40)
        if doc_ids:
            update_id = doc_ids[1] if len(doc_ids) > 1 else doc_ids[0]
            updated_text = "Updated: Artificial intelligence and deep learning are transforming industries."
            updated_embedding = np.random.randn(384).tolist()

            success = client.update_document(
                doc_id=update_id,
                text=updated_text,
                embedding=updated_embedding,
                metadata={"category": "AI-UPDATED", "updated": True}
            )
            print(f"Update successful: {success}")

            updated_doc = client.get_document(update_id)
            if updated_doc:
                print(f"Updated category: {updated_doc['metadata'].get('category')}")

        print("\n12. CLEANUP")
        print("-" * 40)
        if doc_ids:
            success = client.delete_documents(doc_ids)
            if success:
                print(f"Deleted {len(doc_ids)} test documents")
            else:
                print("Failed to delete test documents")

        print("\n13. FINAL HEALTH CHECK")
        print("-" * 40)
        final_health = client.health_check()
        print(f"Status: {final_health['status']}")
        print(f"Final Document Count: {final_health.get('document_count', 0)}")

        print("\n14. SIMILARITY CONVERSION TEST")
        print("-" * 40)
        client.test_similarity_conversion()

        print("\n15. PERFORMANCE TEST")
        print("-" * 40)
        print("Adding 100 test documents for performance measurement...")
        many_docs = [f"Test document {i}" for i in range(100)]
        many_embeddings = np.random.randn(100, 384).tolist()
        many_ids = client.add_documents(many_docs, many_embeddings)

        start_time = time.time()
        test_query = np.random.randn(1, 384).tolist()
        perf_results = client.search_with_embeddings(test_query, n_results=10)
        search_time = time.time() - start_time

        print(f"Search on 100 documents: {search_time:.4f} seconds")
        print(f"Found {perf_results['count']} results")

        client.delete_documents(many_ids)

        print("\n" + "=" * 60)
        print("TEST SUITE COMPLETED SUCCESSFULLY")
        print("=" * 60)

        return True

    except Exception as error:
        print(f"\nTEST FAILED: {error}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_faiss_client()
    sys.exit(0 if success else 1)