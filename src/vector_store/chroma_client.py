# cockatoo_v1/src/vector_store/chroma_client.py

"""
ChromaDB Vector Store Client for cockatoo_v1.
Provides CRUD operations and search capabilities for vector embeddings.
"""

import os
import logging
import uuid
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import chromadb
import numpy as np
from chromadb.config import Settings

os.environ["ANONYMIZED_TELEMETRY"] = "False"

logging.getLogger("chromadb.telemetry").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("chromadb.db").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

try:
    from ..core.constants import DATABASE_DIR, VECTOR_DB_NAME
except (ImportError, ValueError):
    try:
        from src.core.constants import DATABASE_DIR, VECTOR_DB_NAME
    except ImportError:
        DATABASE_DIR = Path.home() / ".cockatoo_v1" / "database"
        VECTOR_DB_NAME = "chroma"
        logger.info(f"Using fallback constants: DATABASE_DIR={DATABASE_DIR}, VECTOR_DB_NAME={VECTOR_DB_NAME}")


class ChromaClient:
    """ChromaDB vector store client for document storage and retrieval."""

    def __init__(self, persist_directory: Optional[Path] = None, collection_name: str = "documents") -> None:
        """Initialize ChromaDB client with persistent storage."""
        if persist_directory is None:
            persist_directory = DATABASE_DIR / VECTOR_DB_NAME

        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.default_top_k = 5

        self.persist_directory.mkdir(parents=True, exist_ok=True)

        settings = Settings(
            anonymized_telemetry=False,
            allow_reset=True,
            is_persistent=True,
            chroma_server_ssl_enabled=False
        )

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=settings
        )

        self.collection = self._get_or_create_collection()
        logger.info(f"ChromaDB client initialized: {self.persist_directory}")

    def _get_or_create_collection(self) -> chromadb.Collection:
        """Retrieve existing collection or create a new one."""
        try:
            collection = self.client.get_collection(name=self.collection_name)
            logger.debug(f"Using existing collection: {self.collection_name}")
            return collection
        except ValueError:
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={
                    "description": "cockatoo_v1 document embeddings",
                    "created_at": datetime.now().isoformat(),
                    "created_by": "ChromaClient"
                }
            )
            logger.info(f"Created new collection: {self.collection_name}")
            return collection

    def add_documents(
        self,
        texts: List[str],
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add documents to the vector store."""
        if not texts:
            return []

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(texts))]

        if metadatas is None:
            metadatas = [{} for _ in range(len(texts))]

        if len(metadatas) != len(texts):
            raise ValueError(f"Metadatas length ({len(metadatas)}) must match texts length ({len(texts)})")

        if embeddings is not None and len(embeddings) != len(texts):
            raise ValueError(f"Embeddings length ({len(embeddings)}) must match texts length ({len(texts)})")

        timestamp = datetime.now().isoformat()
        for i, metadata in enumerate(metadatas):
            metadata["added_at"] = metadata.get("added_at", timestamp)
            metadata["doc_length"] = metadata.get("doc_length", len(texts[i]))

        try:
            if embeddings:
                self.collection.add(embeddings=embeddings, documents=texts, metadatas=metadatas, ids=ids)
            else:
                self.collection.add(documents=texts, metadatas=metadatas, ids=ids)

            logger.info(f"Added {len(texts)} documents to collection '{self.collection_name}'")
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
        """Search for similar documents using a text query."""
        if include is None:
            include = ["documents", "metadatas", "distances"]

        if not query or not query.strip():
            return self._empty_search_result("Empty query")

        try:
            count = self.collection.count()
            if count == 0:
                logger.warning("Collection is empty, no results to return")
                return self._empty_search_result("Collection is empty")

            if n_results > count:
                logger.debug(f"Adjusting n_results from {n_results} to {count} (collection size)")
                n_results = count

            results = self.collection.query(
                query_texts=[query.strip()],
                n_results=n_results,
                where=where,
                where_document=where_document,
                include=include
            )
            return self._format_search_results(results)
        except Exception as error:
            logger.error(f"Search failed: {error}")
            return self._empty_search_result(str(error))

    def search_with_embeddings(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Search using pre-computed embeddings."""
        if not query_embeddings:
            return self._empty_search_result("No query embeddings provided")

        try:
            count = self.collection.count()
            if count == 0:
                logger.warning("Collection is empty, no results to return")
                return self._empty_search_result("Collection is empty")

            if n_results > count:
                logger.debug(f"Adjusting n_results from {n_results} to {count} (collection size)")
                n_results = count

            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"]
            )
            return self._format_search_results(results)
        except Exception as error:
            logger.error(f"Embedding search failed: {error}")
            return self._empty_search_result(str(error))

    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
        threshold: float = 0.3
    ) -> List[Tuple[str, float]]:
        """Perform similarity search with score threshold filtering."""
        if k is None:
            k = self.default_top_k

        if threshold < 0 or threshold > 1:
            raise ValueError("Threshold must be between 0 and 1")

        results = self.search(query=query, n_results=k)

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
        queries: List[str],
        k: int = 5,
        threshold: float = 0.3
    ) -> List[List[Tuple[str, float]]]:
        """Perform batch similarity search for multiple queries."""
        if not queries:
            return []

        results = []
        for query in queries:
            try:
                search_results = self.search(query=query, n_results=k)

                if search_results.get("error") or search_results.get("count", 0) == 0:
                    results.append([])
                    continue

                documents = search_results.get("documents", [])
                distances = search_results.get("distances", [])

                filtered = []
                for doc, dist in zip(documents, distances):
                    similarity = self._distance_to_similarity(dist)
                    if similarity >= threshold:
                        filtered.append((doc, similarity))

                results.append(filtered)

            except Exception as e:
                logger.error(f"Batch search failed for query '{query}': {e}")
                results.append([])

        return results

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a document by its ID."""
        try:
            results = self.collection.get(ids=[doc_id], include=["documents", "metadatas"])
            if results.get("ids") and len(results["ids"]) > 0:
                return {
                    "id": results["ids"][0],
                    "document": results.get("documents", [None])[0],
                    "metadata": results.get("metadatas", [None])[0]
                }
            logger.debug(f"Document not found: {doc_id}")
            return None
        except Exception as error:
            logger.error(f"Failed to retrieve document {doc_id}: {error}")
            return None

    def get_documents(
        self,
        where: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get multiple documents with filtering and pagination."""
        try:
            results = self.collection.get(
                where=where,
                limit=limit,
                offset=offset,
                include=["documents", "metadatas"]
            )

            documents = []
            for i in range(len(results.get("ids", []))):
                documents.append({
                    "id": results["ids"][i],
                    "document": results.get("documents", [None])[i],
                    "metadata": results.get("metadatas", [None])[i]
                })

            return documents
        except Exception as error:
            logger.error(f"Failed to get documents: {error}")
            return []

    def get_all_documents(self, batch_size: int = 1000) -> List[Dict[str, Any]]:
        """Retrieve all documents from the collection using pagination."""
        all_documents = []
        offset = 0

        while True:
            batch = self.get_documents(limit=batch_size, offset=offset)
            if not batch:
                break

            all_documents.extend(batch)

            if len(batch) < batch_size:
                break

            offset += batch_size

            if offset > 100000:
                logger.warning("Reached safety limit for document retrieval")
                break

        return all_documents

    def count_documents(self, where: Optional[Dict[str, Any]] = None) -> int:
        """Count documents in the collection, optionally with filter."""
        try:
            if where:
                results = self.collection.get(where=where)
                return len(results["ids"]) if results and "ids" in results else 0
            else:
                return self.collection.count()
        except Exception as error:
            logger.error(f"Failed to count documents: {error}")
            return 0

    def peek(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get a sample of documents from the collection."""
        try:
            results = self.collection.peek(limit=n)

            documents = []
            for i in range(len(results.get("ids", []))):
                documents.append({
                    "id": results["ids"][i],
                    "document": results.get("documents", [None])[i],
                    "metadata": results.get("metadatas", [None])[i]
                })

            return documents
        except Exception as error:
            logger.error(f"Failed to peek documents: {error}")
            return []

    def update_document(
        self,
        doc_id: str,
        text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None
    ) -> bool:
        """Update an existing document."""
        try:
            update_data = {}
            if text is not None:
                update_data["documents"] = [text]
            if metadata is not None:
                metadata["updated_at"] = datetime.now().isoformat()
                update_data["metadatas"] = [metadata]
            if embedding is not None:
                update_data["embeddings"] = [embedding]

            if not update_data:
                logger.warning("No update data provided")
                return False

            self.collection.update(ids=[doc_id], **update_data)
            logger.info(f"Updated document: {doc_id}")
            return True
        except Exception as error:
            logger.error(f"Failed to update document {doc_id}: {error}")
            return False

    def delete_documents(self, doc_ids: List[str]) -> bool:
        """Delete documents from the vector store."""
        if not doc_ids:
            return True

        try:
            self.collection.delete(ids=doc_ids)
            logger.info(f"Deleted {len(doc_ids)} documents")
            return True
        except Exception as error:
            logger.error(f"Failed to delete documents: {error}")
            return False

    def get_collection_info(self) -> Dict[str, Any]:
        """Retrieve collection information and statistics."""
        try:
            count = self.collection.count()
            return {
                "name": self.collection_name,
                "document_count": count,
                "has_documents": count > 0,
                "persist_directory": str(self.persist_directory),
                "collection_metadata": self.collection.metadata,
                "default_top_k": self.default_top_k
            }
        except Exception as error:
            logger.error(f"Failed to retrieve collection info: {error}")
            return {
                "name": self.collection_name,
                "error": str(error),
                "persist_directory": str(self.persist_directory)
            }

    def reset_collection(self) -> bool:
        """Delete and recreate the collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self._get_or_create_collection()
            logger.info(f"Collection '{self.collection_name}' has been reset")
            return True
        except Exception as error:
            logger.error(f"Failed to reset collection: {error}")
            return False

    def create_new_collection(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Create a new collection."""
        if metadata is None:
            metadata = {}
        metadata["created_at"] = datetime.now().isoformat()
        metadata["created_by"] = "ChromaClient"

        try:
            self.client.create_collection(name=name, metadata=metadata)
            logger.info(f"Created new collection: {name}")
            return True
        except Exception as error:
            logger.error(f"Failed to create collection '{name}': {error}")
            return False

    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the vector store."""
        try:
            info = self.get_collection_info()
            return {
                "status": "healthy",
                "collection_name": self.collection_name,
                "document_count": info.get("document_count", 0),
                "has_documents": info.get("has_documents", False),
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
        """Set the default number of results for search operations."""
        if k <= 0:
            raise ValueError("top_k value must be positive")
        self.default_top_k = k
        logger.info(f"Default top_k set to: {k}")

    def list_collections(self) -> List[str]:
        """List all collections in the database."""
        try:
            collections = self.client.list_collections()
            return [col.name for col in collections]
        except Exception as error:
            logger.error(f"Failed to list collections: {error}")
            return []

    def calculate_similarities(self, query_embedding: List[float], doc_embeddings: List[List[float]]) -> List[float]:
        """Calculate cosine similarities between query and document embeddings."""
        try:
            query_vec = np.array(query_embedding)
            doc_vecs = np.array(doc_embeddings)
            query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
            doc_norms = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-10)
            similarities = np.dot(doc_norms, query_norm)
            return similarities.tolist()
        except Exception as error:
            logger.error(f"Failed to calculate similarities: {error}")
            return []

    def get_embedding_function(self, model_name: str = "all-MiniLM-L6-v2"):
        """Create an embedding function using Sentence Transformers."""
        try:
            from sentence_transformers import SentenceTransformer

            class EmbeddingFunction:
                def __init__(self, model_name: str):
                    self.model = SentenceTransformer(model_name)
                    self.model_name = model_name
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
                        "embedding_dimension": self.model.get_sentence_embedding_dimension(),
                        "max_seq_length": self.model.max_seq_length,
                        "device": str(self.model.device)
                    }

            return EmbeddingFunction(model_name)
        except ImportError:
            error_msg = "sentence-transformers not installed. Install with: pip install sentence-transformers"
            logger.error(error_msg)
            raise ImportError(error_msg)

    def _format_search_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Format search results into a consistent structure."""
        return {
            "ids": results.get("ids", [[]])[0] if results.get("ids") else [],
            "documents": results.get("documents", [[]])[0] if results.get("documents") else [],
            "metadatas": results.get("metadatas", [[]])[0] if results.get("metadatas") else [],
            "distances": results.get("distances", [[]])[0] if results.get("distances") else [],
            "count": len(results.get("ids", [[]])[0]) if results.get("ids") else 0
        }

    def _empty_search_result(self, error_message: str) -> Dict[str, Any]:
        """Return an empty search result with an error message."""
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
        """Convert ChromaDB distance to a similarity score (0-1)."""
        if distance <= 0:
            return 1.0
        similarity = 1.0 / (1.0 + distance)
        return similarity

    def test_similarity_conversion(self):
        """Test function for similarity conversion."""
        test_distances = [0, 0.5, 1.0, 1.5, 2.0]
        print("\nSimilarity Conversion Test:")
        for d in test_distances:
            sim = self._distance_to_similarity(d)
            print(f"  Distance: {d:.1f} -> Similarity: {sim:.3f}")


_chroma_instance: Optional[ChromaClient] = None


def get_chroma_client(
    persist_directory: Optional[Path] = None,
    collection_name: str = "documents"
) -> ChromaClient:
    """Get or create a singleton ChromaClient instance."""
    global _chroma_instance
    if _chroma_instance is None:
        _chroma_instance = ChromaClient(persist_directory, collection_name)
    elif (persist_directory is not None and Path(persist_directory) != _chroma_instance.persist_directory) or \
            collection_name != _chroma_instance.collection_name:
        logger.warning("Requested different config than existing singleton, reinitializing")
        _chroma_instance = ChromaClient(persist_directory, collection_name)
    return _chroma_instance


def test_chroma_client():
    """Test function for ChromaClient."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    print("=" * 60)
    print("CHROMADB CLIENT TEST SUITE")
    print("=" * 60)

    try:
        client = ChromaClient()

        print("\n1. HEALTH CHECK")
        print("-" * 40)
        health = client.health_check()
        print(f"Status: {health['status']}")
        print(f"Document Count: {health.get('document_count', 0)}")
        print(f"Has Documents: {health.get('has_documents', False)}")

        print("\n2. COLLECTION MANAGEMENT")
        print("-" * 40)
        print(f"Collection Name: {client.collection_name}")
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

        print("Adding test documents...")
        doc_ids = client.add_documents(texts=test_documents, metadatas=test_metadata)
        print(f"Added {len(doc_ids)} documents")

        print("\n4. COUNT DOCUMENTS")
        print("-" * 40)
        total_count = client.count_documents()
        filtered_count = client.count_documents(where={"category": "AI"})
        print(f"Total documents: {total_count}")
        print(f"Documents with category AI: {filtered_count}")

        print("\n5. PEEK DOCUMENTS")
        print("-" * 40)
        sample_docs = client.peek(3)
        print(f"Peeked {len(sample_docs)} documents:")
        for i, doc in enumerate(sample_docs, 1):
            print(f"  {i}. ID: {doc['id'][:8]}... - {doc['document'][:50]}...")

        print("\n6. SEARCH OPERATIONS")
        print("-" * 40)
        queries = ["artificial intelligence", "machine learning", "data science"]
        for query in queries:
            print(f"\nQuery: '{query}'")
            results = client.search(query=query, n_results=2)
            print(f"  Found {results['count']} results")
            if results["count"] > 0:
                for i, (doc, metadata) in enumerate(zip(results["documents"], results["metadatas"]), 1):
                    print(f"  {i}. Category: {metadata.get('category', 'N/A')}")
                    print(f"     Document: {doc[:60]}...")

        print("\n7. SIMILARITY SEARCH WITH THRESHOLD")
        print("-" * 40)
        query = "AI and machine learning"
        similarity_results = client.similarity_search(query, k=3, threshold=0.3)
        print(f"Query: '{query}'")
        print(f"Results above threshold 0.3: {len(similarity_results)}")
        for i, (doc, score) in enumerate(similarity_results, 1):
            print(f"  {i}. Similarity: {score:.4f}")
            print(f"     Document: {doc[:70]}...")

        print("\n8. BATCH SEARCH")
        print("-" * 40)
        batch_results = client.batch_similarity_search(queries, k=2, threshold=0.3)
        for i, (q, res) in enumerate(zip(queries, batch_results), 1):
            print(f"  Query {i}: '{q}' - Found {len(res)} results")

        print("\n9. GET DOCUMENTS BY FILTER")
        print("-" * 40)
        filtered_docs = client.get_documents(where={"category": "AI"}, limit=2)
        print(f"Found {len(filtered_docs)} documents with category AI:")
        for i, doc in enumerate(filtered_docs, 1):
            print(f"  {i}. {doc['document'][:60]}...")

        print("\n10. DOCUMENT RETRIEVAL")
        print("-" * 40)
        if doc_ids:
            sample_id = doc_ids[0]
            document = client.get_document(sample_id)
            if document:
                print(f"Retrieved document ID: {document['id']}")
                print(f"Category: {document['metadata'].get('category', 'N/A')}")
                print(f"Document length: {len(document['document'])} characters")

        print("\n11. COLLECTION INFORMATION")
        print("-" * 40)
        info = client.get_collection_info()
        for key, value in info.items():
            if key not in ["sample_document"]:
                print(f"{key}: {value}")

        print("\n12. CLEANUP")
        print("-" * 40)
        if doc_ids:
            success = client.delete_documents(doc_ids)
            print(f"Deleted test documents: {success}")

        print("\n13. FINAL HEALTH CHECK")
        print("-" * 40)
        final_health = client.health_check()
        print(f"Status: {final_health['status']}")
        print(f"Final Document Count: {final_health.get('document_count', 0)}")

        print("\n14. SIMILARITY CONVERSION TEST")
        print("-" * 40)
        client.test_similarity_conversion()

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
    success = test_chroma_client()
    sys.exit(0 if success else 1)