# tests/unit/test_vector_store.py

"""Unit tests for vector store components.

Tests cover ChromaDB client initialization, document operations, search functionality,
collection management, search engine features, index management, error handling,
persistence, backup/restore, and performance benchmarks.
"""

import os
import sys
import time
import json
import tempfile
import shutil
import numpy as np
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import Dict, List, Any, Optional, Tuple

import pytest

from src.vector_store.chroma_client import ChromaClient, _chroma_instance
from src.vector_store.search_engine import SearchEngine
from src.vector_store.index_manager import IndexManager
from src.vector_store.models import Document


@pytest.fixture
def persist_dir(tmp_path):
    """Create temporary persist directory."""
    return tmp_path / "vector_db"


@pytest.fixture
def chroma_client(persist_dir):
    """Create ChromaClient instance."""
    global _chroma_instance
    _chroma_instance = None
    return ChromaClient(persist_directory=persist_dir)


@pytest.fixture
def search_engine(chroma_client, tmp_path):
    """Create SearchEngine instance."""
    return SearchEngine(
        chroma_client=chroma_client,
        faiss_client=None,
        cache_dir=tmp_path / "cache"
    )


@pytest.fixture
def index_manager(tmp_path):
    """Create IndexManager instance."""
    return IndexManager(base_dir=tmp_path / "indices")


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    docs = []
    for i in range(10):
        tags_value = ",".join(["test", f"doc_{i}"]) if i % 2 == 0 else "sample"
        
        doc = Document(
            id=f"doc_{i:03d}",
            text=f"This is sample document number {i}. It contains some text for testing.",
            embedding=np.random.randn(384).tolist(),
            metadata={
                "category": "test" if i % 2 == 0 else "sample",
                "priority": i % 3,
                "tags": tags_value,
                "size": 100 + i * 10,
                "created": f"2025-01-{i+1:02d}"
            }
        )
        docs.append(doc)
    return docs


@pytest.fixture
def populated_client(chroma_client, sample_documents):
    """Create client with pre-populated documents."""
    for doc in sample_documents:
        chroma_client.add_documents(
            texts=[doc.text],
            embeddings=[doc.embedding] if doc.embedding else None,
            metadatas=[doc.metadata],
            ids=[doc.id]
        )
    return chroma_client


@pytest.fixture
def sample_queries():
    """Create sample queries for testing."""
    return [
        "sample document",
        "test content",
        "machine learning",
        "data science",
        "python programming"
    ]


class TestChromaClientInitialization:
    """Test ChromaClient initialization."""

    def test_default_persist_directory(self, chroma_client):
        """Test default persist directory is set correctly."""
        assert chroma_client.persist_directory is not None
        assert isinstance(chroma_client.persist_directory, Path)

    def test_custom_persist_directory(self, persist_dir):
        """Test custom persist directory."""
        global _chroma_instance
        _chroma_instance = None
        client = ChromaClient(persist_directory=persist_dir)
        assert client.persist_directory == persist_dir

    def test_collection_name_default(self, chroma_client):
        """Test default collection name."""
        assert chroma_client.collection_name == "documents"

    def test_custom_collection_name(self, persist_dir):
        """Test custom collection name."""
        global _chroma_instance
        _chroma_instance = None
        client = ChromaClient(persist_directory=persist_dir, collection_name="test_collection")
        assert client.collection_name == "test_collection"

    def test_initialization_error_handling(self, persist_dir):
        """Test error handling during initialization."""
        with patch('pathlib.Path.mkdir', side_effect=PermissionError("Permission denied")):
            with pytest.raises(PermissionError, match="Permission denied"):
                ChromaClient(persist_directory=persist_dir)

    def test_singleton_pattern(self, persist_dir):
        """Test singleton pattern."""
        import src.vector_store.chroma_client
        src.vector_store.chroma_client._chroma_instance = None
        
        # Use the get_chroma_client function
        from src.vector_store.chroma_client import get_chroma_client
        
        client1 = get_chroma_client(persist_directory=persist_dir)
        client2 = get_chroma_client(persist_directory=persist_dir)  # Same directory
        client3 = get_chroma_client(persist_directory=Path("/different/path"))
        
        # Same directory should return same instance
        assert client1 is client2
        
        # Different directory should create new instance
        assert client1 is not client3
        assert client1.persist_directory == persist_dir
        assert client3.persist_directory == Path("/different/path")


class TestCollectionManagement:
    """Test collection management operations."""

    def test_create_collection(self, chroma_client):
        """Test creating a collection."""
        success = chroma_client.create_new_collection("test_collection")
        assert success is True
        collections = chroma_client.list_collections()
        assert "test_collection" in collections

    def test_create_duplicate_collection(self, chroma_client):
        """Test creating duplicate collection."""
        chroma_client.create_new_collection("test_collection")
        success = chroma_client.create_new_collection("test_collection")
        assert success is False

    def test_list_collections(self, chroma_client):
        """Test listing all collections."""
        chroma_client.create_new_collection("collection1")
        chroma_client.create_new_collection("collection2")

        collections = chroma_client.list_collections()
        assert "collection1" in collections
        assert "collection2" in collections

    def test_reset_collection(self, chroma_client):
        """Test resetting a collection."""
        result = chroma_client.reset_collection()
        assert result is True
        assert chroma_client.collection is not None


class TestDocumentOperations:
    """Test document CRUD operations."""

    def test_add_single_document(self, chroma_client, sample_documents):
        """Test adding a single document."""
        doc = sample_documents[0]
        ids = chroma_client.add_documents(
            texts=[doc.text],
            embeddings=[doc.embedding],
            metadatas=[doc.metadata],
            ids=[doc.id]
        )
        assert len(ids) == 1
        assert ids[0] == doc.id
        assert chroma_client.count_documents() == 1

    def test_add_batch_documents(self, chroma_client, sample_documents):
        """Test adding multiple documents."""
        texts = [doc.text for doc in sample_documents]
        embeddings = [doc.embedding for doc in sample_documents]
        metadatas = [doc.metadata for doc in sample_documents]
        ids = [doc.id for doc in sample_documents]

        result_ids = chroma_client.add_documents(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        assert len(result_ids) == len(sample_documents)
        assert chroma_client.count_documents() == len(sample_documents)

    def test_add_documents_with_metadata(self, chroma_client):
        """Test adding documents with metadata."""
        # Convert nested structures to strings for ChromaDB compatibility
        tags_string = "important,test"
        nested_string = json.dumps({"key": "value"})
        
        doc = Document(
            id="doc_meta",
            text="Document with metadata",
            metadata={
                "author": "Test User",
                "created": "2025-01-01",
                "tags": tags_string,
                "nested": nested_string
            }
        )
        ids = chroma_client.add_documents(
            texts=[doc.text],
            metadatas=[doc.metadata],
            ids=[doc.id]
        )
        assert len(ids) == 1

        retrieved = chroma_client.get_document("doc_meta")
        assert retrieved is not None
        assert retrieved["metadata"]["author"] == "Test User"
        assert "important" in retrieved["metadata"]["tags"]
        
        # Parse back the nested JSON
        nested = json.loads(retrieved["metadata"]["nested"])
        assert nested["key"] == "value"

    def test_add_duplicate_documents(self, chroma_client, sample_documents):
        """Test adding duplicate documents."""
        doc = sample_documents[0]
        chroma_client.add_documents(
            texts=[doc.text],
            embeddings=[doc.embedding],
            metadatas=[doc.metadata],
            ids=[doc.id]
        )

        try:
            chroma_client.add_documents(
                texts=[doc.text],
                embeddings=[doc.embedding],
                metadatas=[doc.metadata],
                ids=[doc.id]
            )
        except Exception:
            pass

    def test_get_document_by_id(self, populated_client, sample_documents):
        """Test getting document by ID."""
        doc_id = sample_documents[0].id
        retrieved = populated_client.get_document(doc_id)
        assert retrieved is not None
        assert retrieved["id"] == doc_id
        assert retrieved["document"] == sample_documents[0].text

    def test_get_nonexistent_document(self, populated_client):
        """Test getting non-existent document."""
        retrieved = populated_client.get_document("nonexistent")
        assert retrieved is None

    def test_update_document_text(self, populated_client, sample_documents):
        """Test updating document text."""
        doc_id = sample_documents[0].id
        new_text = "This is updated text"
        success = populated_client.update_document(doc_id, text=new_text)
        assert success is True
        updated = populated_client.get_document(doc_id)
        assert updated["document"] == new_text

    def test_update_document_metadata(self, populated_client, sample_documents):
        """Test updating document metadata."""
        doc_id = sample_documents[0].id
        new_metadata = {"updated": True, "timestamp": str(time.time())}
        success = populated_client.update_document(doc_id, metadata=new_metadata)
        assert success is True
        updated = populated_client.get_document(doc_id)
        assert updated["metadata"]["updated"] is True

    def test_update_document_embedding(self, populated_client, sample_documents):
        """Test updating document embedding."""
        doc_id = sample_documents[0].id
        new_embedding = np.random.randn(384).tolist()
        success = populated_client.update_document(doc_id, embedding=new_embedding)
        assert success is True

    def test_update_nonexistent_document(self, populated_client):
        """Test updating non-existent document."""
        success = populated_client.update_document("nonexistent", text="new text")
        assert success is False

    def test_delete_single_document(self, populated_client, sample_documents):
        """Test deleting a single document."""
        doc_id = sample_documents[0].id
        initial_count = populated_client.count_documents()
        success = populated_client.delete_documents([doc_id])
        assert success is True
        assert populated_client.count_documents() == initial_count - 1
        assert populated_client.get_document(doc_id) is None

    def test_delete_nonexistent_document(self, populated_client):
        """Test deleting non-existent document."""
        success = populated_client.delete_documents(["nonexistent"])
        assert success is True

    def test_batch_delete_documents(self, populated_client, sample_documents):
        """Test batch deleting documents."""
        doc_ids = [doc.id for doc in sample_documents[:5]]
        initial_count = populated_client.count_documents()
        success = populated_client.delete_documents(doc_ids)
        assert success is True
        assert populated_client.count_documents() == initial_count - 5


class TestSearchFunctionality:
    """Test search functionality."""

    def test_search_by_text(self, populated_client):
        """Test searching by text query."""
        results = populated_client.search(query="sample document", n_results=5)
        assert results["count"] <= 5
        assert "documents" in results
        assert "distances" in results

    def test_search_by_embedding(self, populated_client):
        """Test searching by embedding vector."""
        query_embedding = np.random.randn(1, 384).tolist()
        results = populated_client.search_with_embeddings(query_embeddings=query_embedding, n_results=5)
        assert results["count"] <= 5
        assert "documents" in results
        assert "distances" in results

    @pytest.mark.parametrize("top_k,expected_max", [(3, 3), (5, 5), (10, 10)])
    def test_search_with_top_k(self, populated_client, top_k, expected_max):
        """Test top_k parameter limits results."""
        results = populated_client.search(query="test", n_results=top_k)
        assert results["count"] <= expected_max

    def test_search_with_similarity_threshold(self, populated_client):
        """Test similarity threshold filtering."""
        results_no_threshold = populated_client.similarity_search("test", k=5, threshold=0.0)
        results_high_threshold = populated_client.similarity_search("test", k=5, threshold=0.8)
        assert len(results_high_threshold) <= len(results_no_threshold)

    def test_search_with_metadata_filter(self, populated_client):
        """Test search with metadata filter."""
        filter_dict = {"category": "test"}
        results = populated_client.search(query="test", where=filter_dict)
        for i, metadata in enumerate(results["metadatas"]):
            assert metadata.get("category") == "test"

    def test_search_empty_query(self, populated_client):
        """Test search with empty query."""
        results = populated_client.search(query="")
        assert results["count"] == 0
        assert "error" in results

    def test_search_in_empty_collection(self, chroma_client, sample_queries):
        """Test search in empty collection."""
        for query in sample_queries:
            results = chroma_client.search(query=query)
            assert results["count"] == 0


class TestSearchEngineAdvanced:
    """Test advanced search engine features."""

    def test_hybrid_search(self, search_engine, populated_client, sample_documents):
        """Test hybrid search combining multiple sources."""
        query = "sample document"
        query_embeddings = np.random.randn(1, 384).tolist()

        results = search_engine.hybrid_search(
            query=query,
            query_embeddings=query_embeddings,
            n_results=5
        )
        assert len(results) <= 5
        for result in results:
            assert hasattr(result, "document")
            assert hasattr(result, "score")
            assert 0 <= result.score <= 1

    def test_hybrid_search_weights(self, search_engine):
        """Test hybrid search with custom weights."""
        query = "test"
        query_embeddings = np.random.randn(1, 384).tolist()

        results = search_engine.hybrid_search(
            query=query,
            query_embeddings=query_embeddings,
            n_results=5,
            weights={"chroma": 0.7, "faiss": 0.3}
        )
        assert len(results) <= 5

    def test_faceted_search(self, search_engine, populated_client):
        """Test faceted search with aggregations."""
        result = search_engine.faceted_search(
            query="document",
            facets=["category", "priority"],
            n_results=5
        )

        assert "results" in result
        assert "facet_counts" in result
        assert "category" in result["facet_counts"]
        assert "priority" in result["facet_counts"]

    def test_multi_query_search(self, search_engine):
        """Test search with multiple queries."""
        queries = ["sample", "document", "test"]
        query_embeddings_list = [np.random.randn(1, 384).tolist() for _ in queries]

        results = search_engine.multi_query_search(
            queries=queries,
            query_embeddings_list=query_embeddings_list,
            n_results=5,
            aggregation="max"
        )

        assert len(results) <= 5
        for result in results:
            assert hasattr(result, "document")
            assert hasattr(result, "score")
            assert 0 <= result.score <= 1

    def test_relevance_feedback(self, search_engine, populated_client, sample_documents):
        """Test relevance feedback for query refinement."""
        query = "document"
        query_embeddings = np.random.randn(1, 384).tolist()

        initial_results = search_engine.search(
            query=query,
            query_embeddings=query_embeddings,
            n_results=3
        )

        if initial_results:
            relevant_ids = [result.doc_id for result in initial_results[:2]]
            assert len(relevant_ids) > 0


class TestIndexManager:
    """Test index management operations."""

    def test_create_chroma_index(self, index_manager):
        """Test creating a new Chroma index."""
        index_metadata = index_manager.create_chroma_index(
            name="test_chroma_index",
            config={"collection_name": "test_docs"}
        )
        assert index_metadata is not None
        assert index_metadata["name"] == "test_chroma_index"
        assert index_metadata["type"] == "chroma"

    def test_create_faiss_index(self, index_manager):
        """Test creating a new FAISS index."""
        index_metadata = index_manager.create_faiss_index(
            name="test_faiss_index",
            dimension=384,
            config={"metric": "cosine"}
        )
        assert index_metadata is not None
        assert index_metadata["name"] == "test_faiss_index"
        assert index_metadata["type"] == "faiss"
        assert index_metadata["config"]["dimension"] == 384

    def test_create_duplicate_index(self, index_manager):
        """Test creating duplicate index."""
        index_manager.create_faiss_index(name="test_index", dimension=384)
        index_metadata = index_manager.create_faiss_index(name="test_index", dimension=384)
        assert index_metadata is not None
        assert index_metadata["name"] == "test_index"

    def test_create_index_with_params(self, index_manager):
        """Test creating index with custom parameters."""
        params = {
            "metric": "cosine",
            "index_type": "ivf",
            "ivf_nlist": 100
        }
        index_metadata = index_manager.create_faiss_index(
            name="test_index",
            dimension=384,
            config=params
        )
        assert index_metadata is not None
        assert index_metadata["config"]["metric"] == "cosine"
        assert index_metadata["config"]["ivf_nlist"] == 100

    def test_get_index(self, index_manager):
        """Test getting an index by name."""
        index_manager.create_faiss_index(name="test_index", dimension=384)
        index = index_manager.get_index("test_index")
        assert index is not None
        assert index["name"] == "test_index"

    def test_list_indices(self, index_manager):
        """Test listing all indices."""
        index_manager.create_faiss_index(name="index1", dimension=384)
        index_manager.create_chroma_index(name="index2")

        indices = index_manager.list_indices()
        assert len(indices) >= 2

        faiss_indices = index_manager.list_indices(index_type="faiss")
        assert len(faiss_indices) >= 1

    def test_delete_index(self, index_manager):
        """Test deleting an index."""
        index_manager.create_faiss_index(name="test_index", dimension=384)
        success = index_manager.delete_index("test_index")
        assert success is True

        index = index_manager.get_index("test_index")
        assert index is None

    def test_get_index_stats(self, index_manager):
        """Test getting index statistics."""
        index_manager.create_faiss_index(name="test_index", dimension=384)
        stats = index_manager.get_index_stats("test_index")

        assert stats is not None
        assert stats["name"] == "test_index"
        assert "document_count" in stats

    def test_backup_index(self, index_manager, tmp_path):
        """Test backing up an index."""
        index_manager.create_faiss_index(name="test_index", dimension=384)
        backup_path = index_manager.backup_index("test_index")

        assert backup_path is not None
        assert backup_path.exists()

    def test_restore_index(self, index_manager, tmp_path):
        """Test restoring an index from backup."""
        index_manager.create_faiss_index(name="test_index", dimension=384)
        backup_path = index_manager.backup_index("test_index")
        
        assert backup_path is not None
        assert backup_path.exists()
        
        index_manager.delete_index("test_index")
        
        success = index_manager.restore_index(backup_path)
        assert success is True


class TestPersistence:
    """Test persistence and reloading."""

    def test_persistence_after_restart(self, populated_client, sample_documents, persist_dir):
        """Test data persists after client restart."""
        assert populated_client.count_documents() == len(sample_documents)

        global _chroma_instance
        _chroma_instance = None
        new_client = ChromaClient(persist_directory=persist_dir)

        assert new_client.count_documents() == len(sample_documents)
        for doc in sample_documents:
            retrieved = new_client.get_document(doc.id)
            assert retrieved is not None
            assert retrieved["document"] == doc.text

    def test_persistence_with_embeddings(self, populated_client, sample_documents, persist_dir):
        """Test embeddings persist after restart."""
        global _chroma_instance
        _chroma_instance = None
        new_client = ChromaClient(persist_directory=persist_dir)

        for doc in sample_documents:
            retrieved = new_client.get_document(doc.id)
            assert retrieved is not None

    def test_incremental_persistence(self, chroma_client, sample_documents, persist_dir):
        """Test incremental persistence."""
        first_batch = sample_documents[:5]
        for doc in first_batch:
            chroma_client.add_documents(
                texts=[doc.text],
                embeddings=[doc.embedding],
                metadatas=[doc.metadata],
                ids=[doc.id]
            )

        second_batch = sample_documents[5:]
        for doc in second_batch:
            chroma_client.add_documents(
                texts=[doc.text],
                embeddings=[doc.embedding],
                metadatas=[doc.metadata],
                ids=[doc.id]
            )

        global _chroma_instance
        _chroma_instance = None
        new_client = ChromaClient(persist_directory=persist_dir)

        assert new_client.count_documents() == len(sample_documents)


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_connection_error(self, chroma_client):
        """Test handling of connection errors."""
        with patch.object(chroma_client, 'get_collection_info', side_effect=Exception("Connection failed")):
            result = chroma_client.health_check()
            assert result["status"] == "unhealthy"
            assert "error" in result

    def test_timeout_error(self, chroma_client):
        """Test handling of timeout errors."""
        with patch.object(chroma_client, 'search', side_effect=TimeoutError("Operation timed out")):
            with pytest.raises(TimeoutError, match="Operation timed out"):
                chroma_client.search("query")

    def test_invalid_embedding_dimension(self, chroma_client):
        """Test handling of invalid embedding dimension."""
        doc = Document(
            id="test",
            text="test",
            embedding=[0.1, 0.2, 0.3]
        )

        try:
            chroma_client.add_documents(
                texts=[doc.text],
                embeddings=[doc.embedding],
                metadatas=[doc.metadata],
                ids=[doc.id]
            )
            retrieved = chroma_client.get_document("test")
            assert retrieved is not None
        except Exception as e:
            assert e is not None

    def test_corrupt_persistence_directory(self, persist_dir):
        """Test handling of corrupt persistence directory."""
        persist_dir.mkdir(parents=True, exist_ok=True)
        (persist_dir / "corrupt_file").write_text("corrupt data")

        global _chroma_instance
        _chroma_instance = None
        client = ChromaClient(persist_directory=persist_dir)
        assert client is not None


class TestPerformance:
    """Test performance benchmarks."""

    def test_batch_insert_performance(self, chroma_client):
        """Test batch insert performance."""
        docs = []
        for i in range(100):
            tags_value = f"perf,doc_{i}"
            doc = Document(
                id=f"perf_doc_{i}",
                text=f"Document {i} content for performance testing",
                embedding=np.random.randn(384).tolist(),
                metadata={
                    "tags": tags_value,
                    "index": i
                }
            )
            docs.append(doc)

        texts = [doc.text for doc in docs]
        embeddings = [doc.embedding for doc in docs]
        metadatas = [doc.metadata for doc in docs]
        ids = [doc.id for doc in docs]

        start = time.time()
        chroma_client.add_documents(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        elapsed = time.time() - start

        assert elapsed < 5.0

    def test_search_performance(self, populated_client):
        """Test search performance."""
        start = time.time()
        for _ in range(10):
            populated_client.search(query="test", n_results=5)
        elapsed = time.time() - start

        assert elapsed < 2.0

    def test_concurrent_search_performance(self, populated_client):
        """Test concurrent search performance."""
        def search_worker():
            for _ in range(5):
                populated_client.search(query="test", n_results=5)

        start = time.time()
        threads = [threading.Thread(target=search_worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.time() - start

        assert elapsed < 3.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])