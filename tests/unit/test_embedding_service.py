# tests/unit/test_embedding_service.py

"""Unit tests for embedding service.

Tests cover embedding model configuration, caching, generation,
similarity computation, find similar functionality, performance benchmarks,
health checks, GPU support, quantization, and batch processing.
"""

import os
import sys
import time
import json
import hashlib
import threading
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import Dict, List, Any, Optional

import pytest

from src.ai_engine.embedding_service import EmbeddingService, EmbeddingCache


@pytest.fixture
def cache_dir(tmp_path):
    """Create temporary cache directory."""
    return tmp_path / "embedding_cache"


@pytest.fixture
def embedding_service():
    """Create embedding service instance."""
    EmbeddingService._instance = None
    service = EmbeddingService(test_mode=True)
    service.initialize()
    return service


@pytest.fixture
def embedding_service_with_cache(cache_dir):
    """Create embedding service with cache enabled."""
    EmbeddingService._instance = None
    service = EmbeddingService(cache_enabled=True, test_mode=True)
    service.cache = EmbeddingCache(cache_dir=cache_dir)
    service.initialize()
    return service


@pytest.fixture
def sample_texts():
    """Provide sample texts for testing."""
    return [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language for data science.",
        "The sun sets in the west and rises in the east.",
        "Natural language processing enables computers to understand text."
    ]


@pytest.fixture
def sample_embeddings(embedding_service, sample_texts):
    """Generate sample embeddings."""
    return [embedding_service.encode(text) for text in sample_texts]


class TestEmbeddingModelConfig:
    """Test embedding model configuration."""

    def test_default_configuration(self, embedding_service):
        """Test default model configuration."""
        assert embedding_service.model_name == "all-MiniLM-L6-v2"
        assert embedding_service.cache_enabled is True

        info = embedding_service.get_model_info()
        assert info.dimensions == 384
        assert info.context_length == 256
        assert info.normalize_embeddings is True

    def test_custom_configuration(self):
        """Test custom model configuration."""
        EmbeddingService._instance = None
        service = EmbeddingService(
            model_name="all-mpnet-base-v2",
            cache_enabled=False,
            normalize_embeddings=False,
            test_mode=True
        )
        service.initialize()

        assert service.model_name == "all-mpnet-base-v2"
        assert service.cache_enabled is False
        assert service.normalize_embeddings is False

        info = service.get_model_info()
        assert info.dimensions == 768

    def test_available_models_listing(self):
        """Test listing available models."""
        models = EmbeddingService.list_available_models()
        assert len(models) >= 3
        assert "all-MiniLM-L6-v2" in models
        assert "all-mpnet-base-v2" in models
        assert "multi-qa-mpnet-base-dot-v1" in models

    def test_gpu_configuration(self):
        """Test GPU configuration."""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('src.ai_engine.embedding_service.SentenceTransformer') as mock_st:
                mock_st.return_value = MagicMock()
                EmbeddingService._instance = None
                service = EmbeddingService(device="cuda", test_mode=False)
                service.initialize()
                assert service.device == "cuda"

    def test_quantization_configuration(self):
        """Test quantization configuration."""
        EmbeddingService._instance = None
        service = EmbeddingService(quantize=True, test_mode=True)
        service.initialize()
        assert service.quantized is True

    def test_batch_size_configuration(self):
        """Test batch size configuration."""
        EmbeddingService._instance = None
        service = EmbeddingService(batch_size=64, test_mode=True)
        service.initialize()
        assert service.batch_size == 64


class TestEmbeddingCache:
    """Test embedding caching functionality."""

    def test_cache_initialization(self, cache_dir):
        """Test cache initialization."""
        cache = EmbeddingCache(cache_dir=cache_dir, max_size_mb=100)
        assert cache.cache_dir == cache_dir
        assert cache.max_size_bytes == 100 * 1024 * 1024
        assert cache_dir.exists()

    def test_cache_key_generation(self, embedding_service_with_cache):
        """Test cache key generation."""
        cache = embedding_service_with_cache.cache
        text = "test text"
        model = "test-model"

        key1 = cache._generate_key(text, model)
        key2 = cache._generate_key(text, model)
        key3 = cache._generate_key("different text", model)

        assert key1 == key2
        assert key1 != key3
        assert len(key1) == 64

    def test_cache_set_get(self, embedding_service_with_cache, sample_texts):
        """Test setting and getting from cache."""
        service = embedding_service_with_cache
        cache = service.cache
        cache.clear()
        
        emb1 = service.encode(sample_texts[0])
        emb2 = service.encode(sample_texts[0])

        stats = cache.get_stats()
        assert stats['hit_count'] == 1
        assert stats['miss_count'] == 1
        assert stats['entries'] == 1

        np.testing.assert_array_equal(emb1, emb2)

    def test_cache_eviction_lru(self, embedding_service_with_cache):
        """Test LRU cache eviction."""
        service = embedding_service_with_cache
        cache = service.cache
        cache.max_entries = 3
        cache.clear()

        texts = ["text1", "text2", "text3", "text4"]
        for text in texts:
            service.encode(text)

        stats = cache.get_stats()
        assert stats['entries'] == 3
        assert stats['miss_count'] == 4
        assert stats['hit_count'] == 0

        service.encode("text1")
        stats = cache.get_stats()
        assert stats['miss_count'] == 5
        assert stats['hit_count'] == 0

    def test_cache_size_limit_enforcement(self, embedding_service_with_cache, tmp_path):
        """Test cache size limit enforcement."""
        cache = embedding_service_with_cache.cache
        cache.max_size_bytes = 1024

        text = "x" * 1000
        embedding_service_with_cache.encode(text)

        stats = cache.get_stats()
        assert stats['size_bytes'] <= 1024

    def test_cache_corruption_handling(self, embedding_service_with_cache, tmp_path):
        """Test handling of corrupted cache files."""
        service = embedding_service_with_cache
        cache = service.cache

        emb = service.encode("test text")

        for metadata_path in cache.cache_dir.glob("*.json"):
            metadata_path.write_text("corrupted data")

        emb2 = service.encode("test text")
        np.testing.assert_array_equal(emb, emb2)

    def test_cache_persistence(self, embedding_service_with_cache, tmp_path):
        """Test cache persistence across service restarts."""
        cache_dir = tmp_path / "embedding_cache"

        service1 = embedding_service_with_cache
        emb = service1.encode("persistent text")

        EmbeddingService._instance = None
        service2 = EmbeddingService(cache_enabled=True, test_mode=True)
        service2.cache = EmbeddingCache(cache_dir=cache_dir)
        service2.initialize()

        emb2 = service2.encode("persistent text")
        np.testing.assert_array_equal(emb, emb2)

        stats = service2.cache.get_stats()
        assert stats['hit_count'] == 1

    def test_cache_clear(self, embedding_service_with_cache):
        """Test clearing the cache."""
        service = embedding_service_with_cache
        cache = service.cache
        cache.clear()
        
        service.encode("text1")
        service.encode("text2")

        stats = cache.get_stats()
        assert stats['entries'] == 2

        service.clear_cache()
        stats = cache.get_stats()
        assert stats['entries'] == 0
        assert stats['hit_count'] == 0
        assert stats['miss_count'] == 0


class TestEmbeddingGeneration:
    """Test embedding generation functionality."""

    def test_encode_single_text(self, embedding_service, sample_texts):
        """Test encoding a single text."""
        embedding = embedding_service.encode(sample_texts[0])
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)

    def test_encode_batch_texts(self, embedding_service, sample_texts):
        """Test encoding multiple texts in batch."""
        embeddings = embedding_service.encode_batch(sample_texts)
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (len(sample_texts), 384)

    @pytest.mark.parametrize("text", [
        "",
        "!@#$%^&*()",
        "Emoji: 😀🎉🚀",
        "Unicode: こんにちは世界",
        "A" * 100,
    ])
    def test_encode_edge_cases(self, embedding_service, text):
        """Test encoding various edge cases."""
        embedding = embedding_service.encode(text)
        assert embedding.shape == (384,)

    def test_encode_normalization(self, embedding_service):
        """Test embedding normalization."""
        text = "Test text for normalization"
        emb = embedding_service.encode(text, normalize=True)
        norm = np.linalg.norm(emb)
        assert np.abs(norm - 1.0) < 1e-6

    def test_batch_processing_with_varying_lengths(self, embedding_service):
        """Test batch processing with varying text lengths."""
        texts = [
            "Short text",
            "Medium length text with more words",
            "A" * 1000,
            "B" * 5000,
        ]
        embeddings = embedding_service.encode_batch(texts)
        assert embeddings.shape[0] == len(texts)
        assert embeddings.shape[1] == 384

    def test_parallel_encoding(self, embedding_service):
        """Test parallel encoding of texts."""
        texts = ["text" + str(i) for i in range(100)]

        start = time.time()
        embeddings1 = embedding_service.encode_batch(texts, parallel=False)
        sequential_time = time.time() - start

        start = time.time()
        embeddings2 = embedding_service.encode_batch(texts, parallel=True)
        parallel_time = time.time() - start

        assert embeddings1.shape == embeddings2.shape
        assert parallel_time < sequential_time * 0.8


class TestSimilarityComputation:
    """Test similarity computation between embeddings."""

    def test_cosine_similarity(self, embedding_service, sample_embeddings):
        """Test cosine similarity computation."""
        sim_same = embedding_service.compute_similarity(
            sample_embeddings[0], sample_embeddings[0]
        )
        sim_diff = embedding_service.compute_similarity(
            sample_embeddings[0], sample_embeddings[2]
        )

        assert sim_same == pytest.approx(1.0, abs=0.01)
        assert sim_diff < 1.0
        assert sim_diff > -1.0

    @pytest.mark.parametrize("metric", ["cosine", "euclidean", "dot", "manhattan"])
    def test_similarity_metrics(self, embedding_service, sample_embeddings, metric):
        """Test different similarity metrics."""
        sim = embedding_service.compute_similarity(
            sample_embeddings[0], sample_embeddings[1], metric=metric
        )
        assert isinstance(sim, float)

    def test_pairwise_similarity(self, embedding_service, sample_embeddings):
        """Test pairwise similarity computation."""
        matrix = embedding_service.compute_pairwise_similarity(sample_embeddings)
        assert matrix.shape == (5, 5)
        assert np.allclose(matrix.diagonal(), 1.0)

    def test_similarity_matrix_with_threshold(self, embedding_service, sample_embeddings):
        """Test similarity matrix with threshold."""
        matrix = embedding_service.compute_similarity_matrix(
            sample_embeddings, threshold=0.5
        )
        assert (matrix >= 0.5).any()
        assert (matrix < 0.5).any()


class TestFindSimilar:
    """Test find similar functionality."""

    def test_find_similar_by_text(self, embedding_service, sample_texts):
        """Test finding similar texts by query."""
        query = "machine learning"
        results = embedding_service.find_similar(
            query, sample_texts, top_k=3
        )
        assert len(results) == 3
        for text, score in results:
            assert 0 <= score <= 1

    def test_find_similar_by_embedding(self, embedding_service, sample_texts, sample_embeddings):
        """Test finding similar by embedding vector."""
        query_emb = sample_embeddings[0]
        results = embedding_service.find_similar_by_embedding(
            query_emb, sample_texts, sample_embeddings, top_k=3
        )
        assert len(results) == 3
        assert results[0][1] >= results[1][1]

    def test_find_similar_with_threshold(self, embedding_service, sample_texts):
        """Test find similar with similarity threshold."""
        query = "machine learning"
        results = embedding_service.find_similar(
            query, sample_texts, threshold=0.5, top_k=5
        )
        for _, score in results:
            assert score >= 0.5

    def test_find_similar_empty_corpus(self, embedding_service):
        """Test find similar with empty corpus."""
        results = embedding_service.find_similar("test", [])
        assert len(results) == 0

    def test_find_similar_exact_match(self, embedding_service):
        """Test find similar with exact text match."""
        texts = ["exact match text", "different text"]
        results = embedding_service.find_similar("exact match text", texts, top_k=1)
        assert results[0][0] == "exact match text"
        assert results[0][1] == pytest.approx(1.0, abs=0.01)


class TestModelManagement:
    """Test model management functionality."""

    def test_switch_model(self, embedding_service):
        """Test switching between models."""
        embedding_service.switch_model("all-mpnet-base-v2")
        assert embedding_service.model_name == "all-mpnet-base-v2"
        info = embedding_service.get_model_info()
        assert info.dimensions == 768

    def test_switch_to_invalid_model(self, embedding_service):
        """Test switching to invalid model."""
        with pytest.raises(ValueError, match="not available"):
            embedding_service.switch_model("nonexistent-model")

    def test_get_model_info(self, embedding_service):
        """Test getting model information."""
        info = embedding_service.get_model_info("all-MiniLM-L6-v2")
        assert info.name == "all-MiniLM-L6-v2"
        assert info.dimensions == 384
        assert info.context_length == 256
        assert info.normalize_embeddings is True

    def test_list_loaded_models(self, embedding_service):
        """Test listing loaded models."""
        loaded = embedding_service.list_loaded_models()
        assert "all-MiniLM-L6-v2" in loaded

    def test_unload_model(self, embedding_service):
        """Test unloading a model."""
        embedding_service.switch_model("all-mpnet-base-v2")
        embedding_service.unload_model("all-MiniLM-L6-v2")
        loaded = embedding_service.list_loaded_models()
        assert "all-MiniLM-L6-v2" not in loaded


class TestPerformance:
    """Test performance benchmarks."""

    def test_encoding_speed_single(self, embedding_service, sample_texts):
        """Test encoding speed for single text."""
        start = time.time()
        embedding_service.encode(sample_texts[0])
        elapsed = time.time() - start
        assert elapsed < 0.1

    def test_encoding_speed_batch(self, embedding_service, sample_texts):
        """Test encoding speed for batch."""
        start = time.time()
        embedding_service.encode_batch(sample_texts)
        elapsed = time.time() - start
        assert elapsed < 0.5

    def test_throughput_with_batch_sizes(self, embedding_service):
        """Test throughput with different batch sizes."""
        texts = ["text"] * 1000
        batch_sizes = [16, 32, 64, 128]
        throughputs = []

        for batch_size in batch_sizes:
            embedding_service.batch_size = batch_size
            start = time.time()
            embedding_service.encode_batch(texts)
            elapsed = time.time() - start
            throughput = len(texts) / elapsed
            throughputs.append(throughput)

        assert throughputs[-1] > throughputs[0]

    def test_cache_performance_improvement(self, embedding_service_with_cache, sample_texts):
        """Test cache performance improvement."""
        service = embedding_service_with_cache

        start = time.time()
        for text in sample_texts:
            service.encode(text)
        first_pass = time.time() - start

        start = time.time()
        for text in sample_texts:
            service.encode(text)
        second_pass = time.time() - start

        assert second_pass < first_pass * 0.1


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_model_load_failure(self):
        """Test handling of model load failure."""
        with patch('src.ai_engine.embedding_service.SENTENCE_TRANSFORMERS_AVAILABLE', True):
            with patch('src.ai_engine.embedding_service.SentenceTransformer') as mock_st:
                mock_st.side_effect = Exception("Failed to load model")
                EmbeddingService._instance = None
                service = EmbeddingService(test_mode=False)
                
                with pytest.raises(RuntimeError, match="Failed to load model"):
                    service.initialize()

    def test_invalid_input_type(self, embedding_service):
        """Test handling of invalid input type."""
        with pytest.raises(TypeError):
            embedding_service.encode(123)

    def test_memory_error_handling(self, embedding_service):
        """Test handling of memory errors."""
        embedding_service.set_simulate_memory_error(True)
        with pytest.raises(MemoryError):
            embedding_service.encode("test")

    def test_gpu_unavailable_fallback(self):
        """Test fallback when GPU is unavailable."""
        with patch('torch.cuda.is_available', return_value=False):
            with patch('src.ai_engine.embedding_service.SentenceTransformer') as mock_st:
                mock_st.return_value = MagicMock()
                EmbeddingService._instance = None
                service = EmbeddingService(device="cuda", test_mode=False)
                service.initialize()
                assert service.device == "cpu"

    def test_quantization_fallback(self):
        """Test fallback when quantization fails."""
        with patch('sentence_transformers.SentenceTransformer') as mock_st:
            mock_model = MagicMock()
            mock_model.quantize.side_effect = Exception("Quantization failed")
            mock_st.return_value = mock_model

            EmbeddingService._instance = None
            service = EmbeddingService(quantize=True, test_mode=False)
            service.initialize()
            assert service.quantized is False


class TestHealthCheck:
    """Test health check functionality."""

    def test_health_check_success(self, embedding_service):
        """Test successful health check."""
        health = embedding_service.health_check()
        assert health["status"] == "healthy"
        assert health["model_loaded"] is True
        assert "model_info" in health
        assert health["cache_stats"] is not None

    def test_health_check_with_cache(self, embedding_service_with_cache):
        """Test health check with cache."""
        service = embedding_service_with_cache
        service.encode("test")
        health = service.health_check()
        assert health["cache_stats"]["hit_count"] >= 0

    def test_health_check_failure(self, embedding_service):
        """Test failed health check."""
        embedding_service._model = None
        health = embedding_service.health_check()
        assert health["status"] == "unhealthy"
        assert health["model_loaded"] is False

    def test_health_check_performance_metrics(self, embedding_service):
        """Test health check includes performance metrics."""
        health = embedding_service.health_check()
        assert "performance" in health
        assert "total_encodings" in health["performance"]
        assert "average_time_ms" in health["performance"]


class TestConcurrency:
    """Test concurrent access to embedding service."""

    def test_concurrent_encoding(self, embedding_service):
        """Test concurrent encoding from multiple threads."""
        errors = []
        results = []

        def encode_text(thread_id):
            try:
                for i in range(10):
                    text = f"Thread {thread_id} text {i}"
                    emb = embedding_service.encode(text)
                    results.append(emb)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=encode_text, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 50

    def test_concurrent_cache_access(self, embedding_service_with_cache):
        """Test concurrent cache access."""
        service = embedding_service_with_cache
        errors = []

        def access_cache(thread_id):
            try:
                for i in range(20):
                    text = f"text_{thread_id}_{i}"
                    service.encode(text)
                    service.encode(text)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=access_cache, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        stats = service.cache.get_stats()
        assert stats['hit_count'] > 0

    def test_concurrent_model_switching(self, embedding_service):
        """Test concurrent model switching."""
        errors = []

        def switch_and_encode(thread_id):
            try:
                models = ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]
                for i in range(5):
                    model = models[i % 2]
                    embedding_service.switch_model(model)
                    emb = embedding_service.encode(f"test_{thread_id}_{i}")
                    assert emb.shape[0] in (384, 768)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=switch_and_encode, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])