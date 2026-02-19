# cockatoo_v1/src/vector_store/search_engine.py

"""
Search Engine for cockatoo_v1 Vector Store.
Provides advanced search capabilities including hybrid search, filtering,
reranking, caching, and query optimization for both ChromaDB and FAISS.
"""

import logging
import time
import sys
import os
import hashlib
import json
import pickle
import threading
import gc
import atexit
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
import numpy as np
import tempfile
import shutil
import uuid

try:
    from .chroma_client import ChromaClient, get_chroma_client
    from .faiss_client import FAISSClient, get_faiss_client
except ImportError:
    from chroma_client import ChromaClient, get_chroma_client
    from faiss_client import FAISSClient, get_faiss_client

try:
    from ..core.constants import CACHE_DIR
except (ImportError, ValueError):
    try:
        from src.core.constants import CACHE_DIR
    except ImportError:
        CACHE_DIR = Path.home() / ".cockatoo_v1" / "cache"

logger = logging.getLogger(__name__)


class SearchResult:
    """Represents a single search result with metadata."""

    def __init__(
        self,
        document: str,
        score: float,
        metadata: Dict[str, Any],
        doc_id: str,
        source: str = "unknown"
    ):
        """Initialize search result."""
        self.document = document
        self.score = score
        self.metadata = metadata
        self.doc_id = doc_id
        self.source = source
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "document": self.document,
            "score": self.score,
            "metadata": self.metadata,
            "doc_id": self.doc_id,
            "source": self.source,
            "timestamp": self.timestamp.isoformat()
        }

    def __lt__(self, other):
        """For heapq ordering (higher score is better)."""
        return self.score > other.score

    def __repr__(self):
        return f"SearchResult(score={self.score:.4f}, source={self.source}, id={self.doc_id[:8]}...)"


class QueryCache:
    """Cache for search queries and results."""

    def __init__(self, cache_dir: Path, ttl_seconds: int = 3600, max_entries: int = 1000):
        """Initialize query cache."""
        self.cache_dir = Path(cache_dir) / "search_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.memory_cache = {}
        self.cache_lock = threading.RLock()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "entries": 0,
            "size_bytes": 0
        }

        self.index_file = self.cache_dir / "cache_index.json"
        self.cache_index = self._load_index()

        self.cleanup_thread = None
        self.cleanup_running = False
        self._start_cleanup_thread()

        atexit.register(self.stop)

    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """Load cache index from disk."""
        if self.index_file.exists():
            try:
                with open(self.index_file, "r") as f:
                    index = json.load(f)
                logger.debug(f"Loaded cache index with {len(index)} entries")
                return index
            except Exception as error:
                logger.error(f"Failed to load cache index: {error}")

        return {}

    def _save_index(self) -> None:
        """Save cache index to disk."""
        try:
            with open(self.index_file, "w") as f:
                json.dump(self.cache_index, f, indent=2)
        except Exception as error:
            logger.error(f"Failed to save cache index: {error}")

    def _start_cleanup_thread(self) -> None:
        """Start background cleanup thread."""
        self.cleanup_running = True
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_thread.start()

    def _cleanup_worker(self) -> None:
        """Background worker for cache cleanup."""
        while self.cleanup_running:
            try:
                time.sleep(300)
                self.cleanup()
            except Exception as error:
                logger.error(f"Cache cleanup error: {error}")

    def _generate_key(self, query: str, params: Dict[str, Any]) -> str:
        """Generate cache key from query and parameters."""
        content = f"{query}:{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, query: str, params: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """Get cached results for query."""
        key = self._generate_key(query, params)

        with self.cache_lock:
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                if self._is_valid(entry):
                    self.stats["hits"] += 1
                    entry["access_count"] += 1
                    entry["last_accessed"] = datetime.now().isoformat()
                    return entry["results"]

            if key in self.cache_index:
                entry = self.cache_index[key]
                if self._is_valid(entry):
                    cache_file = self.cache_dir / f"{key}.pkl"
                    if cache_file.exists():
                        try:
                            with open(cache_file, "rb") as f:
                                results = pickle.load(f)

                            self.memory_cache[key] = {
                                "results": results,
                                "created_at": entry["created_at"],
                                "expires_at": entry["expires_at"],
                                "access_count": entry.get("access_count", 0) + 1,
                                "last_accessed": datetime.now().isoformat()
                            }

                            self.stats["hits"] += 1
                            entry["access_count"] += 1
                            entry["last_accessed"] = datetime.now().isoformat()
                            self._save_index()

                            return results
                        except Exception as error:
                            logger.error(f"Failed to load cache file: {error}")

            self.stats["misses"] += 1
            return None

    def set(self, query: str, params: Dict[str, Any], results: List[Dict[str, Any]]) -> None:
        """Cache search results."""
        if not results:
            return

        key = self._generate_key(query, params)
        now = datetime.now()

        with self.cache_lock:
            if len(self.cache_index) >= self.max_entries:
                self._evict_lru()

            cache_file = self.cache_dir / f"{key}.pkl"
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump(results, f)

                self.cache_index[key] = {
                    "query": query,
                    "params": params,
                    "created_at": now.isoformat(),
                    "expires_at": (now + timedelta(seconds=self.ttl_seconds)).isoformat(),
                    "last_accessed": now.isoformat(),
                    "access_count": 1,
                    "result_count": len(results),
                    "file_size": cache_file.stat().st_size
                }

                self.memory_cache[key] = {
                    "results": results,
                    "created_at": now.isoformat(),
                    "expires_at": (now + timedelta(seconds=self.ttl_seconds)).isoformat(),
                    "access_count": 1,
                    "last_accessed": now.isoformat()
                }

                self._save_index()
                self.stats["entries"] = len(self.cache_index)
                self.stats["size_bytes"] += cache_file.stat().st_size

            except Exception as error:
                logger.error(f"Failed to cache results: {error}")

    def _is_valid(self, entry: Dict[str, Any]) -> bool:
        """Check if cache entry is still valid."""
        expires_at = datetime.fromisoformat(entry["expires_at"])
        return datetime.now() < expires_at

    def _evict_lru(self) -> None:
        """Evict least recently used cache entries."""
        if not self.cache_index:
            return

        sorted_entries = sorted(
            self.cache_index.items(),
            key=lambda x: x[1].get("last_accessed", x[1]["created_at"])
        )

        to_remove = max(1, len(sorted_entries) // 10)

        for i in range(to_remove):
            key, _ = sorted_entries[i]
            self._remove_entry(key)

    def _remove_entry(self, key: str) -> None:
        """Remove a cache entry."""
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            self.stats["size_bytes"] -= cache_file.stat().st_size
            cache_file.unlink()

        if key in self.cache_index:
            del self.cache_index[key]

        if key in self.memory_cache:
            del self.memory_cache[key]

        self.stats["entries"] = len(self.cache_index)

    def cleanup(self) -> int:
        """Remove expired cache entries."""
        removed = 0
        now = datetime.now()

        with self.cache_lock:
            expired_keys = []
            for key, entry in self.cache_index.items():
                expires_at = datetime.fromisoformat(entry["expires_at"])
                if now > expires_at:
                    expired_keys.append(key)

            for key in expired_keys:
                self._remove_entry(key)
                removed += 1

            if removed > 0:
                self._save_index()
                logger.info(f"Removed {removed} expired cache entries")

        return removed

    def clear(self) -> int:
        """Clear all cache entries."""
        with self.cache_lock:
            count = len(self.cache_index)

            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    cache_file.unlink()
                except:
                    pass

            self.cache_index.clear()
            self.memory_cache.clear()

            self.stats = {
                "hits": 0,
                "misses": 0,
                "entries": 0,
                "size_bytes": 0
            }

            self._save_index()
            logger.info(f"Cleared {count} cache entries")

            return count

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.cache_lock:
            hit_rate = 0
            if self.stats["hits"] + self.stats["misses"] > 0:
                hit_rate = self.stats["hits"] / (self.stats["hits"] + self.stats["misses"])

            return {
                "hits": self.stats["hits"],
                "misses": self.stats["misses"],
                "hit_rate": hit_rate,
                "entries": len(self.cache_index),
                "size_bytes": self.stats["size_bytes"],
                "size_mb": self.stats["size_bytes"] / (1024 * 1024),
                "ttl_seconds": self.ttl_seconds,
                "max_entries": self.max_entries
            }

    def stop(self) -> None:
        """Stop cleanup thread."""
        self.cleanup_running = False
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=2)


class ResultReranker:
    """Reranks search results using various strategies."""

    def __init__(self):
        """Initialize reranker."""
        self.weights = {
            "similarity": 1.0,
            "recency": 0.3,
            "popularity": 0.2,
            "length": 0.1
        }

    def set_weights(self, **weights) -> None:
        """Set reranking weights."""
        for key, value in weights.items():
            if key in self.weights:
                self.weights[key] = value

    def rerank(
        self,
        results: List[SearchResult],
        strategy: str = "hybrid"
    ) -> List[SearchResult]:
        """Rerank search results."""
        if not results:
            return []

        if strategy == "similarity":
            return sorted(results, key=lambda x: x.score, reverse=True)

        elif strategy == "recency":
            return self._rerank_by_recency(results)

        elif strategy == "diversity":
            return self._rerank_by_diversity(results)

        elif strategy == "hybrid":
            return self._rerank_hybrid(results)

        else:
            logger.warning(f"Unknown reranking strategy: {strategy}, using similarity")
            return sorted(results, key=lambda x: x.score, reverse=True)

    def _rerank_by_recency(self, results: List[SearchResult]) -> List[SearchResult]:
        """Rerank by recency (newer documents first)."""
        def recency_score(result: SearchResult) -> float:
            timestamp = result.metadata.get("added_at") or result.metadata.get("timestamp")
            if timestamp:
                try:
                    doc_time = datetime.fromisoformat(timestamp)
                    age = (datetime.now() - doc_time).total_seconds()
                    return 1.0 / (1.0 + age / (30 * 24 * 3600))
                except:
                    pass
            return 0.5

        scored = []
        for result in results:
            combined = (result.score * 0.7) + (recency_score(result) * 0.3)
            scored.append((combined, result))

        scored.sort(reverse=True)
        return [r for _, r in scored]

    def _rerank_by_diversity(self, results: List[SearchResult]) -> List[SearchResult]:
        """Rerank to promote diverse results based on categories."""
        if len(results) <= 3:
            return results

        categories = defaultdict(list)
        for result in results:
            category = result.metadata.get("category", "unknown")
            categories[category].append(result)

        diverse_results = []
        while categories:
            for category in list(categories.keys()):
                if categories[category]:
                    diverse_results.append(categories[category].pop(0))
                if not categories[category]:
                    del categories[category]

        return diverse_results

    def _rerank_hybrid(self, results: List[SearchResult]) -> List[SearchResult]:
        """Hybrid reranking combining multiple factors."""
        scored = []

        for result in results:
            score = result.score * self.weights["similarity"]

            timestamp = result.metadata.get("added_at") or result.metadata.get("timestamp")
            if timestamp:
                try:
                    doc_time = datetime.fromisoformat(timestamp)
                    age_days = (datetime.now() - doc_time).days
                    recency = 1.0 / (1.0 + age_days / 30)
                    score += recency * self.weights["recency"]
                except:
                    pass

            view_count = result.metadata.get("view_count", 0)
            if view_count > 0:
                popularity = min(1.0, view_count / 1000)
                score += popularity * self.weights["popularity"]

            doc_length = len(result.document)
            if 100 <= doc_length <= 1000:
                length_score = 1.0
            elif doc_length < 100:
                length_score = doc_length / 100
            else:
                length_score = 1000 / doc_length
            score += length_score * self.weights["length"]

            scored.append((score, result))

        scored.sort(reverse=True)
        return [r for _, r in scored]


class SearchEngine:
    """Advanced search engine supporting multiple vector stores."""

    def __init__(
        self,
        chroma_client: Optional[ChromaClient] = None,
        faiss_client: Optional[FAISSClient] = None,
        cache_dir: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize search engine."""
        self.chroma = chroma_client or get_chroma_client()
        self.faiss = faiss_client or get_faiss_client()

        self.config = {
            "default_top_k": 10,
            "default_threshold": 0.3,
            "enable_cache": True,
            "enable_reranking": True,
            "default_rerank_strategy": "hybrid",
            "hybrid_weights": {
                "chroma": 0.5,
                "faiss": 0.5
            },
            "timeout_seconds": 30,
            "max_parallel_queries": 5
        }

        if config:
            self.config.update(config)

        cache_path = cache_dir or CACHE_DIR / "search_engine"
        self.cache = QueryCache(
            cache_dir=cache_path,
            ttl_seconds=3600,
            max_entries=1000
        ) if self.config["enable_cache"] else None

        self.reranker = ResultReranker() if self.config["enable_reranking"] else None

        self.stats = {
            "total_searches": 0,
            "total_time_ms": 0,
            "avg_time_ms": 0,
            "chroma_searches": 0,
            "faiss_searches": 0,
            "hybrid_searches": 0,
            "cache_hits": 0,
            "errors": 0
        }

        self.stats_lock = threading.Lock()

        logger.info("Search Engine initialized")

    def search(
        self,
        query: str,
        query_embeddings: Optional[List[List[float]]] = None,
        indices: Optional[List[str]] = None,
        n_results: int = 10,
        threshold: float = 0.3,
        where: Optional[Dict[str, Any]] = None,
        rerank_strategy: str = "hybrid",
        use_cache: bool = True
    ) -> List[SearchResult]:
        """Perform search across configured vector stores."""
        start_time = time.time()

        if indices is None:
            indices = ["chroma", "faiss"]

        if use_cache and self.cache and query_embeddings is None:
            cache_params = {
                "indices": indices,
                "n_results": n_results,
                "threshold": threshold,
                "where": where,
                "rerank_strategy": rerank_strategy
            }
            cached_results = self.cache.get(query, cache_params)
            if cached_results:
                with self.stats_lock:
                    self.stats["cache_hits"] += 1
                    self.stats["total_searches"] += 1

                results = [
                    SearchResult(
                        document=r["document"],
                        score=r["score"],
                        metadata=r["metadata"],
                        doc_id=r["doc_id"],
                        source=r["source"]
                    ) for r in cached_results
                ]

                logger.debug(f"Cache hit for query: {query[:50]}...")
                return results

        try:
            all_results = []

            if "chroma" in indices:
                chroma_results = self._search_chroma(
                    query=query,
                    query_embeddings=query_embeddings,
                    n_results=n_results * 2,
                    threshold=threshold,
                    where=where
                )
                all_results.extend(chroma_results)
                with self.stats_lock:
                    self.stats["chroma_searches"] += 1

            if "faiss" in indices and query_embeddings:
                faiss_results = self._search_faiss(
                    query_embeddings=query_embeddings,
                    n_results=n_results * 2,
                    threshold=threshold,
                    where=where
                )
                all_results.extend(faiss_results)
                with self.stats_lock:
                    self.stats["faiss_searches"] += 1
            elif "faiss" in indices and not query_embeddings:
                logger.debug("FAISS search requires query embeddings, skipping")

            if self.reranker and rerank_strategy:
                all_results = self.reranker.rerank(all_results, rerank_strategy)

            final_results = all_results[:n_results]

            if use_cache and self.cache and query_embeddings is None and final_results:
                cache_data = [r.to_dict() for r in final_results]
                self.cache.set(query, cache_params, cache_data)

            elapsed_ms = (time.time() - start_time) * 1000
            with self.stats_lock:
                self.stats["total_searches"] += 1
                self.stats["total_time_ms"] += elapsed_ms
                self.stats["avg_time_ms"] = self.stats["total_time_ms"] / self.stats["total_searches"]
                if len(indices) > 1:
                    self.stats["hybrid_searches"] += 1

            logger.info(f"Search completed in {elapsed_ms:.2f}ms, found {len(final_results)} results")
            return final_results

        except Exception as error:
            logger.error(f"Search failed: {error}")
            with self.stats_lock:
                self.stats["errors"] += 1
            return []

    def _search_chroma(
        self,
        query: str,
        query_embeddings: Optional[List[List[float]]],
        n_results: int,
        threshold: float,
        where: Optional[Dict[str, Any]]
    ) -> List[SearchResult]:
        """Search in ChromaDB."""
        results = []

        try:
            if query_embeddings:
                chroma_results = self.chroma.search_with_embeddings(
                    query_embeddings=query_embeddings,
                    n_results=n_results,
                    where=where
                )
            else:
                chroma_results = self.chroma.search(
                    query=query,
                    n_results=n_results,
                    where=where
                )

            if chroma_results.get("error"):
                logger.warning(f"Chroma search error: {chroma_results['error']}")
                return []

            for i in range(chroma_results["count"]):
                doc = chroma_results["documents"][i]
                dist = chroma_results["distances"][i]
                metadata = chroma_results["metadatas"][i]
                doc_id = chroma_results["ids"][i]

                similarity = self.chroma._distance_to_similarity(dist)
                if similarity >= threshold:
                    results.append(SearchResult(
                        document=doc,
                        score=similarity,
                        metadata=metadata,
                        doc_id=doc_id,
                        source="chroma"
                    ))

        except Exception as error:
            logger.error(f"Chroma search failed: {error}")

        return results

    def _search_faiss(
        self,
        query_embeddings: List[List[float]],
        n_results: int,
        threshold: float,
        where: Optional[Dict[str, Any]]
    ) -> List[SearchResult]:
        """Search in FAISS."""
        results = []

        try:
            faiss_results = self.faiss.search_with_embeddings(
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where
            )

            if faiss_results.get("error"):
                logger.warning(f"FAISS search error: {faiss_results['error']}")
                return []

            for i in range(faiss_results["count"]):
                doc = faiss_results["documents"][i]
                dist = faiss_results["distances"][i]
                metadata = faiss_results["metadatas"][i]
                doc_id = faiss_results["ids"][i]

                similarity = self.faiss._distance_to_similarity(dist)
                if similarity >= threshold:
                    results.append(SearchResult(
                        document=doc,
                        score=similarity,
                        metadata=metadata,
                        doc_id=doc_id,
                        source="faiss"
                    ))

        except Exception as error:
            logger.error(f"FAISS search failed: {error}")

        return results

    def hybrid_search(
        self,
        query: str,
        query_embeddings: List[List[float]],
        n_results: int = 10,
        threshold: float = 0.3,
        weights: Optional[Dict[str, float]] = None,
        where: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Perform hybrid search combining ChromaDB and FAISS results."""
        if weights is None:
            weights = self.config["hybrid_weights"]

        chroma_results = self._search_chroma(
            query=query,
            query_embeddings=query_embeddings,
            n_results=n_results * 3,
            threshold=threshold,
            where=where
        )

        faiss_results = self._search_faiss(
            query_embeddings=query_embeddings,
            n_results=n_results * 3,
            threshold=threshold,
            where=where
        )

        result_dict = {}

        for result in chroma_results:
            result_dict[result.doc_id] = {
                "result": result,
                "score": result.score * weights.get("chroma", 0.5)
            }

        for result in faiss_results:
            if result.doc_id in result_dict:
                result_dict[result.doc_id]["score"] += result.score * weights.get("faiss", 0.5)
            else:
                result_dict[result.doc_id] = {
                    "result": result,
                    "score": result.score * weights.get("faiss", 0.5)
                }

        sorted_results = sorted(
            result_dict.values(),
            key=lambda x: x["score"],
            reverse=True
        )[:n_results]

        final_results = [item["result"] for item in sorted_results]

        for i, item in enumerate(sorted_results):
            final_results[i].score = item["score"]

        logger.info(f"Hybrid search found {len(final_results)} results")
        return final_results

    def multi_query_search(
        self,
        queries: List[str],
        query_embeddings_list: Optional[List[List[List[float]]]] = None,
        n_results: int = 5,
        threshold: float = 0.3,
        aggregation: str = "max"
    ) -> List[SearchResult]:
        """Search with multiple queries and aggregate results."""
        all_results = []

        for i, query in enumerate(queries):
            embeddings = query_embeddings_list[i] if query_embeddings_list else None

            results = self.search(
                query=query,
                query_embeddings=embeddings,
                n_results=n_results * 2,
                threshold=threshold,
                use_cache=True
            )

            all_results.append(results)

        aggregated = {}

        for query_results in all_results:
            for result in query_results:
                if result.doc_id not in aggregated:
                    aggregated[result.doc_id] = {
                        "result": result,
                        "scores": []
                    }
                aggregated[result.doc_id]["scores"].append(result.score)

        final_results = []
        for doc_id, data in aggregated.items():
            if aggregation == "max":
                combined_score = max(data["scores"])
            elif aggregation == "avg":
                combined_score = sum(data["scores"]) / len(data["scores"])
            elif aggregation == "sum":
                combined_score = sum(data["scores"])
            else:
                combined_score = data["scores"][0]

            result = data["result"]
            result.score = combined_score
            final_results.append(result)

        final_results.sort(key=lambda x: x.score, reverse=True)

        return final_results[:n_results]

    def faceted_search(
        self,
        query: str,
        facets: List[str],
        query_embeddings: Optional[List[List[float]]] = None,
        n_results: int = 20,
        threshold: float = 0.3
    ) -> Dict[str, Any]:
        """Perform faceted search with category counts."""
        results = self.search(
            query=query,
            query_embeddings=query_embeddings,
            n_results=n_results,
            threshold=threshold,
            use_cache=True
        )

        facet_counts = {}
        for facet in facets:
            counts = defaultdict(int)
            for result in results:
                value = result.metadata.get(facet, "unknown")
                counts[value] += 1
            facet_counts[facet] = dict(counts)

        return {
            "results": [r.to_dict() for r in results],
            "facet_counts": facet_counts,
            "total_results": len(results),
            "query": query
        }

    def get_suggestions(self, query: str, max_suggestions: int = 5) -> List[str]:
        """Get query suggestions based on search history."""
        if not self.cache:
            return []

        suggestions = []
        query_lower = query.lower()

        with self.cache.cache_lock:
            for _, entry in self.cache.cache_index.items():
                cached_query = entry.get("query", "")
                if cached_query.lower().startswith(query_lower) and cached_query != query:
                    suggestions.append(cached_query)

        suggestions = list(set(suggestions))
        suggestions.sort(key=lambda q: self._get_query_popularity(q), reverse=True)

        return suggestions[:max_suggestions]

    def _get_query_popularity(self, query: str) -> int:
        """Get popularity score for a query."""
        if not self.cache:
            return 0

        for _, entry in self.cache.cache_index.items():
            if entry.get("query") == query:
                return entry.get("access_count", 0)

        return 0

    def clear_cache(self) -> int:
        """Clear search cache."""
        if self.cache:
            return self.cache.clear()
        return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get search engine statistics."""
        with self.stats_lock:
            stats = self.stats.copy()

        if self.cache:
            stats["cache"] = self.cache.get_stats()

        stats["config"] = self.config

        return stats

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on search engine."""
        try:
            chroma_health = self.chroma.health_check()
            faiss_health = self.faiss.health_check()
            cache_healthy = self.cache is not None

            overall_status = "healthy"
            if chroma_health["status"] != "healthy" or faiss_health["status"] != "healthy":
                overall_status = "degraded"

            return {
                "status": overall_status,
                "chroma": chroma_health,
                "faiss": faiss_health,
                "cache_enabled": cache_healthy,
                "cache_stats": self.cache.get_stats() if self.cache else None,
                "search_stats": self.get_stats(),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as error:
            return {
                "status": "unhealthy",
                "error": str(error),
                "timestamp": datetime.now().isoformat()
            }

    def close(self) -> None:
        """Close all connections and cleanup."""
        if self.cache:
            self.cache.stop()

        gc.collect()
        if os.name == "nt":
            time.sleep(0.5)

    def __del__(self):
        """Cleanup on deletion."""
        self.close()


_search_engine_instance: Optional[SearchEngine] = None


def get_search_engine(
    chroma_client: Optional[ChromaClient] = None,
    faiss_client: Optional[FAISSClient] = None,
    cache_dir: Optional[Path] = None,
    config: Optional[Dict[str, Any]] = None
) -> SearchEngine:
    """Get or create singleton SearchEngine instance."""
    global _search_engine_instance

    if _search_engine_instance is None:
        _search_engine_instance = SearchEngine(chroma_client, faiss_client, cache_dir, config)

    return _search_engine_instance


def test_search_engine():
    """Test function for SearchEngine."""
    import gc

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    print("=" * 60)
    print("SEARCH ENGINE TEST SUITE")
    print("=" * 60)

    test_dir = Path(tempfile.gettempdir()) / f"search_test_{uuid.uuid4().hex[:8]}"
    test_dir.mkdir(parents=True, exist_ok=True)
    print(f"Test directory: {test_dir}")

    engine = None
    chroma = None
    faiss = None

    try:
        chroma = ChromaClient(test_dir / "chroma")
        faiss = FAISSClient(test_dir / "faiss", dimension=384)

        engine = SearchEngine(
            chroma_client=chroma,
            faiss_client=faiss,
            cache_dir=test_dir / "cache"
        )

        print("\n1. HEALTH CHECK")
        print("-" * 40)
        health = engine.health_check()
        print(f"Status: {health['status']}")
        print(f"Chroma: {health['chroma']['status']}")
        print(f"FAISS: {health['faiss']['status']}")

        print("\n2. ADD TEST DOCUMENTS")
        print("-" * 40)

        test_documents = [
            "Artificial intelligence is revolutionizing multiple industries.",
            "Machine learning algorithms require extensive training data.",
            "Natural language processing enables computers to understand human language.",
            "Python is the most popular language for data science.",
            "Vector databases optimize similarity search for AI applications.",
            "Deep learning models need large amounts of computational power.",
            "Neural networks are inspired by the human brain structure.",
            "Data preprocessing is crucial for machine learning success.",
            "Computer vision helps machines interpret visual information.",
            "Reinforcement learning trains agents through trial and error."
        ]

        test_metadata = [
            {"category": "AI", "source": "test", "language": "en", "added_at": "2024-01-01T00:00:00"},
            {"category": "ML", "source": "test", "language": "en", "added_at": "2024-01-02T00:00:00"},
            {"category": "NLP", "source": "test", "language": "en", "added_at": "2024-01-03T00:00:00"},
            {"category": "programming", "source": "test", "language": "en", "added_at": "2024-01-04T00:00:00"},
            {"category": "database", "source": "test", "language": "en", "added_at": "2024-01-05T00:00:00"},
            {"category": "AI", "source": "test", "language": "en", "added_at": "2024-01-06T00:00:00"},
            {"category": "AI", "source": "test", "language": "en", "added_at": "2024-01-07T00:00:00"},
            {"category": "ML", "source": "test", "language": "en", "added_at": "2024-01-08T00:00:00"},
            {"category": "computer vision", "source": "test", "language": "en", "added_at": "2024-01-09T00:00:00"},
            {"category": "ML", "source": "test", "language": "en", "added_at": "2024-01-10T00:00:00"}
        ]

        np.random.seed(42)
        test_embeddings = np.random.randn(len(test_documents), 384).tolist()

        chroma_ids = chroma.add_documents(
            texts=test_documents,
            metadatas=test_metadata
        )
        print(f"Added {len(chroma_ids)} documents to Chroma")

        faiss_ids = faiss.add_documents(
            texts=test_documents,
            embeddings=test_embeddings,
            metadatas=test_metadata
        )
        print(f"Added {len(faiss_ids)} documents to FAISS")

        print("\n3. BASIC SEARCH")
        print("-" * 40)

        queries = [
            "artificial intelligence",
            "machine learning",
            "neural networks",
            "data science"
        ]

        for query in queries:
            print(f"\nQuery: '{query}'")
            results = engine.search(
                query=query,
                n_results=3,
                threshold=0.3
            )

            print(f"  Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"  {i}. Score: {result.score:.4f} | Source: {result.source}")
                print(f"     Category: {result.metadata.get('category', 'N/A')}")
                print(f"     Text: {result.document[:70]}...")

        print("\n4. HYBRID SEARCH")
        print("-" * 40)

        query = "AI and deep learning"
        query_embeddings = np.random.randn(1, 384).tolist()

        hybrid_results = engine.hybrid_search(
            query=query,
            query_embeddings=query_embeddings,
            n_results=5,
            weights={"chroma": 0.6, "faiss": 0.4}
        )

        print(f"Hybrid search for '{query}':")
        for i, result in enumerate(hybrid_results, 1):
            print(f"  {i}. Score: {result.score:.4f} | Source: {result.source}")

        print("\n5. FILTERED SEARCH")
        print("-" * 40)

        filtered_results = engine.search(
            query="machine learning",
            n_results=5,
            where={"category": "ML"},
            threshold=0.3
        )

        print(f"Filtered search (category=ML):")
        for i, result in enumerate(filtered_results, 1):
            print(f"  {i}. Score: {result.score:.4f} | Category: {result.metadata.get('category')}")

        print("\n6. FACETED SEARCH")
        print("-" * 40)

        faceted = engine.faceted_search(
            query="AI",
            facets=["category", "source"],
            n_results=10
        )

        print(f"Faceted search results: {len(faceted['results'])} total")
        print("Category facets:")
        for cat, count in faceted["facet_counts"]["category"].items():
            print(f"  {cat}: {count}")

        print("\n7. MULTI-QUERY SEARCH")
        print("-" * 40)

        multi_results = engine.multi_query_search(
            queries=["artificial intelligence", "machine learning", "deep learning"],
            n_results=5,
            aggregation="max"
        )

        print(f"Multi-query search results:")
        for i, result in enumerate(multi_results, 1):
            print(f"  {i}. Score: {result.score:.4f} | Text: {result.document[:60]}...")

        print("\n8. CACHE TEST")
        print("-" * 40)

        print("First search...")
        start = time.time()
        results1 = engine.search(query="vector database", n_results=3)
        time1 = time.time() - start

        print("Second search (cached)...")
        start = time.time()
        results2 = engine.search(query="vector database", n_results=3)
        time2 = time.time() - start

        print(f"First search: {time1*1000:.2f}ms")
        print(f"Second search: {time2*1000:.2f}ms")

        if time2 > 0:
            print(f"Speedup: {time1/time2:.1f}x")
        else:
            print("Speedup: INFINITE (cached response was instantaneous)")

        print("\n9. RERANKING TEST")
        print("-" * 40)

        strategies = ["similarity", "recency", "diversity", "hybrid"]

        for strategy in strategies:
            results = engine.search(
                query="AI",
                n_results=5,
                rerank_strategy=strategy
            )
            print(f"\nStrategy: {strategy}")
            for i, r in enumerate(results, 1):
                print(f"  {i}. Score: {r.score:.4f} | Category: {r.metadata.get('category')}")

        print("\n10. SUGGESTIONS")
        print("-" * 40)

        suggestions = engine.get_suggestions("ma", max_suggestions=3)
        print(f"Suggestions for 'ma': {suggestions}")

        print("\n11. STATISTICS")
        print("-" * 40)

        stats = engine.get_stats()
        print(f"Total searches: {stats['total_searches']}")
        print(f"Avg time: {stats['avg_time_ms']:.2f}ms")
        print(f"Chroma searches: {stats['chroma_searches']}")
        print(f"FAISS searches: {stats['faiss_searches']}")
        print(f"Hybrid searches: {stats['hybrid_searches']}")
        print(f"Cache hits: {stats.get('cache_hits', 0)}")

        if "cache" in stats:
            print(f"Cache hit rate: {stats['cache']['hit_rate']:.2%}")

        print("\n12. CLEAR CACHE")
        print("-" * 40)

        cleared = engine.clear_cache()
        print(f"Cleared {cleared} cache entries")

        print("\n13. FINAL HEALTH CHECK")
        print("-" * 40)

        final_health = engine.health_check()
        print(f"Status: {final_health['status']}")

        print("\n" + "=" * 60)
        print("TEST SUITE COMPLETED SUCCESSFULLY")
        print("=" * 60)

        return True

    except Exception as error:
        print(f"\nTEST FAILED: {error}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        print("\n14. CLEANUP")
        print("-" * 40)

        if engine:
            try:
                engine.close()
                print("  Engine closed")
            except:
                pass

        if chroma:
            try:
                if hasattr(chroma, "client"):
                    chroma.client.close()
            except:
                pass

        for i in range(3):
            gc.collect()
            if os.name == "nt":
                time.sleep(1)
                print(f"  GC pass {i+1}/3")

        for attempt in range(5):
            try:
                if test_dir.exists():
                    shutil.rmtree(test_dir, ignore_errors=True)
                print(f"  Cleanup successful on attempt {attempt+1}")
                break
            except Exception as e:
                print(f"  Cleanup attempt {attempt+1}: {e}")
                time.sleep(2)
                gc.collect()

        print("  Test completed")


if __name__ == "__main__":
    success = test_search_engine()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")