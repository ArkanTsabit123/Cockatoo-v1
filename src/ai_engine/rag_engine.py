# src/ai_engine/rag_engine.py

"""RAG (Retrieval-Augmented Generation) engine implementation.

Combines document retrieval from vector store with LLM generation
to provide context-aware responses.
"""

import time
import threading
import re
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple, Any, Optional, Union, Generator
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Source:
    """Source document for RAG response."""
    document_id: str
    text: str
    similarity_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "text": self.text,
            "similarity_score": self.similarity_score,
            "metadata": self.metadata
        }


@dataclass
class Document:
    """Document for RAG processing."""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    source: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        data = {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "source": self.source,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
        if self.embedding is not None:
            data["embedding"] = self.embedding.tolist()
        return data
    
    def to_source(self, score: float) -> Source:
        return Source(
            document_id=self.id,
            text=self.content,
            similarity_score=score,
            metadata=self.metadata
        )


@dataclass
class RAGResponse:
    """Response from RAG engine."""
    query: str
    answer: str
    sources: List[Source] = field(default_factory=list)
    confidence: float = 0.0
    processing_time_ms: float = 0.0
    total_tokens: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def format_sources(self) -> str:
        if not self.sources:
            return "No sources available."
        
        lines = ["Sources:"]
        for i, source in enumerate(self.sources, 1):
            preview = source.text[:100] + "..." if len(source.text) > 100 else source.text
            lines.append(f"[{i}] score: {source.similarity_score:.2f} | {preview}")
            if source.metadata:
                meta_str = ", ".join(f"{k}: {v}" for k, v in source.metadata.items())
                lines.append(f"    Metadata: {meta_str}")
        return "\n".join(lines)
    
    def get_citations(self, format: str = "text") -> str:
        if not self.sources:
            return ""
        
        if format == "apa":
            citations = []
            for source in self.sources:
                author = source.metadata.get("author", "Unknown")
                year = source.metadata.get("date", "n.d.")[:4] if source.metadata.get("date") else "n.d."
                title = source.metadata.get("file_name", "Untitled")
                citations.append(f"{author} ({year}). {title}")
            return "\n".join(citations)
        else:
            return "\n".join([
                f"[{i+1}] {s.metadata.get('file_name', 'Unknown')}" 
                for i, s in enumerate(self.sources)
            ])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "sources": [s.to_dict() for s in self.sources],
            "confidence": self.confidence,
            "processing_time_ms": self.processing_time_ms,
            "total_tokens": self.total_tokens,
            "metadata": self.metadata
        }


@dataclass
class RAGConfig:
    """Configuration for RAG engine."""
    top_k: int = 5
    similarity_threshold: float = 0.7
    retrieval_weight: float = 0.7
    llm_weight: float = 0.3
    max_context_tokens: int = 2000
    enable_hybrid_search: bool = True
    deduplicate_sources: bool = True
    enable_query_expansion: bool = False
    rerank_results: bool = False
    enable_cache: bool = False
    cache_ttl_seconds: int = 3600
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def validate(self) -> List[str]:
        errors = []
        
        if not 1 <= self.top_k <= 50:
            errors.append(f"top_k must be between 1 and 50, got {self.top_k}")
        
        if not 0 <= self.similarity_threshold <= 1:
            errors.append(f"similarity_threshold must be between 0 and 1, got {self.similarity_threshold}")
        
        if not 0 <= self.retrieval_weight <= 1 or not 0 <= self.llm_weight <= 1:
            errors.append("weights must be between 0 and 1")
        
        if abs(self.retrieval_weight + self.llm_weight - 1.0) > 0.01:
            errors.append(f"retrieval_weight + llm_weight must sum to 1, got {self.retrieval_weight + self.llm_weight}")
        
        if self.max_context_tokens < 100:
            errors.append(f"max_context_tokens must be at least 100, got {self.max_context_tokens}")
        
        return errors


class Retriever:
    """Document retriever for RAG."""
    
    def __init__(self, vector_store, embedding_service):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self._lock = threading.RLock()
    
    def retrieve(self, query: str, top_k: int = 5, threshold: float = 0.7, 
                filter_metadata: Optional[Dict] = None) -> List[Tuple[Document, float]]:
        """
        Retrieve documents from vector store based on query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            threshold: Similarity threshold
            filter_metadata: Metadata filters
            
        Returns:
            List of (Document, score) tuples
        """
        logger.info(f"Retriever.retrieve called with query: '{query[:50]}...', top_k={top_k}, threshold={threshold}")
        
        try:
            query_emb = self.embedding_service.encode(query)
            documents = []
            
            try:
                if hasattr(self.vector_store, 'get_collection_info'):
                    info = self.vector_store.get_collection_info()
                    logger.info(f"Collection info: {info}")
                    if info.get('document_count', 0) == 0:
                        logger.warning("Collection is empty, no documents to retrieve")
                        return []
                
                if hasattr(self.vector_store, 'count_documents'):
                    count = self.vector_store.count_documents()
                    logger.info(f"Document count: {count}")
                    if count == 0:
                        logger.warning("No documents in store")
                        return []
            except Exception as e:
                logger.warning(f"Failed to check collection: {e}")
            
            if hasattr(self.vector_store, 'search'):
                logger.info(f"Using vector_store.search with query: '{query[:50]}...'")
                results = self.vector_store.search(
                    query=query,
                    n_results=top_k,
                    where=filter_metadata
                )
                
                if isinstance(results, dict):
                    ids = results.get("ids", [])
                    docs = results.get("documents", [])
                    distances = results.get("distances", [])
                    metadatas = results.get("metadatas", [])
                    
                    logger.info(f"Retrieved {len(ids)} documents from vector store via search")
                    
                    if len(ids) == 0:
                        logger.warning("Search returned no results")
                        logger.info("Attempting with empty query to get all documents")
                        all_results = self.vector_store.search("", n_results=top_k)
                        if isinstance(all_results, dict):
                            ids = all_results.get("ids", [])
                            docs = all_results.get("documents", [])
                            distances = all_results.get("distances", [])
                            metadatas = all_results.get("metadatas", [])
                            logger.info(f"Empty query returned {len(ids)} documents")
                    
                    for i, (doc_id, doc_text, distance, metadata) in enumerate(zip(ids, docs, distances, metadatas)):
                        if hasattr(self.vector_store, '_distance_to_similarity'):
                            score = self.vector_store._distance_to_similarity(distance)
                        else:
                            score = 1.0 / (1.0 + distance) if distance > 0 else 1.0
                        
                        logger.debug(f"Document {i}: score={score:.3f}, threshold={threshold}, text='{doc_text[:50]}...'")
                        
                        if score >= threshold:
                            doc = Document(
                                id=doc_id,
                                content=doc_text,
                                metadata=metadata or {}
                            )
                            documents.append((doc, score))
                        else:
                            logger.debug(f"Document {i} skipped: score {score:.3f} < threshold {threshold}")
            
            elif hasattr(self.vector_store, 'similarity_search'):
                logger.info(f"Using vector_store.similarity_search with query: '{query[:50]}...'")
                results = self.vector_store.similarity_search(
                    query=query,
                    k=top_k
                )
                
                for doc, score in results:
                    if score >= threshold:
                        documents.append((doc, score))
            
            elif hasattr(self.vector_store, 'search_with_embeddings'):
                logger.info(f"Using vector_store.search_with_embeddings")
                results = self.vector_store.search_with_embeddings(
                    query_embeddings=[query_emb.tolist()],
                    n_results=top_k,
                    where=filter_metadata
                )
                
                if isinstance(results, dict):
                    ids = results.get("ids", [])
                    docs = results.get("documents", [])
                    distances = results.get("distances", [])
                    metadatas = results.get("metadatas", [])
                    
                    for i, (doc_id, doc_text, distance, metadata) in enumerate(zip(ids, docs, distances, metadatas)):
                        if hasattr(self.vector_store, '_distance_to_similarity'):
                            score = self.vector_store._distance_to_similarity(distance)
                        else:
                            score = 1.0 / (1.0 + distance) if distance > 0 else 1.0
                        
                        if score >= threshold:
                            doc = Document(
                                id=doc_id,
                                content=doc_text,
                                metadata=metadata or {}
                            )
                            documents.append((doc, score))
            
            logger.info(f"After threshold filtering: {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error in retrieve: {e}", exc_info=True)
            return []
    
    def retrieve_by_text(self, query: str, top_k: int = 5, threshold: float = 0.7,
                        filter_metadata: Optional[Dict] = None) -> List[Tuple[Document, float]]:
        """Alias for retrieve method."""
        return self.retrieve(query, top_k, threshold, filter_metadata)
    
    def rerank(self, documents: List[Tuple[Document, float]], query: str) -> List[Tuple[Document, float]]:
        """Rerank documents based on query similarity."""
        if not documents:
            return []
        
        query_emb = self.embedding_service.encode(query)
        
        scored_docs = []
        for doc, original_score in documents:
            if doc.embedding is not None:
                new_score = self.embedding_service.compute_similarity(query_emb, doc.embedding)
                combined_score = 0.7 * new_score + 0.3 * original_score
                scored_docs.append((doc, combined_score))
            else:
                scored_docs.append((doc, original_score))
        
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs
    
    def mmr_retrieve(self, query: str, top_k: int = 5, lambda_param: float = 0.5) -> List[Document]:
        """Maximum Marginal Relevance retrieval for diverse results."""
        candidates = self.retrieve(query, top_k=top_k * 3, threshold=0.0)
        if not candidates:
            return []
        
        query_emb = self.embedding_service.encode(query)
        selected = []
        remaining = candidates.copy()
        
        if remaining:
            scores = []
            for doc, _ in remaining:
                if doc.embedding is not None:
                    score = self.embedding_service.compute_similarity(query_emb, doc.embedding)
                    scores.append((doc, score))
            
            scores.sort(key=lambda x: x[1], reverse=True)
            best_doc, _ = scores[0]
            selected.append(best_doc)
            remaining = [(d, s) for d, s in remaining if d.id != best_doc.id]
        
        while len(selected) < top_k and remaining:
            mmr_scores = []
            
            for doc, _ in remaining:
                if doc.embedding is None:
                    continue
                
                sim_query = self.embedding_service.compute_similarity(query_emb, doc.embedding)
                
                sim_selected = max([
                    self.embedding_service.compute_similarity(doc.embedding, s.embedding)
                    for s in selected if s.embedding is not None
                ]) if selected else 0
                
                mmr = lambda_param * sim_query - (1 - lambda_param) * sim_selected
                mmr_scores.append((doc, mmr))
            
            if mmr_scores:
                mmr_scores.sort(key=lambda x: x[1], reverse=True)
                selected.append(mmr_scores[0][0])
                remaining = [(d, s) for d, s in remaining if d.id != mmr_scores[0][0].id]
            else:
                break
        
        return selected


class ContextBuilder:
    """Build context from documents for LLM."""
    
    def __init__(self, max_tokens: int = 2000):
        self.max_tokens = max_tokens
    
    def build_context(self, documents: List[Tuple[Document, float]], query: str,
                     include_metadata: bool = False, raise_on_overflow: bool = False) -> str:
        """Build context string from retrieved documents."""
        if not documents:
            return "No relevant documents found."
        
        context_parts = []
        current_tokens = 0
        
        separator = "\n\n"
        separator_tokens = self._count_tokens(separator)
        
        for doc, score in documents:
            doc_text = ""
            if include_metadata and doc.metadata:
                meta_parts = []
                for k, v in doc.metadata.items():
                    if k in ['page', 'chapter', 'source', 'file_name']:
                        meta_parts.append(f"[{k.title()}: {v}]")
                if meta_parts:
                    doc_text += " ".join(meta_parts) + "\n"
            
            doc_text += doc.content
            
            prefix = f"Source (relevance: {score:.2f}):\n"
            
            full_doc_text = prefix + doc_text
            full_doc_tokens = self._count_tokens(full_doc_text)
            
            if current_tokens == 0:
                needed_tokens = full_doc_tokens
            else:
                needed_tokens = separator_tokens + full_doc_tokens
            
            if current_tokens + needed_tokens <= self.max_tokens:
                if current_tokens > 0:
                    context_parts.append(separator)
                    current_tokens += separator_tokens
                
                context_parts.append(full_doc_text)
                current_tokens += full_doc_tokens
            else:
                if raise_on_overflow:
                    raise ValueError(f"Context window exceeded: {current_tokens + needed_tokens} > {self.max_tokens}")
                
                remaining_tokens = self.max_tokens - current_tokens - (separator_tokens if current_tokens > 0 else 0)
                
                if remaining_tokens > 10:
                    max_chars = max(1, remaining_tokens * 4)
                    
                    if len(doc_text) > max_chars:
                        truncated_text = doc_text[:max_chars] + "..."
                    else:
                        truncated_text = doc_text
                    
                    if current_tokens > 0:
                        context_parts.append(separator)
                    
                    context_parts.append(prefix + truncated_text)
                break
        
        return "".join(context_parts)

    def _count_tokens(self, text: str) -> int:
        """Simple token counter (approximate)."""
        if not text:
            return 0
        return max(1, (len(text) + 3) // 4)
    
    def format_documents(self, documents: List[Document]) -> str:
        """Format documents for display."""
        parts = []
        for i, doc in enumerate(documents, 1):
            parts.append(f"[{i}] {doc.source or 'Unknown'}: {doc.content[:200]}...")
        return "\n".join(parts)


class Cache:
    """Simple cache for RAG responses."""
    
    def __init__(self, ttl_seconds: int = 3600):
        self.cache: Dict[str, Tuple[RAGResponse, datetime]] = {}
        self.ttl_seconds = ttl_seconds
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[RAGResponse]:
        with self._lock:
            if key in self.cache:
                response, timestamp = self.cache[key]
                if datetime.now() - timestamp < timedelta(seconds=self.ttl_seconds):
                    response.metadata["cache_hit"] = True
                    return response
                else:
                    del self.cache[key]
            return None
    
    def set(self, key: str, response: RAGResponse):
        with self._lock:
            self.cache[key] = (response, datetime.now())
    
    def invalidate(self):
        with self._lock:
            self.cache.clear()
    
    def invalidate_key(self, key: str):
        with self._lock:
            if key in self.cache:
                del self.cache[key]


class PerformanceMetrics:
    """Track performance metrics."""
    
    def __init__(self):
        self.stats = {
            "total_queries": 0,
            "total_tokens": 0,
            "total_processing_time_ms": 0,
            "queries_by_type": {}
        }
        self._lock = threading.RLock()
    
    def record_query(self, processing_time_ms: float, tokens: int, query_type: str = "default"):
        with self._lock:
            self.stats["total_queries"] += 1
            self.stats["total_tokens"] += tokens
            self.stats["total_processing_time_ms"] += processing_time_ms
            
            if query_type not in self.stats["queries_by_type"]:
                self.stats["queries_by_type"][query_type] = {
                    "count": 0,
                    "total_tokens": 0,
                    "total_processing_time_ms": 0
                }
            
            self.stats["queries_by_type"][query_type]["count"] += 1
            self.stats["queries_by_type"][query_type]["total_tokens"] += tokens
            self.stats["queries_by_type"][query_type]["total_processing_time_ms"] += processing_time_ms
    
    def get_metrics(self, by_type: bool = False) -> Dict[str, Any]:
        with self._lock:
            if self.stats["total_queries"] == 0:
                return {
                    "total_queries": 0,
                    "total_tokens": 0,
                    "average_tokens_per_query": 0,
                    "average_processing_time_ms": 0
                }
            
            metrics = {
                "total_queries": self.stats["total_queries"],
                "total_tokens": self.stats["total_tokens"],
                "average_tokens_per_query": self.stats["total_tokens"] / self.stats["total_queries"],
                "average_processing_time_ms": self.stats["total_processing_time_ms"] / self.stats["total_queries"]
            }
            
            if by_type:
                metrics["by_type"] = {}
                for qtype, data in self.stats["queries_by_type"].items():
                    if data["count"] > 0:
                        metrics["by_type"][qtype] = {
                            "count": data["count"],
                            "average_tokens": data["total_tokens"] / data["count"],
                            "average_processing_time_ms": data["total_processing_time_ms"] / data["count"]
                        }
            
            return metrics
    
    def reset(self):
        with self._lock:
            self.stats = {
                "total_queries": 0,
                "total_tokens": 0,
                "total_processing_time_ms": 0,
                "queries_by_type": {}
            }


class ConversationManager:
    """Manage conversation history with context preservation."""
    
    def __init__(self):
        self.conversations: Dict[str, List[Dict]] = {}
        self._lock = threading.RLock()
    
    def start_conversation(self, conversation_id: str):
        with self._lock:
            if conversation_id not in self.conversations:
                self.conversations[conversation_id] = []
                logger.info(f"Started new conversation: {conversation_id}")
    
    def add_to_history(self, conversation_id: str, query: str, response: RAGResponse):
        with self._lock:
            if conversation_id in self.conversations:
                self.conversations[conversation_id].append({
                    "query": query,
                    "answer": response.answer,
                    "timestamp": datetime.now().isoformat(),
                    "sources": [s.to_dict() for s in response.sources]
                })
                
                if len(self.conversations[conversation_id]) > 20:
                    self.conversations[conversation_id] = self.conversations[conversation_id][-20:]
                
                logger.debug(f"Added message to conversation {conversation_id}. Total messages: {len(self.conversations[conversation_id])}")
    
    def get_history(self, conversation_id: str) -> Optional[List[Dict]]:
        with self._lock:
            history = self.conversations.get(conversation_id)
            if history:
                logger.debug(f"Retrieved history for {conversation_id}: {len(history)} messages")
            return history
    
    def clear_conversation(self, conversation_id: str):
        with self._lock:
            if conversation_id in self.conversations:
                del self.conversations[conversation_id]
                logger.info(f"Cleared conversation: {conversation_id}")


class RAGEngine:
    """Main RAG engine combining retrieval and generation."""
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config: Optional[Union[RAGConfig, Dict]] = None,
                 llm_client=None, vector_store=None):
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        if llm_client is None or vector_store is None:
            raise ValueError("LLM client and vector store are required")
        
        if isinstance(config, dict):
            self.config = RAGConfig(**config)
        else:
            self.config = config or RAGConfig()
        
        self.llm_client = llm_client
        self.vector_store = vector_store
        
        self.embedding_service = self._create_mock_embedding_service()
        
        self.retriever = Retriever(vector_store, self.embedding_service)
        self.context_builder = ContextBuilder(self.config.max_context_tokens)
        self.cache = Cache(self.config.cache_ttl_seconds)
        self.metrics = PerformanceMetrics()
        self.conversation_manager = ConversationManager()
        
        self._lock = threading.RLock()
        self._stats = {
            "total_queries": 0,
            "total_tokens": 0,
            "total_processing_time_ms": 0
        }
        self._initialized = True
        
        logger.info("RAGEngine initialized with config: %s", self.config.to_dict())
    
    def _create_mock_embedding_service(self):
        """Create a mock embedding service for testing."""
        class MockEmbeddingService:
            def encode(self, text):
                return np.random.rand(384)
            
            def compute_similarity(self, emb1, emb2):
                return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-10))
        
        return MockEmbeddingService()
    
    def _normalize_llm_response(self, llm_response):
        """
        Normalize LLM response to have consistent attributes.
        Handles differences between LLM client implementations.
        """
        from types import SimpleNamespace
        
        if hasattr(llm_response, 'content'):
            return llm_response
        
        if hasattr(llm_response, 'text'):
            normalized = SimpleNamespace()
            normalized.content = llm_response.text
            normalized.text = llm_response.text
            
            if hasattr(llm_response, 'total_tokens'):
                normalized.total_tokens = llm_response.total_tokens
            elif hasattr(llm_response, 'usage'):
                normalized.usage = llm_response.usage
                if isinstance(llm_response.usage, dict):
                    normalized.total_tokens = llm_response.usage.get('total_tokens', 0)
                else:
                    normalized.total_tokens = 0
            else:
                normalized.total_tokens = 0
            
            if hasattr(llm_response, 'finish_reason'):
                normalized.finish_reason = llm_response.finish_reason
            
            if hasattr(llm_response, 'latency'):
                normalized.latency = llm_response.latency
            
            return normalized
        
        fallback = SimpleNamespace()
        fallback.content = str(llm_response)
        fallback.text = str(llm_response)
        fallback.total_tokens = 0
        return fallback
    
    def query(self, query: str, filter_metadata: Optional[Dict] = None,
              top_k: Optional[int] = None, query_type: str = "default",
              fallback: bool = False, conversation_id: Optional[str] = None,
              include_history: bool = False) -> RAGResponse:
        """
        Main query method for RAG engine.
        
        Args:
            query: User query
            filter_metadata: Metadata filters for retrieval
            top_k: Number of documents to retrieve
            query_type: Type of query for metrics
            fallback: Whether to use fallback if main query fails
            conversation_id: ID for conversation tracking
            include_history: Whether to include conversation history
            
        Returns:
            RAGResponse object
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        start_time = time.time()
        
        if self.config.enable_cache:
            cache_key = f"{query}:{filter_metadata}:{top_k}"
            cached = self.cache.get(cache_key)
            if cached:
                cached.metadata["cache_hit"] = True
                cached.processing_time_ms = (time.time() - start_time) * 1000
                logger.info(f"Cache hit for query: {query[:50]}...")
                return cached
        
        top_k = top_k or self.config.top_k
        threshold = self.config.similarity_threshold
        
        try:
            retrieval_query = query
            
            logger.info(f"Processing query: '{query[:50]}...' with top_k={top_k}, threshold={threshold}")
            
            try:
                collection_info = self.vector_store.get_collection_info()
                logger.info(f"Vector store status: {collection_info}")
                
                direct_results = self.vector_store.search(retrieval_query, n_results=top_k)
                logger.info(f"Direct search returned {direct_results.get('count', 0)} results")
                if direct_results.get('count', 0) > 0:
                    sample = direct_results.get('documents', [''])[0][:100] if direct_results.get('documents') else ''
                    logger.info(f"Sample result: {sample}...")
            except Exception as e:
                logger.warning(f"Debug info failed: {e}")
            
            results = []
            thresholds_to_try = [threshold, 0.3, 0.1] if threshold > 0.1 else [threshold]
            
            for attempt_threshold in thresholds_to_try:
                if results:
                    break
                    
                logger.info(f"Attempting retrieval with threshold: {attempt_threshold}")
                
                try:
                    if hasattr(self.vector_store, 'count_documents'):
                        doc_count = self.vector_store.count_documents()
                        logger.info(f"Total documents in store before retrieval: {doc_count}")
                except:
                    pass
                
                try:
                    direct = self.vector_store.search(retrieval_query, n_results=top_k)
                    direct_count = direct.get('count', 0)
                    logger.info(f"Direct search returned {direct_count} results")
                    if direct_count > 0:
                        logger.info(f"Sample direct result: {direct.get('documents', [''])[0][:100]}...")
                    else:
                        logger.warning("Direct search returned 0 results")
                        
                        empty_results = self.vector_store.search("", n_results=top_k)
                        logger.info(f"Empty query returned {empty_results.get('count', 0)} results")
                except Exception as e:
                    logger.warning(f"Direct search failed: {e}")
                
                results = self.retriever.retrieve_by_text(
                    retrieval_query,
                    top_k=top_k * 2,
                    threshold=attempt_threshold,
                    filter_metadata=filter_metadata
                )
                
                if results:
                    logger.info(f"Found {len(results)} results with threshold {attempt_threshold}")
                else:
                    logger.warning(f"No results found with threshold {attempt_threshold}")
            
            if not results:
                logger.warning("No results found with any threshold, attempting to get all documents")
                try:
                    all_docs = self.vector_store.get_all_documents()
                    logger.info(f"Total documents in store: {len(all_docs)}")
                    
                    if all_docs:
                        for i, doc in enumerate(all_docs[:top_k]):
                            doc_obj = Document(
                                id=doc.get('id', f"doc_{i}"),
                                content=doc.get('document', ''),
                                metadata=doc.get('metadata', {})
                            )
                            results.append((doc_obj, 0.1))
                        logger.info(f"Using {len(results)} documents as fallback")
                except Exception as e:
                    logger.warning(f"Failed to get all documents: {e}")
            
            if self.config.rerank_results and results:
                results = self.retriever.rerank(results, retrieval_query)
            
            sources = [doc.to_source(score) for doc, score in results]
            
            if self.config.deduplicate_sources:
                sources = self._deduplicate_sources(sources)
            
            context = self.context_builder.build_context(
                results,
                query,
                include_metadata=True
            )
            
            user_name = None
            if conversation_id and include_history:
                history = self.conversation_manager.get_history(conversation_id)
                
                conversation_context = ""
                if history:
                    for i, item in enumerate(history):
                        conversation_context += f"User {i+1}: {item['query']}\n"
                        conversation_context += f"Assistant {i+1}: {item['answer']}\n\n"
                        
                        query_lower = item['query'].lower()
                        
                        patterns = [
                            (r"my name is (\w+)", "my name is"),
                            (r"my name's (\w+)", "my name's"), 
                            (r"i am (\w+)", "i am"),
                            (r"i'm (\w+)", "i'm"),
                            (r"call me (\w+)", "call me"),
                            (r"name[:\s]+(\w+)", "name:"),
                            (r"(\w+) is my name", "is my name")
                        ]
                        
                        for pattern, desc in patterns:
                            match = re.search(pattern, query_lower, re.IGNORECASE)
                            if match:
                                user_name = match.group(1).capitalize()
                                logger.info(f"Extracted user name: {user_name} using pattern: {desc}")
                                break
                
                is_name_query = any(phrase in query.lower() for phrase in 
                                   ["my name", "what is my name", "who am i", "name"])
                
                if is_name_query:
                    if user_name:
                        system_message = f"The user's name is {user_name}. You MUST answer with their name when asked about it. Do not introduce yourself as a chatbot."
                        logger.info(f"Using name-specific prompt for user: {user_name}")
                    else:
                        system_message = "The user has not provided their name yet."
                else:
                    if user_name:
                        system_message = f"Remember that the user's name is {user_name}."
                    else:
                        system_message = "Remember the conversation history."
                
                if conversation_context:
                    full_context = f"""CONVERSATION HISTORY:
{conversation_context}

{system_message}

DOCUMENT CONTEXT:
{context}

CURRENT QUESTION: {query}

INSTRUCTIONS:
1. Answer naturally based on the conversation history and document context
2. If asked about the user's name, answer with their actual name
3. Do not introduce yourself when answering about the user's name
4. Use information from the conversation history

Your answer:"""
                else:
                    full_context = f"""DOCUMENT CONTEXT:
{context}

QUESTION: {query}

Answer based on the document context above."""
                
                prompt = self._build_prompt(query, full_context)
            else:
                prompt = self._build_prompt(query, context)
            
            logger.info(f"Sending prompt to LLM (length: {len(prompt)} chars)")
            llm_response = self.llm_client.generate(prompt)
            
            llm_response = self._normalize_llm_response(llm_response)
            
            confidence = self._calculate_confidence(sources, llm_response.content)
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            response = RAGResponse(
                query=query,
                answer=llm_response.content,
                sources=sources,
                confidence=confidence,
                processing_time_ms=processing_time_ms,
                total_tokens=getattr(llm_response, 'total_tokens', 0),
                metadata={
                    "top_k": top_k,
                    "threshold": threshold,
                    "num_sources": len(sources),
                    "filter": filter_metadata,
                    "fallback_used": False,
                    "retrieved_count": len(results)
                }
            )
            
            if conversation_id:
                self.conversation_manager.add_to_history(conversation_id, query, response)
            
            self.metrics.record_query(processing_time_ms, response.total_tokens, query_type)
            with self._lock:
                self._stats["total_queries"] += 1
                self._stats["total_tokens"] += response.total_tokens
                self._stats["total_processing_time_ms"] += processing_time_ms
            
            if self.config.enable_cache:
                self.cache.set(cache_key, response)
            
            logger.info(f"Query completed in {processing_time_ms:.2f}ms with {len(sources)} sources")
            return response
            
        except Exception as e:
            logger.error("RAG query failed: %s", str(e), exc_info=True)
            
            if fallback:
                try:
                    fallback_results = self._fallback_search(query)
                    sources = [doc.to_source(0.5) for doc, _ in fallback_results]
                    
                    processing_time_ms = (time.time() - start_time) * 1000
                    
                    response = RAGResponse(
                        query=query,
                        answer="I found some information but could not process it fully.",
                        sources=sources,
                        confidence=0.3,
                        processing_time_ms=processing_time_ms,
                        total_tokens=0,
                        metadata={
                            "top_k": top_k,
                            "threshold": threshold,
                            "num_sources": len(sources),
                            "filter": filter_metadata,
                            "fallback_used": True,
                            "error": str(e)
                        }
                    )
                    return response
                except Exception as fallback_error:
                    logger.error("Fallback search also failed: %s", str(fallback_error))
            
            raise RuntimeError(f"RAG query failed: {str(e)}") from e
    
    def query_streaming(self, query: str, filter_metadata: Optional[Dict] = None,
                    top_k: Optional[int] = None, conversation_id: Optional[str] = None,
                    include_history: bool = False) -> Generator[str, None, None]:
        """Stream query response token by token."""
        start_time = time.time()
        
        top_k = top_k or self.config.top_k
        threshold = self.config.similarity_threshold
        
        results = self.retriever.retrieve_by_text(
            query,
            top_k=top_k,
            threshold=threshold,
            filter_metadata=filter_metadata
        )
        
        context = self.context_builder.build_context(results, query)
        
        if conversation_id and include_history:
            history = self.conversation_manager.get_history(conversation_id)
            if history:
                conversation_context = ""
                for item in history:
                    conversation_context += f"User: {item['query']}\n"
                    conversation_context += f"Assistant: {item['answer']}\n"
                
                context = f"Previous conversation:\n{conversation_context}\n\nDocument context:\n{context}"
        
        prompt = self._build_prompt(query, context)
        
        full_answer = []
        for chunk in self.llm_client.generate_stream(prompt):
            full_answer.append(chunk)
            yield chunk
        
        if conversation_id:
            response = RAGResponse(
                query=query,
                answer="".join(full_answer),
                sources=[doc.to_source(score) for doc, score in results],
                processing_time_ms=(time.time() - start_time) * 1000
            )
            self.conversation_manager.add_to_history(conversation_id, query, response)
        
        processing_time_ms = (time.time() - start_time) * 1000
        with self._lock:
            self._stats["total_queries"] += 1
            self._stats["total_processing_time_ms"] += processing_time_ms
    
    def _expand_query(self, query: str) -> List[str]:
        """Expand query with alternatives for better retrieval."""
        try:
            prompt = f"""Generate 3 alternative ways to ask this question for better search results:
Original: {query}

Alternative queries (one per line, numbered):"""
            
            response = self.llm_client.generate(prompt)
            
            if hasattr(response, 'content'):
                response_text = response.content
            elif hasattr(response, 'text'):
                response_text = response.text
            else:
                response_text = str(response)
            
            lines = response_text.strip().split('\n')
            alternatives = []
            for line in lines:
                if '.' in line:
                    line = line.split('.', 1)[1].strip()
                if line and line not in alternatives:
                    alternatives.append(line)
            
            return [query] + alternatives[:3]
        except Exception as e:
            logger.warning("Query expansion failed: %s", str(e))
            return [query]
    
    def _deduplicate_sources(self, sources: List[Source]) -> List[Source]:
        """Remove duplicate sources based on text content."""
        seen = set()
        unique = []
        
        for source in sources:
            fingerprint = source.text[:100]
            if fingerprint not in seen:
                seen.add(fingerprint)
                unique.append(source)
        
        return unique
    
    def _fallback_search(self, query: str) -> List[Tuple[Document, float]]:
        """Fallback search when main retrieval fails."""
        return self.retriever.retrieve_by_text(query, top_k=3, threshold=0.3)
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Build prompt for LLM with instructions."""
        
        is_name_query = any(phrase in query.lower() for phrase in 
                           ["my name", "what is my name", "who am i", "name"])
        
        if is_name_query and "CONVERSATION HISTORY" in context and "user's name is" in context:
            return f"""{context}

Remember: The user provided their name in the conversation history.
Answer with their actual name, not your own name.
Be direct and concise."""
        else:
            return f"""You are a helpful assistant. Answer the question based on the provided context.

Context:
{context}

Question: {query}

Answer:"""
    
    def _calculate_confidence(self, sources: List[Source], answer: str) -> float:
        """Calculate confidence score for response."""
        if not sources:
            return 0.0
        
        source_conf = sum(s.similarity_score for s in sources) / len(sources)
        answer_conf = min(1.0, len(answer) / 500) if answer else 0.0
        
        return 0.7 * source_conf + 0.3 * answer_conf
    
    def start_conversation(self, conversation_id: str):
        """Start a new conversation session."""
        self.conversation_manager.start_conversation(conversation_id)
    
    def get_conversation_history(self, conversation_id: str) -> Optional[List[Dict]]:
        """Get conversation history by ID."""
        return self.conversation_manager.get_history(conversation_id)
    
    def clear_conversation(self, conversation_id: str):
        """Clear a conversation."""
        self.conversation_manager.clear_conversation(conversation_id)
    
    def get_performance_metrics(self, by_type: bool = False) -> Dict[str, Any]:
        """Get performance metrics."""
        metrics = self.metrics.get_metrics(by_type)
        with self._lock:
            metrics.update(self._stats)
        return metrics
    
    def reset_metrics(self):
        """Reset all metrics."""
        self.metrics.reset()
        with self._lock:
            self._stats = {
                "total_queries": 0,
                "total_tokens": 0,
                "total_processing_time_ms": 0
            }
    
    def health_check(self, include_performance: bool = False) -> Dict[str, Any]:
        """Check health of all components."""
        components = {
            "llm": True,
            "vector_store": True,
            "config": True
        }
        
        health_score = 1.0
        
        try:
            self.llm_client.generate("test", max_tokens=5)
        except Exception as e:
            logger.warning(f"LLM health check failed: {e}")
            components["llm"] = False
            health_score -= 0.5
        
        try:
            self.vector_store.get_collection_info()
        except Exception as e:
            logger.warning(f"Vector store health check failed: {e}")
            components["vector_store"] = False
            health_score -= 0.5
        
        health = {
            "status": "healthy" if health_score >= 1.0 else "degraded" if health_score > 0 else "unhealthy",
            "health_score": max(0, health_score),
            "components": components,
            "timestamp": time.time()
        }
        
        if include_performance:
            health["performance"] = self.get_performance_metrics()
        
        return health
    
    def invalidate_cache(self):
        """Invalidate all cached responses."""
        self.cache.invalidate()
        logger.info("Cache invalidated")


def get_rag_engine(embedding_service=None, llm_client=None, vector_store=None,
                   config: Optional[Union[RAGConfig, Dict]] = None) -> RAGEngine:
    """Get RAG engine instance (singleton)."""
    return RAGEngine(config=config, llm_client=llm_client, vector_store=vector_store)