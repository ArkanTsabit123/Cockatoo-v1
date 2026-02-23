# src/ai_engine/rag_engine.py

"""RAG (Retrieval-Augmented Generation) engine.

Combines retrieval from vector store with LLM generation.
"""

import time
import threading
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
    """Document for RAG."""
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
        query_emb = self.embedding_service.encode(query)
        
        results = self.vector_store.search(
            query_emb, 
            top_k=top_k,
            filter_metadata=filter_metadata
        )
        
        documents = []
        for doc_id, score in results:
            if score >= threshold:
                doc_data = self.vector_store.get_document(doc_id)
                if doc_data:
                    doc = Document(
                        id=doc_data["id"],
                        content=doc_data["content"],
                        metadata=doc_data.get("metadata", {}),
                        source=doc_data.get("source"),
                        created_at=datetime.fromisoformat(doc_data["created_at"]) if "created_at" in doc_data else datetime.now(),
                        updated_at=datetime.fromisoformat(doc_data["updated_at"]) if "updated_at" in doc_data else datetime.now()
                    )
                    if "embedding" in doc_data:
                        doc.embedding = np.array(doc_data["embedding"])
                    documents.append((doc, score))
        
        return documents
    
    def retrieve_by_text(self, query: str, top_k: int = 5, threshold: float = 0.7,
                        filter_metadata: Optional[Dict] = None) -> List[Tuple[Document, float]]:
        return self.retrieve(query, top_k, threshold, filter_metadata)
    
    def rerank(self, documents: List[Tuple[Document, float]], query: str) -> List[Tuple[Document, float]]:
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
        if not text:
            return 0
        return max(1, (len(text) + 3) // 4)
    
    def format_documents(self, documents: List[Document]) -> str:
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
    """Manage conversation history."""
    
    def __init__(self):
        self.conversations: Dict[str, List[Dict]] = {}
        self._lock = threading.RLock()
    
    def start_conversation(self, conversation_id: str):
        with self._lock:
            if conversation_id not in self.conversations:
                self.conversations[conversation_id] = []
    
    def add_to_history(self, conversation_id: str, query: str, response: RAGResponse):
        with self._lock:
            if conversation_id in self.conversations:
                self.conversations[conversation_id].append({
                    "query": query,
                    "answer": response.answer,
                    "timestamp": datetime.now().isoformat(),
                    "sources": [s.to_dict() for s in response.sources]
                })
    
    def get_history(self, conversation_id: str) -> Optional[List[Dict]]:
        with self._lock:
            return self.conversations.get(conversation_id)
    
    def clear_conversation(self, conversation_id: str):
        with self._lock:
            if conversation_id in self.conversations:
                del self.conversations[conversation_id]
    
    def build_context_with_history(self, conversation_id: str, current_query: str) -> List[Dict]:
        history = self.get_history(conversation_id)
        if not history:
            return [{"role": "user", "content": current_query}]
        
        messages = []
        for item in history:
            messages.append({"role": "user", "content": item["query"]})
            messages.append({"role": "assistant", "content": item["answer"]})
        messages.append({"role": "user", "content": current_query})
        
        return messages


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
        class MockEmbeddingService:
            def encode(self, text):
                return np.random.rand(384)
            
            def compute_similarity(self, emb1, emb2):
                return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
        
        return MockEmbeddingService()
    
    def query(self, query: str, filter_metadata: Optional[Dict] = None,
              top_k: Optional[int] = None, query_type: str = "default",
              fallback: bool = False, conversation_id: Optional[str] = None,
              include_history: bool = False) -> RAGResponse:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        start_time = time.time()
        
        if self.config.enable_cache:
            cache_key = f"{query}:{filter_metadata}:{top_k}"
            cached = self.cache.get(cache_key)
            if cached:
                cached.metadata["cache_hit"] = True
                cached.processing_time_ms = (time.time() - start_time) * 1000
                return cached
        
        top_k = top_k or self.config.top_k
        threshold = self.config.similarity_threshold
        
        try:
            if self.config.enable_query_expansion:
                expanded_queries = self._expand_query(query)
                retrieval_query = expanded_queries[0] if expanded_queries else query
            else:
                retrieval_query = query
            
            results = self.retriever.retrieve_by_text(
                retrieval_query,
                top_k=top_k,
                threshold=threshold,
                filter_metadata=filter_metadata
            )
            
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
            
            if conversation_id and include_history:
                history = self.conversation_manager.build_context_with_history(conversation_id, query)
                history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history])
                context = f"Conversation history:\n{history_text}\n\nCurrent context:\n{context}"
            
            prompt = self._build_prompt(query, context)
            llm_response = self.llm_client.generate(prompt)
            
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
                    "fallback_used": False
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
            
            return response
            
        except Exception as e:
            logger.error("RAG query failed: %s", str(e))
            
            if fallback:
                try:
                    fallback_results = self._fallback_search(query)
                    sources = [doc.to_source(0.5) for doc, _ in fallback_results]
                    
                    processing_time_ms = (time.time() - start_time) * 1000
                    
                    response = RAGResponse(
                        query=query,
                        answer="I found some information but couldn't process it fully.",
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
                       top_k: Optional[int] = None) -> Generator[str, None, None]:
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
        prompt = self._build_prompt(query, context)
        
        full_answer = []
        for chunk in self.llm_client.generate_stream(prompt):
            full_answer.append(chunk)
            yield chunk
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        with self._lock:
            self._stats["total_queries"] += 1
            self._stats["total_processing_time_ms"] += processing_time_ms
    
    def _expand_query(self, query: str) -> List[str]:
        try:
            prompt = f"""Generate 3 alternative ways to ask this question for better search results:
Original: {query}

Alternative queries (one per line, numbered):"""
            
            response = self.llm_client.generate(prompt)
            
            lines = response.content.strip().split('\n')
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
        seen = set()
        unique = []
        
        for source in sources:
            fingerprint = source.text[:100]
            if fingerprint not in seen:
                seen.add(fingerprint)
                unique.append(source)
        
        return unique
    
    def _fallback_search(self, query: str) -> List[Tuple[Document, float]]:
        return self.retriever.retrieve_by_text(query, top_k=3, threshold=0.3)
    
    def _build_prompt(self, query: str, context: str) -> str:
        return f"""You are a helpful assistant. Answer the question based on the provided context.

Context:
{context}

Question: {query}

Answer:"""
    
    def _calculate_confidence(self, sources: List[Source], answer: str) -> float:
        if not sources:
            return 0.0
        
        source_conf = sum(s.similarity_score for s in sources) / len(sources)
        answer_conf = min(1.0, len(answer) / 500) if answer else 0.0
        
        return 0.7 * source_conf + 0.3 * answer_conf
    
    def start_conversation(self, conversation_id: str):
        self.conversation_manager.start_conversation(conversation_id)
    
    def get_conversation_history(self, conversation_id: str) -> Optional[List[Dict]]:
        return self.conversation_manager.get_history(conversation_id)
    
    def clear_conversation(self, conversation_id: str):
        self.conversation_manager.clear_conversation(conversation_id)
    
    def get_performance_metrics(self, by_type: bool = False) -> Dict[str, Any]:
        metrics = self.metrics.get_metrics(by_type)
        with self._lock:
            metrics.update(self._stats)
        return metrics
    
    def reset_metrics(self):
        self.metrics.reset()
        with self._lock:
            self._stats = {
                "total_queries": 0,
                "total_tokens": 0,
                "total_processing_time_ms": 0
            }
    
    def health_check(self, include_performance: bool = False) -> Dict[str, Any]:
        components = {
            "llm": True,
            "vector_store": True,
            "config": True
        }
        
        health_score = 1.0
        
        try:
            self.llm_client.generate("test")
        except:
            components["llm"] = False
            health_score -= 0.5
        
        try:
            self.vector_store.get_stats()
        except:
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
        self.cache.invalidate()


def get_rag_engine(embedding_service=None, llm_client=None, vector_store=None,
                   config: Optional[Union[RAGConfig, Dict]] = None) -> RAGEngine:
    """Get RAG engine instance."""
    return RAGEngine(config=config, llm_client=llm_client, vector_store=vector_store)