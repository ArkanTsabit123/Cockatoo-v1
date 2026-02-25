# src/ai_engine/__init__.py

"""AI Engine package for intelligent document processing.

This package provides comprehensive AI capabilities including:
- Embedding generation and caching
- LLM integration with multiple providers
- Model management and downloading
- Prompt templating system
- RAG (Retrieval Augmented Generation) engine
- Document summarization
- Automatic tagging and keyword extraction
- Conversation memory management
"""

__version__ = "1.0.0"
__author__ = "AI Engine Team"

__all__ = [
    # Embedding Service
    "EmbeddingService",
    "EmbeddingCache",
    "EmbeddingModelInfo",
    "get_embedding_service",
    
    # LLM Clients
    "LLMClient",
    "OpenAIClient",
    "AnthropicClient",
    "OllamaClient",
    "LocalLLMClient",
    "LLMConfig",
    "LLMResponse",
    "LLMModel",
    "CircuitBreaker",
    "RateLimiter",
    "RequestPriority",
    "get_llm_client",
    
    # Model Management
    "ModelManager",
    "ModelRegistry",
    "ModelCache",
    "ModelInfo",
    "ModelType",
    "DownloadProgress",
    "ModelMetrics",
    "get_model_manager",
    
    # Prompt Management
    "PromptManager",
    "PromptTemplate",
    "get_prompt_manager",
    
    # RAG Engine
    "RAGEngine",
    "RAGConfig",
    "RAGResponse",
    "Document",
    "Source",
    "Retriever",
    "ContextBuilder",
    "ConversationManager",
    "get_rag_engine",
    
    # Summarization
    "Summarizer",
    "ChunkedSummarizer",
    "SummaryResponse",
    "SummaryStyle",
    "get_summarizer",
    
    # Tagging
    "Tagger",
    "TaxonomyManager",
    "KeywordExtractor",
    "Tag",
    "TaggingConfig",
    "get_tagger",
    
    # Conversation Memory
    "ConversationMemory",
    "ConversationEntry",
    "ConversationSummary",
    "ConversationConfig",
    "MemoryStore",
    "get_conversation_memory"
]


def __getattr__(name):
    """Lazy load modules when attributes are accessed."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    
    # Embedding Service
    if name in ["EmbeddingService", "EmbeddingCache", "EmbeddingModelInfo", "get_embedding_service"]:
        from .embedding_service import (
            EmbeddingService, 
            EmbeddingCache, 
            ModelInfo as EmbeddingModelInfo, 
            get_embedding_service
        )
        
        module_mapping = {
            "EmbeddingService": EmbeddingService,
            "EmbeddingCache": EmbeddingCache,
            "EmbeddingModelInfo": EmbeddingModelInfo,
            "get_embedding_service": get_embedding_service
        }
        return module_mapping[name]
    
    # LLM Clients
    if name in ["LLMClient", "OpenAIClient", "AnthropicClient", "OllamaClient", 
                "LocalLLMClient", "LLMConfig", "LLMResponse", "LLMModel", 
                "CircuitBreaker", "RateLimiter", "RequestPriority", "get_llm_client"]:
        from .llm_client import (
            LLMClient, OpenAIClient, AnthropicClient, OllamaClient,
            LocalLLMClient, LLMConfig, LLMResponse, LLMModel,
            CircuitBreaker, RateLimiter, RequestPriority, get_llm_client
        )
        
        module_mapping = {
            "LLMClient": LLMClient,
            "OpenAIClient": OpenAIClient,
            "AnthropicClient": AnthropicClient,
            "OllamaClient": OllamaClient,
            "LocalLLMClient": LocalLLMClient,
            "LLMConfig": LLMConfig,
            "LLMResponse": LLMResponse,
            "LLMModel": LLMModel,
            "CircuitBreaker": CircuitBreaker,
            "RateLimiter": RateLimiter,
            "RequestPriority": RequestPriority,
            "get_llm_client": get_llm_client
        }
        return module_mapping[name]
    
    # Model Management
    if name in ["ModelManager", "ModelRegistry", "ModelCache", "ModelInfo", 
                "ModelType", "DownloadProgress", "ModelMetrics", "get_model_manager"]:
        from .model_manager import (
            ModelManager, ModelRegistry, ModelCache, ModelInfo,
            ModelType, DownloadProgress, ModelMetrics, get_model_manager
        )
        
        module_mapping = {
            "ModelManager": ModelManager,
            "ModelRegistry": ModelRegistry,
            "ModelCache": ModelCache,
            "ModelInfo": ModelInfo,
            "ModelType": ModelType,
            "DownloadProgress": DownloadProgress,
            "ModelMetrics": ModelMetrics,
            "get_model_manager": get_model_manager
        }
        return module_mapping[name]
    
    # Prompt Management
    if name in ["PromptManager", "PromptTemplate", "get_prompt_manager"]:
        from .prompt_templates import PromptManager, PromptTemplate, get_prompt_manager
        
        module_mapping = {
            "PromptManager": PromptManager,
            "PromptTemplate": PromptTemplate,
            "get_prompt_manager": get_prompt_manager
        }
        return module_mapping[name]
    
    # RAG Engine
    if name in ["RAGEngine", "RAGConfig", "RAGResponse", "Document", "Source",
                "Retriever", "ContextBuilder", "ConversationManager", "get_rag_engine"]:
        from .rag_engine import (
            RAGEngine, RAGConfig, RAGResponse, Document, Source,
            Retriever, ContextBuilder, ConversationManager, get_rag_engine
        )
        
        module_mapping = {
            "RAGEngine": RAGEngine,
            "RAGConfig": RAGConfig,
            "RAGResponse": RAGResponse,
            "Document": Document,
            "Source": Source,
            "Retriever": Retriever,
            "ContextBuilder": ContextBuilder,
            "ConversationManager": ConversationManager,
            "get_rag_engine": get_rag_engine
        }
        return module_mapping[name]
    
    # Summarization
    if name in ["Summarizer", "ChunkedSummarizer", "SummaryResponse", 
                "SummaryStyle", "get_summarizer"]:
        from .summarizer import (
            Summarizer, ChunkedSummarizer, SummaryResponse,
            SummaryStyle, get_summarizer
        )
        
        module_mapping = {
            "Summarizer": Summarizer,
            "ChunkedSummarizer": ChunkedSummarizer,
            "SummaryResponse": SummaryResponse,
            "SummaryStyle": SummaryStyle,
            "get_summarizer": get_summarizer
        }
        return module_mapping[name]
    
    # Tagging
    if name in ["Tagger", "TaxonomyManager", "KeywordExtractor", 
                "Tag", "TaggingConfig", "get_tagger"]:
        from .tagging import (
            Tagger, TaxonomyManager, KeywordExtractor, Tag,
            TaggingConfig, get_tagger
        )
        
        module_mapping = {
            "Tagger": Tagger,
            "TaxonomyManager": TaxonomyManager,
            "KeywordExtractor": KeywordExtractor,
            "Tag": Tag,
            "TaggingConfig": TaggingConfig,
            "get_tagger": get_tagger
        }
        return module_mapping[name]
    
    # Conversation Memory
    if name in ["ConversationMemory", "ConversationEntry", "ConversationSummary",
                "ConversationConfig", "MemoryStore", "get_conversation_memory"]:
        from .conversation_memory import (
            ConversationMemory, ConversationEntry, ConversationSummary,
            ConversationConfig, MemoryStore, get_conversation_memory
        )
        
        module_mapping = {
            "ConversationMemory": ConversationMemory,
            "ConversationEntry": ConversationEntry,
            "ConversationSummary": ConversationSummary,
            "ConversationConfig": ConversationConfig,
            "MemoryStore": MemoryStore,
            "get_conversation_memory": get_conversation_memory
        }
        return module_mapping[name]