# cockatoo_v1/src/vector_store/__init__.py

"""
Vector Store package for Cockatoo V1.
Provides ChromaDB and FAISS vector database clients with search capabilities,
index management, and advanced search functionality.
"""

from .chroma_client import ChromaClient, get_chroma_client
from .faiss_client import FAISSClient, get_faiss_client
from .index_manager import IndexManager, get_index_manager
from .search_engine import SearchEngine, get_search_engine

__all__ = [
    "ChromaClient",
    "get_chroma_client",
    "FAISSClient",
    "get_faiss_client",
    "IndexManager",
    "get_index_manager",
    "SearchEngine",
    "get_search_engine",
]

__version__ = "1.0.0"