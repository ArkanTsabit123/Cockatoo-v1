# src/__init__.py

"""
cockatoo_v1 - Document AI Assistant
A comprehensive document processing and conversational AI system.
"""

__version__ = "1.0.0"
__author__ = "cockatoo_v1 Team"
__description__ = "Document AI Assistant for processing and querying documents"

# Import all modules for easier access
from . import ai_engine
from . import core
from . import database
from . import document_processing
from . import plugins
from . import storage
from . import ui
from . import utilities
from . import vector_store

__all__ = [
    'ai_engine',
    'core',
    'database',
    'document_processing',
    'plugins',
    'storage',
    'ui',
    'utilities',
    'vector_store',
    '__version__',
    '__author__',
    '__description__'
]