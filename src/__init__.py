# src/__init__.py

"""
cockatoo_v1 - Document AI Assistant

A comprehensive document processing and conversational AI system.
"""

__version__ = "1.0.0"
__author__ = "cockatoo_v1 Team"
__description__ = "Document AI Assistant for processing and querying documents"

# Import core modules for easier access
from . import database
from . import core
from . import utilities

__all__ = [
    'database',
    'core',
    'utilities',
    '__version__',
    '__author__',
    '__description__'
]