# cockatoo_v1/src/document_processing/extractors/__init__.py

"""
Extractors package for Cockatoo document processing.
Contains format-specific document extractors with registry system.
"""

import logging
from typing import Dict, List, Type, Optional, Any, Callable

from .base_extractor import BaseExtractor
from .pdf_extractor import PDFExtractor
from .txt_extractor import TXTExtractor
from .docx_extractor import DOCXExtractor
from .csv_extractor import CSVExtractor
from .epub_extractor import EPUBExtractor
from .html_extractor import HTMLExtractor
from .image_extractor import ImageExtractor
from .json_extractor import JSONExtractor
from .markdown_extractor import MarkdownExtractor
from .pptx_extractor import PPTXExtractor
from .xlsx_extractor import XLSXExtractor
from .web_extractor import WebExtractor


# ====== GLOBAL REGISTRIES ======
_EXTRACTOR_REGISTRY: Dict[str, Type[BaseExtractor]] = {}
_FACTORY_REGISTRY: Dict[str, Callable] = {}


# ====== SPECIALIZED FACTORY FUNCTIONS ======
# DEFINE THESE FIRST before they are used in _initialize_registries

def create_pdf_extractor(
    use_pdfplumber: bool = True, 
    use_pypdf2: bool = True
) -> PDFExtractor:
    """Create a PDF extractor with custom settings."""
    return PDFExtractor(use_pdfplumber=use_pdfplumber, use_pypdf2=use_pypdf2)


def create_image_extractor(
    use_ocr: bool = True, 
    ocr_languages: Optional[List[str]] = None
) -> ImageExtractor:
    """Create an image extractor with custom settings."""
    return ImageExtractor(use_ocr=use_ocr, ocr_languages=ocr_languages)


def create_web_extractor(
    timeout: int = 30, 
    max_retries: int = 3
) -> WebExtractor:
    """Create a web extractor with custom settings."""
    return WebExtractor(timeout=timeout, max_retries=max_retries)


# Simple factory functions for backward compatibility
def create_docx_extractor():
    """Create DOCX extractor instance."""
    return DOCXExtractor()


def create_txt_extractor():
    """Create TXT extractor instance."""
    return TXTExtractor()


def create_markdown_extractor():
    """Create Markdown extractor instance."""
    return MarkdownExtractor()


def create_html_extractor():
    """Create HTML extractor instance."""
    return HTMLExtractor()


def create_csv_extractor():
    """Create CSV extractor instance."""
    return CSVExtractor()


def create_epub_extractor():
    """Create EPUB extractor instance."""
    return EPUBExtractor()


def create_json_extractor():
    """Create JSON extractor instance."""
    return JSONExtractor()


def create_pptx_extractor():
    """Create PPTX extractor instance."""
    return PPTXExtractor()


def create_xlsx_extractor():
    """Create XLSX extractor instance."""
    return XLSXExtractor()


def _initialize_registries():
    """Initialize the registries with all extractors."""
    # Main extractor registry
    _EXTRACTOR_REGISTRY.update({
        # Document formats
        '.pdf': PDFExtractor,
        '.txt': TXTExtractor,
        '.docx': DOCXExtractor,
        '.csv': CSVExtractor,
        '.epub': EPUBExtractor,
        '.html': HTMLExtractor,
        '.json': JSONExtractor,
        '.md': MarkdownExtractor,
        '.pptx': PPTXExtractor,
        '.xlsx': XLSXExtractor,
        
        # Additional formats from extractor_map
        '.text': TXTExtractor,
        '.rtf': TXTExtractor,
        '.log': TXTExtractor,
        '.doc': DOCXExtractor,
        '.ppt': PPTXExtractor,
        '.xls': XLSXExtractor,
        '.xlsm': XLSXExtractor,
        '.xltx': XLSXExtractor,
        '.xltm': XLSXExtractor,
        '.xlt': XLSXExtractor,
        '.htm': HTMLExtractor,
        '.xhtml': HTMLExtractor,
        '.shtml': HTMLExtractor,
        '.php': HTMLExtractor,
        '.asp': HTMLExtractor,
        '.jsp': HTMLExtractor,
        '.epub3': EPUBExtractor,
        '.tsv': CSVExtractor,
        '.jsonld': JSONExtractor,
        '.geojson': JSONExtractor,
        '.topojson': JSONExtractor,
        '.jsonl': JSONExtractor,
        '.markdown': MarkdownExtractor,
        '.mdown': MarkdownExtractor,
        '.mkd': MarkdownExtractor,
        '.mkdn': MarkdownExtractor,
        '.mdwn': MarkdownExtractor,
        '.mdt': MarkdownExtractor,
        '.mdtext': MarkdownExtractor,
        '.rst': TXTExtractor,
        '.jpg': ImageExtractor,
        '.jpeg': ImageExtractor,
        '.png': ImageExtractor,
        '.gif': ImageExtractor,
        '.bmp': ImageExtractor,
        '.tiff': ImageExtractor,
        '.tif': ImageExtractor,
        '.webp': ImageExtractor,
        '.ico': ImageExtractor,
        '.svg': ImageExtractor,
        '.ini': TXTExtractor,
        '.cfg': TXTExtractor,
        '.conf': TXTExtractor,
        '.yaml': TXTExtractor,
        '.yml': TXTExtractor,
        '.xml': TXTExtractor,
    })
    
    # Factory function registry
    _FACTORY_REGISTRY.update({
        'pdf': create_pdf_extractor,
        'image': create_image_extractor,
        'web': create_web_extractor,
        'docx': create_docx_extractor,
        'txt': create_txt_extractor,
        'markdown': create_markdown_extractor,
        'html': create_html_extractor,
        'csv': create_csv_extractor,
        'epub': create_epub_extractor,
        'json': create_json_extractor,
        'pptx': create_pptx_extractor,
        'xlsx': create_xlsx_extractor,
    })


# Initialize registries
_initialize_registries()


# ====== REGISTRY MANAGEMENT FUNCTIONS ======
def register_extractor(
    extension: str, 
    extractor_class: Type[BaseExtractor], 
    factory_func: Optional[Callable] = None
) -> None:
    """
    Dynamically register a new extractor.
    
    Args:
        extension: File extension (with or without dot)
        extractor_class: Extractor class
        factory_func: Optional factory function
    
    Example:
        >>> register_extractor('.myformat', MyExtractor, create_my_extractor)
    """
    # Normalize extension
    if not extension.startswith('.'):
        extension = f'.{extension}'
    
    extension = extension.lower()
    _EXTRACTOR_REGISTRY[extension] = extractor_class
    
    # Register factory function if provided
    if factory_func:
        key = extension.lstrip('.')
        _FACTORY_REGISTRY[key] = factory_func


def unregister_extractor(extension: str) -> None:
    """
    Remove an extractor from the registry.
    
    Args:
        extension: File extension to remove
    """
    extension = extension.lower()
    if not extension.startswith('.'):
        extension = f'.{extension}'
    
    if extension in _EXTRACTOR_REGISTRY:
        del _EXTRACTOR_REGISTRY[extension]
    
    # Also remove from factory registry if present
    key = extension.lstrip('.')
    if key in _FACTORY_REGISTRY:
        del _FACTORY_REGISTRY[key]


def get_registered_extensions() -> List[str]:
    """
    Get all registered file extensions.
    
    Returns:
        List of registered file extensions (with dots)
    """
    return sorted(list(_EXTRACTOR_REGISTRY.keys()))


def get_registry_stats() -> Dict[str, Any]:
    """
    Get statistics about the extractor registry.
    
    Returns:
        Dictionary with registry statistics
    """
    return {
        'total_extractors': len(_EXTRACTOR_REGISTRY),
        'total_factories': len(_FACTORY_REGISTRY),
        'extensions': get_registered_extensions(),
        'factory_keys': list(_FACTORY_REGISTRY.keys()),
    }


# ====== EXTRACTOR FACTORY FUNCTIONS ======
def get_extractor(file_extension: str, url: str = None) -> BaseExtractor:
    """
    Get appropriate extractor for file extension or URL.
    
    Args:
        file_extension: File extension with dot (e.g., '.pdf')
        url: Optional URL string (takes precedence over file_extension)
        
    Returns:
        Appropriate extractor instance
        
    Raises:
        ValueError: If no extractor found
    """
    # First, check if this is a URL/web content
    if url or (file_extension and _is_url_like(file_extension)):
        return WebExtractor()
    
    file_extension = file_extension.lower()
    
    # Check registry first
    if file_extension in _EXTRACTOR_REGISTRY:
        extractor_class = _EXTRACTOR_REGISTRY[file_extension]
        
        # Try factory function first
        key = file_extension.lstrip('.')
        if key in _FACTORY_REGISTRY:
            try:
                return _FACTORY_REGISTRY[key]()
            except Exception as e:
                logging.warning(f"Factory function failed for {key}, falling back to direct instantiation: {e}")
                pass  # Fall back to direct instantiation
        
        # Direct instantiation
        return extractor_class()
    
    raise ValueError(
        f"No extractor found for file extension: {file_extension}. "
        f"Supported formats: {get_registered_extensions()}"
    )


def get_extractor_for_file(file_path: str) -> BaseExtractor:
    """
    Convenience function to get extractor for a file path.
    
    Args:
        file_path: Path to file
        
    Returns:
        Appropriate extractor instance
    """
    from pathlib import Path
    path = Path(file_path)
    
    # Check if it's a URL
    if _is_url_like(file_path):
        return WebExtractor()
    
    # Otherwise, use file extension
    return get_extractor(path.suffix)


def create_extractor(
    file_extension: str, 
    use_registry: bool = True,
    **kwargs
) -> Optional[BaseExtractor]:
    """
    Advanced factory function to create extractor with custom parameters.
    
    Args:
        file_extension: File extension
        use_registry: Whether to use registry or direct class
        **kwargs: Arguments for extractor constructor
        
    Returns:
        Extractor instance or None
    """
    file_extension = file_extension.lower()
    
    if use_registry and file_extension in _EXTRACTOR_REGISTRY:
        extractor_class = _EXTRACTOR_REGISTRY[file_extension]
        return extractor_class(**kwargs)
    
    return None


# ====== UTILITY FUNCTIONS ======
def _is_url_like(text: str) -> bool:
    """
    Check if text looks like a URL.
    
    Args:
        text: Text to check
        
    Returns:
        True if text appears to be a URL
    """
    if not text:
        return False
    
    text_lower = text.lower()
    return (text_lower.startswith('http://') or 
            text_lower.startswith('https://') or
            text_lower.startswith('www.') or
            '://' in text_lower)


def get_supported_extensions() -> Dict[str, List[str]]:
    """
    Get all supported file extensions grouped by category.
    
    Returns:
        Dictionary mapping categories to lists of extensions
    """
    return {
        "Documents": ['.pdf', '.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.xls'],
        "Text Files": ['.txt', '.text', '.rtf', '.log', '.ini', '.cfg', '.conf'],
        "Web Pages": ['.html', '.htm', '.xhtml', '.shtml'],
        "Ebooks": ['.epub'],
        "Data Files": ['.csv', '.tsv', '.json', '.jsonl'],
        "Markdown": ['.md', '.markdown', '.mdown', '.mkd'],
        "Images": ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'],
        "URLs": ['http://', 'https://', 'www.'],
    }


def validate_file_extension(file_extension: str) -> bool:
    """
    Validate if file extension is supported.
    
    Args:
        file_extension: File extension with dot
        
    Returns:
        True if extension is supported
    """
    try:
        get_extractor(file_extension)
        return True
    except ValueError:
        return False


# ====== MODULE EXPORTS ======
__all__ = [
    # Extractor classes
    'BaseExtractor',
    'PDFExtractor',
    'TXTExtractor',
    'DOCXExtractor',
    'CSVExtractor',
    'EPUBExtractor',
    'HTMLExtractor',
    'ImageExtractor',
    'JSONExtractor',
    'MarkdownExtractor',
    'PPTXExtractor',
    'XLSXExtractor',
    'WebExtractor',
    
    # Registry functions
    'register_extractor',
    'unregister_extractor',
    'get_registered_extensions',
    'get_registry_stats',
    
    # Factory functions
    'get_extractor',
    'get_extractor_for_file',
    'create_extractor',
    'create_pdf_extractor',
    'create_image_extractor',
    'create_web_extractor',
    
    # Utility functions
    'get_supported_extensions',
    'validate_file_extension',
]