# cockatoo_v1/src/document_processing/__init__.py

"""
Document Processing Module for Cockatoo_V1.

This package provides comprehensive document processing functionality including:
- Format-specific text extraction
- Text cleaning and normalization
- Metadata extraction
- Intelligent text chunking
- Complete processing pipeline orchestration

Author: Cockatoo_V1 Development Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Cockatoo_V1 Development Team"
__license__ = "MIT"

# Core processing components
from .processor import (
    DocumentProcessor,
    ProcessingConfig,
    ProcessingResult,
    process_document,
    batch_process_documents,
    process_directory_contents
)

# Text cleaning utilities
from .cleaning import (
    TextCleaner,
    CleaningConfig,
    clean_text,
    batch_clean_text,
    create_custom_cleaner,
    get_presets,
    analyze_text_cleaning
)

# Metadata extraction
from .metadata import (
    MetadataExtractor,
    DocumentMetadata,
    extract_metadata,
    batch_extract_metadata,
    metadata_to_json,
    metadata_from_json
)

# Text chunking
from .chunking import (
    TextChunker,
    TextChunk,
    ChunkingConfig,
    ChunkingStrategy,
    chunk_text,
    analyze_text_for_chunking,
    get_optimal_chunk_size
)

# Format-specific extractors - UPDATED BASED ON EXTRACTORS/__INIT__.PY
try:
    from .extractors import (
        BaseExtractor,
        PDFExtractor,
        DOCXExtractor,
        TXTExtractor,
        EPUBExtractor,
        JSONExtractor,
        HTMLExtractor,
        ImageExtractor,
        CSVExtractor,
        MarkdownExtractor,
        PPTXExtractor,
        XLSXExtractor,
        WebExtractor,
        get_extractor_for_file,
        get_supported_extensions,  # Changed from get_supported_formats
        validate_file_extension    # New function
    )
    HAS_EXTRACTORS = True
except ImportError:
    HAS_EXTRACTORS = False

# Utilities
from .utilities import (
    validate_file_format,
    detect_file_type,
    calculate_file_stats,
    normalize_file_path
)

# Exceptions
from .exceptions import (
    DocumentProcessingError,
    UnsupportedFormatError,
    ExtractionError,
    CleaningError,
    ChunkingError,
    MetadataError
)

# Define what gets imported with "from document_processing import *"
__all__ = [
    # Core processing
    'DocumentProcessor',
    'ProcessingConfig',
    'ProcessingResult',
    'process_document',
    'batch_process_documents',
    'process_directory_contents',
    
    # Text cleaning
    'TextCleaner',
    'CleaningConfig',
    'clean_text',
    'batch_clean_text',
    'create_custom_cleaner',
    'get_presets',
    'analyze_text_cleaning',
    
    # Metadata extraction
    'MetadataExtractor',
    'DocumentMetadata',
    'extract_metadata',
    'batch_extract_metadata',
    'metadata_to_json',
    'metadata_from_json',
    
    # Text chunking
    'TextChunker',
    'TextChunk',
    'ChunkingConfig',
    'ChunkingStrategy',
    'chunk_text',
    'analyze_text_for_chunking',
    'get_optimal_chunk_size',
    
    # Extractors (if available) - UPDATED
    'BaseExtractor',
    'PDFExtractor',
    'DOCXExtractor',
    'TXTExtractor',
    'EPUBExtractor',
    'JSONExtractor',       # Replaced MDExtractor
    'HTMLExtractor',
    'ImageExtractor',
    'CSVExtractor',
    'MarkdownExtractor',   # New
    'PPTXExtractor',       # New
    'XLSXExtractor',       # New
    'WebExtractor',
    'get_extractor_for_file',
    'get_supported_extensions',  # Changed
    'validate_file_extension',   # New
    
    # Utilities
    'validate_file_format',
    'detect_file_type',
    'calculate_file_stats',
    'normalize_file_path',
    
    # Exceptions
    'DocumentProcessingError',
    'UnsupportedFormatError',
    'ExtractionError',
    'CleaningError',
    'ChunkingError',
    'MetadataError',
    
    # Constants
    'HAS_EXTRACTORS',
]

# Module initialization
def init_module() -> bool:
    """
    Initialize the document processing module.
    
    This function can be called to perform any necessary setup,
    such as verifying dependencies or setting up default configurations.
    
    Returns:
        True if initialization successful, False otherwise.
    """
    import logging
    
    logger = logging.getLogger("document_processing")
    logger.info(f"Document Processing Module v{__version__} initialized")
    
    # Check for optional dependencies
    if not HAS_EXTRACTORS:
        logger.warning("Extractor submodule not available. Some format support may be limited.")
    
    # Verify core dependencies
    try:
        import tiktoken
        logger.debug("tiktoken available for token counting")
    except ImportError:
        logger.warning("tiktoken not installed. Token counting will use approximations.")
    
    try:
        import pdfplumber
        logger.debug("pdfplumber available for PDF processing")
    except ImportError:
        logger.warning("pdfplumber not installed. PDF support will be limited.")
    
    try:
        from docx import Document
        logger.debug("python-docx available for DOCX processing")
    except ImportError:
        logger.warning("python-docx not installed. DOCX support will be limited.")
    
    return True

# Constants - UPDATED WITH EXTRACTORS INFO
SUPPORTED_FORMATS = {
    'pdf': ['.pdf'],
    'docx': ['.docx', '.doc'],
    'txt': ['.txt', '.text', '.md', '.markdown', '.rst', '.rtf', '.log', '.ini', '.cfg', '.conf', '.yaml', '.yml', '.xml'],
    'epub': ['.epub', '.epub3'],
    'html': ['.html', '.htm', '.xhtml', '.shtml', '.php', '.asp', '.jsp'],
    'csv': ['.csv', '.tsv'],
    'image': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp', '.ico', '.svg'],
    'json': ['.json', '.jsonld', '.geojson', '.topojson', '.jsonl'],
    'markdown': ['.md', '.markdown', '.mdown', '.mkd', '.mkdn', '.mdwn', '.mdt', '.mdtext'],
    'pptx': ['.pptx', '.ppt'],
    'xlsx': ['.xlsx', '.xls', '.xlsm', '.xltx', '.xltm', '.xlt'],
    'web': ['http://', 'https://', 'ftp://', 'sftp://']
}

# Default configurations
DEFAULT_CLEANING_CONFIG = {
    'normalize_whitespace': True,
    'fix_encoding': True,
    'remove_control_chars': True,
    'clean_html': True,
    'remove_excessive_newlines': True,
    'normalize_quotes': True,
    'fix_hyphenation': True,
}

DEFAULT_CHUNKING_CONFIG = {
    'chunk_size': 500,
    'chunk_overlap': 50,
    'strategy': 'semantic',
    'separator': '\n\n',
}

DEFAULT_PROCESSING_CONFIG = {
    'chunk_size': 500,
    'chunk_overlap': 50,
    'max_file_size_mb': 100,
    'enable_cleaning': True,
    'enable_chunking': True,
    'enable_metadata_extraction': True,
}

# Package metadata
PACKAGE_INFO = {
    'name': 'document_processing',
    'version': __version__,
    'author': __author__,
    'description': 'Comprehensive document processing for Cockatoo_V1',
    'supported_formats': SUPPORTED_FORMATS,
}

# Helper functions
def get_package_info() -> dict:
    """
    Get package information and configuration.
    
    Returns:
        Dictionary with package metadata and configurations.
    """
    info = {
        **PACKAGE_INFO,
        'default_configs': {
            'cleaning': DEFAULT_CLEANING_CONFIG,
            'chunking': DEFAULT_CHUNKING_CONFIG,
            'processing': DEFAULT_PROCESSING_CONFIG,
        },
        'has_extractors': HAS_EXTRACTORS,
    }
    
    # Add extractor info if available
    if HAS_EXTRACTORS:
        from .extractors import get_supported_extensions, get_registry_stats
        try:
            info['extractor_formats'] = get_supported_extensions()
            info['extractor_stats'] = get_registry_stats()
        except Exception:
            pass
    
    return info

def list_supported_formats() -> dict:
    """
    Get all supported file formats.
    
    Returns:
        Dictionary mapping format names to list of extensions.
    """
    # Try to get from extractors module first
    if HAS_EXTRACTORS:
        try:
            from .extractors import get_supported_extensions
            return get_supported_extensions()
        except ImportError:
            pass
    
    # Fallback to local mapping
    return SUPPORTED_FORMATS.copy()

def check_dependencies() -> dict:
    """
    Check for required and optional dependencies.
    
    Returns:
        Dictionary with dependency status.
    """
    dependencies = {
        'required': {},
        'optional': {},
        'missing': [],
    }
    
    # Required dependencies (for core functionality)
    required_packages = [
        ('re', 'regex', 'built-in'),
        ('json', 'json', 'built-in'),
        ('dataclasses', 'dataclasses', 'built-in'),
        ('typing', 'typing', 'built-in'),
        ('logging', 'logging', 'built-in'),
    ]
    
    for import_name, package_name, status in required_packages:
        try:
            __import__(import_name)
            dependencies['required'][package_name] = {'status': 'installed', 'version': 'built-in' if status == 'built-in' else 'unknown'}
        except ImportError:
            dependencies['required'][package_name] = {'status': 'missing'}
            dependencies['missing'].append(package_name)
    
    # Optional dependencies for extractors
    optional_packages = [
        ('tiktoken', 'tiktoken', 'Token counting'),
        ('pdfplumber', 'pdfplumber', 'PDF processing'),
        ('docx', 'python-docx', 'DOCX processing'),
        ('ebooklib', 'ebooklib', 'EPUB processing'),
        ('yaml', 'pyyaml', 'YAML parsing'),
        ('frontmatter', 'python-frontmatter', 'Markdown frontmatter'),
        ('PIL', 'pillow', 'Image processing'),
        ('pytesseract', 'pytesseract', 'OCR'),
        ('pandas', 'pandas', 'CSV/Excel processing'),
        ('bs4', 'beautifulsoup4', 'HTML parsing'),
        ('openpyxl', 'openpyxl', 'Excel processing'),
        ('pptx', 'python-pptx', 'PowerPoint processing'),
        ('pdf2image', 'pdf2image', 'PDF to image conversion'),
        ('requests', 'requests', 'Web content fetching'),
        ('lxml', 'lxml', 'XML/HTML parsing'),
    ]
    
    for import_name, package_name, purpose in optional_packages:
        try:
            __import__(import_name)
            dependencies['optional'][package_name] = {
                'status': 'installed',
                'purpose': purpose,
            }
        except ImportError:
            dependencies['optional'][package_name] = {
                'status': 'missing',
                'purpose': purpose,
            }
    
    return dependencies

# Initialize module on import
init_module()

# Clean up namespace
del init_module