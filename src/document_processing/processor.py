# cockatoo_v1/src/document_processing/processor.py

"""
processor.py
Main Document Processing Pipeline for Cockatoo_V1.

This module orchestrates the complete document processing workflow including:
1. Document ingestion and validation
2. Format-specific text extraction
3. Text cleaning and normalization
4. Intelligent chunking
5. Embedding generation
6. Storage in vector database

Author: Cockatoo_V1 Development Team
Version: 1.0.0
"""

import os
import sys
import re
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union, Generator
from dataclasses import dataclass, asdict, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# Setup logger
logger = logging.getLogger(__name__)

# Import local modules
from .cleaning import TextCleaner, CleaningConfig
from .metadata import MetadataExtractor, DocumentMetadata
from .chunking import TextChunker, TextChunk, ChunkingConfig, ChunkingStrategy

# Updated based on extractors/__init__.py
try:
    from .extractors import (
        PDFExtractor, DOCXExtractor, TXTExtractor, EPUBExtractor,
        JSONExtractor, HTMLExtractor, ImageExtractor, CSVExtractor,
        MarkdownExtractor, PPTXExtractor, XLSXExtractor, WebExtractor,
        get_extractor_for_file, get_supported_extensions, validate_file_extension
    )
    HAS_EXTRACTORS = True
except ImportError as e:
    HAS_EXTRACTORS = False
    logger.warning(f"Extractor modules not available: {e}. Some functionality will be limited.")


@dataclass
class ProcessingConfig:
    """
    Configuration for document processing pipeline.
    """
    # Processing parameters
    chunk_size: int = 500  # tokens
    chunk_overlap: int = 50  # tokens
    max_file_size_mb: int = 100  # Maximum file size to process
    max_workers: int = 4  # Maximum concurrent workers
    
    # Processing options
    enable_cleaning: bool = True
    enable_chunking: bool = True
    enable_metadata_extraction: bool = True
    enable_embedding_generation: bool = True
    enable_vector_storage: bool = True
    
    # Text processing
    cleaning_config: Optional[Dict] = None
    chunking_config: Optional[Dict] = None
    
    # Output options
    save_intermediate: bool = False
    intermediate_dir: Optional[str] = None
    verbose: bool = False
    
    def __post_init__(self):
        """Initialize default configurations."""
        if self.cleaning_config is None:
            self.cleaning_config = {
                'normalize_whitespace': True,
                'fix_encoding': True,
                'remove_control_chars': True,
                'clean_html': True,
                'remove_excessive_newlines': True,
            }
        
        if self.chunking_config is None:
            self.chunking_config = {
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'separator': '\n\n',
                'preserve_formatting': False,
            }
        
        if self.intermediate_dir is None:
            self.intermediate_dir = 'data/processed'


@dataclass
class ProcessingResult:
    """
    Results from document processing pipeline.
    """
    # Document identification
    document_id: str
    file_path: str
    file_type: str
    
    # Processing status
    status: str  # 'pending', 'processing', 'completed', 'failed'
    error_message: Optional[str] = None
    
    # Processing metrics
    processing_time: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Processing results
    extracted_text: Optional[str] = None
    cleaned_text: Optional[str] = None
    chunks: List[Dict] = field(default_factory=list)
    embeddings: List[List[float]] = field(default_factory=list)
    vector_ids: List[str] = field(default_factory=list)
    
    # Metadata
    metadata: Optional[Dict] = None
    chunk_count: int = 0
    word_count: int = 0
    token_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        
        # Convert datetime objects to ISO strings
        for key in ['start_time', 'end_time']:
            if result.get(key):
                result[key] = result[key].isoformat()
        
        return result
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate processing metrics."""
        return {
            'processing_time': self.processing_time,
            'chunk_count': self.chunk_count,
            'word_count': self.word_count,
            'token_count': self.token_count,
            'average_chunk_size': self.word_count / self.chunk_count if self.chunk_count > 0 else 0,
            'success_rate': 1.0 if self.status == 'completed' else 0.0,
        }


class DocumentProcessor:
    """
    Main document processing pipeline orchestrator.
    
    Handles complete document processing workflow from ingestion to storage.
    """
    
    # Updated based on extractors/__init__.py
    SUPPORTED_FORMATS = {
        'pdf': ['.pdf'],
        'docx': ['.docx', '.doc'],
        'txt': ['.txt', '.text', '.rtf', '.log', '.ini', '.cfg', '.conf', '.yaml', '.yml', '.xml'],
        'epub': ['.epub', '.epub3'],
        'html': ['.html', '.htm', '.xhtml', '.shtml', '.php', '.asp', '.jsp'],
        'csv': ['.csv', '.tsv'],
        'image': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.ico', '.svg'],
        'json': ['.json', '.jsonld', '.geojson', '.topojson', '.jsonl'],
        'markdown': ['.md', '.markdown', '.mdown', '.mkd', '.mkdn', '.mdwn', '.mdt', '.mdtext'],
        'pptx': ['.pptx', '.ppt'],
        'xlsx': ['.xlsx', '.xls', '.xlsm', '.xltx', '.xltm', '.xlt'],
        'web': ['http://', 'https://', 'www.']
    }
    
    # Format to extractor mapping
    EXTRACTOR_MAPPING = {
        'pdf': 'PDFExtractor',
        'docx': 'DOCXExtractor',
        'txt': 'TXTExtractor',
        'epub': 'EPUBExtractor',
        'html': 'HTMLExtractor',
        'csv': 'CSVExtractor',
        'image': 'ImageExtractor',
        'json': 'JSONExtractor',
        'markdown': 'MarkdownExtractor',
        'pptx': 'PPTXExtractor',
        'xlsx': 'XLSXExtractor',
        'web': 'WebExtractor',
    }
    
    def __init__(self, config: Optional[Union[ProcessingConfig, Dict]] = None):
        """
        Initialize document processor with configuration.
        
        Args:
            config: Processing configuration. Can be ProcessingConfig object or dict.
                   Uses defaults if None.
        """
        # Handle if config is a dict (from tests)
        if config is not None and isinstance(config, dict):
            self.config = ProcessingConfig(**config)
        elif config is None:
            self.config = ProcessingConfig()
        else:
            self.config = config
        
        # Initialize components
        self._initialize_components()
        
        # Create intermediate directory if needed
        if self.config.save_intermediate:
            Path(self.config.intermediate_dir).mkdir(parents=True, exist_ok=True)
        
        # Statistics tracking
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'total_processing_time': 0.0,
            'format_stats': {},
        }
        
        logger.info(f"DocumentProcessor initialized with config: {self.config}")
    
    def _initialize_components(self):
        """Initialize processing components."""
        # Initialize text cleaner
        if self.config.enable_cleaning:
            cleaning_config = CleaningConfig(**self.config.cleaning_config)
            self.text_cleaner = TextCleaner(config=cleaning_config)
        else:
            self.text_cleaner = None
            logger.warning("Text cleaning disabled")
        
        # Initialize text chunker
        if self.config.enable_chunking:
            chunking_config = ChunkingConfig(**self.config.chunking_config)
            self.text_chunker = TextChunker(config=chunking_config)
        else:
            self.text_chunker = None
            logger.warning("Text chunking disabled")
        
        # Initialize metadata extractor
        if self.config.enable_metadata_extraction:
            self.metadata_extractor = MetadataExtractor()
        else:
            self.metadata_extractor = None
            logger.warning("Metadata extraction disabled")
    
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """
        Get all supported file formats and extensions.
        
        Returns:
            Dictionary mapping format names to list of extensions.
        """
        return self.SUPPORTED_FORMATS
    
    def is_supported_format(self, file_path: Union[str, Path]) -> bool:
        """
        Check if a file format is supported.
        
        Args:
            file_path: Path to the file.
            
        Returns:
            True if format is supported, False otherwise.
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        # Check for web URLs
        if self._is_url_like(str(file_path)):
            return True
        
        for extensions in self.SUPPORTED_FORMATS.values():
            if extension in extensions:
                return True
        
        return False
    
    def get_format_type(self, file_path: Union[str, Path]) -> Optional[str]:
        """
        Get the format type of a file.
        
        Args:
            file_path: Path to the file.
            
        Returns:
            Format type (e.g., 'pdf', 'docx') or None if not supported.
        """
        file_path = Path(file_path)
        file_str = str(file_path)
        
        # Check if it's a URL/web content
        if self._is_url_like(file_str):
            return 'web'
        
        extension = file_path.suffix.lower()
        
        for format_type, extensions in self.SUPPORTED_FORMATS.items():
            if extension in extensions:
                return format_type
        
        return None
    
    def _is_url_like(self, text: str) -> bool:
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
    
    def validate_file(self, file_path: Union[str, Path]) -> Tuple[bool, Optional[str]]:
        """
        Validate a file for processing.
        
        Args:
            file_path: Path to the file.
            
        Returns:
            Tuple of (is_valid, error_message).
        """
        file_path = Path(file_path)
        
        # Check if file exists (skip for URLs)
        if not self._is_url_like(str(file_path)) and not file_path.exists():
            return False, f"File does not exist: {file_path}"
        
        # Check if file is readable (for local files only)
        if not self._is_url_like(str(file_path)) and not os.access(file_path, os.R_OK):
            return False, f"File is not readable: {file_path}"
        
        # Check file size for local files
        if not self._is_url_like(str(file_path)):
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            if file_size_mb > self.config.max_file_size_mb:
                return False, f"File too large: {file_size_mb:.1f}MB > {self.config.max_file_size_mb}MB limit"
        
        # Check if format is supported
        if not self.is_supported_format(file_path):
            return False, f"Unsupported file format: {file_path.suffix or 'URL'}"
        
        return True, None
    
    def generate_document_id(self, file_path: Union[str, Path]) -> str:
        """
        Generate a unique document ID.
        
        Args:
            file_path: Path to the file.
            
        Returns:
            Unique document ID.
        """
        file_path = Path(file_path)
        
        # For URLs, use the URL itself
        if self._is_url_like(str(file_path)):
            unique_string = str(file_path)
        else:
            # Use hash of file path and modification time for uniqueness
            unique_string = f"{file_path.absolute()}:{file_path.stat().st_mtime}"
        
        return hashlib.sha256(unique_string.encode()).hexdigest()[:16]
    
    def process_document(self, file_path: Union[str, Path]) -> ProcessingResult:
        """
        Process a single document through the complete pipeline.
        
        Args:
            file_path: Path to the document file.
            
        Returns:
            ProcessingResult with all processing details.
        """
        file_path = Path(file_path)
        start_time = datetime.now()
        
        # Generate document ID
        doc_id = self.generate_document_id(file_path)
        
        # Initialize result object
        result = ProcessingResult(
            document_id=doc_id,
            file_path=str(file_path.absolute()),
            file_type=file_path.suffix.lower() if not self._is_url_like(str(file_path)) else 'web',
            status='processing',
            start_time=start_time,
        )
        
        logger.info(f"Processing document: {file_path.name if not self._is_url_like(str(file_path)) else file_path} (ID: {doc_id})")
        
        try:
            # Step 1: Validate file
            is_valid, error = self.validate_file(file_path)
            if not is_valid:
                result.status = 'failed'
                result.error_message = error
                logger.error(f"Validation failed for {file_path}: {error}")
                return result
            
            # Step 2: Extract metadata
            if self.config.enable_metadata_extraction and self.metadata_extractor:
                try:
                    metadata = self.metadata_extractor.extract_metadata(file_path)
                    result.metadata = metadata.to_dict()
                    
                    # Extract word count from metadata if available
                    if metadata.word_count > 0:
                        result.word_count = metadata.word_count
                    
                    logger.debug(f"Metadata extracted: {metadata.title if metadata.title else 'No title'}")
                except Exception as e:
                    logger.warning(f"Metadata extraction failed for {file_path}: {e}")
            
            # Step 3: Extract text content
            extracted_text = self._extract_text(file_path, result)
            if not extracted_text:
                result.status = 'failed'
                result.error_message = "Failed to extract text content"
                logger.error(f"Text extraction failed for {file_path}")
                return result
            
            result.extracted_text = extracted_text
            
            # Update word count if not already set
            if result.word_count == 0:
                result.word_count = len(re.findall(r'\b\w+\b', extracted_text))
            
            # Step 4: Clean text
            if self.config.enable_cleaning and self.text_cleaner:
                try:
                    cleaned_text = self.text_cleaner.clean_text(extracted_text, verbose=self.config.verbose)
                    result.cleaned_text = cleaned_text
                    
                    if self.config.verbose:
                        logger.info(f"Text cleaning complete. Original: {len(extracted_text)} chars, Cleaned: {len(cleaned_text)} chars")
                except Exception as e:
                    logger.warning(f"Text cleaning failed for {file_path}: {e}")
                    # Continue with original text
                    result.cleaned_text = extracted_text
            else:
                result.cleaned_text = extracted_text
            
            # Step 5: Chunk text
            if self.config.enable_chunking and self.text_chunker:
                text_to_chunk = result.cleaned_text or result.extracted_text
                
                try:
                    chunks = self.text_chunker.chunk_text(text_to_chunk)
                    result.chunks = chunks
                    result.chunk_count = len(chunks)
                    
                    # Calculate token count
                    result.token_count = sum(chunk.get('token_count', 0) for chunk in chunks) if chunks else 0
                    
                    logger.info(f"Text chunking complete. Created {result.chunk_count} chunks")
                except Exception as e:
                    logger.error(f"Text chunking failed for {file_path}: {e}")
                    result.status = 'failed'
                    result.error_message = f"Text chunking failed: {str(e)}"
                    return result
            else:
                # If chunking is disabled, treat entire text as one chunk
                text = result.cleaned_text or result.extracted_text
                result.chunks = [{
                    'text': text,
                    'chunk_index': 0,
                    'start_char': 0,
                    'end_char': len(text),
                    'token_count': len(re.findall(r'\b\w+\b', text)),
                }]
                result.chunk_count = 1
                result.token_count = result.word_count
            
            # Step 6: Generate embeddings (if enabled)
            if self.config.enable_embedding_generation:
                embeddings = self._generate_embeddings(result.chunks)
                result.embeddings = embeddings
                
                logger.info(f"Generated {len(embeddings)} embeddings")
            
            # Step 7: Store in vector database (if enabled)
            if self.config.enable_vector_storage:
                vector_ids = self._store_in_vector_db(result.chunks, result.embeddings, result.metadata)
                result.vector_ids = vector_ids
                
                logger.info(f"Stored {len(vector_ids)} chunks in vector database")
            
            # Step 8: Save intermediate results (if enabled)
            if self.config.save_intermediate:
                self._save_intermediate_results(result)
            
            # Update result status
            result.status = 'completed'
            result.end_time = datetime.now()
            result.processing_time = (result.end_time - start_time).total_seconds()
            
            # Update statistics
            self._update_statistics(result, success=True)
            
            logger.info(f"Document processing completed: {file_path.name if not self._is_url_like(str(file_path)) else file_path} "
                       f"(Time: {result.processing_time:.2f}s, "
                       f"Chunks: {result.chunk_count}, "
                       f"Words: {result.word_count})")
            
        except Exception as e:
            # Handle any unexpected errors
            result.status = 'failed'
            result.error_message = str(e)
            result.end_time = datetime.now()
            result.processing_time = (result.end_time - start_time).total_seconds()
            
            self._update_statistics(result, success=False)
            
            logger.error(f"Document processing failed for {file_path}: {e}", exc_info=True)
        
        return result
    
    def _extract_text(self, file_path: Path, result: ProcessingResult) -> Optional[str]:
        """
        Extract text from file using appropriate extractor.
        
        Args:
            file_path: Path to the file.
            result: ProcessingResult for logging.
            
        Returns:
            Extracted text or None if extraction failed.
        """
        # Check if it's a URL
        if self._is_url_like(str(file_path)):
            try:
                if HAS_EXTRACTORS:
                    extractor = WebExtractor()
                    text = extractor.extract(str(file_path))
                    if text and text.strip():
                        logger.info(f"Extracted text from URL: {len(text)} chars")
                        return text
            except Exception as e:
                logger.warning(f"Web extraction failed for {file_path}: {e}")
            
            # Fallback for URLs
            return f"URL content from: {file_path}"
        
        format_type = self.get_format_type(file_path)
        
        if not format_type:
            logger.error(f"Unsupported format for {file_path}")
            return None
        
        # Try to use extractor from extractors module
        if HAS_EXTRACTORS:
            try:
                extractor = get_extractor_for_file(str(file_path))
                text = extractor.extract(file_path)
                if text and text.strip():
                    logger.info(f"Extracted text using {format_type} extractor: {len(text)} chars")
                    return text
            except Exception as e:
                logger.warning(f"Extractor failed for {file_path}: {e}")
        
        # Fallback: Use simple file reading for text-based formats
        try:
            encoding = self._detect_encoding(file_path)
            
            with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                content = f.read()
                
            if content.strip():
                logger.info(f"Extracted text using fallback method: {len(content)} chars")
                return content
            else:
                logger.warning(f"Empty content for {file_path}")
                return None
                
        except Exception as e:
            logger.error(f"Fallback extraction failed for {file_path}: {e}")
            return None
    
    def _detect_encoding(self, file_path: Path) -> str:
        """
        Detect file encoding.
        
        Args:
            file_path: Path to the file.
            
        Returns:
            Detected encoding.
        """
        # Try common encodings
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    f.read(1024)  # Read a sample
                return encoding
            except UnicodeDecodeError:
                continue
        
        # Default to utf-8 with errors ignored
        return 'utf-8'
    
    def _generate_embeddings(self, chunks: List[Dict]) -> List[List[float]]:
        """
        Generate embeddings for text chunks.
        
        Args:
            chunks: List of text chunks.
            
        Returns:
            List of embeddings.
        """
        if not chunks:
            return []
        
        # This is a placeholder. In a real implementation, this would use
        # sentence-transformers, OpenAI embeddings, or similar.
        
        embeddings = []
        
        for chunk in chunks:
            text = chunk.get('text', '')
            
            # Placeholder: Return dummy embeddings
            # In production, replace with actual embedding generation
            if text:
                # Simple hash-based dummy embedding for demonstration
                import struct
                
                # Create a deterministic "embedding" based on text hash
                text_hash = hashlib.md5(text.encode()).digest()
                # Convert first 16 bytes to 8 floats
                floats = []
                for i in range(0, 16, 2):
                    byte_pair = text_hash[i:i+2]
                    if len(byte_pair) == 2:
                        # Convert to float between 0 and 1
                        value = struct.unpack('H', byte_pair)[0] / 65535.0
                        floats.append(value)
                
                # Pad or truncate to 384 dimensions (common embedding size)
                if len(floats) < 384:
                    # Repeat pattern to fill
                    while len(floats) < 384:
                        floats.extend(floats[:min(len(floats), 384 - len(floats))])
                else:
                    floats = floats[:384]
                
                embeddings.append(floats)
            else:
                embeddings.append([0.0] * 384)
        
        logger.info(f"Generated {len(embeddings)} dummy embeddings (placeholder)")
        return embeddings
    
    def _store_in_vector_db(self, chunks: List[Dict], embeddings: List[List[float]], 
                           metadata: Optional[Dict]) -> List[str]:
        """
        Store chunks and embeddings in vector database.
        
        Args:
            chunks: List of text chunks.
            embeddings: List of embeddings.
            metadata: Document metadata.
            
        Returns:
            List of vector IDs.
        """
        if not chunks:
            return []
        
        # This is a placeholder. In a real implementation, this would store
        # in ChromaDB, FAISS, Pinecone, etc.
        
        vector_ids = []
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Generate unique ID for this chunk
            chunk_id = f"{metadata.get('document_id', 'doc')}_{i}" if metadata else f"chunk_{i}"
            vector_ids.append(chunk_id)
            
            # In production, this would actually store in vector DB
            # Example for ChromaDB:
            # collection.add(
            #     documents=[chunk['text']],
            #     embeddings=[embedding],
            #     metadatas=[{**chunk, **metadata}],
            #     ids=[chunk_id]
            # )
        
        logger.info(f"Stored {len(vector_ids)} chunks in vector database (placeholder)")
        return vector_ids
    
    def _save_intermediate_results(self, result: ProcessingResult):
        """
        Save intermediate processing results.
        
        Args:
            result: Processing result to save.
        """
        try:
            intermediate_dir = Path(self.config.intermediate_dir)
            
            # Save metadata
            metadata_file = intermediate_dir / f"{result.document_id}_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(result.metadata or {}, f, indent=2, ensure_ascii=False)
            
            # Save extracted text
            if result.extracted_text:
                text_file = intermediate_dir / f"{result.document_id}_extracted.txt"
                with open(text_file, 'w', encoding='utf-8') as f:
                    f.write(result.extracted_text)
            
            # Save cleaned text
            if result.cleaned_text:
                cleaned_file = intermediate_dir / f"{result.document_id}_cleaned.txt"
                with open(cleaned_file, 'w', encoding='utf-8') as f:
                    f.write(result.cleaned_text)
            
            # Save chunks
            if result.chunks:
                chunks_file = intermediate_dir / f"{result.document_id}_chunks.json"
                with open(chunks_file, 'w', encoding='utf-8') as f:
                    json.dump(result.chunks, f, indent=2, ensure_ascii=False)
            
            # Save result summary
            summary_file = intermediate_dir / f"{result.document_id}_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Saved intermediate results for {result.document_id}")
            
        except Exception as e:
            logger.warning(f"Failed to save intermediate results: {e}")
    
    def _update_statistics(self, result: ProcessingResult, success: bool):
        """
        Update processing statistics.
        
        Args:
            result: Processing result.
            success: Whether processing was successful.
        """
        self.stats['total_processed'] += 1
        
        if success:
            self.stats['successful'] += 1
            self.stats['total_processing_time'] += result.processing_time
        else:
            self.stats['failed'] += 1
        
        # Update format statistics
        file_type = result.file_type
        if file_type not in self.stats['format_stats']:
            self.stats['format_stats'][file_type] = {
                'total': 0,
                'successful': 0,
                'failed': 0,
                'total_time': 0.0,
            }
        
        self.stats['format_stats'][file_type]['total'] += 1
        if success:
            self.stats['format_stats'][file_type]['successful'] += 1
            self.stats['format_stats'][file_type]['total_time'] += result.processing_time
        else:
            self.stats['format_stats'][file_type]['failed'] += 1
    
    def batch_process(self, file_paths: List[Union[str, Path]], 
                     max_workers: Optional[int] = None) -> List[ProcessingResult]:
        """
        Process multiple documents concurrently.
        
        Args:
            file_paths: List of file paths to process.
            max_workers: Maximum number of concurrent workers. Uses config value if None.
            
        Returns:
            List of ProcessingResult objects.
        """
        if not file_paths:
            return []
        
        max_workers = max_workers or self.config.max_workers
        results = []
        
        logger.info(f"Starting batch processing of {len(file_paths)} documents with {max_workers} workers")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self.process_document, file_path): file_path
                for file_path in file_paths
            }
            
            # Process results as they complete
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                
                try:
                    result = future.result(timeout=300)  # 5 minute timeout
                    results.append(result)
                    
                    logger.info(f"Completed: {Path(file_path).name if not self._is_url_like(str(file_path)) else file_path} - {result.status}")
                    
                except Exception as e:
                    # Create a failed result for this document
                    failed_result = ProcessingResult(
                        document_id=self.generate_document_id(file_path),
                        file_path=str(Path(file_path).absolute()),
                        file_type=Path(file_path).suffix.lower() if not self._is_url_like(str(file_path)) else 'web',
                        status='failed',
                        error_message=f"Processing failed: {str(e)}",
                    )
                    results.append(failed_result)
                    
                    self._update_statistics(failed_result, success=False)
                    logger.error(f"Processing failed for {file_path}: {e}")
        
        # Log batch summary
        successful = sum(1 for r in results if r.status == 'completed')
        failed = len(results) - successful
        
        logger.info(f"Batch processing completed: {successful} successful, {failed} failed")
        
        return results
    
    def process_directory(self, directory_path: Union[str, Path], 
                         recursive: bool = True,
                         pattern: Optional[str] = None) -> List[ProcessingResult]:
        """
        Process all documents in a directory.
        
        Args:
            directory_path: Path to directory.
            recursive: Whether to process subdirectories recursively.
            pattern: Glob pattern for file matching.
            
        Returns:
            List of ProcessingResult objects.
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            logger.error(f"Directory does not exist: {directory_path}")
            return []
        
        # Find all files
        files = []
        
        if recursive:
            search_pattern = pattern or "**/*"
            for ext_list in self.SUPPORTED_FORMATS.values():
                for ext in ext_list:
                    if ext.startswith('.'):  # Only process file extensions, not URLs
                        files.extend(directory_path.glob(f"{search_pattern}{ext}"))
        else:
            for ext_list in self.SUPPORTED_FORMATS.values():
                for ext in ext_list:
                    if ext.startswith('.'):  # Only process file extensions, not URLs
                        files.extend(directory_path.glob(f"*{ext}"))
        
        # Remove duplicates and sort
        files = list(set(files))
        files.sort()
        
        logger.info(f"Found {len(files)} documents in {directory_path}")
        
        # Process files
        return self.batch_process(files)
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics.
        
        Returns:
            Dictionary with processing statistics.
        """
        stats = self.stats.copy()
        
        # Calculate additional metrics
        if stats['total_processed'] > 0:
            stats['success_rate'] = stats['successful'] / stats['total_processed']
            if stats['successful'] > 0:
                stats['average_processing_time'] = stats['total_processing_time'] / stats['successful']
            else:
                stats['average_processing_time'] = 0.0
        else:
            stats['success_rate'] = 0.0
            stats['average_processing_time'] = 0.0
        
        return stats
    
    def reset_statistics(self):
        """Reset processing statistics."""
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'total_processing_time': 0.0,
            'format_stats': {},
        }
    
    def export_results(self, results: List[ProcessingResult], 
                      output_format: str = 'json',
                      output_file: Optional[Union[str, Path]] = None) -> Optional[str]:
        """
        Export processing results.
        
        Args:
            results: List of ProcessingResult objects.
            output_format: Output format ('json', 'csv', 'text').
            output_file: Path to output file. If None, returns as string.
            
        Returns:
            Exported content if output_file is None, otherwise None.
        """
        if not results:
            return None
        
        # Convert results to dictionaries
        result_dicts = [r.to_dict() for r in results]
        
        if output_format == 'json':
            content = json.dumps(result_dicts, indent=2, ensure_ascii=False)
        
        elif output_format == 'csv':
            import csv
            import io
            
            # Create CSV content
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header (use keys from first result)
            if result_dicts:
                headers = result_dicts[0].keys()
                writer.writerow(headers)
                
                # Write rows
                for result in result_dicts:
                    writer.writerow([str(result.get(h, '')) for h in headers])
            
            content = output.getvalue()
        
        else:  # text format
            lines = []
            for result in results:
                file_name = Path(result.file_path).name if not self._is_url_like(result.file_path) else result.file_path
                lines.append(f"Document: {file_name}")
                lines.append(f"  ID: {result.document_id}")
                lines.append(f"  Status: {result.status}")
                lines.append(f"  Processing Time: {result.processing_time:.2f}s")
                lines.append(f"  Chunks: {result.chunk_count}")
                lines.append(f"  Words: {result.word_count}")
                if result.error_message:
                    lines.append(f"  Error: {result.error_message}")
                lines.append("")
            
            content = "\n".join(lines)
        
        # Write to file or return as string
        if output_file:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Exported {len(results)} results to {output_file}")
            return None
        else:
            return content


# ========== CONVENIENCE FUNCTIONS ==========

def process_document(file_path: Union[str, Path], 
                    config: Optional[ProcessingConfig] = None) -> ProcessingResult:
    """
    Convenience function for processing a single document.
    
    Args:
        file_path: Path to the document file.
        config: Optional processing configuration.
        
    Returns:
        ProcessingResult object.
    """
    processor = DocumentProcessor(config)
    return processor.process_document(file_path)


def batch_process_documents(file_paths: List[Union[str, Path]], 
                           config: Optional[ProcessingConfig] = None,
                           max_workers: Optional[int] = None) -> List[ProcessingResult]:
    """
    Convenience function for batch processing documents.
    
    Args:
        file_paths: List of file paths to process.
        config: Optional processing configuration.
        max_workers: Maximum number of concurrent workers.
        
    Returns:
        List of ProcessingResult objects.
    """
    processor = DocumentProcessor(config)
    return processor.batch_process(file_paths, max_workers)


def process_directory_contents(directory_path: Union[str, Path], 
                              config: Optional[ProcessingConfig] = None,
                              recursive: bool = True) -> List[ProcessingResult]:
    """
    Convenience function for processing all documents in a directory.
    
    Args:
        directory_path: Path to directory.
        config: Optional processing configuration.
        recursive: Whether to process subdirectories.
        
    Returns:
        List of ProcessingResult objects.
    """
    processor = DocumentProcessor(config)
    return processor.process_directory(directory_path, recursive)


# ========== SINGLETON ACCESSOR ==========

_processor_instance = None

def get_processor(config: Optional[Union[ProcessingConfig, Dict]] = None) -> DocumentProcessor:
    """
    Get or create a singleton DocumentProcessor instance.
    
    Args:
        config: Optional processing configuration. Can be ProcessingConfig object or dict.
        
    Returns:
        DocumentProcessor instance.
    """
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = DocumentProcessor(config)
    elif config is not None:
        # If config provided and different, create new instance
        _processor_instance = DocumentProcessor(config)
    return _processor_instance


# ========== TESTING AND VALIDATION ==========

def test_processor() -> Dict[str, bool]:
    """Test document processor functionality."""
    import tempfile
    
    test_results = {}
    
    # Create test files
    test_files = []
    
    # Test TXT file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write("Test document content for processor testing.\n" * 100)
        test_files.append(f.name)
    
    # Test simple processing
    try:
        processor = DocumentProcessor(ProcessingConfig(
            max_workers=1,
            enable_cleaning=False,
            enable_chunking=False,
            enable_embedding_generation=False,
            enable_vector_storage=False,
            verbose=False,
        ))
        
        result = processor.process_document(test_files[0])
        
        test_results['single_document_processing'] = (
            result.status == 'completed' and
            result.extracted_text is not None and
            result.word_count > 0
        )
    except Exception as e:
        test_results['single_document_processing'] = False
        logger.error(f"Single document processing test failed: {e}")
    
    # Test batch processing
    try:
        results = processor.batch_process(test_files)
        test_results['batch_processing'] = len(results) == 1 and results[0].status == 'completed'
    except Exception as e:
        test_results['batch_processing'] = False
        logger.error(f"Batch processing test failed: {e}")
    
    # Test statistics
    try:
        stats = processor.get_processing_stats()
        test_results['statistics_tracking'] = (
            stats['total_processed'] == 2 and  # Single + batch
            'success_rate' in stats
        )
    except Exception as e:
        test_results['statistics_tracking'] = False
        logger.error(f"Statistics tracking test failed: {e}")
    
    # Clean up
    for file_path in test_files:
        try:
            os.unlink(file_path)
        except:
            pass
    
    return test_results


# ========== MAIN MODULE EXECUTION ==========

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Document Processing Pipeline")
    parser.add_argument("--process", type=str, help="Process a single document")
    parser.add_argument("--batch", type=str, nargs='+', help="Process multiple documents")
    parser.add_argument("--directory", type=str, help="Process all documents in directory")
    parser.add_argument("--recursive", action="store_true", help="Process directories recursively")
    parser.add_argument("--config", type=str, help="Path to config file (JSON/YAML)")
    parser.add_argument("--output", type=str, help="Output file for results")
    parser.add_argument("--format", type=str, choices=['json', 'csv', 'text'], default='json',
                       help="Output format")
    parser.add_argument("--stats", action="store_true", help="Show processing statistics")
    parser.add_argument("--test", action="store_true", help="Run unit tests")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if args.test:
        print("Running document processor tests...")
        results = test_processor()
        passed = sum(results.values())
        total = len(results)
        print(f"Tests passed: {passed}/{total} ({passed/total*100:.1f}%)")
        
        for test, success in results.items():
            status = "✓" if success else "✗"
            print(f"  {status} {test}")
    
    elif args.process:
        file_path = Path(args.process)
        
        if not file_path.exists() and not (str(file_path).startswith('http://') or str(file_path).startswith('https://')):
            print(f"Error: File not found: {file_path}")
            exit(1)
        
        print(f"Processing document: {file_path}")
        
        # Load config if provided
        config = None
        if args.config:
            try:
                config_path = Path(args.config)
                if config_path.exists():
                    import yaml
                    with open(config_path, 'r') as f:
                        config_data = yaml.safe_load(f)
                    config = ProcessingConfig(**config_data)
            except Exception as e:
                print(f"Warning: Could not load config: {e}")
        
        # Process document
        result = process_document(file_path, config)
        
        # Output results
        if args.output:
            processor = DocumentProcessor(config)
            processor.export_results([result], args.format, args.output)
            print(f"Results exported to: {args.output}")
        else:
            print("\n=== PROCESSING RESULTS ===")
            print(f"Document: {result.file_path}")
            print(f"Status: {result.status}")
            print(f"Processing Time: {result.processing_time:.2f}s")
            print(f"Chunks: {result.chunk_count}")
            print(f"Words: {result.word_count}")
            
            if result.error_message:
                print(f"Error: {result.error_message}")
            
            if result.metadata and 'title' in result.metadata:
                print(f"Title: {result.metadata['title']}")
    
    elif args.batch:
        file_paths = [Path(f) for f in args.batch]
        
        # Load config if provided
        config = None
        if args.config:
            try:
                config_path = Path(args.config)
                if config_path.exists():
                    import yaml
                    with open(config_path, 'r') as f:
                        config_data = yaml.safe_load(f)
                    config = ProcessingConfig(**config_data)
            except Exception as e:
                print(f"Warning: Could not load config: {e}")
        
        print(f"Processing {len(file_paths)} documents...")
        
        # Process documents
        results = batch_process_documents(file_paths, config)
        
        # Output results
        if args.output:
            processor = DocumentProcessor(config)
            processor.export_results(results, args.format, args.output)
            print(f"Results exported to: {args.output}")
        else:
            successful = sum(1 for r in results if r.status == 'completed')
            failed = len(results) - successful
            
            print(f"\n=== BATCH PROCESSING SUMMARY ===")
            print(f"Total: {len(results)}")
            print(f"Successful: {successful}")
            print(f"Failed: {failed}")
            
            if failed > 0:
                print("\nFailed documents:")
                for result in results:
                    if result.status != 'completed':
                        file_name = Path(result.file_path).name if not result.file_path.startswith('http') else result.file_path
                        print(f"  • {file_name}: {result.error_message}")
    
    elif args.directory:
        directory_path = Path(args.directory)
        
        if not directory_path.exists():
            print(f"Error: Directory not found: {directory_path}")
            exit(1)
        
        # Load config if provided
        config = None
        if args.config:
            try:
                config_path = Path(args.config)
                if config_path.exists():
                    import yaml
                    with open(config_path, 'r') as f:
                        config_data = yaml.safe_load(f)
                    config = ProcessingConfig(**config_data)
            except Exception as e:
                print(f"Warning: Could not load config: {e}")
        
        print(f"Processing documents in directory: {directory_path}")
        
        # Process directory
        results = process_directory_contents(directory_path, config, args.recursive)
        
        # Output results
        if args.output:
            processor = DocumentProcessor(config)
            processor.export_results(results, args.format, args.output)
            print(f"Results exported to: {args.output}")
        else:
            successful = sum(1 for r in results if r.status == 'completed')
            failed = len(results) - successful
            
            print(f"\n=== DIRECTORY PROCESSING SUMMARY ===")
            print(f"Total documents found: {len(results)}")
            print(f"Successful: {successful}")
            print(f"Failed: {failed}")
            
            # Show statistics by format
            if results:
                processor = DocumentProcessor(config)
                stats = processor.get_processing_stats()
                
                if stats['format_stats']:
                    print("\nStatistics by format:")
                    for format_type, format_stats in stats['format_stats'].items():
                        if format_stats['total'] > 0:
                            success_rate = format_stats['successful'] / format_stats['total']
                            print(f"  {format_type}: {format_stats['successful']}/{format_stats['total']} "
                                  f"({success_rate:.1%})")
    
    elif args.stats:
        print("Document Processor Statistics:")
        print("===============================")
        
        processor = DocumentProcessor()
        stats = processor.get_processing_stats()
        
        print(f"Total Processed: {stats['total_processed']}")
        print(f"Successful: {stats['successful']}")
        print(f"Failed: {stats['failed']}")
        print(f"Success Rate: {stats.get('success_rate', 0):.1%}")
        print(f"Average Processing Time: {stats.get('average_processing_time', 0):.2f}s")
        
        if stats['format_stats']:
            print("\nBy Format:")
            for format_type, format_stats in stats['format_stats'].items():
                if format_stats['total'] > 0:
                    success_rate = format_stats['successful'] / format_stats['total']
                    print(f"  {format_type}: {format_stats['successful']}/{format_stats['total']} "
                          f"({success_rate:.1%})")
    
    else:
        # Show usage and supported formats
        print("=" * 70)
        print("DOCUMENT PROCESSING PIPELINE")
        print("=" * 70)
        
        processor = DocumentProcessor()
        supported_formats = processor.get_supported_formats()
        
        print("\nSupported Formats:")
        for format_name, extensions in supported_formats.items():
            if extensions and not all(ext.startswith('http') for ext in extensions):
                print(f"  • {format_name.upper()}: {', '.join(ext for ext in extensions if ext.startswith('.'))}")
        
        print("\nSupported URLs:")
        print("  • Web pages (http://, https://, www.)")
        
        print("\nUsage examples:")
        print("  python processor.py --process document.pdf")
        print("  python processor.py --process https://example.com")
        print("  python processor.py --batch file1.txt file2.pdf")
        print("  python processor.py --directory ./documents --recursive")
        print("  python processor.py --process doc.pdf --output results.json --format json")
        print("  python processor.py --config config.yaml --process document.docx")
        print("  python processor.py --stats")
        print("  python processor.py --test")
        
        print("\nPipeline Steps:")
        print("  1. Document validation")
        print("  2. Metadata extraction")
        print("  3. Text extraction")
        print("  4. Text cleaning")
        print("  5. Text chunking")
        print("  6. Embedding generation")
        print("  7. Vector storage")