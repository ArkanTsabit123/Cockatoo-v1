# Cockatoo_V1/src/document_processing/processor.py

"""
Main Document Processing Pipeline
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from .extractors import create_extractor, get_supported_extensions
from .extractors.pdf_extractor import PDFExtractor  # <-- IMPORT INI
from .cleaning import clean_text_pipeline
from .chunking import chunk_text
from ..ai_engine.embedding_service import EmbeddingService
from ..vector_store.chroma_client import ChromaClient
from ..database.sqlite_client import SQLiteClient


logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Main document processing orchestrator"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize document processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {
            'chunk_size': 500,
            'chunk_overlap': 50,
            'max_file_size_mb': 100,
            'supported_formats': ['.pdf', '.docx', '.txt', '.epub', '.md', '.html', '.jpg', '.png', '.csv'],
            'ocr_enabled': True,
            'ocr_languages': ['eng', 'ind']
        }
        
        # Initialize extractor registry
        self._init_extractor_registry()
        
        # Initialize services (lazy loading untuk menghindari error jika belum ada)
        self.embedding_service = None
        self.vector_store = None
        self.database = None
        
        logger.info("DocumentProcessor initialized with config: %s", self.config)
    
    def extract_text(self, file_path: Path) -> str:
        """Public API method - delegates to extractor."""
        extractor = self._get_extractor(file_path)
        if extractor:
            result = extractor.extract(str(file_path))
            return result.get('text', '')
        return ""

    def chunk_text(self, text: str, chunk_size: int = None, chunk_overlap: int = None) -> List[Dict]:
        """Public API method - delegates to chunking module."""
        from .chunking import chunk_text as chunk_text_func
        return chunk_text_func(
            text=text,
            chunk_size=chunk_size or self.config['chunk_size'],
            chunk_overlap=chunk_overlap or self.config['chunk_overlap']
        )

    def clean_text(self, text: str) -> str:
        """Public API method - delegates to cleaning module."""
        from .cleaning import clean_text_pipeline
        return clean_text_pipeline(text)

    def _init_extractor_registry(self):
        """Initialize extractor registry."""
        self.extractor_registry = {}
        
        # Register extractors
        try:
            # PDF Extractor
            self.extractor_registry['.pdf'] = PDFExtractor()
            logger.debug("Registered PDF extractor")
        except ImportError as e:
            logger.warning(f"PDF extractor not available: {e}")
        
        # Add other extractors as they become available
        # self.extractor_registry['.docx'] = DOCXExtractor()
        # self.extractor_registry['.txt'] = TXTExtractor()
        # etc...
    
    def process_document(self, file_path: Path) -> Dict[str, Any]:
        """
        Complete document processing pipeline.
        
        Args:
            file_path: Path to document file
            
        Returns:
            Dictionary with processing results
        """
        start_time = datetime.now()
        results = {
            'success': False,
            'document_id': str(file_path),
            'error': None,
            'processing_time': 0,
            'chunks_created': 0,
            'warnings': []
        }
        
        try:
            # Validate file
            validation = self._validate_file(file_path)
            if not validation['is_valid']:
                raise ValueError(f"File validation failed: {', '.join(validation['errors'])}")
            
            logger.info(f"Processing document: {file_path.name}")
            
            # Get extractor (gunakan registry Anda atau factory)
            extractor = self._get_extractor(file_path)
            if not extractor:
                raise ValueError(f"No extractor found for file: {file_path}")
            
            # Extract text and metadata
            try:
                extraction_result = extractor.extract(str(file_path))
            except AttributeError:
                # Fallback to old method if extract() doesn't take string
                extraction_result = extractor.extract_with_metadata(file_path)
            
            if not extraction_result.get('text', '').strip():
                results['warnings'].append("Extracted text is empty or whitespace only")
            
            raw_text = extraction_result.get('text', '')
            metadata = extraction_result.get('metadata', {})
            
            # Add file metadata
            metadata.update({
                'file_path': str(file_path),
                'file_name': file_path.name,
                'file_extension': file_path.suffix.lower(),
                'file_size': file_path.stat().st_size,
                'processing_timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"Extracted {len(raw_text)} characters from {file_path.name}")
            
            # Clean text
            cleaned_text = clean_text_pipeline(raw_text)
            if len(cleaned_text) < len(raw_text) * 0.1:  # Jika >90% hilang
                results['warnings'].append(f"Text cleaning removed {len(raw_text) - len(cleaned_text)} characters")
            
            logger.debug(f"Text cleaned: {len(cleaned_text)} characters remaining")
            
            # Chunk text
            chunks = chunk_text(
                text=cleaned_text,
                chunk_size=self.config['chunk_size'],
                chunk_overlap=self.config['chunk_overlap']
            )
            
            if not chunks:
                results['warnings'].append("No chunks created from text")
            
            logger.info(f"Created {len(chunks)} chunks from document")
            
            # Generate embeddings
            chunk_texts = [chunk.get('text', '') for chunk in chunks]
            
            # Initialize embedding service if needed
            if self.embedding_service is None:
                self.embedding_service = EmbeddingService()
            
            embeddings = self.embedding_service.generate_embeddings(chunk_texts)
            
            # Store in vector database
            if self.vector_store is None:
                self.vector_store = ChromaClient()
            
            vector_ids = self.vector_store.add_documents(
                texts=chunk_texts,
                embeddings=embeddings,
                metadatas=[{
                    **metadata,
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'original_file': str(file_path),
                    'processed_at': datetime.now().isoformat(),
                    'chunk_length': len(chunk.get('text', ''))
                } for i, chunk in enumerate(chunks)]
            )
            
            # Store in SQLite database
            if self.database is None:
                self.database = SQLiteClient()
            
            doc_id = self.database.add_document(
                file_path=str(file_path),
                file_name=file_path.name,
                file_type=metadata['file_extension'],
                file_size=metadata['file_size'],
                chunk_count=len(chunks),
                vector_ids=vector_ids,
                metadata=metadata
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            results.update({
                'success': True,
                'document_id': doc_id,
                'chunks_created': len(chunks),
                'processing_time': processing_time,
                'metadata': metadata,
                'vector_ids': vector_ids,
                'text_sample': cleaned_text[:500] + "..." if len(cleaned_text) > 500 else cleaned_text
            })
            
            logger.info(
                f"Successfully processed {file_path.name}: "
                f"{len(chunks)} chunks, {processing_time:.2f}s"
            )
            
        except Exception as e:
            error_msg = f"Failed to process {file_path}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            results['error'] = error_msg
        
        finally:
            results['processing_time'] = (datetime.now() - start_time).total_seconds()
        
        return results
    
    def _get_extractor(self, file_path: Path):
        """
        Get extractor for file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Extractor instance or None
        """
        extension = file_path.suffix.lower()
        
        # Coba dari registry dulu
        if extension in self.extractor_registry:
            return self.extractor_registry[extension]
        
        # Fallback ke factory function Anda
        try:
            return create_extractor(file_path)
        except Exception as e:
            logger.warning(f"Factory extractor failed for {extension}: {e}")
            return None
    
    def _validate_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Comprehensive file validation.
        
        Args:
            file_path: Path to validate
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            'is_valid': False,
            'errors': [],
            'warnings': [],
            'file_exists': False,
            'file_readable': False,
            'format_supported': False,
            'size_valid': False
        }
        
        # Check existence
        if not file_path.exists():
            validation['errors'].append(f"File not found: {file_path}")
            return validation
        
        validation['file_exists'] = True
        
        # Check if it's a file (not directory)
        if not file_path.is_file():
            validation['errors'].append(f"Not a file: {file_path}")
            return validation
        
        # Check file size
        try:
            file_size = file_path.stat().st_size
            max_size = self.config.get('max_file_size_mb', 100) * 1024 * 1024
            
            if file_size == 0:
                validation['errors'].append("File is empty (0 bytes)")
                return validation
            
            if file_size > max_size:
                validation['errors'].append(
                    f"File too large: {file_size / (1024*1024):.1f}MB "
                    f"(max: {self.config.get('max_file_size_mb', 100)}MB)"
                )
                return validation
            
            validation['size_valid'] = True
            
            if file_size > 50 * 1024 * 1024:  # 50MB
                validation['warnings'].append(
                    f"Large file: {file_size / (1024*1024):.1f}MB - processing may be slow"
                )
        
        except OSError as e:
            validation['errors'].append(f"Cannot access file size: {e}")
            return validation
        
        # Check readability
        try:
            with open(file_path, 'rb') as f:
                f.read(1)  # Try to read 1 byte
            validation['file_readable'] = True
        except Exception as e:
            validation['errors'].append(f"File not readable: {e}")
            return validation
        
        # Check format support
        extension = file_path.suffix.lower()
        supported = self.config.get('supported_formats', get_supported_extensions())
        
        if extension in supported:
            validation['format_supported'] = True
        else:
            validation['errors'].append(
                f"Unsupported file format: {extension}. "
                f"Supported: {', '.join(supported)}"
            )
            return validation
        
        # File-specific validation
        if extension == '.pdf':
            try:
                # Gunakan PDFExtractor untuk validasi PDF
                pdf_validator = PDFExtractor()
                pdf_validation = pdf_validator.validate_pdf(file_path)
                
                if not pdf_validation.get('is_valid', True):
                    validation['errors'].extend(pdf_validation.get('errors', []))
                
                validation['warnings'].extend(pdf_validation.get('warnings', []))
                
                if not pdf_validation.get('has_text', True):
                    validation['warnings'].append(
                        "PDF may not have extractable text (could be scanned document)"
                    )
            
            except Exception as e:
                validation['warnings'].append(f"PDF validation failed: {e}")
        
        validation['is_valid'] = len(validation['errors']) == 0
        return validation
    
    def process_batch(self, file_paths: List[Path]) -> List[Dict[str, Any]]:
        """
        Process multiple documents with improved error handling.
        
        Args:
            file_paths: List of file paths
            
        Returns:
            List of processing results
        """
        results = []
        total_files = len(file_paths)
        
        logger.info(f"Starting batch processing of {total_files} files")
        
        for i, file_path in enumerate(file_paths, 1):
            logger.info(f"Processing file {i}/{total_files}: {file_path.name}")
            
            try:
                result = self.process_document(file_path)
                results.append(result)
                
                if result['success']:
                    logger.info(f"✓ Success: {file_path.name}")
                else:
                    logger.error(f"✗ Failed: {file_path.name} - {result.get('error', 'Unknown error')}")
            
            except Exception as e:
                logger.error(f"✗ Exception processing {file_path.name}: {e}", exc_info=True)
                results.append({
                    'success': False,
                    'document_id': str(file_path),
                    'error': str(e),
                    'warnings': []
                })
        
        # Summary
        successful = sum(1 for r in results if r.get('success', False))
        failed = total_files - successful
        
        logger.info(
            f"Batch processing complete: {successful}/{total_files} successful, "
            f"{failed} failed"
        )
        
        return results
    
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported file formats.
        
        Returns:
            List of supported file extensions
        """
        return self.config.get('supported_formats', get_supported_extensions())
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics with more details.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'supported_formats': self.get_supported_formats(),
            'chunk_size': self.config.get('chunk_size', 500),
            'chunk_overlap': self.config.get('chunk_overlap', 50),
            'max_file_size_mb': self.config.get('max_file_size_mb', 100),
            'ocr_enabled': self.config.get('ocr_enabled', True),
            'ocr_languages': self.config.get('ocr_languages', ['eng', 'ind']),
            'registered_extractors': list(self.extractor_registry.keys())
        }
    
    def validate_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Public method to validate a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Validation results
        """
        return self._validate_file(file_path)


_processor_instance = None

def get_processor(config: Optional[Dict[str, Any]] = None) -> DocumentProcessor:
    """
    Get or create DocumentProcessor instance.
    
    Args:
        config: Optional configuration
        
    Returns:
        DocumentProcessor instance
    """
    global _processor_instance
    
    if _processor_instance is None:
        _processor_instance = DocumentProcessor(config)
    elif config:
        # Update config if provided
        _processor_instance.config.update(config)
    
    return _processor_instance


def test_processor():
    """Test function for processor."""
    import sys
    
    if len(sys.argv) > 1:
        test_file = Path(sys.argv[1])
        
        if test_file.exists():
            processor = DocumentProcessor()
            
            # Validate first
            validation = processor.validate_file(test_file)
            print("Validation result:")
            print(f"  Valid: {validation['is_valid']}")
            
            if validation['errors']:
                print("  Errors:")
                for error in validation['errors']:
                    print(f"    - {error}")
            
            if validation['warnings']:
                print("  Warnings:")
                for warning in validation['warnings']:
                    print(f"    - {warning}")
            
            if validation['is_valid']:
                print("\nProcessing file...")
                result = processor.process_document(test_file)
                print("\nProcessing result:")
                print(f"  Success: {result['success']}")
                print(f"  Document ID: {result.get('document_id')}")
                print(f"  Chunks created: {result.get('chunks_created', 0)}")
                print(f"  Processing time: {result.get('processing_time', 0):.2f}s")
                
                if result.get('warnings'):
                    print("  Warnings:")
                    for warning in result['warnings']:
                        print(f"    - {warning}")
                
                if result.get('text_sample'):
                    print(f"\n  Text sample (first 500 chars):")
                    print(f"    {result['text_sample']}")
            else:
                print("File validation failed, not processing.")
        
        else:
            print(f"File not found: {test_file}")
    
    else:
        print("Usage: python processor.py <file_path>")
        print("Supported formats:", get_supported_extensions())


if __name__ == "__main__":
    test_processor()