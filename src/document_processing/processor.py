# src/document_processing/processor.py

"""Main document processing pipeline orchestrator."""

import os
import re
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict, field
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

from .cleaning import TextCleaner, CleaningConfig
from .metadata import MetadataExtractor, DocumentMetadata
from .chunking import TextChunker, ChunkingConfig

try:
    from .extractors import (
        PDFExtractor, DOCXExtractor, TXTExtractor, EPUBExtractor,
        JSONExtractor, HTMLExtractor, ImageExtractor, CSVExtractor,
        MarkdownExtractor, PPTXExtractor, XLSXExtractor, WebExtractor,
        get_extractor_for_file
    )
    HAS_EXTRACTORS = True
except ImportError as e:
    HAS_EXTRACTORS = False
    logger.warning(f"Extractor modules not available: {e}. Some functionality will be limited.")


@dataclass
class ProcessingConfig:
    """Configuration for document processing pipeline."""
    chunk_size: int = 500
    chunk_overlap: int = 50
    max_file_size_mb: int = 100
    max_workers: int = 4
    
    enable_cleaning: bool = True
    enable_chunking: bool = True
    enable_metadata_extraction: bool = True
    enable_embedding_generation: bool = True
    enable_vector_storage: bool = True
    
    cleaning_config: Optional[Dict] = None
    chunking_config: Optional[Dict] = None
    
    save_intermediate: bool = False
    intermediate_dir: Optional[str] = None
    verbose: bool = False
    
    def __post_init__(self):
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
    """Results from document processing pipeline."""
    document_id: str
    file_path: str
    file_type: str
    
    status: str
    error_message: Optional[str] = None
    
    processing_time: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    extracted_text: Optional[str] = None
    cleaned_text: Optional[str] = None
    chunks: List[Dict] = field(default_factory=list)
    embeddings: List[List[float]] = field(default_factory=list)
    vector_ids: List[str] = field(default_factory=list)
    
    metadata: Optional[Dict] = None
    chunk_count: int = 0
    word_count: int = 0
    token_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        
        for key in ['start_time', 'end_time']:
            if result.get(key):
                result[key] = result[key].isoformat()
        
        return result
    
    def calculate_metrics(self) -> Dict[str, Any]:
        return {
            'processing_time': self.processing_time,
            'chunk_count': self.chunk_count,
            'word_count': self.word_count,
            'token_count': self.token_count,
            'average_chunk_size': self.word_count / self.chunk_count if self.chunk_count > 0 else 0,
            'success_rate': 1.0 if self.status == 'completed' else 0.0,
        }


class DocumentProcessor:
    """Main document processing pipeline orchestrator."""
    
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
    
    def __init__(self, config: Optional[Union[ProcessingConfig, Dict]] = None, vector_store=None):
        if config is not None and isinstance(config, dict):
            self.config = ProcessingConfig(**config)
        elif config is None:
            self.config = ProcessingConfig()
        else:
            self.config = config
        
        self.vector_store = vector_store
        
        self._initialize_components()
        
        if self.config.save_intermediate:
            Path(self.config.intermediate_dir).mkdir(parents=True, exist_ok=True)
        
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'total_processing_time': 0.0,
            'format_stats': {},
        }
        
        logger.info(f"DocumentProcessor initialized with config: {self.config}")
    
    def _initialize_components(self):
        if self.config.enable_cleaning:
            cleaning_config = CleaningConfig(**self.config.cleaning_config)
            self.text_cleaner = TextCleaner(config=cleaning_config)
        else:
            self.text_cleaner = None
            logger.warning("Text cleaning disabled")
        
        if self.config.enable_chunking:
            chunking_config = ChunkingConfig(**self.config.chunking_config)
            self.text_chunker = TextChunker(config=chunking_config)
        else:
            self.text_chunker = None
            logger.warning("Text chunking disabled")
        
        if self.config.enable_metadata_extraction:
            self.metadata_extractor = MetadataExtractor()
        else:
            self.metadata_extractor = None
            logger.warning("Metadata extraction disabled")
    
    def get_supported_formats(self) -> Dict[str, List[str]]:
        return self.SUPPORTED_FORMATS
    
    def is_supported_format(self, file_path: Union[str, Path]) -> bool:
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        if self._is_url_like(str(file_path)):
            return True
        
        for extensions in self.SUPPORTED_FORMATS.values():
            if extension in extensions:
                return True
        
        return False
    
    def get_format_type(self, file_path: Union[str, Path]) -> Optional[str]:
        file_path = Path(file_path)
        file_str = str(file_path)
        
        if self._is_url_like(file_str):
            return 'web'
        
        extension = file_path.suffix.lower()
        
        for format_type, extensions in self.SUPPORTED_FORMATS.items():
            if extension in extensions:
                return format_type
        
        return None
    
    def _is_url_like(self, text: str) -> bool:
        if not text:
            return False
        
        text_lower = text.lower()
        return (text_lower.startswith('http://') or 
                text_lower.startswith('https://') or
                text_lower.startswith('www.') or
                '://' in text_lower)
    
    def validate_file(self, file_path: Union[str, Path]) -> Tuple[bool, Optional[str]]:
        file_path = Path(file_path)
        
        if not self._is_url_like(str(file_path)) and not file_path.exists():
            return False, f"File does not exist: {file_path}"
        
        if not self._is_url_like(str(file_path)) and not os.access(file_path, os.R_OK):
            return False, f"File is not readable: {file_path}"
        
        if not self._is_url_like(str(file_path)):
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            if file_size_mb > self.config.max_file_size_mb:
                return False, f"File too large: {file_size_mb:.1f}MB > {self.config.max_file_size_mb}MB limit"
        
        if not self.is_supported_format(file_path):
            return False, f"Unsupported file format: {file_path.suffix or 'URL'}"
        
        return True, None
    
    def generate_document_id(self, file_path: Union[str, Path]) -> str:
        file_path = Path(file_path)
        
        if self._is_url_like(str(file_path)):
            unique_string = str(file_path)
        else:
            try:
                mtime = file_path.stat().st_mtime
                unique_string = f"{file_path.absolute()}:{mtime}"
            except:
                unique_string = str(file_path.absolute())
        
        return hashlib.sha256(unique_string.encode()).hexdigest()[:16]
    
    def process_document(self, file_path: Union[str, Path]) -> ProcessingResult:
        file_path = Path(file_path)
        start_time = datetime.now()
        
        doc_id = self.generate_document_id(file_path)
        
        result = ProcessingResult(
            document_id=doc_id,
            file_path=str(file_path.absolute()),
            file_type=file_path.suffix.lower() if not self._is_url_like(str(file_path)) else 'web',
            status='processing',
            start_time=start_time,
        )
        
        logger.info(f"Processing document: {file_path.name if not self._is_url_like(str(file_path)) else file_path} (ID: {doc_id})")
        
        try:
            is_valid, error = self.validate_file(file_path)
            if not is_valid:
                result.status = 'failed'
                result.error_message = error
                logger.error(f"Validation failed for {file_path}: {error}")
                return result
            
            if self.config.enable_metadata_extraction and self.metadata_extractor:
                try:
                    metadata = self.metadata_extractor.extract_metadata(file_path)
                    result.metadata = metadata.to_dict()
                    
                    if metadata.processing_word_count > 0:
                        result.word_count = metadata.processing_word_count
                    
                    logger.debug(f"Metadata extracted: {metadata.content_title if metadata.content_title else 'No title'}")
                except Exception as e:
                    logger.warning(f"Metadata extraction failed for {file_path}: {e}")
            
            extracted_text = self._extract_text(file_path, result)
            if not extracted_text:
                result.status = 'failed'
                result.error_message = "Failed to extract text content"
                logger.error(f"Text extraction failed for {file_path}")
                return result
            
            result.extracted_text = extracted_text
            
            if result.word_count == 0:
                result.word_count = len(re.findall(r'\b\w+\b', extracted_text))
            
            if self.config.enable_cleaning and self.text_cleaner:
                try:
                    cleaned_text = self.text_cleaner.clean_text(extracted_text, verbose=self.config.verbose)
                    result.cleaned_text = cleaned_text
                    
                    if self.config.verbose:
                        logger.info(f"Text cleaning complete. Original: {len(extracted_text)} chars, Cleaned: {len(cleaned_text)} chars")
                except Exception as e:
                    logger.warning(f"Text cleaning failed for {file_path}: {e}")
                    result.cleaned_text = extracted_text
            else:
                result.cleaned_text = extracted_text
            
            if self.config.enable_chunking and self.text_chunker:
                text_to_chunk = result.cleaned_text or result.extracted_text
                
                try:
                    chunks = self.text_chunker.chunk_text(text_to_chunk)
                    result.chunks = chunks
                    result.chunk_count = len(chunks)
                    
                    result.token_count = sum(chunk.get('token_count', 0) for chunk in chunks) if chunks else 0
                    
                    logger.info(f"Text chunking complete. Created {result.chunk_count} chunks")
                except Exception as e:
                    logger.error(f"Text chunking failed for {file_path}: {e}")
                    result.status = 'failed'
                    result.error_message = f"Text chunking failed: {str(e)}"
                    return result
            else:
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
            
            if self.config.enable_embedding_generation:
                embeddings = self._generate_embeddings(result.chunks)
                result.embeddings = embeddings
                
                logger.info(f"Generated {len(embeddings)} embeddings")
            
            if self.config.enable_vector_storage and self.vector_store:
                vector_ids = self._store_in_vector_db(result.chunks, result.embeddings, result.metadata)
                result.vector_ids = vector_ids
                
                logger.info(f"Stored {len(vector_ids)} chunks in vector database")
            elif self.config.enable_vector_storage and not self.vector_store:
                logger.warning("Vector storage enabled but no vector_store provided")
            
            if self.config.save_intermediate:
                self._save_intermediate_results(result)
            
            result.status = 'completed'
            result.end_time = datetime.now()
            result.processing_time = (result.end_time - start_time).total_seconds()
            
            self._update_statistics(result, success=True)
            
            logger.info(f"Document processing completed: {file_path.name if not self._is_url_like(str(file_path)) else file_path} "
                       f"(Time: {result.processing_time:.2f}s, "
                       f"Chunks: {result.chunk_count}, "
                       f"Words: {result.word_count})")
            
        except Exception as e:
            result.status = 'failed'
            result.error_message = str(e)
            result.end_time = datetime.now()
            result.processing_time = (result.end_time - start_time).total_seconds()
            
            self._update_statistics(result, success=False)
            
            logger.error(f"Document processing failed for {file_path}: {e}", exc_info=True)
        
        return result
    
    def _extract_text(self, file_path: Path, result: ProcessingResult) -> Optional[str]:
        if self._is_url_like(str(file_path)):
            try:
                if HAS_EXTRACTORS:
                    extractor = WebExtractor()
                    extraction_result = extractor.extract(str(file_path))
                    
                    if isinstance(extraction_result, dict):
                        text = extraction_result.get('text', '')
                    else:
                        text = str(extraction_result)
                    
                    if text and text.strip():
                        logger.info(f"Extracted text from URL: {len(text)} chars")
                        return text
            except Exception as e:
                logger.warning(f"Web extraction failed for {file_path}: {e}")
            
            return f"URL content from: {file_path}"
        
        format_type = self.get_format_type(file_path)
        
        if not format_type:
            logger.error(f"Unsupported format for {file_path}")
            return None
        
        if HAS_EXTRACTORS:
            try:
                extractor = get_extractor_for_file(str(file_path))
                extraction_result = extractor.extract(file_path)
                
                if isinstance(extraction_result, dict):
                    text = extraction_result.get('text', '')
                    if not text and 'content' in extraction_result:
                        text = extraction_result['content']
                    
                    if result.metadata is None:
                        result.metadata = {}
                    
                    if 'metadata' in extraction_result:
                        result.metadata.update(extraction_result['metadata'])
                    
                    logger.info(f"Extracted text using {format_type} extractor (dict format): {len(text)} chars")
                else:
                    text = str(extraction_result)
                    logger.info(f"Extracted text using {format_type} extractor (string format): {len(text)} chars")
                
                if text and text.strip():
                    return text
                else:
                    logger.warning(f"Extractor returned empty text for {file_path}")
                    
            except Exception as e:
                logger.warning(f"Extractor failed for {file_path}: {e}")
        
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
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    f.read(1024)
                return encoding
            except UnicodeDecodeError:
                continue
        
        return 'utf-8'
    
    def _generate_embeddings(self, chunks: List[Dict]) -> List[List[float]]:
        if not chunks:
            return []
        
        embeddings = []
        
        for chunk in chunks:
            text = chunk.get('text', '')
            
            if text:
                import struct
                
                text_hash = hashlib.md5(text.encode()).digest()
                floats = []
                for i in range(0, 16, 2):
                    byte_pair = text_hash[i:i+2]
                    if len(byte_pair) == 2:
                        value = struct.unpack('H', byte_pair)[0] / 65535.0
                        floats.append(value)
                
                if len(floats) < 384:
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
        if not chunks:
            return []
        
        vector_ids = []
        
        base_metadata = {}
        if metadata:
            for key, value in metadata.items():
                if value is None:
                    base_metadata[key] = ""
                elif isinstance(value, (str, int, float, bool)):
                    base_metadata[key] = value
                elif isinstance(value, (list, dict)):
                    base_metadata[key] = str(value)
                else:
                    base_metadata[key] = str(value)
        
        if 'custom_fields' in base_metadata:
            base_metadata['custom_fields'] = str(base_metadata['custom_fields'])
        
        for i, chunk in enumerate(chunks):
            doc_id = metadata.get('core_document_id', 'doc') if metadata else 'doc'
            chunk_id = f"{doc_id}_{i}"
            vector_ids.append(chunk_id)
            
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({
                'chunk_index': str(i),
                'chunk_text_preview': chunk['text'][:100] if chunk['text'] else "",
                'chunk_token_count': str(chunk.get('token_count', 0)),
            })
            
            clean_metadata = {}
            for k, v in chunk_metadata.items():
                if v is None:
                    clean_metadata[k] = ""
                elif isinstance(v, (dict, list)):
                    clean_metadata[k] = str(v)
                elif isinstance(v, bool):
                    clean_metadata[k] = str(v).lower()
                else:
                    clean_metadata[k] = str(v)
            
            try:
                if not hasattr(self, 'vector_store') or not self.vector_store:
                    logger.warning("No vector_store available, skipping storage")
                    continue
                    
                self.vector_store.collection.add(
                    documents=[chunk['text']],
                    embeddings=[embeddings[i]] if embeddings else None,
                    metadatas=[clean_metadata],
                    ids=[chunk_id]
                )
            except Exception as e:
                logger.error(f"Failed to store chunk {i}: {e}")
                continue
        
        logger.info(f"Stored {len(vector_ids)} chunks in vector database")
        return vector_ids
    
    def _save_intermediate_results(self, result: ProcessingResult):
        try:
            intermediate_dir = Path(self.config.intermediate_dir)
            
            metadata_file = intermediate_dir / f"{result.document_id}_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(result.metadata or {}, f, indent=2, ensure_ascii=False)
            
            if result.extracted_text:
                text_file = intermediate_dir / f"{result.document_id}_extracted.txt"
                with open(text_file, 'w', encoding='utf-8') as f:
                    f.write(result.extracted_text)
            
            if result.cleaned_text:
                cleaned_file = intermediate_dir / f"{result.document_id}_cleaned.txt"
                with open(cleaned_file, 'w', encoding='utf-8') as f:
                    f.write(result.cleaned_text)
            
            if result.chunks:
                chunks_file = intermediate_dir / f"{result.document_id}_chunks.json"
                with open(chunks_file, 'w', encoding='utf-8') as f:
                    json.dump(result.chunks, f, indent=2, ensure_ascii=False)
            
            summary_file = intermediate_dir / f"{result.document_id}_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Saved intermediate results for {result.document_id}")
            
        except Exception as e:
            logger.warning(f"Failed to save intermediate results: {e}")
    
    def _update_statistics(self, result: ProcessingResult, success: bool):
        self.stats['total_processed'] += 1
        
        if success:
            self.stats['successful'] += 1
            self.stats['total_processing_time'] += result.processing_time
        else:
            self.stats['failed'] += 1
        
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
        if not file_paths:
            return []
        
        max_workers = max_workers or self.config.max_workers
        results = []
        
        logger.info(f"Starting batch processing of {len(file_paths)} documents with {max_workers} workers")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self.process_document, file_path): file_path
                for file_path in file_paths
            }
            
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                
                try:
                    result = future.result(timeout=300)
                    results.append(result)
                    
                    logger.info(f"Completed: {Path(file_path).name if not self._is_url_like(str(file_path)) else file_path} - {result.status}")
                    
                except Exception as e:
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
        
        successful = sum(1 for r in results if r.status == 'completed')
        failed = len(results) - successful
        
        logger.info(f"Batch processing completed: {successful} successful, {failed} failed")
        
        return results
    
    def process_directory(self, directory_path: Union[str, Path], 
                         recursive: bool = True,
                         pattern: Optional[str] = None) -> List[ProcessingResult]:
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            logger.error(f"Directory does not exist: {directory_path}")
            return []
        
        files = []
        
        if recursive:
            search_pattern = pattern or "**/*"
            for ext_list in self.SUPPORTED_FORMATS.values():
                for ext in ext_list:
                    if ext.startswith('.'):
                        files.extend(directory_path.glob(f"{search_pattern}{ext}"))
        else:
            for ext_list in self.SUPPORTED_FORMATS.values():
                for ext in ext_list:
                    if ext.startswith('.'):
                        files.extend(directory_path.glob(f"*{ext}"))
        
        files = list(set(files))
        files.sort()
        
        logger.info(f"Found {len(files)} documents in {directory_path}")
        
        return self.batch_process(files)
    
    def get_processing_stats(self) -> Dict[str, Any]:
        stats = self.stats.copy()
        
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
        if not results:
            return None
        
        result_dicts = [r.to_dict() for r in results]
        
        if output_format == 'json':
            content = json.dumps(result_dicts, indent=2, ensure_ascii=False)
        
        elif output_format == 'csv':
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            if result_dicts:
                headers = result_dicts[0].keys()
                writer.writerow(headers)
                
                for result in result_dicts:
                    writer.writerow([str(result.get(h, '')) for h in headers])
            
            content = output.getvalue()
        
        else:
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
        
        if output_file:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Exported {len(results)} results to {output_file}")
            return None
        else:
            return content


def process_document(file_path: Union[str, Path], 
                    config: Optional[ProcessingConfig] = None) -> ProcessingResult:
    processor = DocumentProcessor(config)
    return processor.process_document(file_path)


def batch_process_documents(file_paths: List[Union[str, Path]], 
                           config: Optional[ProcessingConfig] = None,
                           max_workers: Optional[int] = None) -> List[ProcessingResult]:
    processor = DocumentProcessor(config)
    return processor.batch_process(file_paths, max_workers)


def process_directory_contents(directory_path: Union[str, Path], 
                              config: Optional[ProcessingConfig] = None,
                              recursive: bool = True) -> List[ProcessingResult]:
    processor = DocumentProcessor(config)
    return processor.process_directory(directory_path, recursive)


_processor_instance = None

def get_processor(config: Optional[Union[ProcessingConfig, Dict]] = None, vector_store=None) -> DocumentProcessor:
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = DocumentProcessor(config, vector_store)
    elif config is not None or vector_store is not None:
        _processor_instance = DocumentProcessor(config, vector_store)
    return _processor_instance


def test_processor() -> Dict[str, bool]:
    import tempfile
    
    test_results = {}
    
    test_files = []
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write("Test document content for processor testing.\n" * 100)
        test_files.append(f.name)
    
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
    
    try:
        results = processor.batch_process(test_files)
        test_results['batch_processing'] = len(results) == 1 and results[0].status == 'completed'
    except Exception as e:
        test_results['batch_processing'] = False
        logger.error(f"Batch processing test failed: {e}")
    
    try:
        stats = processor.get_processing_stats()
        test_results['statistics_tracking'] = (
            stats['total_processed'] == 2 and
            'success_rate' in stats
        )
    except Exception as e:
        test_results['statistics_tracking'] = False
        logger.error(f"Statistics tracking test failed: {e}")
    
    for file_path in test_files:
        try:
            os.unlink(file_path)
        except:
            pass
    
    return test_results


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
        
        result = process_document(file_path, config)
        
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
            
            if result.metadata and 'content_title' in result.metadata:
                print(f"Title: {result.metadata['content_title']}")
    
    elif args.batch:
        file_paths = [Path(f) for f in args.batch]
        
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
        
        results = batch_process_documents(file_paths, config)
        
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
        
        results = process_directory_contents(directory_path, config, args.recursive)
        
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