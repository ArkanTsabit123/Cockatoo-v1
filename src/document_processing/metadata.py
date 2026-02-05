# cockatoo_v1/src/document_processing/metadata.py

"""
metadata.py
Metadata Extraction and Management for Cockatoo_V1 Document Processing Pipeline.

This module provides comprehensive metadata extraction from various document formats,
including automatic detection of titles, authors, dates, and other metadata fields.

Author: Cockatoo_V1 Development Team
Version: 1.0.0
"""

import os
import re
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict, field
import logging
from enum import Enum

# Third-party imports for specialized metadata extraction
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False
    pdfplumber = None

try:
    from docx import Document as DocxDocument
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False
    DocxDocument = None

try:
    import ebooklib
    from ebooklib import epub
    HAS_EPUB = True
except ImportError:
    HAS_EPUB = False
    epub = None

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    yaml = None

try:
    import frontmatter
    HAS_FRONTMATTER = True
except ImportError:
    HAS_FRONTMATTER = False
    frontmatter = None

logger = logging.getLogger(__name__)

class DocumentType(Enum):
    """Supported document types for metadata extraction."""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    EPUB = "epub"
    MD = "md"  # Markdown
    HTML = "html"
    JSON = "json"
    YAML = "yaml"
    PPTX = "pptx"
    XLSX = "xlsx"
    CSV = "csv"
    IMAGE = "image"
    UNKNOWN = "unknown"

@dataclass
class DocumentMetadata:
    """
    Comprehensive metadata structure for documents.
    Based on database schema from blueprint.
    """
    # Core file metadata
    file_path: str
    file_name: str
    file_type: str
    file_size: int
    upload_date: Optional[datetime] = None
    
    # Content metadata
    title: Optional[str] = None
    author: Optional[str] = None
    date: Optional[datetime] = None
    publisher: Optional[str] = None
    language: Optional[str] = None
    abstract: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    
    # Processing metadata
    processing_status: str = "pending"
    processing_error: Optional[str] = None
    chunk_count: int = 0
    word_count: int = 0
    page_count: Optional[int] = None
    
    # Technical metadata
    encoding: Optional[str] = None
    created_date: Optional[datetime] = None
    modified_date: Optional[datetime] = None
    software: Optional[str] = None
    version: Optional[str] = None
    
    # Custom metadata
    custom_fields: Dict[str, Any] = field(default_factory=dict)
    
    # Internal tracking
    is_indexed: bool = False
    indexed_at: Optional[datetime] = None
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    
    def __post_init__(self):
        """Initialize optional fields."""
        if self.upload_date is None:
            self.upload_date = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for serialization."""
        result = asdict(self)
        
        # Convert datetime objects to ISO format strings
        for key, value in result.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
        
        return result
    
    def to_json(self) -> str:
        """Convert metadata to JSON string."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentMetadata':
        """Create metadata from dictionary."""
        # Convert ISO strings back to datetime
        datetime_fields = ['upload_date', 'date', 'created_date', 'modified_date', 'indexed_at', 'last_accessed']
        
        for field in datetime_fields:
            if field in data and data[field] and isinstance(data[field], str):
                try:
                    data[field] = datetime.fromisoformat(data[field].replace('Z', '+00:00'))
                except (ValueError, TypeError):
                    data[field] = None
        
        return cls(**data)

class MetadataExtractor:
    """
    Main class for extracting metadata from various document formats.
    Uses format-specific extractors with fallback strategies.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize metadata extractor with configuration.
        
        Args:
            config: Optional configuration dictionary.
        """
        self.config = config or {}
        self.extractors = self._initialize_extractors()
        
        # Common patterns for metadata extraction
        self.patterns = {
            'title': [
                r'^#{1,3}\s+(.+)$',  # Markdown headers
                r'<title>(.+?)</title>',  # HTML title
                r'^(?:[A-Z][A-Z\s]{10,})$',  # ALL CAPS titles
                r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){2,}$',  # Title case lines
            ],
            'author': [
                r'by\s+([A-Z][a-z]+\s+[A-Z][a-z]+)',  # "by John Doe"
                r'author[:\s]+([^\n]+)',  # "Author: John Doe"
                r'©\s*\d{4}\s+([^\n]+)',  # Copyright notice
                r'written by\s+([^\n]+)',  # "Written by John Doe"
            ],
            'date': [
                r'\b\d{4}-\d{2}-\d{2}\b',  # YYYY-MM-DD
                r'\b\d{2}/\d{2}/\d{4}\b',  # MM/DD/YYYY
                r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b',  # Month Day, Year
                r'\b\d{4}\b',  # Just year
            ],
            'email': [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ],
            'url': [
                r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
            ],
            'isbn': [
                r'\b(?:ISBN(?:-1[03])?:? )?(?=[0-9X]{10}$|(?=(?:[0-9]+[- ]){3})[- 0-9X]{13}$|97[89][0-9]{10}$|(?=(?:[0-9]+[- ]){4})[- 0-9]{17}$)(?:97[89][- ]?)?[0-9]{1,5}[- ]?[0-9]+[- ]?[0-9]+[- ]?[0-9X]\b'
            ],
            'doi': [
                r'\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b'
            ]
        }
        
        # Language detection patterns
        self.language_patterns = {
            'en': [r'\bthe\b', r'\band\b', r'\bof\b'],  # English common words
            'id': [r'\bdan\b', r'\byang\b', r'\bdi\b'],  # Indonesian
            'es': [r'\bel\b', r'\bla\b', r'\by\b'],  # Spanish
            'fr': [r'\ble\b', r'\bla\b', r'\bet\b'],  # French
            'de': [r'\bder\b', r'\bdie\b', r'\bden\b'],  # German
        }
    
    def _initialize_extractors(self) -> Dict[DocumentType, Any]:
        """Initialize format-specific extractors."""
        extractors = {}
        
        if HAS_PDFPLUMBER:
            extractors[DocumentType.PDF] = self._extract_pdf_metadata
        else:
            logger.warning("pdfplumber not installed. PDF metadata extraction disabled.")
        
        if HAS_DOCX:
            extractors[DocumentType.DOCX] = self._extract_docx_metadata
        else:
            logger.warning("python-docx not installed. DOCX metadata extraction disabled.")
        
        if HAS_EPUB:
            extractors[DocumentType.EPUB] = self._extract_epub_metadata
        else:
            logger.warning("ebooklib not installed. EPUB metadata extraction disabled.")
        
        # Always available extractors
        extractors[DocumentType.TXT] = self._extract_txt_metadata
        extractors[DocumentType.MD] = self._extract_md_metadata
        extractors[DocumentType.HTML] = self._extract_html_metadata
        extractors[DocumentType.JSON] = self._extract_json_metadata
        extractors[DocumentType.YAML] = self._extract_yaml_metadata
        extractors[DocumentType.CSV] = self._extract_csv_metadata
        extractors[DocumentType.PPTX] = self._extract_pptx_metadata
        extractors[DocumentType.XLSX] = self._extract_xlsx_metadata
        extractors[DocumentType.IMAGE] = self._extract_image_metadata
        
        return extractors
    
    def extract_metadata(self, file_path: Union[str, Path]) -> DocumentMetadata:
        """
        Extract metadata from a document file.
        
        Args:
            file_path: Path to the document file.
            
        Returns:
            DocumentMetadata object with extracted metadata.
            
        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If file type is not supported.
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Get basic file metadata
        file_stats = file_path.stat()
        file_type = self._detect_file_type(file_path)
        
        # Create base metadata object
        metadata = DocumentMetadata(
            file_path=str(file_path),
            file_name=file_path.name,
            file_type=file_type.value,
            file_size=file_stats.st_size,
            created_date=datetime.fromtimestamp(file_stats.st_ctime),
            modified_date=datetime.fromtimestamp(file_stats.st_mtime),
        )
        
        try:
            # Use format-specific extractor if available
            if file_type in self.extractors:
                extractor = self.extractors[file_type]
                format_metadata = extractor(file_path)
                
                # Merge extracted metadata
                for key, value in format_metadata.items():
                    if value is not None and value != "":
                        if hasattr(metadata, key):
                            setattr(metadata, key, value)
                        else:
                            metadata.custom_fields[key] = value
            
            # Apply fallback extraction for missing fields
            self._apply_fallback_extraction(file_path, metadata)
            
            # Calculate word count if not already done
            if metadata.word_count == 0:
                metadata.word_count = self._count_words_in_file(file_path)
            
            # Detect language if not specified
            if not metadata.language:
                metadata.language = self._detect_language(file_path)
            
            # Generate keywords if empty
            if not metadata.keywords:
                metadata.keywords = self._extract_keywords(file_path, metadata.title)
            
            # Set processing status
            metadata.processing_status = "completed"
            
        except Exception as e:
            metadata.processing_status = "failed"
            metadata.processing_error = str(e)
            logger.error(f"Error extracting metadata from {file_path}: {e}")
        
        return metadata
    
    def _detect_file_type(self, file_path: Path) -> DocumentType:
        """Detect document type from file extension."""
        extension = file_path.suffix.lower()
        
        type_mapping = {
            '.pdf': DocumentType.PDF,
            '.docx': DocumentType.DOCX,
            '.doc': DocumentType.DOCX,
            '.txt': DocumentType.TXT,
            '.text': DocumentType.TXT,
            '.epub': DocumentType.EPUB,
            '.md': DocumentType.MD,
            '.markdown': DocumentType.MD,
            '.html': DocumentType.HTML,
            '.htm': DocumentType.HTML,
            '.json': DocumentType.JSON,
            '.yaml': DocumentType.YAML,
            '.yml': DocumentType.YAML,
            '.pptx': DocumentType.PPTX,
            '.ppt': DocumentType.PPTX,
            '.xlsx': DocumentType.XLSX,
            '.xls': DocumentType.XLSX,
            '.csv': DocumentType.CSV,
            '.jpg': DocumentType.IMAGE,
            '.jpeg': DocumentType.IMAGE,
            '.png': DocumentType.IMAGE,
            '.gif': DocumentType.IMAGE,
            '.bmp': DocumentType.IMAGE,
            '.tiff': DocumentType.IMAGE,
            '.tif': DocumentType.IMAGE,
        }
        
        return type_mapping.get(extension, DocumentType.UNKNOWN)
    
    def _extract_pdf_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from PDF files."""
        metadata = {}
        
        try:
            with pdfplumber.open(file_path) as pdf:
                # Extract PDF info metadata
                if pdf.metadata:
                    pdf_info = pdf.metadata
                    
                    # Map PDF metadata fields to our schema
                    field_mapping = {
                        'Title': 'title',
                        'Author': 'author',
                        'Subject': 'abstract',
                        'Keywords': 'keywords',
                        'Producer': 'software',
                        'Creator': 'software',
                        'CreationDate': 'created_date',
                        'ModDate': 'modified_date',
                    }
                    
                    for pdf_field, our_field in field_mapping.items():
                        if pdf_field in pdf_info and pdf_info[pdf_field]:
                            value = pdf_info[pdf_field]
                            
                            # Parse dates
                            if 'Date' in pdf_field and isinstance(value, str):
                                try:
                                    value = self._parse_pdf_date(value)
                                except (ValueError, TypeError):
                                    pass
                            
                            metadata[our_field] = value
                
                # Extract page count
                metadata['page_count'] = len(pdf.pages)
                
                # Try to extract title from first page if not in metadata
                if 'title' not in metadata or not metadata['title']:
                    if pdf.pages:
                        first_page = pdf.pages[0]
                        text = first_page.extract_text()
                        if text:
                            # Look for title-like text (first line or centered text)
                            lines = text.split('\n')
                            if lines:
                                potential_title = lines[0].strip()
                                if len(potential_title) < 200 and self._looks_like_title(potential_title):
                                    metadata['title'] = potential_title
                
        except Exception as e:
            logger.warning(f"Error extracting PDF metadata from {file_path}: {e}")
        
        return metadata
    
    def _parse_pdf_date(self, pdf_date: str) -> Optional[datetime]:
        """Parse PDF date string to datetime."""
        try:
            # PDF dates format: D:YYYYMMDDHHmmSSOHH'mm'
            if pdf_date.startswith('D:'):
                date_str = pdf_date[2:]
                year = int(date_str[0:4])
                month = int(date_str[4:6]) if len(date_str) >= 6 else 1
                day = int(date_str[6:8]) if len(date_str) >= 8 else 1
                hour = int(date_str[8:10]) if len(date_str) >= 10 else 0
                minute = int(date_str[10:12]) if len(date_str) >= 12 else 0
                second = int(date_str[12:14]) if len(date_str) >= 14 else 0
                
                return datetime(year, month, day, hour, minute, second)
        except (ValueError, IndexError):
            pass
        
        return None
    
    def _extract_docx_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from DOCX files."""
        metadata = {}
        
        try:
            doc = DocxDocument(file_path)
            
            # Extract core properties
            core_props = doc.core_properties
            
            if core_props.title:
                metadata['title'] = core_props.title
            if core_props.author:
                metadata['author'] = core_props.author
            if core_props.subject:
                metadata['abstract'] = core_props.subject
            if core_props.keywords:
                metadata['keywords'] = core_props.keywords.split(',')
            if core_props.language:
                metadata['language'] = core_props.language
            if core_props.created:
                metadata['created_date'] = core_props.created
            if core_props.modified:
                metadata['modified_date'] = core_props.modified
            if core_props.last_modified_by:
                metadata['software'] = f"Microsoft Word ({core_props.last_modified_by})"
            
            # Extract page count (approximate)
            metadata['page_count'] = len(doc.element.xpath('//w:pgSz'))
            
            # Try to extract title from document if not in properties
            if 'title' not in metadata or not metadata['title']:
                # Look for heading 1 or large text at beginning
                for paragraph in doc.paragraphs[:10]:
                    if paragraph.text and len(paragraph.text) < 200:
                        if paragraph.style.name.startswith('Heading') or self._looks_like_title(paragraph.text):
                            metadata['title'] = paragraph.text.strip()
                            break
            
        except Exception as e:
            logger.warning(f"Error extracting DOCX metadata from {file_path}: {e}")
        
        return metadata
    
    def _extract_epub_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from EPUB files."""
        metadata = {}
        
        try:
            book = epub.read_epub(file_path)
            
            # Extract Dublin Core metadata
            dc_metadata = book.get_metadata('DC', {})
            
            # Map Dublin Core fields
            dc_mapping = {
                'title': 'title',
                'creator': 'author',
                'description': 'abstract',
                'publisher': 'publisher',
                'date': 'date',
                'language': 'language',
                'identifier': 'isbn',  # Might be ISBN
                'subject': 'keywords',
            }
            
            for dc_field, our_field in dc_mapping.items():
                if dc_field in dc_metadata:
                    values = dc_metadata[dc_field]
                    if values:
                        if our_field in ['keywords', 'categories']:
                            metadata[our_field] = values
                        else:
                            metadata[our_field] = values[0]
            
            # Extract page count from spine
            metadata['page_count'] = len(book.spine)
            
        except Exception as e:
            logger.warning(f"Error extracting EPUB metadata from {file_path}: {e}")
        
        return metadata
    
    def _extract_txt_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from plain text files."""
        metadata = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(5000)  # Read first 5000 chars for metadata extraction
                
                # Look for metadata patterns
                for field, patterns in self.patterns.items():
                    for pattern in patterns:
                        matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
                        if matches:
                            if field == 'keywords' or field == 'categories':
                                metadata[field] = list(set(matches))
                            else:
                                metadata[field] = matches[0]
                            break
                
                # Try to extract title from first few lines
                if 'title' not in metadata:
                    lines = content.split('\n')
                    for i, line in enumerate(lines[:10]):
                        line = line.strip()
                        if line and self._looks_like_title(line):
                            metadata['title'] = line
                            break
                
        except Exception as e:
            logger.warning(f"Error extracting TXT metadata from {file_path}: {e}")
        
        return metadata
    
    def _extract_md_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from Markdown files."""
        metadata = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Try frontmatter extraction if available
            if HAS_FRONTMATTER:
                try:
                    post = frontmatter.loads(content)
                    if post.metadata:
                        for key, value in post.metadata.items():
                            if key in ['title', 'author', 'date', 'categories', 'tags', 'keywords']:
                                metadata[key] = value
                except Exception:
                    pass
            
            # Fallback: Look for YAML frontmatter
            if not metadata and content.startswith('---'):
                yaml_end = content.find('---', 3)
                if yaml_end != -1:
                    yaml_content = content[3:yaml_end]
                    if HAS_YAML:
                        try:
                            yaml_metadata = yaml.safe_load(yaml_content)
                            if yaml_metadata:
                                for key, value in yaml_metadata.items():
                                    if key in ['title', 'author', 'date', 'categories', 'tags', 'keywords']:
                                        metadata[key] = value
                        except yaml.YAMLError:
                            pass
            
            # Extract title from first heading
            if 'title' not in metadata:
                heading_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
                if heading_match:
                    metadata['title'] = heading_match.group(1).strip()
            
        except Exception as e:
            logger.warning(f"Error extracting MD metadata from {file_path}: {e}")
        
        return metadata
    
    def _extract_html_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from HTML files."""
        metadata = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Extract from meta tags
            meta_patterns = {
                'title': r'<title>(.+?)</title>',
                'description': r'<meta\s+name=["\']description["\']\s+content=["\'](.+?)["\']',
                'keywords': r'<meta\s+name=["\']keywords["\']\s+content=["\'](.+?)["\']',
                'author': r'<meta\s+name=["\']author["\']\s+content=["\'](.+?)["\']',
                'date': r'<meta\s+name=["\']date["\']\s+content=["\'](.+?)["\']',
                'og:title': r'<meta\s+property=["\']og:title["\']\s+content=["\'](.+?)["\']',
            }
            
            for field, pattern in meta_patterns.items():
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    value = match.group(1).strip()
                    if field == 'keywords':
                        metadata['keywords'] = [k.strip() for k in value.split(',')]
                    elif field == 'og:title' and 'title' not in metadata:
                        metadata['title'] = value
                    elif field != 'og:title':
                        metadata[field] = value
            
            # Try to extract from h1 if title not found
            if 'title' not in metadata:
                h1_match = re.search(r'<h1[^>]*>(.+?)</h1>', content, re.IGNORECASE)
                if h1_match:
                    # Remove HTML tags from h1 content
                    h1_text = re.sub(r'<[^>]+>', '', h1_match.group(1)).strip()
                    if h1_text:
                        metadata['title'] = h1_text
            
        except Exception as e:
            logger.warning(f"Error extracting HTML metadata from {file_path}: {e}")
        
        return metadata
    
    def _extract_json_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from JSON files."""
        metadata = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Common JSON metadata field names
            common_fields = {
                'title': ['title', 'name', 'headline', 'subject'],
                'author': ['author', 'creator', 'by', 'writer'],
                'date': ['date', 'created', 'published', 'timestamp'],
                'abstract': ['abstract', 'summary', 'description', 'content'],
                'keywords': ['keywords', 'tags', 'categories', 'topics'],
            }
            
            def find_in_dict(data_dict, field_names):
                """Recursively find field in dictionary."""
                if isinstance(data_dict, dict):
                    for field in field_names:
                        if field in data_dict:
                            value = data_dict[field]
                            if value:
                                return value
                    # Recursively search nested dictionaries
                    for value in data_dict.values():
                        if isinstance(value, dict):
                            result = find_in_dict(value, field_names)
                            if result:
                                return result
                return None
            
            for metadata_field, json_fields in common_fields.items():
                value = find_in_dict(data, json_fields)
                if value:
                    metadata[metadata_field] = value
            
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning(f"Error parsing JSON metadata from {file_path}: {e}")
        except Exception as e:
            logger.warning(f"Error extracting JSON metadata from {file_path}: {e}")
        
        return metadata
    
    def _extract_yaml_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from YAML files."""
        metadata = {}
        
        if not HAS_YAML:
            return metadata
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            if isinstance(data, dict):
                # Common YAML metadata field names
                common_fields = {
                    'title': ['title', 'name', 'Title', 'Name'],
                    'author': ['author', 'Author', 'by', 'By'],
                    'date': ['date', 'Date', 'created', 'Created'],
                    'description': ['description', 'Description', 'summary', 'Summary'],
                    'tags': ['tags', 'Tags', 'keywords', 'Keywords'],
                }
                
                for metadata_field, yaml_fields in common_fields.items():
                    for field in yaml_fields:
                        if field in data:
                            value = data[field]
                            if value:
                                if metadata_field == 'tags' and isinstance(value, list):
                                    metadata['keywords'] = value
                                elif metadata_field == 'tags' and isinstance(value, str):
                                    metadata['keywords'] = [tag.strip() for tag in value.split(',')]
                                else:
                                    metadata[metadata_field] = value
                                break
            
        except yaml.YAMLError as e:
            logger.warning(f"Error parsing YAML metadata from {file_path}: {e}")
        except Exception as e:
            logger.warning(f"Error extracting YAML metadata from {file_path}: {e}")
        
        return metadata
    
    def _extract_csv_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from CSV files."""
        metadata = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                # Read first few lines to get column names
                lines = []
                for i in range(5):
                    line = f.readline()
                    if not line:
                        break
                    lines.append(line)
                
                if lines:
                    # Try to detect column names (usually first line)
                    first_line = lines[0].strip()
                    if first_line:
                        # Simple check: if first line looks like column headers
                        columns = first_line.split(',')
                        if len(columns) > 1 and all(col.strip() for col in columns):
                            metadata['columns'] = [col.strip() for col in columns]
                            metadata['row_count_estimate'] = sum(1 for _ in open(file_path)) - 1
        except Exception as e:
            logger.warning(f"Error extracting CSV metadata from {file_path}: {e}")
        
        return metadata
    
    def _extract_pptx_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from PowerPoint files."""
        metadata = {}
        
        try:
            from pptx import Presentation
            
            prs = Presentation(file_path)
            
            # Extract slide count
            metadata['page_count'] = len(prs.slides)
            
            # Try to extract title from first slide
            if prs.slides:
                first_slide = prs.slides[0]
                for shape in first_slide.shapes:
                    if hasattr(shape, "text") and shape.text:
                        text = shape.text.strip()
                        if text and self._looks_like_title(text):
                            metadata['title'] = text
                            break
            
        except ImportError:
            logger.warning("python-pptx not installed. PPTX metadata extraction disabled.")
        except Exception as e:
            logger.warning(f"Error extracting PPTX metadata from {file_path}: {e}")
        
        return metadata
    
    def _extract_xlsx_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from Excel files."""
        metadata = {}
        
        try:
            import pandas as pd
            
            # Read Excel file properties
            excel_file = pd.ExcelFile(file_path)
            
            # Extract sheet names
            metadata['sheets'] = excel_file.sheet_names
            
            # Try to get basic info from first sheet
            try:
                df = pd.read_excel(file_path, sheet_name=0, nrows=5)
                metadata['columns'] = df.columns.tolist()
                metadata['row_count_estimate'] = sum(1 for _ in open(file_path, 'rb')) // 100  # Rough estimate
            except:
                pass
            
        except ImportError:
            logger.warning("pandas not installed. Excel metadata extraction disabled.")
        except Exception as e:
            logger.warning(f"Error extracting Excel metadata from {file_path}: {e}")
        
        return metadata
    
    def _extract_image_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from image files."""
        metadata = {}
        
        try:
            from PIL import Image
            import PIL.ExifTags
            
            with Image.open(file_path) as img:
                # Basic image info
                metadata['width'] = img.width
                metadata['height'] = img.height
                metadata['format'] = img.format
                metadata['mode'] = img.mode
                
                # Try to extract EXIF data
                if hasattr(img, '_getexif') and img._getexif():
                    exif = img._getexif()
                    exif_tags = {
                        PIL.ExifTags.TAGS.get(tag, tag): value
                        for tag, value in exif.items()
                    }
                    
                    # Map common EXIF tags
                    exif_mapping = {
                        'DateTime': 'created_date',
                        'Artist': 'author',
                        'ImageDescription': 'description',
                        'Copyright': 'copyright',
                    }
                    
                    for exif_tag, our_field in exif_mapping.items():
                        if exif_tag in exif_tags:
                            value = exif_tags[exif_tag]
                            if value:
                                metadata[our_field] = value
                
        except ImportError:
            logger.warning("Pillow not installed. Image metadata extraction disabled.")
        except Exception as e:
            logger.warning(f"Error extracting image metadata from {file_path}: {e}")
        
        return metadata
    
    def _apply_fallback_extraction(self, file_path: Path, metadata: DocumentMetadata):
        """Apply fallback extraction methods for missing metadata."""
        try:
            # Read file content for pattern matching
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(10000)  # Read first 10k chars
            
            # Extract title if missing
            if not metadata.title:
                metadata.title = self._extract_title_from_content(content)
            
            # Extract author if missing
            if not metadata.author:
                metadata.author = self._extract_author_from_content(content)
            
            # Extract date if missing
            if not metadata.date:
                metadata.date = self._extract_date_from_content(content)
            
            # Extract additional metadata
            self._extract_additional_metadata(content, metadata)
            
        except Exception as e:
            logger.warning(f"Error in fallback extraction for {file_path}: {e}")
    
    def _extract_title_from_content(self, content: str) -> Optional[str]:
        """Extract title from content using various heuristics."""
        lines = content.split('\n')
        
        # Strategy 1: First non-empty line that looks like a title
        for line in lines[:20]:  # Check first 20 lines
            line = line.strip()
            if line and self._looks_like_title(line):
                return line
        
        # Strategy 2: Longest line in first 50 lines (often the title)
        first_lines = lines[:50]
        if first_lines:
            non_empty_lines = [line.strip() for line in first_lines if line.strip()]
            if non_empty_lines:
                return max(non_empty_lines, key=len)
        
        return None
    
    def _looks_like_title(self, text: str) -> bool:
        """Determine if text looks like a title."""
        if not text or len(text) > 200:
            return False
        
        # Check for common title characteristics
        words = text.split()
        
        # Too few words
        if len(words) < 2:
            return False
        
        # Check capitalization (titles often have major words capitalized)
        capitalized_words = sum(1 for word in words if word and word[0].isupper())
        capitalization_ratio = capitalized_words / len(words)
        
        # Check for common non-title patterns
        non_title_patterns = [
            r'^\d+\.',  # Numbered list
            r'^[A-Za-z]\)',  # Letter list
            r'^\s*$',  # Empty
            r'^[\-\*\+]\s',  # Bullet point
        ]
        
        for pattern in non_title_patterns:
            if re.match(pattern, text):
                return False
        
        # Titles typically have moderate capitalization
        return 0.3 < capitalization_ratio < 0.9
    
    def _extract_author_from_content(self, content: str) -> Optional[str]:
        """Extract author from content."""
        # Look for author patterns
        author_patterns = [
            r'by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})',
            r'Author[:\s]+([^\n]+)',
            r'©\s*\d{4}\s+([^\n]+)',
            r'Written by\s+([^\n]+)',
            r'Created by\s+([^\n]+)',
        ]
        
        for pattern in author_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                author = match.group(1).strip()
                if len(author.split()) <= 4:  # Reasonable name length
                    return author
        
        return None
    
    def _extract_date_from_content(self, content: str) -> Optional[datetime]:
        """Extract date from content."""
        date_patterns = [
            r'\b\d{4}-\d{2}-\d{2}\b',
            r'\b\d{2}/\d{2}/\d{4}\b',
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b',
            r'\b\d{4}\b',
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, content)
            if matches:
                for match in matches:
                    try:
                        # Try different date formats
                        for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%B %d, %Y', '%b %d, %Y', '%Y']:
                            try:
                                return datetime.strptime(match, fmt)
                            except ValueError:
                                continue
                    except (ValueError, TypeError):
                        continue
        
        return None
    
    def _extract_additional_metadata(self, content: str, metadata: DocumentMetadata):
        """Extract additional metadata fields from content."""
        # Extract emails
        email_matches = re.findall(self.patterns['email'][0], content)
        if email_matches:
            metadata.custom_fields['emails'] = list(set(email_matches))
        
        # Extract URLs
        url_matches = re.findall(self.patterns['url'][0], content)
        if url_matches:
            metadata.custom_fields['urls'] = list(set(url_matches))
        
        # Extract ISBN if present
        isbn_matches = re.findall(self.patterns['isbn'][0], content)
        if isbn_matches:
            metadata.custom_fields['isbn'] = isbn_matches[0]
        
        # Extract DOI if present
        doi_matches = re.findall(self.patterns['doi'][0], content, re.IGNORECASE)
        if doi_matches:
            metadata.custom_fields['doi'] = doi_matches[0]
    
    def _count_words_in_file(self, file_path: Path) -> int:
        """Count words in a file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                words = re.findall(r'\b\w+\b', content)
                return len(words)
        except Exception:
            return 0
    
    def _detect_language(self, file_path: Path) -> str:
        """Detect language of document content."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(5000)  # Sample first 5000 chars
                
                # Count matches for each language
                language_scores = {}
                for lang, patterns in self.language_patterns.items():
                    score = 0
                    for pattern in patterns:
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        score += len(matches)
                    language_scores[lang] = score
                
                # Return language with highest score
                if language_scores:
                    max_lang = max(language_scores.items(), key=lambda x: x[1])
                    if max_lang[1] > 0:  # Only return if we found matches
                        return max_lang[0]
                
        except Exception:
            pass
        
        return 'en'  # Default to English
    
    def _extract_keywords(self, file_path: Path, title: Optional[str] = None) -> List[str]:
        """Extract keywords from document content."""
        keywords = set()
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(10000)  # Sample first 10000 chars
            
            # Add words from title
            if title:
                title_words = re.findall(r'\b[A-Za-z]{4,}\b', title)
                keywords.update(title_words)
            
            # Extract capitalized phrases (potential proper nouns/terms)
            capitalized_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', content)
            keywords.update(phrase.lower() for phrase in capitalized_phrases[:10])
            
            # Extract words that appear multiple times (potential keywords)
            words = re.findall(r'\b[a-z]{4,}\b', content.lower())
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # Add frequent words (excluding common stopwords)
            common_words = {'the', 'and', 'that', 'for', 'with', 'this', 'have', 'from', 'they', 'which'}
            for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]:
                if freq >= 3 and word not in common_words:
                    keywords.add(word)
            
        except Exception:
            pass
        
        return list(keywords)[:10]  # Return top 10 keywords

# ========== CONVENIENCE FUNCTIONS ==========

def extract_metadata(file_path: Union[str, Path], config: Optional[Dict] = None) -> DocumentMetadata:
    """
    Convenience function for one-off metadata extraction.
    
    Args:
        file_path: Path to the document file.
        config: Optional extractor configuration.
        
    Returns:
        DocumentMetadata object.
    """
    extractor = MetadataExtractor(config)
    return extractor.extract_metadata(file_path)

def batch_extract_metadata(file_paths: List[Union[str, Path]], config: Optional[Dict] = None) -> List[DocumentMetadata]:
    """
    Extract metadata from multiple files.
    
    Args:
        file_paths: List of file paths.
        config: Optional extractor configuration.
        
    Returns:
        List of DocumentMetadata objects.
    """
    extractor = MetadataExtractor(config)
    results = []
    
    for file_path in file_paths:
        try:
            metadata = extractor.extract_metadata(file_path)
            results.append(metadata)
        except Exception as e:
            logger.error(f"Failed to extract metadata from {file_path}: {e}")
    
    return results

def metadata_to_json(metadata: DocumentMetadata, indent: int = 2) -> str:
    """Convert metadata to JSON string."""
    return metadata.to_json()

def metadata_from_json(json_str: str) -> DocumentMetadata:
    """Create metadata from JSON string."""
    data = json.loads(json_str)
    return DocumentMetadata.from_dict(data)

# ========== TESTING AND VALIDATION ==========

def test_metadata_extraction() -> Dict[str, bool]:
    """Test metadata extraction with sample files."""
    import tempfile
    
    test_results = {}
    
    # Create test files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write("""Test Document Title

By John Doe

Created: 2023-10-15

This is a test document for metadata extraction.
It contains some sample content for testing purposes.

Keywords: testing, metadata, extraction, python""")
        txt_file = f.name
    
    # Test TXT extraction
    try:
        metadata = extract_metadata(txt_file)
        test_results['txt_extraction'] = (
            metadata.title == "Test Document Title" and
            metadata.author == "John Doe" and
            metadata.file_type == "txt"
        )
    except Exception as e:
        test_results['txt_extraction'] = False
        logger.error(f"TXT extraction test failed: {e}")
    
    # Clean up
    try:
        os.unlink(txt_file)
    except:
        pass
    
    return test_results

# ========== MAIN MODULE EXECUTION ==========

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Metadata Extraction Module")
    parser.add_argument("--test", action="store_true", help="Run unit tests")
    parser.add_argument("--extract", type=str, help="Extract metadata from file")
    parser.add_argument("--batch", type=str, nargs='+', help="Extract metadata from multiple files")
    parser.add_argument("--output", type=str, choices=['json', 'yaml', 'text'], default='text',
                       help="Output format")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    if args.test:
        print("Running metadata extraction tests...")
        results = test_metadata_extraction()
        passed = sum(results.values())
        total = len(results)
        print(f"Tests passed: {passed}/{total} ({passed/total*100:.1f}%)")
        
        for test, success in results.items():
            status = "✓" if success else "✗"
            print(f"  {status} {test}")
    
    elif args.extract:
        file_path = Path(args.extract)
        
        if not file_path.exists():
            print(f"Error: File not found: {file_path}")
            exit(1)
        
        print(f"Extracting metadata from: {file_path}")
        
        try:
            metadata = extract_metadata(file_path)
            
            if args.output == 'json':
                print(metadata.to_json())
            elif args.output == 'yaml':
                if HAS_YAML:
                    import yaml
                    print(yaml.dump(metadata.to_dict(), allow_unicode=True, default_flow_style=False))
                else:
                    print("YAML output requires PyYAML. Install with: pip install pyyaml")
            else:
                print("\n=== METADATA EXTRACTION RESULTS ===")
                print(f"File: {metadata.file_name}")
                print(f"Type: {metadata.file_type}")
                print(f"Size: {metadata.file_size:,} bytes")
                
                if metadata.title:
                    print(f"Title: {metadata.title}")
                if metadata.author:
                    print(f"Author: {metadata.author}")
                if metadata.date:
                    print(f"Date: {metadata.date}")
                if metadata.language:
                    print(f"Language: {metadata.language}")
                if metadata.keywords:
                    print(f"Keywords: {', '.join(metadata.keywords)}")
                if metadata.word_count:
                    print(f"Word Count: {metadata.word_count:,}")
                if metadata.page_count:
                    print(f"Page Count: {metadata.page_count}")
                
                print(f"Processing Status: {metadata.processing_status}")
                if metadata.processing_error:
                    print(f"Error: {metadata.processing_error}")
        
        except Exception as e:
            print(f"Error extracting metadata: {e}")
    
    elif args.batch:
        file_paths = [Path(f) for f in args.batch]
        
        print(f"Extracting metadata from {len(file_paths)} files...")
        
        results = batch_extract_metadata(file_paths)
        
        print(f"\n=== BATCH EXTRACTION SUMMARY ===")
        print(f"Total files: {len(file_paths)}")
        print(f"Successfully extracted: {len(results)}")
        
        for metadata in results:
            print(f"\n{metadata.file_name}:")
            if metadata.title:
                print(f"  Title: {metadata.title}")
            print(f"  Status: {metadata.processing_status}")
    
    else:
        # Show usage examples
        print("=" * 70)
        print("METADATA EXTRACTION MODULE")
        print("=" * 70)
        print("\nUsage examples:")
        print("  python metadata.py --extract document.pdf")
        print("  python metadata.py --extract document.pdf --output json")
        print("  python metadata.py --batch file1.txt file2.pdf")
        print("  python metadata.py --test")
        print("\nAvailable document types:")
        print("  • PDF (.pdf)")
        print("  • DOCX (.docx, .doc)")
        print("  • TXT (.txt, .text, .rtf, .log)")
        print("  • EPUB (.epub)")
        print("  • Markdown (.md, .markdown)")
        print("  • HTML (.html, .htm)")
        print("  • JSON (.json)")
        print("  • YAML (.yaml, .yml)")
        print("  • PowerPoint (.pptx, .ppt)")
        print("  • Excel (.xlsx, .xls)")
        print("  • CSV (.csv)")
        print("  • Images (.jpg, .png, .gif, .bmp)")