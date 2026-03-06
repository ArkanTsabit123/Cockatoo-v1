# src/document_processing/extractors.py

"""Basic extractors for document processing."""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class BaseExtractor:
    """Base class for all extractors."""
    
    def extract(self, file_path: Path) -> str:
        """Extract text from file."""
        raise NotImplementedError


class TXTExtractor(BaseExtractor):
    """Extractor for text files."""
    
    def extract(self, file_path: Path) -> str:
        """Extract text from text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Try common encodings
            encodings = ['latin-1', 'cp1252', 'iso-8859-1']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
            # Last resort: read with errors ignored
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()


class PDFExtractor(BaseExtractor):
    """Placeholder PDF extractor."""
    
    def extract(self, file_path: Path) -> str:
        """Extract text from PDF."""
        logger.warning(f"PDF extraction not fully implemented for {file_path.name}")
        return f"[PDF content from {file_path.name} - extract with pdfplumber or PyPDF2 for full text]"


class DOCXExtractor(BaseExtractor):
    """Placeholder DOCX extractor."""
    
    def extract(self, file_path: Path) -> str:
        """Extract text from DOCX."""
        logger.warning(f"DOCX extraction not fully implemented for {file_path.name}")
        return f"[DOCX content from {file_path.name} - extract with python-docx for full text]"


class HTMLExtractor(BaseExtractor):
    """Placeholder HTML extractor."""
    
    def extract(self, file_path: Path) -> str:
        """Extract text from HTML."""
        logger.warning(f"HTML extraction not fully implemented for {file_path.name}")
        return f"[HTML content from {file_path.name} - extract with beautifulsoup4 for full text]"


class MarkdownExtractor(BaseExtractor):
    """Placeholder Markdown extractor."""
    
    def extract(self, file_path: Path) -> str:
        """Extract text from Markdown."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.warning(f"Failed to extract Markdown from {file_path.name}: {e}")
            return f"[Markdown content from {file_path.name}]"


class JSONExtractor(BaseExtractor):
    """Placeholder JSON extractor."""
    
    def extract(self, file_path: Path) -> str:
        """Extract text from JSON."""
        try:
            import json
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return json.dumps(data, indent=2)
        except Exception as e:
            logger.warning(f"Failed to extract JSON from {file_path.name}: {e}")
            return f"[JSON content from {file_path.name}]"


class CSVExtractor(BaseExtractor):
    """Placeholder CSV extractor."""
    
    def extract(self, file_path: Path) -> str:
        """Extract text from CSV."""
        try:
            import csv
            rows = []
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    rows.append(', '.join(row))
            return '\n'.join(rows)
        except Exception as e:
            logger.warning(f"Failed to extract CSV from {file_path.name}: {e}")
            return f"[CSV content from {file_path.name}]"


class EPUBExtractor(BaseExtractor):
    """Placeholder EPUB extractor."""
    
    def extract(self, file_path: Path) -> str:
        """Extract text from EPUB."""
        logger.warning(f"EPUB extraction not fully implemented for {file_path.name}")
        return f"[EPUB content from {file_path.name} - extract with ebooklib for full text]"


class ImageExtractor(BaseExtractor):
    """Placeholder Image extractor."""
    
    def extract(self, file_path: Path) -> str:
        """Extract text from Image."""
        logger.warning(f"Image extraction not fully implemented for {file_path.name}")
        return f"[Image content from {file_path.name} - use OCR (pytesseract) for text extraction]"


class PPTXExtractor(BaseExtractor):
    """Placeholder PowerPoint extractor."""
    
    def extract(self, file_path: Path) -> str:
        """Extract text from PowerPoint."""
        logger.warning(f"PPTX extraction not fully implemented for {file_path.name}")
        return f"[PPTX content from {file_path.name} - extract with python-pptx for full text]"


class XLSXExtractor(BaseExtractor):
    """Placeholder Excel extractor."""
    
    def extract(self, file_path: Path) -> str:
        """Extract text from Excel."""
        logger.warning(f"XLSX extraction not fully implemented for {file_path.name}")
        return f"[XLSX content from {file_path.name} - extract with openpyxl or pandas for full text]"


class WebExtractor(BaseExtractor):
    """Placeholder Web extractor."""
    
    def extract(self, url: str) -> str:
        """Extract text from web URL."""
        logger.warning(f"Web extraction not fully implemented for {url}")
        return f"[Web content from {url} - fetch with requests and parse with beautifulsoup4 for full text]"


def get_extractor_for_file(file_path: Path) -> BaseExtractor:
    """Get appropriate extractor for file."""
    file_path = Path(file_path)
    file_str = str(file_path)
    
    # Check if it's a URL
    if file_str.startswith(('http://', 'https://', 'www.')):
        return WebExtractor()
    
    ext = file_path.suffix.lower()
    
    extractors = {
        '.txt': TXTExtractor,
        '.text': TXTExtractor,
        '.md': MarkdownExtractor,
        '.markdown': MarkdownExtractor,
        '.py': TXTExtractor,
        '.json': JSONExtractor,
        '.yaml': TXTExtractor,
        '.yml': TXTExtractor,
        '.xml': TXTExtractor,
        '.pdf': PDFExtractor,
        '.docx': DOCXExtractor,
        '.doc': DOCXExtractor,
        '.html': HTMLExtractor,
        '.htm': HTMLExtractor,
        '.csv': CSVExtractor,
        '.epub': EPUBExtractor,
        '.jpg': ImageExtractor,
        '.jpeg': ImageExtractor,
        '.png': ImageExtractor,
        '.gif': ImageExtractor,
        '.bmp': ImageExtractor,
        '.tiff': ImageExtractor,
        '.webp': ImageExtractor,
        '.pptx': PPTXExtractor,
        '.ppt': PPTXExtractor,
        '.xlsx': XLSXExtractor,
        '.xls': XLSXExtractor,
    }
    
    extractor_class = extractors.get(ext, TXTExtractor)
    return extractor_class()


def get_supported_extensions() -> Dict[str, List[str]]:
    """Get supported file extensions."""
    return {
        'text': ['.txt', '.py', '.json', '.yaml', '.yml', '.xml'],
        'markdown': ['.md', '.markdown'],
        'pdf': ['.pdf'],
        'document': ['.docx', '.doc'],
        'spreadsheet': ['.xlsx', '.xls', '.csv'],
        'presentation': ['.pptx', '.ppt'],
        'image': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'],
        'ebook': ['.epub'],
        'html': ['.html', '.htm'],
        'web': ['http://', 'https://', 'www.'],
    }


def validate_file_extension(file_path: Path) -> bool:
    """Validate if file extension is supported."""
    file_path = Path(file_path)
    file_str = str(file_path)
    
    # Check for URLs
    if file_str.startswith(('http://', 'https://', 'www.')):
        return True
    
    ext = file_path.suffix.lower()
    for extensions in get_supported_extensions().values():
        if ext in extensions:
            return True
    return False


def get_registry_stats() -> Dict[str, Any]:
    """Get extractor registry statistics."""
    return {
        'total_extractors': 12,
        'supported_formats': list(get_supported_extensions().keys()),
        'total_extensions': sum(len(ext) for ext in get_supported_extensions().values()),
    }