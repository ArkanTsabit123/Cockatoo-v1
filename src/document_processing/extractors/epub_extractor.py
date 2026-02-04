# cockatoo_v1/src/document_processing/extractors/epub_extractor.py

"""
EPUB ebook extractor using EbookLib.
"""

import os
import logging
from typing import Dict, Any, List, Union, Optional
from pathlib import Path
import zipfile
import tempfile
import shutil

from .base_extractor import BaseExtractor

logger = logging.getLogger(__name__)

try:
    import ebooklib
    from ebooklib import epub
    HAS_EBOOKLIB = True
except ImportError:
    HAS_EBOOKLIB = False
    logger.warning("EbookLib not installed. EPUB support will be limited.")

try:
    from bs4 import BeautifulSoup
    HAS_BEAUTIFULSOUP = True
except ImportError:
    HAS_BEAUTIFULSOUP = False


class EPUBExtractor(BaseExtractor):
    """
    EPUB ebook extractor.
    """
    
    def __init__(self):
        """Initialize EPUB extractor."""
        super().__init__()
        if not HAS_EBOOKLIB:
            self.logger.warning(
                "EbookLib is not installed. Install with: pip install EbookLib"
            )
        if not HAS_BEAUTIFULSOUP:
            self.logger.warning(
                "BeautifulSoup4 is not installed. HTML parsing will be limited. "
                "Install with: pip install beautifulsoup4"
            )
    
    def get_supported_formats(self) -> List[str]:
        """
        Return list of supported file formats.
        
        Returns:
            List of supported file extensions
        """
        return ['.epub', '.epub3']
    
    def extract(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Extract text and metadata from EPUB file.
        
        Args:
            file_path: Path to EPUB file
        
        Returns:
            Dictionary containing:
                - text: Extracted text content
                - metadata: EPUB metadata
                - chapters: List of chapters with text
                - table_of_contents: Table of contents structure
                - images: Information about images
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"EPUB file not found: {file_path}")
        
        logger.info(f"Extracting EPUB: {file_path}")
        
        result = {
            "text": "",
            "metadata": self.get_basic_metadata(file_path),
            "chapters": [],
            "table_of_contents": [],
            "images": [],
            "styles": [],
            "fonts": [],
            "extraction_method": "ebooklib" if HAS_EBOOKLIB else "zip_parsing",
        }
        
        if HAS_EBOOKLIB:
            try:
                result = self._extract_with_ebooklib(file_path, result)
            except Exception as e:
                logger.error(f"EbookLib extraction failed: {e}")
                result = self._extract_with_zip(file_path, result)
        else:
            result = self._extract_with_zip(file_path, result)
        
        # Clean and post-process
        result["text"] = self.clean_text(result["text"])
        
        # Add language detection and summary
        if result["text"]:
            result["metadata"]["language"] = self.detect_language(result["text"])
            result["metadata"]["summary"] = self.extract_summary(result["text"])
        
        return result
    
    def _extract_with_ebooklib(self, file_path: Path, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract using EbookLib library.
        
        Args:
            file_path: Path to EPUB file
            result: Result dictionary to update
            
        Returns:
            Updated result dictionary
        """
        book = epub.read_epub(str(file_path))
        
        # Extract metadata
        metadata = {}
        
        # Dublin Core metadata
        if hasattr(book, 'metadata'):
            for key, values in book.metadata.items():
                if values:
                    metadata[key] = [str(v) for v in values]
        
        # Map common metadata fields
        result["metadata"].update({
            "title": self._get_metadata_value(book, 'title'),
            "creator": self._get_metadata_value(book, 'creator'),
            "author": self._get_metadata_value(book, 'creator'),  # Alias
            "publisher": self._get_metadata_value(book, 'publisher'),
            "description": self._get_metadata_value(book, 'description'),
            "language": self._get_metadata_value(book, 'language'),
            "identifier": self._get_metadata_value(book, 'identifier'),
            "date": self._get_metadata_value(book, 'date'),
            "rights": self._get_metadata_value(book, 'rights'),
            "subject": self._get_metadata_value(book, 'subject', multi=True),
            "type": self._get_metadata_value(book, 'type'),
        })
        
        # Extract text from all items
        chapters = []
        all_text = []
        
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                # Parse HTML/XML content
                content = item.get_content().decode('utf-8', errors='ignore')
                
                # Extract text using BeautifulSoup if available
                if HAS_BEAUTIFULSOUP:
                    soup = BeautifulSoup(content, 'html.parser')
                    
                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()
                    
                    text = soup.get_text(separator='\n', strip=True)
                else:
                    # Simple regex-based text extraction
                    import re
                    text = re.sub(r'<[^>]+>', ' ', content)
                    text = re.sub(r'\s+', ' ', text).strip()
                
                if text.strip():
                    chapter_info = {
                        "id": item.get_id(),
                        "file_name": item.get_name(),
                        "media_type": item.get_media_type(),
                        "text": text,
                        "text_length": len(text),
                        "word_count": len(text.split()),
                    }
                    
                    chapters.append(chapter_info)
                    all_text.append(text)
        
        result["chapters"] = chapters
        result["text"] = "\n\n".join(all_text)
        
        # Extract table of contents
        toc = []
        try:
            if book.toc:
                toc = self._extract_toc_structure(book.toc)
        except:
            pass
        
        result["table_of_contents"] = toc
        
        # Count images and other resources
        images = []
        styles = []
        fonts = []
        
        for item in book.get_items():
            item_type = item.get_type()
            media_type = item.get_media_type()
            
            if item_type == ebooklib.ITEM_IMAGE:
                images.append({
                    "id": item.get_id(),
                    "file_name": item.get_name(),
                    "media_type": media_type,
                    "size": len(item.get_content()) if hasattr(item, 'get_content') else 0,
                })
            elif item_type == ebooklib.ITEM_STYLE:
                styles.append({
                    "id": item.get_id(),
                    "file_name": item.get_name(),
                    "media_type": media_type,
                })
            elif media_type in ['application/vnd.ms-opentype', 
                              'application/font-woff',
                              'application/font-sfnt']:
                fonts.append({
                    "id": item.get_id(),
                    "file_name": item.get_name(),
                    "media_type": media_type,
                })
        
        result["images"] = images
        result["styles"] = styles
        result["fonts"] = fonts
        
        # Extract cover image if available
        cover = book.get_metadata('OPF', 'cover')
        if cover:
            result["metadata"]["cover_id"] = cover[0][0]
        
        return result
    
    def _get_metadata_value(self, book, field: str, multi: bool = False) -> Union[str, List[str], None]:
        """
        Get metadata value from book.
        
        Args:
            book: EbookLib book object
            field: Metadata field name
            multi: Whether to return multiple values
            
        Returns:
            Metadata value(s)
        """
        try:
            values = book.get_metadata('DC', field)
            if not values:
                return [] if multi else ""
            
            if multi:
                return [str(v[0]) for v in values]
            else:
                return str(values[0][0])
        except:
            return [] if multi else ""
    
    def _extract_toc_structure(self, toc_items, level: int = 0) -> List[Dict[str, Any]]:
        """
        Recursively extract table of contents structure.
        
        Args:
            toc_items: TOC items from EbookLib
            level: Current depth level
            
        Returns:
            List of TOC entries
        """
        structure = []
        
        for item in toc_items:
            if isinstance(item, tuple) or isinstance(item, list):
                # EbookLib TOC entry: (section, href, title, children)
                if len(item) >= 3:
                    toc_entry = {
                        "title": item[2] if len(item) > 2 else "",
                        "href": item[1] if len(item) > 1 else "",
                        "level": level,
                        "children": [],
                    }
                    
                    # Recursively process children
                    if len(item) > 3:
                        toc_entry["children"] = self._extract_toc_structure(item[3], level + 1)
                    
                    structure.append(toc_entry)
            elif hasattr(item, 'title'):
                # Alternative TOC structure
                toc_entry = {
                    "title": getattr(item, 'title', ''),
                    "href": getattr(item, 'href', ''),
                    "level": level,
                    "children": [],
                }
                
                if hasattr(item, 'children'):
                    toc_entry["children"] = self._extract_toc_structure(item.children, level + 1)
                
                structure.append(toc_entry)
        
        return structure
    
    def _extract_with_zip(self, file_path: Path, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback extraction by parsing EPUB as ZIP archive.
        
        Args:
            file_path: Path to EPUB file
            result: Result dictionary to update
            
        Returns:
            Updated result dictionary
        """
        all_text = []
        
        try:
            with zipfile.ZipFile(file_path, 'r') as epub_zip:
                # List all files in EPUB
                file_list = epub_zip.namelist()
                
                # Find OPF file (container)
                container_xml = None
                if 'META-INF/container.xml' in file_list:
                    with epub_zip.open('META-INF/container.xml') as f:
                        container_xml = f.read().decode('utf-8', errors='ignore')
                
                # Extract text from XHTML/HTML files
                text_files = [f for f in file_list if f.endswith(('.xhtml', '.html', '.htm', '.xml'))]
                
                for text_file in text_files[:50]:  # Limit to first 50 files
                    try:
                        with epub_zip.open(text_file) as f:
                            content = f.read().decode('utf-8', errors='ignore')
                            
                            # Simple text extraction
                            import re
                            text = re.sub(r'<[^>]+>', ' ', content)
                            text = re.sub(r'\s+', ' ', text).strip()
                            
                            if text:
                                all_text.append(text)
                    except:
                        continue
                
                result["text"] = " ".join(all_text)
                
        except zipfile.BadZipFile:
            logger.error(f"File is not a valid ZIP archive: {file_path}")
            result["text"] = ""
        except Exception as e:
            logger.error(f"Failed to parse EPUB as ZIP: {e}")
            result["text"] = ""
        
        return result
    
    def extract_chapter(self, file_path: Union[str, Path], chapter_id: str) -> Dict[str, Any]:
        """
        Extract specific chapter from EPUB.
        
        Args:
            file_path: Path to EPUB file
            chapter_id: Chapter identifier
            
        Returns:
            Chapter content
        """
        if not HAS_EBOOKLIB:
            return {"error": "EbookLib required for chapter extraction"}
        
        file_path = Path(file_path)
        chapter_info = {}
        
        try:
            book = epub.read_epub(str(file_path))
            
            item = book.get_item_with_id(chapter_id)
            if item:
                content = item.get_content().decode('utf-8', errors='ignore')
                
                if HAS_BEAUTIFULSOUP:
                    soup = BeautifulSoup(content, 'html.parser')
                    
                    # Get chapter title
                    title_elem = soup.find(['h1', 'h2', 'h3', 'title'])
                    title = title_elem.get_text() if title_elem else ""
                    
                    # Extract text
                    for script in soup(["script", "style"]):
                        script.decompose()
                    
                    text = soup.get_text(separator='\n', strip=True)
                    
                    chapter_info = {
                        "id": chapter_id,
                        "title": title,
                        "text": text,
                        "html_content": content[:1000] + "..." if len(content) > 1000 else content,
                        "word_count": len(text.split()),
                    }
                
        except Exception as e:
            logger.error(f"Failed to extract chapter: {e}")
            chapter_info = {"error": str(e)}
        
        return chapter_info