# cockatoo_v1/src/document_processing/extractors/pdf_extractor.py

"""
PDF document extractor using PyPDF2 and pdfplumber for text extraction.
"""

import os
import logging
import re
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

from ..extractors.base_extractor import BaseExtractor

logger = logging.getLogger(__name__)


class PDFExtractor(BaseExtractor):
    """
    PDF document extractor that combines PyPDF2 and pdfplumber
    for optimal text extraction.
    """
    
    def __init__(self, use_pdfplumber: bool = True, use_pypdf2: bool = True):
        """
        Initialize PDF extractor.
        
        Args:
            use_pdfplumber: Use pdfplumber for detailed text extraction
            use_pypdf2: Use PyPDF2 for metadata and fallback extraction
        """
        super().__init__()
        self.use_pdfplumber = use_pdfplumber and HAS_PDFPLUMBER
        self.use_pypdf2 = use_pypdf2 and HAS_PYPDF2
        
        if not self.use_pdfplumber and not self.use_pypdf2:
            raise ImportError(
                "Neither PyPDF2 nor pdfplumber is available. "
                "Please install at least one: pip install pypdf2 pdfplumber"
            )
    
    def get_supported_formats(self) -> List[str]:
        """
        Return list of supported file formats.
        
        Returns:
            List of supported file extensions
        """
        return ['.pdf', '.PDF']  # Both lowercase and uppercase
    
    def validate_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Validate if file is a valid PDF and can be processed.
        
        Args:
            file_path: Path to file
            
        Returns:
            Validation results dictionary
        """
        return self.validate_pdf(file_path)
    
    def extract(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Extract text and metadata from PDF file.
        
        Args:
            file_path: Path to PDF file
        
        Returns:
            Dictionary containing:
                - text: Extracted text content
                - metadata: PDF metadata
                - page_count: Number of pages
                - has_tables: Whether PDF contains tables
                - extraction_method: Method used for extraction
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        logger.info(f"Extracting PDF: {file_path}")
        
        result = {
            "text": "",
            "metadata": {},
            "page_count": 0,
            "has_tables": False,
            "extraction_method": "none",
            "pages": []
        }
        
        # Extract with pdfplumber (preferred for text quality)
        if self.use_pdfplumber:
            try:
                result = self._extract_with_pdfplumber(file_path, result)
                result["extraction_method"] = "pdfplumber"
            except Exception as e:
                logger.warning(f"pdfplumber extraction failed: {e}")
                if self.use_pypdf2:
                    logger.info("Falling back to PyPDF2")
        
        # Extract with PyPDF2 (fallback or for metadata)
        if result["text"] == "" and self.use_pypdf2:
            try:
                result = self._extract_with_pypdf2(file_path, result)
                if result["extraction_method"] == "none":
                    result["extraction_method"] = "pypdf2"
            except Exception as e:
                logger.error(f"PyPDF2 extraction also failed: {e}")
        
        if result["text"] == "":
            logger.error(f"Failed to extract text from PDF: {file_path}")
        
        # Clean the extracted text
        if result["text"]:
            result["text"] = self.clean_text(result["text"])
        
        return result
    
    def _extract_with_pdfplumber(self, file_path: Path, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract text using pdfplumber.
        
        Args:
            file_path: Path to PDF file
            result: Result dictionary to update
        
        Returns:
            Updated result dictionary
        """
        extracted_text = []
        tables_found = False
        
        with pdfplumber.open(file_path) as pdf:
            result["page_count"] = len(pdf.pages)
            
            for page_num, page in enumerate(pdf.pages):
                page_text = ""
                
                # Extract text
                page_text = page.extract_text() or ""
                
                # Check for tables
                tables = page.extract_tables()
                if tables:
                    tables_found = True
                    # Add table text
                    for table in tables:
                        for row in table:
                            if row:
                                # Filter out None values and join
                                row_text = " | ".join(str(cell) for cell in row if cell)
                                page_text += f"\n{row_text}\n"
                
                # Store page information
                result["pages"].append({
                    "page_number": page_num + 1,
                    "text": page_text,
                    "width": page.width,
                    "height": page.height,
                    "has_tables": bool(tables)
                })
                
                extracted_text.append(page_text)
            
            # Extract metadata
            try:
                result["metadata"] = {
                    "producer": pdf.metadata.get("Producer", ""),
                    "creator": pdf.metadata.get("Creator", ""),
                    "creation_date": pdf.metadata.get("CreationDate", ""),
                    "modification_date": pdf.metadata.get("ModDate", ""),
                    "title": pdf.metadata.get("Title", ""),
                    "author": pdf.metadata.get("Author", ""),
                    "subject": pdf.metadata.get("Subject", ""),
                    "keywords": pdf.metadata.get("Keywords", ""),
                }
            except:
                result["metadata"] = {}
        
        result["text"] = "\n\n".join(extracted_text)
        result["has_tables"] = tables_found
        
        return result
    
    def _extract_with_pypdf2(self, file_path: Path, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract text using PyPDF2.
        
        Args:
            file_path: Path to PDF file
            result: Result dictionary to update
        
        Returns:
            Updated result dictionary
        """
        extracted_text = []
        
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            result["page_count"] = len(pdf_reader.pages)
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text() or ""
                
                result["pages"].append({
                    "page_number": page_num + 1,
                    "text": page_text
                })
                
                extracted_text.append(page_text)
            
            # Extract metadata
            try:
                metadata = pdf_reader.metadata
                result["metadata"] = {
                    "producer": metadata.get("/Producer", "") if metadata else "",
                    "creator": metadata.get("/Creator", "") if metadata else "",
                    "creation_date": metadata.get("/CreationDate", "") if metadata else "",
                    "title": metadata.get("/Title", "") if metadata else "",
                    "author": metadata.get("/Author", "") if metadata else "",
                    "subject": metadata.get("/Subject", "") if metadata else "",
                }
            except:
                result["metadata"] = {}
        
        result["text"] = "\n\n".join(extracted_text)
        
        return result
    
    def extract_metadata_only(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Extract only metadata from PDF (faster than full extraction).
        
        Args:
            file_path: Path to PDF file
        
        Returns:
            Dictionary with PDF metadata
        """
        file_path = Path(file_path)
        metadata = {}
        
        try:
            if self.use_pypdf2:
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    if pdf_reader.metadata:
                        raw_metadata = pdf_reader.metadata
                        metadata = {
                            "producer": raw_metadata.get("/Producer", ""),
                            "creator": raw_metadata.get("/Creator", ""),
                            "creation_date": raw_metadata.get("/CreationDate", ""),
                            "title": raw_metadata.get("/Title", ""),
                            "author": raw_metadata.get("/Author", ""),
                            "subject": raw_metadata.get("/Subject", ""),
                            "page_count": len(pdf_reader.pages),
                            "encrypted": pdf_reader.is_encrypted,
                        }
        except Exception as e:
            logger.warning(f"Failed to extract metadata with PyPDF2: {e}")
        
        return metadata
    
    def extract_images(self, file_path: Union[str, Path], output_dir: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
        """
        Extract images from PDF (requires pdfplumber).
        
        Args:
            file_path: Path to PDF file
            output_dir: Directory to save extracted images
        
        Returns:
            List of extracted image information
        """
        if not self.use_pdfplumber:
            raise RuntimeError("Image extraction requires pdfplumber")
        
        images = []
        file_path = Path(file_path)
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_images = page.images
                
                for img_num, img in enumerate(page_images):
                    img_info = {
                        "page": page_num + 1,
                        "image_number": img_num + 1,
                        "x0": img["x0"],
                        "y0": img["y0"],
                        "x1": img["x1"],
                        "y1": img["y1"],
                        "width": img["width"],
                        "height": img["height"],
                        "stream": img.get("stream")
                    }
                    
                    # Save image if output directory specified
                    if output_dir and img.get("stream"):
                        img_filename = f"page_{page_num+1}_img_{img_num+1}.png"
                        img_path = output_dir / img_filename
                        
                        try:
                            # This is simplified - actual image extraction needs more work
                            img_info["saved_path"] = str(img_path)
                        except Exception as e:
                            logger.warning(f"Failed to save image: {e}")
                    
                    images.append(img_info)
        
        return images
    
    def validate_pdf(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Validate PDF file structure and content.
        
        Args:
            file_path: Path to PDF file
        
        Returns:
            Validation results
        """
        file_path = Path(file_path)
        validation = {
            "is_valid": False,
            "errors": [],
            "warnings": [],
            "page_count": 0,
            "file_size": file_path.stat().st_size if file_path.exists() else 0,
            "is_encrypted": False,
            "has_text": False,
        }
        
        if not file_path.exists():
            validation["errors"].append("File does not exist")
            return validation
        
        # Check with PyPDF2
        if self.use_pypdf2:
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    
                    validation["page_count"] = len(pdf_reader.pages)
                    validation["is_encrypted"] = pdf_reader.is_encrypted
                    
                    # Check if any page has text
                    for page in pdf_reader.pages[:3]:  # Check first 3 pages
                        if page.extract_text().strip():
                            validation["has_text"] = True
                            break
                    
                    validation["is_valid"] = True
                    
                    if validation["is_encrypted"]:
                        validation["warnings"].append("PDF is encrypted, text extraction may fail")
                    
                    if not validation["has_text"]:
                        validation["warnings"].append("PDF appears to have no extractable text (may be scanned)")
            
            except Exception as e:
                validation["errors"].append(f"PyPDF2 validation failed: {e}")
        
        return validation
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted PDF text - remove excessive whitespace and normalize.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace and normalize line endings
        text = ' '.join(text.split())
        
        # Remove control characters (keep only printable chars)
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
        
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove multiple consecutive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def clean_text(self, text: str) -> str:
        """
        Public method to clean text (overrides base class).
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        return self._clean_text(text)


# ====== FACTORY FUNCTIONS ======

def create_pdf_extractor(use_pdfplumber: bool = True, use_pypdf2: bool = True) -> PDFExtractor:
    """
    Factory function to create PDF extractor instance.
    
    Args:
        use_pdfplumber: Whether to use pdfplumber
        use_pypdf2: Whether to use PyPDF2
    
    Returns:
        PDFExtractor instance
    """
    return PDFExtractor(use_pdfplumber=use_pdfplumber, use_pypdf2=use_pypdf2)


# Register with the extractor system
_EXTRACTOR_REGISTRY = {
    'pdf': create_pdf_extractor,
    'PDF': create_pdf_extractor,
}