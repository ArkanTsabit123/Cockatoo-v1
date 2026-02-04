# cockatoo_v1/src/document_processing/extractors/base_extractor.py

"""
Base extractor class providing common functionality for all document extractors.
"""

import os
import logging
import mimetypes
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)


class BaseExtractor(ABC):
    """
    Abstract base class for document extractors.
    All document extractors should inherit from this class.
    """
    
    def __init__(self):
        """Initialize the base extractor."""
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """
        Return list of supported file formats/extensions.
        
        Returns:
            List of supported file extensions (e.g., ['.pdf', '.PDF'])
        """
        pass
    
    @abstractmethod
    def extract(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Extract text and metadata from a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary containing:
                - text: Extracted text content
                - metadata: File metadata
                - Additional format-specific information
        """
        pass
    
    def validate_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Validate if file can be processed by this extractor.
        
        Args:
            file_path: Path to file
            
        Returns:
            Dictionary with validation results:
                - is_valid: Boolean indicating if file is valid
                - errors: List of error messages
                - warnings: List of warning messages
                - file_size: Size in bytes
                - mime_type: MIME type if detectable
        """
        file_path = Path(file_path)
        validation = {
            "is_valid": False,
            "errors": [],
            "warnings": [],
            "file_size": 0,
            "mime_type": None,
            "supported": False,
        }
        
        # Check if file exists
        if not file_path.exists():
            validation["errors"].append(f"File does not exist: {file_path}")
            return validation
        
        # Check file size
        try:
            file_size = file_path.stat().st_size
            validation["file_size"] = file_size
            
            if file_size == 0:
                validation["errors"].append("File is empty")
                return validation
            
            # Check if file is too large (100MB limit)
            if file_size > 100 * 1024 * 1024:  # 100MB
                validation["warnings"].append(
                    f"File size ({file_size / (1024*1024):.2f} MB) exceeds recommended limit"
                )
        except Exception as e:
            validation["errors"].append(f"Cannot read file size: {e}")
            return validation
        
        # Check file extension
        file_extension = file_path.suffix.lower()
        supported_formats = self.get_supported_formats()
        
        if file_extension not in [ext.lower() for ext in supported_formats]:
            validation["errors"].append(
                f"File extension '{file_extension}' not supported. "
                f"Supported formats: {supported_formats}"
            )
            return validation
        
        # Detect MIME type
        try:
            mime_type, _ = mimetypes.guess_type(str(file_path))
            validation["mime_type"] = mime_type
        except:
            pass
        
        # File passed basic validation
        validation["is_valid"] = True
        validation["supported"] = True
        
        return validation
    
    def calculate_file_hash(self, file_path: Union[str, Path]) -> str:
        """
        Calculate MD5 hash of file for identification.
        
        Args:
            file_path: Path to file
            
        Returns:
            MD5 hash string
        """
        file_path = Path(file_path)
        hash_md5 = hashlib.md5()
        
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            self.logger.error(f"Failed to calculate file hash: {e}")
            return ""
    
    def get_basic_metadata(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get basic file metadata (size, timestamps, etc.).
        
        Args:
            file_path: Path to file
            
        Returns:
            Dictionary with basic metadata
        """
        file_path = Path(file_path)
        metadata = {}
        
        try:
            stat_info = file_path.stat()
            
            metadata = {
                "file_name": file_path.name,
                "file_path": str(file_path.resolve()),
                "file_size": stat_info.st_size,
                "file_size_human": self._format_file_size(stat_info.st_size),
                "created": stat_info.st_ctime,
                "modified": stat_info.st_mtime,
                "accessed": stat_info.st_atime,
                "file_hash": self.calculate_file_hash(file_path),
                "file_extension": file_path.suffix.lower(),
                "parent_directory": str(file_path.parent),
            }
        except Exception as e:
            self.logger.error(f"Failed to get basic metadata: {e}")
        
        return metadata
    
    def _format_file_size(self, size_bytes: int) -> str:
        """
        Format file size in human-readable format.
        
        Args:
            size_bytes: Size in bytes
            
        Returns:
            Formatted string (e.g., "1.5 MB")
        """
        if size_bytes == 0:
            return "0 B"
        
        units = ["B", "KB", "MB", "GB", "TB"]
        unit_index = 0
        
        while size_bytes >= 1024 and unit_index < len(units) - 1:
            size_bytes /= 1024
            unit_index += 1
        
        return f"{size_bytes:.2f} {units[unit_index]}"
    
    def clean_text(self, text: str) -> str:
        """
        Clean extracted text by removing extra whitespace, etc.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove null characters
        text = text.replace('\x00', '')
        
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove multiple consecutive newlines
        import re
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Remove extra spaces
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def extract_summary(self, text: str, max_length: int = 500) -> str:
        """
        Extract a simple summary from text (first few sentences).
        
        Args:
            text: Full text
            max_length: Maximum summary length
            
        Returns:
            Summary text
        """
        if not text:
            return ""
        
        # Simple sentence splitting
        import re
        sentences = re.split(r'[.!?]+', text)
        
        summary = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                if len(summary) + len(sentence) < max_length:
                    summary += sentence + ". "
                else:
                    break
        
        return summary.strip()
    
    def detect_language(self, text: str) -> str:
        """
        Detect language of text (simplified version).
        
        Args:
            text: Text to analyze
            
        Returns:
            Language code (e.g., 'en', 'id', 'unknown')
        """
        if not text or len(text) < 20:
            return "unknown"
        
        # Simple keyword-based detection
        # This is a simplified version - consider using langdetect for production
        text_lower = text.lower()
        
        # Common words for different languages
        english_words = ['the', 'and', 'that', 'have', 'for', 'with', 'this']
        indonesian_words = ['dan', 'yang', 'di', 'dengan', 'untuk', 'dari', 'ini']
        
        english_count = sum(1 for word in english_words if word in text_lower)
        indonesian_count = sum(1 for word in indonesian_words if word in text_lower)
        
        if english_count > indonesian_count:
            return "en"
        elif indonesian_count > english_count:
            return "id"
        else:
            return "unknown"