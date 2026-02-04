# cockatoo_v1/src/document_processing/extractors/txt_extractor.py

"""
Plain text document extractor.
"""

import os
import logging
from typing import Dict, Any, List, Union
from pathlib import Path
import chardet

from .base_extractor import BaseExtractor

logger = logging.getLogger(__name__)


class TXTExtractor(BaseExtractor):
    """
    Plain text document extractor with encoding detection.
    """
    
    def get_supported_formats(self) -> List[str]:
        """
        Return list of supported file formats.
        
        Returns:
            List of supported file extensions
        """
        return ['.txt', '.text', '.log', '.md', '.rst', '.ini', '.cfg', '.conf']
    
    def extract(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Extract text from plain text file.
        
        Args:
            file_path: Path to text file
        
        Returns:
            Dictionary containing:
                - text: Extracted text content
                - metadata: File metadata
                - encoding: Detected encoding
                - line_count: Number of lines
                - word_count: Number of words
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Text file not found: {file_path}")
        
        logger.info(f"Extracting text file: {file_path}")
        
        result = {
            "text": "",
            "metadata": self.get_basic_metadata(file_path),
            "encoding": "unknown",
            "line_count": 0,
            "word_count": 0,
            "char_count": 0,
        }
        
        # Detect encoding
        encoding = self._detect_encoding(file_path)
        result["encoding"] = encoding
        
        # Read file with detected encoding
        try:
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                text = f.read()
                
            # Clean text
            cleaned_text = self.clean_text(text)
            
            result.update({
                "text": cleaned_text,
                "line_count": len(text.splitlines()),
                "word_count": len(cleaned_text.split()),
                "char_count": len(cleaned_text),
            })
            
            # Extract additional metadata
            result["metadata"].update({
                "title": self._extract_title(cleaned_text),
                "language": self.detect_language(cleaned_text),
                "has_bom": self._has_bom(file_path),
                "summary": self.extract_summary(cleaned_text),
            })
            
        except UnicodeDecodeError:
            # Fallback to binary read and decode with errors='replace'
            with open(file_path, 'rb') as f:
                binary_data = f.read()
            text = binary_data.decode('utf-8', errors='replace')
            result["text"] = self.clean_text(text)
            result["encoding"] = "utf-8 (fallback)"
            
        except Exception as e:
            logger.error(f"Failed to extract text from {file_path}: {e}")
            raise
        
        return result
    
    def _detect_encoding(self, file_path: Path) -> str:
        """
        Detect file encoding using chardet.
        
        Args:
            file_path: Path to file
            
        Returns:
            Detected encoding string
        """
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Read first 10KB for detection
            
            if not raw_data:
                return 'utf-8'
            
            detection = chardet.detect(raw_data)
            encoding = detection.get('encoding', 'utf-8')
            confidence = detection.get('confidence', 0)
            
            # If confidence is low, default to utf-8
            if confidence < 0.5:
                encoding = 'utf-8'
            
            # Common encodings to check
            common_encodings = ['utf-8', 'utf-8-sig', 'iso-8859-1', 'windows-1252']
            
            if encoding.lower() in ['ascii', 'utf-8']:
                # Check for BOM (Byte Order Mark)
                if raw_data[:3] == b'\xef\xbb\xbf':
                    return 'utf-8-sig'
                return 'utf-8'
            
            return encoding.lower()
            
        except Exception as e:
            logger.warning(f"Encoding detection failed: {e}")
            return 'utf-8'
    
    def _has_bom(self, file_path: Path) -> bool:
        """
        Check if file has BOM (Byte Order Mark).
        
        Args:
            file_path: Path to file
            
        Returns:
            True if file has BOM
        """
        try:
            with open(file_path, 'rb') as f:
                bom = f.read(3)
            return bom == b'\xef\xbb\xbf'
        except:
            return False
    
    def _extract_title(self, text: str) -> str:
        """
        Extract title from text content.
        
        Args:
            text: Text content
            
        Returns:
            Extracted title
        """
        if not text:
            return ""
        
        lines = text.split('\n')
        
        # Look for title in first few non-empty lines
        for line in lines[:10]:
            line = line.strip()
            if line and len(line) < 200:  # Not too long for a title
                # Remove common prefixes
                if line.lower().startswith(('title:', 'subject:', 'name:')):
                    return line.split(':', 1)[1].strip()
                return line
        
        # If no suitable title found, use first 50 chars
        first_line = text.strip().split('\n')[0]
        return first_line[:50] + ("..." if len(first_line) > 50 else "")
    
    def extract_lines(self, file_path: Union[str, Path], start_line: int = 0, end_line: int = None) -> List[str]:
        """
        Extract specific lines from text file.
        
        Args:
            file_path: Path to text file
            start_line: Starting line number (0-indexed)
            end_line: Ending line number (exclusive)
            
        Returns:
            List of lines
        """
        file_path = Path(file_path)
        encoding = self._detect_encoding(file_path)
        
        lines = []
        try:
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                for i, line in enumerate(f):
                    if i < start_line:
                        continue
                    if end_line is not None and i >= end_line:
                        break
                    lines.append(line.rstrip('\n'))
        except Exception as e:
            logger.error(f"Failed to extract lines: {e}")
        
        return lines