# cockatoo_v1/src/document_processing/cleaning.py

"""
cleaning.py
Text Cleaning Utilities for Cockatoo_V1 Document Processing Pipeline.

This module provides comprehensive text cleaning and normalization functions
for preparing document text for embedding and analysis.

Author: Cockatoo_V1 Development Team
Version: 1.0.0
"""

import re
import unicodedata
import html
import string
from typing import Optional, List, Dict, Callable, Union, Any
from functools import wraps
import logging
from dataclasses import dataclass, field
import argparse

logger = logging.getLogger(__name__)


@dataclass
class CleaningConfig:
    """Configuration for text cleaning operations."""
    
    # Core cleaning options
    normalize_whitespace: bool = True
    fix_encoding: bool = True
    remove_control_chars: bool = True
    normalize_unicode: bool = True
    clean_html: bool = True
    remove_excessive_newlines: bool = True
    normalize_quotes: bool = True
    fix_hyphenation: bool = True
    
    # Advanced options
    lowercase: bool = False
    remove_numbers: bool = False
    remove_punctuation: bool = False
    keep_specific_punctuation: List[str] = field(default_factory=list)
    preserve_case_for: List[str] = field(default_factory=list)  # Words to preserve case (like acronyms)
    
    # Language specific
    language: str = "en"
    remove_stopwords: bool = False
    
    def __post_init__(self):
        """Initialize default values."""
        if not self.keep_specific_punctuation:
            self.keep_specific_punctuation = ['.', ',', '?', '!', ':', ';', '-', "'", '"']
        if not self.preserve_case_for:
            self.preserve_case_for = []


class TextCleaner:
    """
    Main text cleaning class with configurable cleaning pipelines.
    
    This class provides a flexible interface for applying multiple cleaning
    operations to text with configurable parameters.
    """
    
    # Common control characters to remove
    CONTROL_CHARS = {
        '\x00', '\x01', '\x02', '\x03', '\x04', '\x05', '\x06', '\x07',
        '\x08', '\x0b', '\x0c', '\x0e', '\x0f', '\x10', '\x11', '\x12',
        '\x13', '\x14', '\x15', '\x16', '\x17', '\x18', '\x19', '\x1a',
        '\x1b', '\x1c', '\x1d', '\x1e', '\x1f', '\x7f'
    }
    
    # Unicode categories to preserve (not control characters)
    PRESERVE_CATEGORIES = {'Ll', 'Lu', 'Lt', 'Lm', 'Lo', 'Nd', 'Nl', 'No',
                          'Pd', 'Ps', 'Pe', 'Pc', 'Po', 'Sm', 'Sc', 'Sk', 'So',
                          'Zs', 'Zl', 'Zp', 'Mn', 'Mc', 'Me'}
    
    # Threshold for large text processing (1MB)
    LARGE_TEXT_THRESHOLD = 1000000
    
    def __init__(self, config: Optional[CleaningConfig] = None):
        """
        Initialize text cleaner with configuration.
        
        Args:
            config: Optional configuration for cleaning operations.
                   Uses default config if not provided.
        """
        self.config = config or CleaningConfig()
        self.cleaning_pipeline = self._build_pipeline()
        
        # Compile regex patterns for performance
        self._whitespace_regex = re.compile(r'\s+')
        self._html_tag_regex = re.compile(r'<[^>]+>')
        self._multi_newline_regex = re.compile(r'\n{3,}')
        self._hyphen_newline_regex = re.compile(r'(\w)-\n(\w)')
        
        # Language-specific stopwords
        self._stopwords = self._load_stopwords()
    
    def _build_pipeline(self) -> List[Callable[[str], str]]:
        """
        Build cleaning pipeline based on configuration.
        
        Returns:
            List of cleaning functions to apply in sequence.
        """
        pipeline = []
        
        # Add configured cleaning steps in optimal order
        if self.config.fix_encoding:
            pipeline.append(self.fix_encoding)
        
        if self.config.remove_control_chars:
            pipeline.append(self.remove_control_characters)
        
        if self.config.normalize_unicode:
            pipeline.append(self.normalize_unicode)
        
        if self.config.clean_html:
            pipeline.append(self.clean_html_tags)
        
        if self.config.normalize_whitespace:
            pipeline.append(self.normalize_whitespace)
        
        if self.config.normalize_quotes:
            pipeline.append(self.normalize_quotes)
        
        if self.config.fix_hyphenation:
            pipeline.append(self.fix_hyphenation)
        
        if self.config.remove_excessive_newlines:
            pipeline.append(self.remove_excessive_newlines)
        
        # Language processing (after basic cleaning)
        if self.config.lowercase:
            pipeline.append(self._lowercase_text)
        
        if self.config.remove_punctuation:
            pipeline.append(self._remove_punctuation)
        
        if self.config.remove_numbers:
            pipeline.append(self._remove_numbers)
        
        if self.config.remove_stopwords:
            pipeline.append(self._remove_stopwords)
        
        return pipeline
    
    def _load_stopwords(self) -> set:
        """Load language-specific stopwords."""
        # Basic English stopwords
        english_stopwords = {
            'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'until',
            'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
            'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
            'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
            'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
            'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
            'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
            'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
        }
        
        # Add language-specific stopwords here
        stopword_sets = {
            'en': english_stopwords,
            'id': {'dan', 'atau', 'tetapi', 'jika', 'karena', 'sebagai', 'sampai'},
            # Add more languages as needed
        }
        
        return stopword_sets.get(self.config.language, english_stopwords)
    
    def clean_text(self, text: str, verbose: bool = False) -> str:
        """
        Apply complete cleaning pipeline to text.
        
        Args:
            text: Input text to clean.
            verbose: If True, log each cleaning step.
            
        Returns:
            Cleaned text with all transformations applied.
            
        Raises:
            TypeError: If input is None.
            ValueError: If input text is not a string.
        """
        if text is None:
            raise TypeError("Input text cannot be None")
        
        if not isinstance(text, str):
            raise ValueError(f"Expected string input, got {type(text).__name__}")
        
        if not text.strip():
            return text
        
        # For very large texts, use optimized processing (from FILE 1)
        if len(text) > self.LARGE_TEXT_THRESHOLD:
            return self._clean_large_text(text, verbose)
        
        cleaned = text
        
        if verbose:
            logger.info(f"Starting text cleaning pipeline (length: {len(cleaned)})")
        
        for i, cleaner in enumerate(self.cleaning_pipeline, 1):
            try:
                original_length = len(cleaned)
                cleaned = cleaner(cleaned)
                
                if verbose:
                    change = original_length - len(cleaned)
                    logger.debug(f"Step {i}: {cleaner.__name__} | "
                                f"Length: {original_length} -> {len(cleaned)} | "
                                f"Change: {change:+d}")
                    
            except Exception as e:
                logger.warning(f"Error in cleaning step {cleaner.__name__}: {e}")
                # Continue with next cleaning step
                continue
        
        if verbose:
            logger.info(f"Cleaning complete. Final length: {len(cleaned)}")
            reduction = len(text) - len(cleaned)
            if len(text) > 0:
                logger.info(f"Characters removed: {reduction} ({reduction/len(text)*100:.1f}%)")
            else:
                logger.info(f"Characters removed: {reduction}")
        
        return cleaned
    
    def _clean_large_text(self, text: str, verbose: bool = False) -> str:
        """
        Clean very large text efficiently by processing in chunks.
        
        Args:
            text: Large text to clean.
            verbose: If True, log progress.
            
        Returns:
            Cleaned text.
        """
        if verbose:
            logger.info(f"Processing large text ({len(text):,} characters) in chunks")
        
        # Split into manageable chunks (preserve paragraph boundaries)
        chunks = []
        chunk_size = 100000  # 100KB chunks
        start = 0
        
        while start < len(text):
            # Try to find paragraph boundary for splitting
            end = min(start + chunk_size, len(text))
            if end < len(text):
                # Look for paragraph break near the end
                para_break = text.rfind('\n\n', start + chunk_size - 1000, end)
                if para_break != -1:
                    end = para_break + 2  # Include the newlines
            
            chunks.append(text[start:end])
            start = end
        
        if verbose:
            logger.info(f"Split into {len(chunks)} chunks for processing")
        
        # Process each chunk
        cleaned_chunks = []
        for i, chunk in enumerate(chunks):
            if verbose and i % 10 == 0:
                logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            
            cleaned_chunk = chunk
            for cleaner in self.cleaning_pipeline:
                try:
                    cleaned_chunk = cleaner(cleaned_chunk)
                except Exception:
                    continue  # Skip failed cleaning steps for this chunk
            
            cleaned_chunks.append(cleaned_chunk)
        
        # Recombine chunks
        result = ''.join(cleaned_chunks)
        
        if verbose:
            logger.info(f"Large text processing complete. Final length: {len(result)}")
        
        return result
    
    def batch_clean(self, texts: List[str], verbose: bool = False) -> List[str]:
        """
        Clean a batch of texts efficiently.
        
        Args:
            texts: List of texts to clean.
            verbose: If True, log progress.
            
        Returns:
            List of cleaned texts.
            
        Raises:
            ValueError: If any input is not a string.
        """
        cleaned_texts = []
        
        for i, text in enumerate(texts):
            try:
                if verbose and i % 100 == 0:
                    logger.info(f"Processing text {i+1}/{len(texts)}")
                
                cleaned = self.clean_text(text, verbose=False)
                cleaned_texts.append(cleaned)
                
            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping text at index {i}: {e}")
                cleaned_texts.append("")  # Append empty string for failed cases
            except Exception as e:
                logger.warning(f"Unexpected error for text at index {i}: {e}")
                cleaned_texts.append(text)  # Keep original text (from FILE 2)
        
        return cleaned_texts
    
    def clean_text_with_report(self, text: str) -> Dict[str, Any]:
        """
        Clean text and return detailed report of changes.
        
        Args:
            text: Input text to clean.
            
        Returns:
            Dictionary with cleaned text and cleaning report.
        """
        if not text:
            return {
                'cleaned_text': text,
                'original_length': 0,
                'final_length': 0,
                'characters_removed': 0,
                'steps_applied': []
            }
        
        report = {
            'original_text': text[:100] + "..." if len(text) > 100 else text,  # Truncated (from FILE 1)
            'original_full_length': len(text),
            'steps_applied': [],
            'intermediate_results': []
        }
        
        cleaned = text
        
        for cleaner in self.cleaning_pipeline:
            step_name = cleaner.__name__
            try:
                before_length = len(cleaned)
                cleaned = cleaner(cleaned)
                after_length = len(cleaned)
                
                report['steps_applied'].append({
                    'step': step_name,
                    'before_length': before_length,
                    'after_length': after_length,
                    'characters_removed': before_length - after_length,
                    'success': True
                })
                
                # Limit intermediate results to prevent memory issues (from FILE 1)
                if len(report['intermediate_results']) < 5:
                    report['intermediate_results'].append({
                        'step': step_name,
                        'text': cleaned[:100] + "..." if len(cleaned) > 100 else cleaned
                    })
                
            except Exception as e:
                report['steps_applied'].append({
                    'step': step_name,
                    'error': str(e),
                    'success': False
                })
        
        report['cleaned_text'] = cleaned
        report['final_length'] = len(cleaned)
        report['characters_removed'] = len(text) - len(cleaned)
        if len(text) > 0:
            report['reduction_percentage'] = (len(text) - len(cleaned)) / len(text) * 100
        else:
            report['reduction_percentage'] = 0.0
        
        return report
    
    # ========== CORE CLEANING FUNCTIONS ==========
    
    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """
        Normalize whitespace characters in text.
        
        Args:
            text: Input text with potentially irregular whitespace.
            
        Returns:
            Text with normalized whitespace.
        """
        if not text:
            return text
        
        # Replace all whitespace variations with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    @staticmethod
    def fix_encoding(text: str) -> str:
        """
        Fix common encoding issues in text.
        
        Args:
            text: Input text with potential encoding issues.
            
        Returns:
            Text with encoding issues resolved.
        """
        if not text:
            return text
        
        # Common encoding fixes
        replacements = [
            ('â€œ', '"'),  # Left double quote
            ('â€', '"'),   # Right double quote
            ('â€˜', "'"),  # Left single quote
            ('â€™', "'"),  # Right single quote
            ('â€"', '—'),  # Em dash
            ('â€"', '–'),  # En dash
            ('â€¢', '•'),  # Bullet
            ('â„¢', '™'),  # Trademark
            ('Â©', '©'),   # Copyright
            ('Â®', '®'),   # Registered
            ('Ã©', 'é'),   # e acute
            ('Ã¨', 'è'),   # e grave
            ('Ãª', 'ê'),   # e circumflex
            ('Ã±', 'ñ'),   # n tilde
            ('Ã¶', 'ö'),   # o umlaut
            ('Ã¼', 'ü'),   # u umlaut
            ('Ã¡', 'á'),   # a acute
            ('Ã ', 'à'),   # a grave
            ('Ã¢', 'â'),   # a circumflex
            ('Ã£', 'ã'),   # a tilde
            ('Ã§', 'ç'),   # c cedilla
            ('Ã­', 'í'),   # i acute
            ('Ã¬', 'ì'),   # i grave
            ('Ã®', 'î'),   # i circumflex
            ('Ã¯', 'ï'),   # i umlaut
            ('Ã³', 'ó'),   # o acute
            ('Ã²', 'ò'),   # o grave
            ('Ã´', 'ô'),   # o circumflex
            ('Ãµ', 'õ'),   # o tilde
            ('Ãº', 'ú'),   # u acute
            ('Ã¹', 'ù'),   # u grave
            ('Ã»', 'û'),   # u circumflex
        ]
        
        for wrong, correct in replacements:
            text = text.replace(wrong, correct)
        
        # Try UTF-8 encoding/decoding
        try:
            text = text.encode('utf-8', errors='replace').decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            # Fallback: remove problematic characters
            text = text.encode('utf-8', errors='ignore').decode('utf-8')
        
        return text
    
    @staticmethod
    def remove_control_characters(text: str) -> str:
        """
        Remove control characters from text.
        
        Args:
            text: Input text potentially containing control characters.
            
        Returns:
            Text with control characters removed.
        """
        if not text:
            return text
        
        # Remove common control characters
        control_chars = ''.join(
            chr(i) for i in range(32) if chr(i) not in ['\n', '\r', '\t']
        )
        
        for char in control_chars:
            text = text.replace(char, '')
        
        # Remove Unicode control characters using categories
        cleaned_chars = []
        for char in text:
            cat = unicodedata.category(char)
            if cat[0] != 'C' or char in ['\n', '\r', '\t']:
                cleaned_chars.append(char)
        
        return ''.join(cleaned_chars)
    
    @staticmethod
    def normalize_unicode(text: str) -> str:
        """
        Normalize Unicode characters to consistent forms.
        
        Args:
            text: Input text with Unicode characters.
            
        Returns:
            Text with normalized Unicode.
        """
        if not text:
            return text
        
        # Normalize to NFC (Canonical Decomposition followed by Canonical Composition)
        text = unicodedata.normalize('NFC', text)
        
        return text
    
    @staticmethod
    def clean_html_tags(text: str) -> str:
        """
        Remove or decode HTML tags and entities from text.
        
        Args:
            text: Input text potentially containing HTML.
            
        Returns:
            Text with HTML tags removed and entities decoded.
        """
        if not text:
            return text
        
        # Decode HTML entities
        text = html.unescape(text)
        
        # Remove HTML tags (including self-closing)
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove common HTML artifacts
        html_entities = {
            '&nbsp;': ' ',
            '&amp;': '&',
            '&lt;': '<',
            '&gt;': '>',
            '&quot;': '"',
            '&apos;': "'",
            '&cent;': '¢',
            '&pound;': '£',
            '&yen;': '¥',
            '&euro;': '€',
            '&copy;': '©',
            '&reg;': '®',
            '&#xA0;': ' ',  # Non-breaking space
            '&#x2019;': "'",  # Right single quote
            '&#x201C;': '"',  # Left double quote
            '&#x201D;': '"',  # Right double quote
        }
        
        for entity, replacement in html_entities.items():
            text = text.replace(entity, replacement)
        
        # Remove multiple spaces that might result from HTML removal
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    @staticmethod
    def remove_excessive_newlines(text: str) -> str:
        """
        Remove excessive consecutive newlines from text.
        
        Args:
            text: Input text with potentially excessive newlines.
            
        Returns:
            Text with normalized newlines.
        """
        if not text:
            return text
        
        # Replace 3+ newlines with 2 newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove leading/trailing newlines
        text = text.strip('\n')
        
        return text
    
    @staticmethod
    def normalize_quotes(text: str) -> str:
        """
        Normalize various quote characters to standard ASCII quotes.
        
        Args:
            text: Input text with various quote characters.
            
        Returns:
            Text with normalized quotes.
        """
        if not text:
            return text
        
        # Curly quotes to straight quotes
        quote_replacements = {
            '“': '"',  # Left double quotation mark
            '”': '"',  # Right double quotation mark
            '‘': "'",  # Left single quotation mark
            '’': "'",  # Right single quotation mark
            '«': '"',  # Left-pointing double angle quote
            '»': '"',  # Right-pointing double angle quote
            '‹': "'",  # Left-pointing single angle quote
            '›': "'",  # Right-pointing single angle quote
            '„': '"',  # Double low-9 quotation mark
            '‟': '"',  # Double high-reversed-9 quotation mark
            '‚': "'",  # Single low-9 quotation mark
            '‛': "'",  # Single high-reversed-9 quotation mark
        }
        
        for curly, straight in quote_replacements.items():
            text = text.replace(curly, straight)
        
        return text
    
    @staticmethod
    def fix_hyphenation(text: str) -> str:
        """
        Fix hyphenation issues in text.
        
        Args:
            text: Input text with potential hyphenation issues.
            
        Returns:
            Text with fixed hyphenation.
        """
        if not text:
            return text
        
        # Remove soft hyphens (Unicode U+00AD)
        text = text.replace('\xad', '')
        
        # Remove discretionary hyphens (U+00AD in some contexts)
        text = text.replace('\u00AD', '')
        
        # Join words broken by hyphens at line endings
        text = re.sub(r'(\w)-\n(\w)', r'\1\2\n', text)
        
        # Remove hyphenation markers (common in some document formats)
        text = re.sub(r'(\w+)-(\w+)\n', r'\1\2\n', text)
        
        return text
    
    # ========== ADVANCED CLEANING FUNCTIONS ==========
    
    def _lowercase_text(self, text: str) -> str:
        """Convert text to lowercase, preserving specified words."""
        if not text or not self.config.lowercase:
            return text
        
        if not self.config.preserve_case_for:
            return text.lower()
        
        # Preserve case for specific words (like acronyms)
        words = text.split()
        preserved_words = set(word.lower() for word in self.config.preserve_case_for)
        
        processed_words = []
        for word in words:
            if word.lower() in preserved_words:
                processed_words.append(word)  # Keep original case
            else:
                processed_words.append(word.lower())
        
        return ' '.join(processed_words)
    
    def _remove_punctuation(self, text: str) -> str:
        """Remove punctuation from text, keeping specified punctuation."""
        if not text or not self.config.remove_punctuation:
            return text
        
        if not self.config.keep_specific_punctuation:
            # Remove all punctuation
            return text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove all punctuation except specified characters
        all_punctuation = string.punctuation
        punctuation_to_remove = ''.join(
            char for char in all_punctuation 
            if char not in self.config.keep_specific_punctuation
        )
        
        return text.translate(str.maketrans('', '', punctuation_to_remove))
    
    def _remove_numbers(self, text: str) -> str:
        """Remove numbers from text."""
        if not text or not self.config.remove_numbers:
            return text
        
        return re.sub(r'\d+', '', text)
    
    def _remove_stopwords(self, text: str) -> str:
        """Remove stopwords from text."""
        if not text or not self.config.remove_stopwords:
            return text
        
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in self._stopwords]
        
        return ' '.join(filtered_words)


# ========== CONVENIENCE FUNCTIONS ==========

def clean_text(
    text: str, 
    config: Optional[CleaningConfig] = None,
    verbose: bool = False
) -> str:
    """
    Convenience function for one-off text cleaning.
    
    Args:
        text: Input text to clean.
        config: Optional configuration for cleaning pipeline.
        verbose: If True, log cleaning steps.
        
    Returns:
        Cleaned text.
        
    Examples:
        >>> clean_text("  Hello   World  ")
        'Hello World'
        >>> clean_text("<p>Hello &amp; World</p>")
        'Hello & World'
    """
    cleaner = TextCleaner(config)
    return cleaner.clean_text(text, verbose)


def batch_clean_text(
    texts: List[str], 
    config: Optional[CleaningConfig] = None,
    verbose: bool = False
) -> List[str]:
    """
    Clean a batch of texts efficiently.
    
    Args:
        texts: List of texts to clean.
        config: Optional configuration for cleaning pipeline.
        verbose: If True, log progress.
        
    Returns:
        List of cleaned texts.
    """
    cleaner = TextCleaner(config)
    return cleaner.batch_clean(texts, verbose)


def create_custom_cleaner(config: CleaningConfig) -> Callable[[str], str]:
    """
    Create a custom cleaning function with specific configuration.
    
    Args:
        config: Configuration for the cleaning pipeline.
        
    Returns:
        Callable cleaning function.
    """
    cleaner = TextCleaner(config)
    return cleaner.clean_text


def analyze_text_cleaning(text: str, config: Optional[CleaningConfig] = None) -> Dict[str, Any]:
    """
    Analyze text and provide detailed cleaning report.
    
    Args:
        text: Text to analyze and clean.
        config: Optional cleaning configuration.
        
    Returns:
        Dictionary with analysis and cleaning report.
    """
    cleaner = TextCleaner(config)
    return cleaner.clean_text_with_report(text)


# ========== SPECIALIZED CLEANERS ==========

def get_presets() -> Dict[str, CleaningConfig]:
    """
    Get preset cleaning configurations for common use cases.
    
    Returns:
        Dictionary of preset configurations.
    """
    return {
        # Minimal cleaning - just fix encoding and whitespace
        'minimal': CleaningConfig(
            normalize_whitespace=True,
            fix_encoding=True,
            remove_control_chars=False,
            normalize_unicode=False,
            clean_html=False,
            remove_excessive_newlines=False,
            normalize_quotes=False,
            fix_hyphenation=False
        ),
        
        # Document cleaning - for PDFs, DOCX, etc.
        'document': CleaningConfig(
            normalize_whitespace=True,
            fix_encoding=True,
            remove_control_chars=True,
            normalize_unicode=True,
            clean_html=True,
            remove_excessive_newlines=True,
            normalize_quotes=True,
            fix_hyphenation=True
        ),
        
        # Web content cleaning
        'web': CleaningConfig(
            normalize_whitespace=True,
            fix_encoding=True,
            remove_control_chars=True,
            normalize_unicode=True,
            clean_html=True,  # Important for web content
            remove_excessive_newlines=True,
            normalize_quotes=True,
            fix_hyphenation=True
        ),
        
        # For embedding/RAG - optimized for vector search
        'embedding': CleaningConfig(
            normalize_whitespace=True,
            fix_encoding=True,
            remove_control_chars=True,
            normalize_unicode=True,
            clean_html=True,
            remove_excessive_newlines=True,
            normalize_quotes=True,
            fix_hyphenation=True,
            lowercase=True,
            remove_punctuation=False,
            remove_numbers=False,
            remove_stopwords=False
        ),
        
        # For NLP preprocessing
        'nlp': CleaningConfig(
            normalize_whitespace=True,
            fix_encoding=True,
            remove_control_chars=True,
            normalize_unicode=True,
            clean_html=True,
            remove_excessive_newlines=True,
            normalize_quotes=True,
            fix_hyphenation=True,
            lowercase=True,
            remove_punctuation=True,
            remove_numbers=True,
            remove_stopwords=True
        )
    }


# ========== TESTING AND VALIDATION ==========

def _test_cleaning_functions() -> Dict[str, bool]:
    """
    Test all cleaning functions with sample inputs.
    
    Returns:
        Dictionary with test results.
    """
    test_cases = [
        # (function_name, input_text, expected_output)
        ("normalize_whitespace", "  Hello   World  ", "Hello World"),
        ("normalize_whitespace", "\tTab\tSeparated\t", "Tab Separated"),
        ("fix_encoding", "Hello â€œWorldâ€", 'Hello "World"'),
        ("remove_control_characters", "Hello\x00World\x07", "HelloWorld"),
        ("normalize_unicode", "café", "café"),  # Should stay the same in NFC
        ("clean_html_tags", "<p>Hello &amp; World</p>", "Hello & World"),
        ("remove_excessive_newlines", "Line1\n\n\nLine2", "Line1\n\nLine2"),
        ("normalize_quotes", "Hello \"World\"", 'Hello "World"'),
        ("normalize_quotes", "Hello 'World'", "Hello 'World'"),
        ("fix_hyphenation", "hy-\nphenation", "hyphenation\n"),
    ]
    
    results = {}
    cleaner = TextCleaner()
    
    for func_name, input_text, expected in test_cases:
        try:
            func = getattr(cleaner, func_name)
            result = func(input_text)
            results[func_name] = result == expected
            
            if result != expected:
                logger.warning(f"Test failed for {func_name}: "
                              f"Expected '{expected}', got '{result}'")
                
        except Exception as e:
            results[func_name] = False
            logger.error(f"Error testing {func_name}: {e}")
    
    return results


# ========== MAIN MODULE EXECUTION ==========

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text Cleaning Utilities")
    parser.add_argument("--test", action="store_true", help="Run unit tests")
    parser.add_argument("--sample", action="store_true", help="Show sample usage")
    parser.add_argument("--presets", action="store_true", help="List available presets")
    parser.add_argument("--clean", type=str, help="Text to clean")
    parser.add_argument("--config", type=str, default="document", 
                       help="Cleaning preset (minimal, document, web, embedding, nlp)")
    
    args = parser.parse_args()
    
    if args.test:
        print("Running cleaning function tests...")
        results = _test_cleaning_functions()
        passed = sum(results.values())
        total = len(results)
        print(f"Tests passed: {passed}/{total} ({passed/total*100:.1f}%)")
        
        if passed < total:
            print("\nFailed tests:")
            for func, success in results.items():
                if not success:
                    print(f"  - {func}")
    
    elif args.presets:
        print("Available cleaning presets:")
        presets = get_presets()
        for name, config in presets.items():
            print(f"\n{name.upper()}:")
            print(f"  normalize_whitespace: {config.normalize_whitespace}")
            print(f"  fix_encoding: {config.fix_encoding}")
            print(f"  remove_control_chars: {config.remove_control_chars}")
            print(f"  clean_html: {config.clean_html}")
            print(f"  lowercase: {config.lowercase}")
            print(f"  remove_stopwords: {config.remove_stopwords}")
    
    elif args.clean:
        presets = get_presets()
        config = presets.get(args.config, CleaningConfig())
        
        print(f"\nOriginal text ({len(args.clean)} chars):")
        print(f"'{args.clean}'")
        
        cleaned = clean_text(args.clean, config=config, verbose=True)
        
        print(f"\nCleaned text ({len(cleaned)} chars):")
        print(f"'{cleaned}'")
        
        print(f"\nSummary:")
        reduction = len(args.clean) - len(cleaned)
        print(f"  Characters removed: {reduction}")
        if len(args.clean) > 0:
            print(f"  Reduction: {reduction / len(args.clean) * 100:.1f}%")
    
    else:
        # Default: show sample usage
        sample_text = """
        <p>This is a <strong>sample</strong> text with   multiple   spaces, 
        HTML &amp; entities, and "curly quotes".</p>
        
        It also has\x00control\x07chars, and
        excessive
        
        
        newlines.
        
        Some encoding issues: café, naïve, résumé.
        """
        
        print("=" * 70)
        print("TEXT CLEANING UTILITIES - SAMPLE USAGE")
        print("=" * 70)
        
        print("\nOriginal text:")
        print(repr(sample_text))
        
        print("\n\nCleaned with 'document' preset:")
        cleaned = clean_text(sample_text, verbose=True)
        print(repr(cleaned))
        
        print("\n\nTest results:")
        test_results = _test_cleaning_functions()
        for func, success in test_results.items():
            status = "✓" if success else "✗"
            print(f"  {status} {func}")