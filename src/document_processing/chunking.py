# cockatoo_v1/src/document_processing/chunking.py

"""
chunking.py
Intelligent Text Chunking for Cockatoo_V1 Document Processing Pipeline.

This module provides sophisticated text chunking strategies for preparing
document content for embedding and RAG (Retrieval Augmented Generation).

Author: Cockatoo_V1 Development Team
Version: 1.0.0
"""

import re
import math
import time
from typing import List, Dict, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from collections import deque
import json
import argparse

logger = logging.getLogger(__name__)

try:
    import tiktoken  # For accurate token counting
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False
    logger.warning("tiktoken not installed. Using approximate token counting.")


class ChunkingStrategy(Enum):
    """Available text chunking strategies."""
    FIXED_SIZE = "fixed_size"  # Fixed token/size chunks
    SEMANTIC = "semantic"  # Chunk at semantic boundaries
    OVERLAP = "overlap"  # Overlapping chunks
    RECURSIVE = "recursive"  # Recursive chunking
    SLIDING_WINDOW = "sliding_window"  # Sliding window approach
    PARAGRAPH = "paragraph"  # Paragraph-based chunking
    SENTENCE = "sentence"  # Sentence-based chunking


@dataclass
class ChunkingConfig:
    """
    Configuration for text chunking.
    """
    # Core chunking parameters
    chunk_size: int = 500  # Target chunk size in tokens
    chunk_overlap: int = 50  # Overlap between chunks in tokens
    min_chunk_size: int = 50  # Minimum chunk size in tokens
    
    # Strategy selection
    strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC
    
    # Text segmentation
    separator: str = "\n\n"  # Default separator for splitting
    separators: Optional[List[str]] = None  # Multiple separators in order of preference
    keep_separator: bool = True  # Whether to keep separator in chunks
    
    # Advanced options
    tokenizer: str = "cl100k_base"  # Tokenizer for accurate token counting
    encoding_name: str = "cl100k_base"  # Encoding for tokenizer
    max_tokens_per_chunk: Optional[int] = None  # Hard limit
    
    # Semantic chunking options
    semantic_threshold: float = 0.5  # Threshold for semantic boundaries
    preserve_formatting: bool = False  # Preserve original formatting
    
    # Language-specific
    language: str = "en"  # Language for sentence/paragraph detection
    
    def __post_init__(self):
        """Initialize default values."""
        if self.separators is None:
            self.separators = [
                "\n\n",  # Paragraph breaks
                "\n",  # Line breaks
                ". ",  # Sentences
                "! ",  # Exclamations
                "? ",  # Questions
                "; ",  # Semicolons
                ": ",  # Colons
                ", ",  # Commas
                " ",  # Spaces
                "",  # Any character
            ]
        
        if self.max_tokens_per_chunk is None:
            self.max_tokens_per_chunk = self.chunk_size * 2


class TextChunk:
    """
    Represents a single text chunk with metadata.
    """
    
    def __init__(self, text: str, chunk_index: int = 0, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a text chunk.
        
        Args:
            text: The chunk text content.
            chunk_index: Position of chunk in document.
            metadata: Additional metadata about the chunk.
        """
        self.text = text
        self.chunk_index = chunk_index
        self.metadata = metadata or {}
        
        # Calculate metrics
        self.char_count = len(text)
        self.word_count = len(re.findall(r'\b\w+\b', text))
        self.token_count = 0  # Will be calculated with tokenizer
        self.line_count = text.count('\n') + 1
        
        # Position information
        self.start_char = self.metadata.get('start_char', 0)
        self.end_char = self.metadata.get('end_char', self.start_char + self.char_count)
        
        # Quality metrics
        self.coherence_score = self.metadata.get('coherence_score', 0.0)
        self.completeness_score = self.metadata.get('completeness_score', 0.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary."""
        return {
            'text': self.text,
            'chunk_index': self.chunk_index,
            'char_count': self.char_count,
            'word_count': self.word_count,
            'token_count': self.token_count,
            'line_count': self.line_count,
            'start_char': self.start_char,
            'end_char': self.end_char,
            'coherence_score': self.coherence_score,
            'completeness_score': self.completeness_score,
            'metadata': self.metadata,
        }
    
    def __repr__(self) -> str:
        return f"TextChunk(index={self.chunk_index}, tokens={self.token_count}, words={self.word_count})"
    
    def __len__(self) -> int:
        return len(self.text)


class TextChunker:
    """
    Main class for intelligent text chunking.
    
    Provides multiple chunking strategies with configurable parameters.
    """
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        """
        Initialize text chunker with configuration.
        
        Args:
            config: Chunking configuration. Uses defaults if None.
        """
        self.config = config or ChunkingConfig()
        
        # Initialize tokenizer
        self.tokenizer = None
        if HAS_TIKTOKEN:
            try:
                self.tokenizer = tiktoken.get_encoding(self.config.tokenizer)
                logger.debug(f"Tokenizer {self.config.tokenizer} initialized")
            except Exception as e:
                logger.warning(f"Failed to load tokenizer {self.config.tokenizer}: {e}")
                logger.warning("Falling back to approximate token counting")
        else:
            logger.info("Using approximate token counting (tiktoken not available)")
        
        # Initialize language-specific patterns
        self._init_language_patterns()
        
        logger.info(f"TextChunker initialized with strategy: {self.config.strategy.value}")
        logger.info(f"Chunk size: {self.config.chunk_size} tokens, Overlap: {self.config.chunk_overlap} tokens")
    
    def _init_language_patterns(self) -> None:
        """Initialize language-specific patterns for sentence/paragraph detection."""
        # Sentence boundary patterns by language
        self.sentence_patterns = {
            'en': r'(?<=[.!?])\s+(?=[A-Z])',
            'id': r'(?<=[.!?])\s+(?=[A-Z])',
            'es': r'(?<=[.!?])\s+(?=[A-Z])',
            'fr': r'(?<=[.!?])\s+(?=[A-Z])',
            'de': r'(?<=[.!?])\s+(?=[A-Z])',
            'zh': r'(?<=[。！？])\s*',
            'ja': r'(?<=[。！？])\s*',
            'ko': r'(?<=[。！？])\s*',
        }
        
        # Paragraph patterns
        self.paragraph_pattern = r'\n\s*\n'
        
        # Get pattern for current language
        self.sentence_pattern = self.sentence_patterns.get(
            self.config.language, 
            self.sentence_patterns['en']  # Default to English
        )
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tokenizer or approximation.
        
        Args:
            text: Text to count tokens for.
            
        Returns:
            Number of tokens.
        """
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception as e:
                logger.warning(f"Token counting failed: {e}")
        
        # Fallback approximation: 1 token ≈ 4 characters for English
        # This is a rough approximation and should be replaced with proper tokenization
        if not text:
            return 0
        return max(1, len(text) // 4)
    
    def chunk_text(self, text: str, strategy: Optional[ChunkingStrategy] = None) -> List[Dict[str, Any]]:
        """
        Chunk text using specified strategy.
        
        Args:
            text: Text to chunk.
            strategy: Chunking strategy. Uses config strategy if None.
            
        Returns:
            List of chunk dictionaries.
        """
        if not text or not text.strip():
            return []
        
        strategy = strategy or self.config.strategy
        
        # Pre-process text
        processed_text = self._preprocess_text(text)
        
        # Apply chunking strategy
        chunk_methods = {
            ChunkingStrategy.FIXED_SIZE: self._chunk_fixed_size,
            ChunkingStrategy.SEMANTIC: self._chunk_semantic,
            ChunkingStrategy.OVERLAP: self._chunk_overlap,
            ChunkingStrategy.RECURSIVE: self._chunk_recursive,
            ChunkingStrategy.SLIDING_WINDOW: self._chunk_sliding_window,
            ChunkingStrategy.PARAGRAPH: self._chunk_by_paragraph,
            ChunkingStrategy.SENTENCE: self._chunk_by_sentence,
        }
        
        chunk_method = chunk_methods.get(strategy, self._chunk_fixed_size)
        chunks = chunk_method(processed_text)
        
        # Post-process chunks
        processed_chunks = self._postprocess_chunks(chunks, text)
        
        # Convert to dictionaries
        chunk_dicts = [chunk.to_dict() for chunk in processed_chunks]
        
        logger.info(f"Created {len(chunk_dicts)} chunks using {strategy.value} strategy")
        logger.debug(f"Total tokens: {sum(c['token_count'] for c in chunk_dicts)}")
        
        return chunk_dicts
    
    def _preprocess_text(self, text: str) -> str:
        """
        Pre-process text before chunking.
        
        Args:
            text: Original text.
            
        Returns:
            Pre-processed text.
        """
        if not text:
            return text
        
        # Remove excessive whitespace but preserve paragraph breaks
        if not self.config.preserve_formatting:
            # Normalize line endings
            text = text.replace('\r\n', '\n').replace('\r', '\n')
            
            # Preserve paragraph breaks (multiple newlines)
            text = re.sub(r'\n{3,}', '\n\n', text)
            
            # Clean up excessive spaces
            text = re.sub(r'[ \t]+', ' ', text)
            
            # Remove leading/trailing whitespace per line
            lines = text.split('\n')
            lines = [line.strip() for line in lines]
            text = '\n'.join(lines)
        
        return text
    
    def _postprocess_chunks(self, chunks: List[TextChunk], original_text: str) -> List[TextChunk]:
        """
        Post-process chunks after chunking.
        
        Args:
            chunks: List of text chunks.
            original_text: Original text for position calculation.
            
        Returns:
            Processed chunks with metadata.
        """
        processed_chunks = []
        
        for i, chunk in enumerate(chunks):
            # Update chunk index
            chunk.chunk_index = i
            
            # Calculate token count if not already done
            if chunk.token_count == 0:
                chunk.token_count = self.count_tokens(chunk.text)
            
            # Calculate position in original text
            if original_text and chunk.text:
                # Find the chunk in original text (approximate)
                chunk_text_clean = chunk.text.strip()[:100]  # Use first 100 chars for matching
                if chunk_text_clean:
                    pos = original_text.find(chunk_text_clean)
                    if pos != -1:
                        chunk.start_char = pos
                        chunk.end_char = pos + len(chunk.text)
                    else:
                        # Fallback: estimate based on average position
                        if chunks:
                            avg_chars_per_chunk = len(original_text) / len(chunks)
                            chunk.start_char = int(i * avg_chars_per_chunk)
                            chunk.end_char = int((i + 1) * avg_chars_per_chunk)
            
            # Calculate quality scores
            chunk.coherence_score = self._calculate_coherence(chunk.text)
            chunk.completeness_score = self._calculate_completeness(chunk.text)
            
            processed_chunks.append(chunk)
        
        return processed_chunks
    
    def _chunk_fixed_size(self, text: str) -> List[TextChunk]:
        """
        Chunk text into fixed-size chunks.
        
        Args:
            text: Text to chunk.
            
        Returns:
            List of text chunks.
        """
        # If text is smaller than chunk size, return as single chunk
        if self.count_tokens(text) <= self.config.chunk_size:
            return [TextChunk(text)]
        
        # Split by preferred separators
        current_chunk = ""
        current_tokens = 0
        sentences = self._split_by_separators(text, self.config.separators or [])
        
        chunks = []
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            # If single sentence is larger than chunk size, split it
            if sentence_tokens > self.config.chunk_size:
                if current_chunk:
                    chunks.append(TextChunk(current_chunk))
                    current_chunk = ""
                    current_tokens = 0
                
                # Split large sentence
                sub_chunks = self._split_large_sentence(sentence)
                chunks.extend(sub_chunks)
                continue
            
            # Add sentence to current chunk
            if current_tokens + sentence_tokens <= self.config.chunk_size:
                current_chunk += sentence
                current_tokens += sentence_tokens
            else:
                # Save current chunk and start new one
                if current_chunk:
                    chunks.append(TextChunk(current_chunk))
                
                current_chunk = sentence
                current_tokens = sentence_tokens
        
        # Add final chunk
        if current_chunk:
            chunks.append(TextChunk(current_chunk))
        
        return chunks
    
    def _chunk_semantic(self, text: str) -> List[TextChunk]:
        """
        Chunk text at semantic boundaries (paragraphs, sentences).
        
        Args:
            text: Text to chunk.
            
        Returns:
            List of text chunks.
        """
        # First, try to chunk by paragraphs
        paragraphs = re.split(self.paragraph_pattern, text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        if not paragraphs:
            return [TextChunk(text)]
        
        chunks = []
        current_chunk_parts = []
        current_tokens = 0
        
        for paragraph in paragraphs:
            paragraph_tokens = self.count_tokens(paragraph)
            
            # If paragraph fits in chunk, add it
            if current_tokens + paragraph_tokens <= self.config.chunk_size:
                current_chunk_parts.append(paragraph)
                current_tokens += paragraph_tokens
            else:
                # If paragraph alone is too large, split it
                if paragraph_tokens > self.config.chunk_size:
                    if current_chunk_parts:
                        chunks.append(TextChunk("\n\n".join(current_chunk_parts)))
                        current_chunk_parts = []
                        current_tokens = 0
                    
                    # Split large paragraph
                    sub_chunks = self._split_paragraph_semantically(paragraph)
                    chunks.extend(sub_chunks)
                else:
                    # Save current chunk and start new one with this paragraph
                    if current_chunk_parts:
                        chunks.append(TextChunk("\n\n".join(current_chunk_parts)))
                    
                    current_chunk_parts = [paragraph]
                    current_tokens = paragraph_tokens
        
        # Add final chunk
        if current_chunk_parts:
            chunks.append(TextChunk("\n\n".join(current_chunk_parts)))
        
        return chunks
    
    def _chunk_overlap(self, text: str) -> List[TextChunk]:
        """
        Create overlapping chunks for better context retention.
        
        Args:
            text: Text to chunk.
            
        Returns:
            List of overlapping text chunks.
        """
        # First create fixed-size chunks
        base_chunks = self._chunk_fixed_size(text)
        
        if len(base_chunks) <= 1:
            return base_chunks
        
        overlapping_chunks = []
        
        for i in range(len(base_chunks) - 1):
            current_chunk = base_chunks[i]
            next_chunk = base_chunks[i + 1]
            
            # Create overlapping chunk
            overlap_text = self._create_overlap(current_chunk.text, next_chunk.text)
            overlapping_chunks.append(TextChunk(overlap_text))
        
        # Add last chunk without overlap at the end
        overlapping_chunks.append(base_chunks[-1])
        
        return overlapping_chunks
    
    def _chunk_recursive(self, text: str) -> List[TextChunk]:
        """
        Recursively split text until chunks are within size limits.
        
        Args:
            text: Text to chunk.
            
        Returns:
            List of text chunks.
        """
        # If text is small enough, return as single chunk
        if self.count_tokens(text) <= self.config.chunk_size:
            return [TextChunk(text)]
        
        # Try to split by different separators in order of preference
        for separator in self.config.separators or []:
            if separator and separator in text:
                parts = text.split(separator)
                
                # Only split if it creates meaningful parts
                meaningful_parts = [p for p in parts if p.strip()]
                if len(meaningful_parts) > 1:
                    chunks = []
                    
                    # Reconstruct parts with separator
                    for i, part in enumerate(meaningful_parts):
                        if self.config.keep_separator and i < len(meaningful_parts) - 1:
                            part_with_separator = part + separator
                        else:
                            part_with_separator = part
                        
                        # Recursively chunk each part
                        if part_with_separator.strip():
                            sub_chunks = self._chunk_recursive(part_with_separator)
                            chunks.extend(sub_chunks)
                    
                    if chunks:
                        return chunks
        
        # If no good split found, split by character count as last resort
        return self._split_by_characters(text)
    
    def _chunk_sliding_window(self, text: str) -> List[TextChunk]:
        """
        Create chunks using a sliding window approach.
        
        Args:
            text: Text to chunk.
            
        Returns:
            List of text chunks.
        """
        chunks = []
        text_length = len(text)
        
        if text_length == 0:
            return chunks
        
        # Approximate character counts
        chunk_size_chars = self.config.chunk_size * 4
        overlap_chars = self.config.chunk_overlap * 4
        
        # Calculate step size
        step_size = max(1, chunk_size_chars - overlap_chars)
        
        # Create sliding windows
        for i in range(0, text_length, step_size):
            end = min(i + chunk_size_chars, text_length)
            chunk_text = text[i:end]
            
            if chunk_text.strip():
                chunks.append(TextChunk(chunk_text, metadata={'window_start': i, 'window_end': end}))
        
        return chunks
    
    def _chunk_by_paragraph(self, text: str) -> List[TextChunk]:
        """
        Chunk text by paragraphs.
        
        Args:
            text: Text to chunk.
            
        Returns:
            List of paragraph chunks.
        """
        paragraphs = re.split(self.paragraph_pattern, text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        if not paragraphs:
            return [TextChunk(text)]
        
        chunks = []
        current_chunk_parts = []
        current_tokens = 0
        
        for paragraph in paragraphs:
            paragraph_tokens = self.count_tokens(paragraph)
            
            # If paragraph is too large, split it
            if paragraph_tokens > self.config.chunk_size:
                if current_chunk_parts:
                    chunks.append(TextChunk("\n\n".join(current_chunk_parts)))
                    current_chunk_parts = []
                    current_tokens = 0
                
                sub_chunks = self._split_paragraph_semantically(paragraph)
                chunks.extend(sub_chunks)
            # If adding paragraph would exceed chunk size, start new chunk
            elif current_tokens + paragraph_tokens > self.config.chunk_size and current_chunk_parts:
                chunks.append(TextChunk("\n\n".join(current_chunk_parts)))
                current_chunk_parts = [paragraph]
                current_tokens = paragraph_tokens
            else:
                current_chunk_parts.append(paragraph)
                current_tokens += paragraph_tokens
        
        # Add final chunk
        if current_chunk_parts:
            chunks.append(TextChunk("\n\n".join(current_chunk_parts)))
        
        return chunks
    
    def _chunk_by_sentence(self, text: str) -> List[TextChunk]:
        """
        Chunk text by sentences.
        
        Args:
            text: Text to chunk.
            
        Returns:
            List of sentence chunks.
        """
        # Split by sentences
        sentences = re.split(self.sentence_pattern, text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return [TextChunk(text)]
        
        chunks = []
        current_chunk_parts = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            # If sentence is too large, split it
            if sentence_tokens > self.config.chunk_size:
                if current_chunk_parts:
                    chunks.append(TextChunk(" ".join(current_chunk_parts)))
                    current_chunk_parts = []
                    current_tokens = 0
                
                sub_chunks = self._split_large_sentence(sentence)
                chunks.extend(sub_chunks)
            # If adding sentence would exceed chunk size, start new chunk
            elif current_tokens + sentence_tokens > self.config.chunk_size and current_chunk_parts:
                chunks.append(TextChunk(" ".join(current_chunk_parts)))
                current_chunk_parts = [sentence]
                current_tokens = sentence_tokens
            else:
                current_chunk_parts.append(sentence)
                current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk_parts:
            chunks.append(TextChunk(" ".join(current_chunk_parts)))
        
        return chunks
    
    def _split_by_separators(self, text: str, separators: List[str]) -> List[str]:
        """
        Split text by multiple separators in order of preference.
        
        Args:
            text: Text to split.
            separators: List of separators in order of preference.
            
        Returns:
            List of text segments.
        """
        if not separators:
            return [text]
        
        # Try each separator
        for separator in separators:
            if separator and separator in text:
                parts = text.split(separator)
                
                # Reconstruct with separator if keeping it
                result = []
                for i, part in enumerate(parts):
                    if part.strip():
                        if self.config.keep_separator and i < len(parts) - 1:
                            result.append(part + separator)
                        else:
                            result.append(part)
                
                if len(result) > 1:
                    return result
        
        # If no separator found, return as single segment
        return [text]
    
    def _split_large_sentence(self, sentence: str) -> List[TextChunk]:
        """
        Split a large sentence into smaller chunks.
        
        Args:
            sentence: Large sentence to split.
            
        Returns:
            List of smaller text chunks.
        """
        # Try to split by clauses
        clause_separators = [', ', '; ', ': ', ' - ']
        
        # Build regex pattern for clause separators
        clause_pattern = '|'.join([re.escape(sep) for sep in clause_separators])
        
        # Split by clause separators
        import re
        clauses = re.split(f'({clause_pattern})', sentence)
        
        # Reconstruct clauses with separators
        reconstructed_clauses = []
        i = 0
        while i < len(clauses):
            clause = clauses[i].strip()
            if clause:
                if i + 1 < len(clauses) and clauses[i + 1] in clause_separators:
                    reconstructed_clauses.append(clause + clauses[i + 1])
                    i += 2
                else:
                    reconstructed_clauses.append(clause)
                    i += 1
            else:
                i += 1
        
        if len(reconstructed_clauses) > 1:
            chunks = []
            current_chunk_parts = []
            current_tokens = 0
            
            for clause in reconstructed_clauses:
                clause_tokens = self.count_tokens(clause)
                
                # If clause alone is too large, split by words
                if clause_tokens > self.config.chunk_size:
                    if current_chunk_parts:
                        chunks.append(TextChunk("".join(current_chunk_parts)))
                        current_chunk_parts = []
                        current_tokens = 0
                    
                    # Split clause by words
                    word_chunks = self._split_by_words(clause)
                    chunks.extend(word_chunks)
                # Add to current chunk if it fits
                elif current_tokens + clause_tokens <= self.config.chunk_size:
                    current_chunk_parts.append(clause)
                    current_tokens += clause_tokens
                else:
                    if current_chunk_parts:
                        chunks.append(TextChunk("".join(current_chunk_parts)))
                    
                    current_chunk_parts = [clause]
                    current_tokens = clause_tokens
            
            if current_chunk_parts:
                chunks.append(TextChunk("".join(current_chunk_parts)))
            
            if chunks:
                return chunks
        
        # If no clause separators or splitting didn't work, split by words
        return self._split_by_words(sentence)
    
    def _split_paragraph_semantically(self, paragraph: str) -> List[TextChunk]:
        """
        Split paragraph at semantic boundaries.
        
        Args:
            paragraph: Paragraph to split.
            
        Returns:
            List of text chunks.
        """
        # First try to split by sentences
        sentences = re.split(self.sentence_pattern, paragraph)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) > 1:
            chunks = []
            current_chunk_parts = []
            current_tokens = 0
            
            for sentence in sentences:
                sentence_tokens = self.count_tokens(sentence)
                
                if current_tokens + sentence_tokens <= self.config.chunk_size:
                    current_chunk_parts.append(sentence)
                    current_tokens += sentence_tokens
                else:
                    if current_chunk_parts:
                        chunks.append(TextChunk(" ".join(current_chunk_parts)))
                    
                    current_chunk_parts = [sentence]
                    current_tokens = sentence_tokens
            
            if current_chunk_parts:
                chunks.append(TextChunk(" ".join(current_chunk_parts)))
            
            return chunks
        
        # If only one sentence or sentence splitting failed, split by words
        return self._split_by_words(paragraph)
    
    def _split_by_words(self, text: str) -> List[TextChunk]:
        """
        Split text by words to create chunks of appropriate size.
        
        Args:
            text: Text to split.
            
        Returns:
            List of text chunks.
        """
        words = text.split()
        if not words:
            return [TextChunk(text)]
        
        chunks = []
        current_chunk_words = []
        current_tokens = 0
        
        for word in words:
            word_with_space = word + " "
            word_tokens = self.count_tokens(word_with_space)
            
            if current_tokens + word_tokens <= self.config.chunk_size:
                current_chunk_words.append(word)
                current_tokens += word_tokens
            else:
                if current_chunk_words:
                    chunks.append(TextChunk(" ".join(current_chunk_words)))
                
                current_chunk_words = [word]
                current_tokens = word_tokens
        
        if current_chunk_words:
            chunks.append(TextChunk(" ".join(current_chunk_words)))
        
        return chunks
    
    def _split_by_characters(self, text: str) -> List[TextChunk]:
        """
        Split text by character count as last resort.
        
        Args:
            text: Text to split.
            
        Returns:
            List of text chunks.
        """
        chunks = []
        text_length = len(text)
        
        if text_length == 0:
            return chunks
        
        chunk_size_chars = max(100, self.config.chunk_size * 4)  # Minimum 100 chars
        
        for i in range(0, text_length, chunk_size_chars):
            chunk_text = text[i:i + chunk_size_chars]
            if chunk_text.strip():
                chunks.append(TextChunk(chunk_text))
        
        return chunks
    
    def _create_overlap(self, chunk1_text: str, chunk2_text: str) -> str:
        """
        Create overlapping chunk between two chunks.
        
        Args:
            chunk1_text: First chunk text.
            chunk2_text: Second chunk text.
            
        Returns:
            Overlapping chunk text.
        """
        # Calculate overlap in tokens
        target_overlap_tokens = self.config.chunk_overlap
        
        # Get end of first chunk (last n tokens)
        chunk1_tokens = self.count_tokens(chunk1_text)
        if chunk1_tokens <= target_overlap_tokens:
            overlap_start_text = chunk1_text
        else:
            # Approximate character position for overlap
            overlap_chars = int((target_overlap_tokens / max(1, chunk1_tokens)) * len(chunk1_text))
            overlap_start_text = chunk1_text[-overlap_chars:]
        
        # Get beginning of second chunk (first n tokens)
        chunk2_tokens = self.count_tokens(chunk2_text)
        if chunk2_tokens <= target_overlap_tokens:
            overlap_end_text = chunk2_text
        else:
            # Approximate character position for overlap
            overlap_chars = int((target_overlap_tokens / max(1, chunk2_tokens)) * len(chunk2_text))
            overlap_end_text = chunk2_text[:overlap_chars]
        
        # Combine for overlapping chunk
        return overlap_start_text + overlap_end_text
    
    def _calculate_coherence(self, text: str) -> float:
        """
        Calculate coherence score for text chunk.
        
        Args:
            text: Text to evaluate.
            
        Returns:
            Coherence score between 0 and 1.
        """
        if not text:
            return 0.0
        
        # Simple heuristics for coherence
        score = 0.0
        
        # Check for complete sentences
        sentences = re.split(self.sentence_pattern, text)
        sentences = [s for s in sentences if s.strip()]
        
        if sentences:
            complete_sentences = sum(1 for s in sentences if len(s.split()) >= 3)
            score += (complete_sentences / len(sentences)) * 0.4
        
        # Check for paragraph structure
        if '\n\n' in text:
            score += 0.3
        
        # Check for transition words (indicates flow)
        transition_words = ['however', 'therefore', 'moreover', 'furthermore', 
                           'consequently', 'similarly', 'additionally']
        has_transitions = any(word in text.lower() for word in transition_words)
        if has_transitions:
            score += 0.2
        
        # Check length (neither too short nor too long)
        token_count = self.count_tokens(text)
        if self.config.min_chunk_size <= token_count <= self.config.chunk_size:
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_completeness(self, text: str) -> float:
        """
        Calculate completeness score for text chunk.
        
        Args:
            text: Text to evaluate.
            
        Returns:
            Completeness score between 0 and 1.
        """
        if not text:
            return 0.0
        
        # Simple heuristics for completeness
        score = 0.0
        
        # Check if chunk starts with capital letter (likely start of sentence/paragraph)
        if text and text[0].isupper():
            score += 0.2
        
        # Check if chunk ends with sentence terminator
        if text and text[-1] in '.!?':
            score += 0.3
        
        # Check for balanced parentheses/quotes (indicates complete thought)
        open_paren = text.count('(')
        close_paren = text.count(')')
        if open_paren == close_paren:
            score += 0.2
        
        open_quote = text.count('"')
        if open_quote % 2 == 0:  # Even number of quotes
            score += 0.1
        
        # Check for reasonable length
        word_count = len(re.findall(r'\b\w+\b', text))
        if word_count >= 10:  # At least 10 words
            score += 0.2
        
        return min(score, 1.0)
    
    def analyze_text_structure(self, text: str) -> Dict[str, Any]:
        """
        Analyze text structure for chunking optimization.
        
        Args:
            text: Text to analyze.
            
        Returns:
            Dictionary with structure analysis.
        """
        if not text:
            return {
                'total_chars': 0,
                'total_words': 0,
                'total_tokens': 0,
                'paragraphs': 0,
                'sentences': 0,
                'avg_sentence_length': 0,
                'avg_paragraph_length': 0,
                'recommended_strategy': self.config.strategy.value,
                'estimated_chunks': 0,
            }
        
        analysis = {
            'total_chars': len(text),
            'total_words': len(re.findall(r'\b\w+\b', text)),
            'total_tokens': self.count_tokens(text),
            'recommended_strategy': self.config.strategy.value,
        }
        
        # Calculate sentence statistics
        sentences = re.split(self.sentence_pattern, text)
        sentences = [s for s in sentences if s.strip()]
        if sentences:
            analysis['sentences'] = len(sentences)
            analysis['avg_sentence_length'] = sum(len(s) for s in sentences) / len(sentences)
        else:
            analysis['sentences'] = 0
            analysis['avg_sentence_length'] = 0
        
        # Calculate paragraph statistics
        paragraphs = re.split(self.paragraph_pattern, text)
        paragraphs = [p for p in paragraphs if p.strip()]
        if paragraphs:
            analysis['paragraphs'] = len(paragraphs)
            analysis['avg_paragraph_length'] = sum(len(p) for p in paragraphs) / len(paragraphs)
        else:
            analysis['paragraphs'] = 0
            analysis['avg_paragraph_length'] = 0
        
        # Estimate number of chunks
        estimated_chunks = max(1, analysis['total_tokens'] // max(1, self.config.chunk_size))
        analysis['estimated_chunks'] = estimated_chunks
        
        # Recommend strategy based on text structure
        if analysis['paragraphs'] > 1 and analysis['avg_paragraph_length'] < self.config.chunk_size * 4:
            analysis['recommended_strategy'] = ChunkingStrategy.PARAGRAPH.value
        elif analysis['sentences'] > 1 and analysis['avg_sentence_length'] < self.config.chunk_size * 2:
            analysis['recommended_strategy'] = ChunkingStrategy.SENTENCE.value
        elif analysis['total_tokens'] > self.config.chunk_size * 10:
            analysis['recommended_strategy'] = ChunkingStrategy.RECURSIVE.value
        
        return analysis
    
    def get_chunking_statistics(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate statistics for chunking results.
        
        Args:
            chunks: List of chunk dictionaries.
            
        Returns:
            Dictionary with chunking statistics.
        """
        if not chunks:
            return {}
        
        stats = {
            'total_chunks': len(chunks),
            'total_tokens': 0,
            'total_words': 0,
            'total_chars': 0,
            'avg_tokens_per_chunk': 0,
            'avg_words_per_chunk': 0,
            'avg_chars_per_chunk': 0,
            'min_tokens': float('inf'),
            'max_tokens': 0,
            'min_words': float('inf'),
            'max_words': 0,
            'coherence_scores': [],
            'completeness_scores': [],
        }
        
        token_counts = []
        word_counts = []
        char_counts = []
        
        for chunk in chunks:
            tokens = chunk.get('token_count', 0)
            words = chunk.get('word_count', 0)
            chars = chunk.get('char_count', 0)
            
            token_counts.append(tokens)
            word_counts.append(words)
            char_counts.append(chars)
            
            stats['total_tokens'] += tokens
            stats['total_words'] += words
            stats['total_chars'] += chars
            
            stats['min_tokens'] = min(stats['min_tokens'], tokens)
            stats['max_tokens'] = max(stats['max_tokens'], tokens)
            stats['min_words'] = min(stats['min_words'], words)
            stats['max_words'] = max(stats['max_words'], words)
            
            if 'coherence_score' in chunk:
                stats['coherence_scores'].append(chunk['coherence_score'])
            if 'completeness_score' in chunk:
                stats['completeness_scores'].append(chunk['completeness_score'])
        
        if token_counts:
            stats['avg_tokens_per_chunk'] = sum(token_counts) / len(token_counts)
            stats['avg_words_per_chunk'] = sum(word_counts) / len(word_counts)
            stats['avg_chars_per_chunk'] = sum(char_counts) / len(char_counts)
        
        if stats['coherence_scores']:
            stats['avg_coherence'] = sum(stats['coherence_scores']) / len(stats['coherence_scores'])
        if stats['completeness_scores']:
            stats['avg_completeness'] = sum(stats['completeness_scores']) / len(stats['completeness_scores'])
        
        # Calculate distribution
        stats['token_distribution'] = {
            'under_100': sum(1 for t in token_counts if t < 100),
            '100_300': sum(1 for t in token_counts if 100 <= t < 300),
            '300_500': sum(1 for t in token_counts if 300 <= t < 500),
            '500_700': sum(1 for t in token_counts if 500 <= t < 700),
            'over_700': sum(1 for t in token_counts if t >= 700),
        }
        
        return stats


# ========== CONVENIENCE FUNCTIONS ==========

def chunk_text(text: str, config: Optional[ChunkingConfig] = None) -> List[Dict[str, Any]]:
    """
    Convenience function for chunking text.
    
    Args:
        text: Text to chunk.
        config: Optional chunking configuration.
        
    Returns:
        List of chunk dictionaries.
    """
    chunker = TextChunker(config)
    return chunker.chunk_text(text)


def analyze_text_for_chunking(text: str, config: Optional[ChunkingConfig] = None) -> Dict[str, Any]:
    """
    Analyze text structure for optimal chunking.
    
    Args:
        text: Text to analyze.
        config: Optional chunking configuration.
        
    Returns:
        Dictionary with analysis results.
    """
    chunker = TextChunker(config)
    return chunker.analyze_text_structure(text)


def get_optimal_chunk_size(text: str, target_chunks: int = 10) -> int:
    """
    Calculate optimal chunk size for given text and target chunk count.
    
    Args:
        text: Text to analyze.
        target_chunks: Desired number of chunks.
        
    Returns:
        Recommended chunk size in tokens.
    """
    chunker = TextChunker()
    total_tokens = chunker.count_tokens(text)
    
    if target_chunks <= 0:
        target_chunks = 10
    
    optimal_size = max(100, total_tokens // target_chunks)
    
    # Round to nearest 50
    optimal_size = ((optimal_size + 25) // 50) * 50
    
    return optimal_size


# ========== TESTING AND VALIDATION ==========

def _test_chunking() -> Dict[str, bool]:
    """Test text chunking functionality."""
    test_results = {}
    
    # Sample text for testing
    sample_text = """
    This is a test document for chunking functionality. It contains multiple paragraphs
    to test different chunking strategies.
    
    The second paragraph discusses various aspects of text processing. Chunking is
    important for efficient information retrieval and natural language processing tasks.
    
    Finally, the third paragraph concludes the document. It provides a summary of
    the key points discussed above and suggests areas for future work.
    
    Additional content to ensure we have enough text for proper chunking tests.
    This should create at least 3-4 chunks with default settings.
    """
    
    # Test 1: Fixed size chunking
    try:
        config = ChunkingConfig(strategy=ChunkingStrategy.FIXED_SIZE, chunk_size=100)
        chunker = TextChunker(config)
        chunks = chunker.chunk_text(sample_text)
        
        test_results['fixed_size_chunking'] = (
            len(chunks) >= 2 and  # Should create multiple chunks
            all('text' in chunk for chunk in chunks) and
            all(chunk.get('token_count', 0) > 0 for chunk in chunks)
        )
    except Exception as e:
        test_results['fixed_size_chunking'] = False
        logger.error(f"Fixed size chunking test failed: {e}")
    
    # Test 2: Paragraph chunking
    try:
        config = ChunkingConfig(strategy=ChunkingStrategy.PARAGRAPH)
        chunker = TextChunker(config)
        chunks = chunker.chunk_text(sample_text)
        
        test_results['paragraph_chunking'] = (
            len(chunks) >= 3 and  # Should have at least 3 paragraphs
            all('text' in chunk for chunk in chunks)
        )
    except Exception as e:
        test_results['paragraph_chunking'] = False
        logger.error(f"Paragraph chunking test failed: {e}")
    
    # Test 3: Sentence chunking
    try:
        config = ChunkingConfig(strategy=ChunkingStrategy.SENTENCE)
        chunker = TextChunker(config)
        chunks = chunker.chunk_text(sample_text)
        
        test_results['sentence_chunking'] = (
            len(chunks) >= 5 and  # Should have multiple sentences
            all('text' in chunk for chunk in chunks)
        )
    except Exception as e:
        test_results['sentence_chunking'] = False
        logger.error(f"Sentence chunking test failed: {e}")
    
    # Test 4: Text analysis
    try:
        chunker = TextChunker()
        analysis = chunker.analyze_text_structure(sample_text)
        
        test_results['text_analysis'] = (
            'total_tokens' in analysis and
            'sentences' in analysis and
            'paragraphs' in analysis and
            analysis['total_tokens'] > 0
        )
    except Exception as e:
        test_results['text_analysis'] = False
        logger.error(f"Text analysis test failed: {e}")
    
    return test_results


# ========== MAIN MODULE EXECUTION ==========

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intelligent Text Chunking")
    parser.add_argument("--text", type=str, help="Text to chunk (or file path)")
    parser.add_argument("--file", type=str, help="File containing text to chunk")
    parser.add_argument("--strategy", type=str, choices=[s.value for s in ChunkingStrategy],
                       default="semantic", help="Chunking strategy")
    parser.add_argument("--chunk-size", type=int, default=500, help="Target chunk size in tokens")
    parser.add_argument("--overlap", type=int, default=50, help="Overlap between chunks")
    parser.add_argument("--output", type=str, help="Output file for chunks (JSON)")
    parser.add_argument("--analyze", action="store_true", help="Analyze text structure only")
    parser.add_argument("--stats", action="store_true", help="Show chunking statistics")
    parser.add_argument("--test", action="store_true", help="Run unit tests")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    
    if args.test:
        print("Running chunking tests...")
        results = _test_chunking()
        passed = sum(results.values())
        total = len(results)
        print(f"Tests passed: {passed}/{total} ({passed/total*100:.1f}%)")
        
        for test, success in results.items():
            status = "✓" if success else "✗"
            print(f"  {status} {test}")
    
    elif args.text or args.file:
        # Get text from argument or file
        text = ""
        
        if args.file:
            try:
                with open(args.file, 'r', encoding='utf-8') as f:
                    text = f.read()
                print(f"Read {len(text)} characters from {args.file}")
            except Exception as e:
                print(f"Error reading file: {e}")
                exit(1)
        else:
            text = args.text
        
        if not text.strip():
            print("Error: No text to chunk")
            exit(1)
        
        # Create configuration
        strategy = ChunkingStrategy(args.strategy)
        config = ChunkingConfig(
            strategy=strategy,
            chunk_size=args.chunk_size,
            chunk_overlap=args.overlap,
        )
        
        # Create chunker
        chunker = TextChunker(config)
        
        if args.analyze:
            # Analyze text only
            print("\n=== TEXT STRUCTURE ANALYSIS ===")
            analysis = chunker.analyze_text_structure(text)
            
            print(f"Total characters: {analysis['total_chars']:,}")
            print(f"Total words: {analysis['total_words']:,}")
            print(f"Total tokens: {analysis['total_tokens']:,}")
            print(f"Paragraphs: {analysis['paragraphs']}")
            print(f"Sentences: {analysis['sentences']}")
            if analysis['sentences'] > 0:
                print(f"Average sentence length: {analysis['avg_sentence_length']:.0f} chars")
            if analysis['paragraphs'] > 0:
                print(f"Average paragraph length: {analysis['avg_paragraph_length']:.0f} chars")
            print(f"Estimated chunks: {analysis['estimated_chunks']}")
            print(f"Recommended strategy: {analysis['recommended_strategy']}")
            
        else:
            # Chunk the text
            print(f"\nChunking text with {args.strategy} strategy...")
            print(f"Target chunk size: {args.chunk_size} tokens")
            print(f"Overlap: {args.overlap} tokens")
            
            start_time = time.time()
            chunks = chunker.chunk_text(text)
            elapsed_time = time.time() - start_time
            
            print(f"\nCreated {len(chunks)} chunks in {elapsed_time:.2f} seconds")
            
            # Show chunking statistics
            if args.stats:
                stats = chunker.get_chunking_statistics(chunks)
                
                print("\n=== CHUNKING STATISTICS ===")
                print(f"Total tokens: {stats['total_tokens']:,}")
                print(f"Average tokens per chunk: {stats['avg_tokens_per_chunk']:.1f}")
                print(f"Min tokens: {stats['min_tokens']}")
                print(f"Max tokens: {stats['max_tokens']}")
                
                if 'avg_coherence' in stats:
                    print(f"Average coherence: {stats['avg_coherence']:.2f}")
                if 'avg_completeness' in stats:
                    print(f"Average completeness: {stats['avg_completeness']:.2f}")
                
                print("\nToken distribution:")
                for range_name, count in stats['token_distribution'].items():
                    print(f"  {range_name}: {count} chunks")
            
            # Show sample chunks
            print("\n=== SAMPLE CHUNKS ===")
            for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
                chunk_text = chunk['text']
                preview = chunk_text[:100] + "..." if len(chunk_text) > 100 else chunk_text
                print(f"\nChunk {i + 1} (Tokens: {chunk.get('token_count', 'N/A')}, "
                      f"Words: {chunk.get('word_count', 'N/A')}):")
                print(f"  {preview}")
            
            if len(chunks) > 3:
                print(f"\n... and {len(chunks) - 3} more chunks")
            
            # Save to file if requested
            if args.output:
                try:
                    with open(args.output, 'w', encoding='utf-8') as f:
                        json.dump(chunks, f, indent=2, ensure_ascii=False)
                    print(f"\nChunks saved to: {args.output}")
                except Exception as e:
                    print(f"Error saving chunks: {e}")
    
    else:
        # Show usage
        print("=" * 70)
        print("INTELLIGENT TEXT CHUNKING")
        print("=" * 70)
        
        print("\nAvailable chunking strategies:")
        for strategy in ChunkingStrategy:
            print(f"  • {strategy.value}")
        
        print("\nUsage examples:")
        print("  python chunking.py --text \"Your text here\"")
        print("  python chunking.py --file document.txt")
        print("  python chunking.py --file doc.txt --strategy paragraph")
        print("  python chunking.py --file doc.txt --chunk-size 300 --overlap 30")
        print("  python chunking.py --file doc.txt --analyze")
        print("  python chunking.py --file doc.txt --stats")
        print("  python chunking.py --file doc.txt --output chunks.json")
        print("  python chunking.py --test")
        
        print("\nDefault configuration:")
        print(f"  Chunk size: {ChunkingConfig().chunk_size} tokens")
        print(f"  Overlap: {ChunkingConfig().chunk_overlap} tokens")
        print(f"  Strategy: {ChunkingConfig().strategy.value}")