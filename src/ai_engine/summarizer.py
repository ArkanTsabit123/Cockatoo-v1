# src/ai_engine/summarizer.py

"""Document summarization functionality.

Provides extractive and abstractive summarization for documents.
"""

import time
import math
from enum import Enum
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SummaryStyle(Enum):
    """Summary style enumeration."""
    CONCISE = "concise"
    DETAILED = "detailed"
    BULLET_POINTS = "bullet_points"
    PARAGRAPH = "paragraph"
    EXECUTIVE = "executive"


@dataclass
class SummaryResponse:
    """Response from summarizer."""
    summary: str
    key_points: List[str]
    original_length: int
    summary_length: int
    compression_ratio: float
    style: SummaryStyle
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["style"] = self.style.value
        return data


class Summarizer:
    """Document summarizer."""
    
    def __init__(self, llm_client, config: Optional[Dict] = None):
        """Initialize summarizer.
        
        Args:
            llm_client: LLM client instance
            config: Configuration dictionary
        """
        self.llm_client = llm_client
        self.config = config or {}
        
        # Import here to avoid circular imports
        from ai_engine.prompt_templates import get_prompt_manager
        self.prompt_manager = get_prompt_manager()
    
    def summarize(self, text: str, max_length: Optional[int] = None,
                 min_length: Optional[int] = None,
                 style: Union[str, SummaryStyle] = SummaryStyle.CONCISE) -> SummaryResponse:
        """Summarize text.
        
        Args:
            text: Input text
            max_length: Maximum summary length
            min_length: Minimum summary length
            style: Summary style
            
        Returns:
            SummaryResponse object
        """
        if isinstance(style, str):
            style = SummaryStyle(style)
        
        # Prepare prompt
        style_desc = self._get_style_description(style)
        
        prompt = f"""Please summarize the following text in a {style_desc} manner:

{text}

Summary:"""
        
        # Add length constraints
        if max_length:
            prompt = prompt.replace("Summary:", f"Keep the summary under {max_length} words.\n\nSummary:")
        if min_length:
            prompt = prompt.replace("Summary:", f"Ensure the summary is at least {min_length} words.\n\nSummary:")
        
        # Generate summary
        start_time = time.time()
        response = self.llm_client.generate(prompt)
        latency = time.time() - start_time
        
        summary = response.text.strip()
        
        # Extract key points
        key_points = self._extract_key_points(text, summary)
        
        # Calculate metrics
        original_length = len(text.split())
        summary_length = len(summary.split())
        compression_ratio = summary_length / original_length if original_length > 0 else 0
        
        return SummaryResponse(
            summary=summary,
            key_points=key_points,
            original_length=original_length,
            summary_length=summary_length,
            compression_ratio=compression_ratio,
            style=style,
            metadata={
                "latency": latency,
                "token_usage": response.usage
            }
        )
    
    def summarize_batch(self, texts: List[str], max_length: Optional[int] = None,
                       style: Union[str, SummaryStyle] = SummaryStyle.CONCISE) -> List[SummaryResponse]:
        """Summarize multiple texts.
        
        Args:
            texts: List of input texts
            max_length: Maximum summary length
            style: Summary style
            
        Returns:
            List of SummaryResponse objects
        """
        results = []
        for text in texts:
            try:
                result = self.summarize(text, max_length, style=style)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to summarize text: {e}")
                results.append(None)
        return results
    
    def summarize_with_points(self, text: str, num_points: int = 5) -> Dict[str, Any]:
        """Summarize and extract key points.
        
        Args:
            text: Input text
            num_points: Number of key points
            
        Returns:
            Dictionary with summary and key points
        """
        prompt = f"""Please provide:
1. A concise summary of the following text
2. {num_points} key points from the text

Text: {text}

Summary:
Key Points:"""
        
        response = self.llm_client.generate(prompt)
        
        # Parse response (simple parsing)
        parts = response.text.split("Key Points:")
        summary = parts[0].replace("Summary:", "").strip() if len(parts) > 0 else ""
        
        key_points = []
        if len(parts) > 1:
            points_text = parts[1].strip()
            for line in points_text.split("\n"):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith("-") or line.startswith("•")):
                    # Remove numbering/bullets
                    clean_line = line.lstrip("0123456789.-• ").strip()
                    if clean_line:
                        key_points.append(clean_line)
        
        return {
            "summary": summary,
            "key_points": key_points[:num_points]
        }
    
    def extractive_summarize(self, text: str, ratio: float = 0.3) -> str:
        """Extractive summarization (selects important sentences).
        
        Args:
            text: Input text
            ratio: Compression ratio (0-1)
            
        Returns:
            Extractive summary
        """
        # Simple extractive summarization using sentence scoring
        sentences = text.split(". ")
        if not sentences:
            return text
        
        num_sentences = max(1, int(len(sentences) * ratio))
        
        # Score sentences by length and position
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            # Position score (earlier sentences are more important)
            position_score = 1.0 - (i / len(sentences))
            # Length score (medium length sentences are better)
            length = len(sentence.split())
            length_score = min(1.0, length / 20) if length < 20 else 1.0
            
            total_score = (position_score + length_score) / 2
            scored_sentences.append((sentence, total_score))
        
        # Sort by score and take top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        selected = [s[0] for s in scored_sentences[:num_sentences]]
        
        # Restore original order
        ordered = []
        for sentence in sentences:
            if sentence in selected:
                ordered.append(sentence)
        
        return ". ".join(ordered) + "."
    
    def abstractive_summarize(self, text: str, max_length: Optional[int] = None) -> str:
        """Abstractive summarization (generates new text).
        
        Args:
            text: Input text
            max_length: Maximum summary length
            
        Returns:
            Abstractive summary
        """
        prompt = f"""Generate a concise abstractive summary of the following text:

{text}

Summary:"""
        
        if max_length:
            prompt = prompt.replace("Summary:", f"Keep the summary under {max_length} words.\n\nSummary:")
        
        response = self.llm_client.generate(prompt)
        return response.text.strip()
    
    def get_summary_stats(self, text: str) -> Dict[str, Any]:
        """Get summary statistics.
        
        Args:
            text: Input text
            
        Returns:
            Statistics dictionary
        """
        sentences = text.split(". ")
        words = text.split()
        
        return {
            "characters": len(text),
            "words": len(words),
            "sentences": len(sentences),
            "avg_word_length": sum(len(w) for w in words) / len(words) if words else 0,
            "avg_sentence_length": len(words) / len(sentences) if sentences else 0
        }
    
    def _get_style_description(self, style: SummaryStyle) -> str:
        """Get description for summary style."""
        descriptions = {
            SummaryStyle.CONCISE: "concise and to the point",
            SummaryStyle.DETAILED: "detailed and comprehensive",
            SummaryStyle.BULLET_POINTS: "bullet point format",
            SummaryStyle.PARAGRAPH: "single paragraph",
            SummaryStyle.EXECUTIVE: "executive summary style with key highlights"
        }
        return descriptions.get(style, "concise")
    
    def _extract_key_points(self, text: str, summary: str) -> List[str]:
        """Extract key points from text.
        
        Args:
            text: Original text
            summary: Generated summary
            
        Returns:
            List of key points
        """
        prompt = f"""Based on the following text and its summary, extract 3-5 key points:

Text: {text}
Summary: {summary}

Key points (as a numbered list):"""
        
        try:
            response = self.llm_client.generate(prompt, max_tokens=300)
            
            # Parse numbered list
            points = []
            for line in response.text.split("\n"):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith("-") or line.startswith("•")):
                    clean_line = line.lstrip("0123456789.-• ").strip()
                    if clean_line:
                        points.append(clean_line)
            
            return points[:5]
        except Exception as e:
            logger.error(f"Failed to extract key points: {e}")
            return []


class ChunkedSummarizer:
    """Summarizer for long documents using chunking."""
    
    def __init__(self, summarizer: Summarizer, chunk_size: int = 1000,
                 chunk_overlap: int = 100):
        """Initialize chunked summarizer.
        
        Args:
            summarizer: Base summarizer instance
            chunk_size: Size of each chunk in words
            chunk_overlap: Overlap between chunks
        """
        self.summarizer = summarizer
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def summarize_long(self, text: str, max_length: Optional[int] = None,
                      style: Union[str, SummaryStyle] = SummaryStyle.CONCISE) -> SummaryResponse:
        """Summarize long document.
        
        Args:
            text: Long input text
            max_length: Maximum summary length
            style: Summary style
            
        Returns:
            SummaryResponse object
        """
        # Split into chunks
        chunks = self._chunk_text(text)
        
        # Summarize each chunk
        chunk_summaries = []
        for chunk in chunks:
            try:
                response = self.summarizer.summarize(chunk, style=style)
                chunk_summaries.append(response.summary)
            except Exception as e:
                logger.error(f"Failed to summarize chunk: {e}")
                chunk_summaries.append("")
        
        # Combine chunk summaries
        combined = " ".join(chunk_summaries)
        
        # Final summary
        final_response = self.summarizer.summarize(combined, max_length, style=style)
        
        return final_response
    
    def hierarchical_summarize(self, text: str, depth: int = 2) -> str:
        """Hierarchical summarization.
        
        Args:
            text: Input text
            depth: Number of summarization levels
            
        Returns:
            Hierarchical summary
        """
        current_text = text
        
        for level in range(depth):
            logger.info(f"Hierarchical summarization level {level + 1}/{depth}")
            
            # Split into chunks
            chunks = self._chunk_text(current_text, chunk_size=self.chunk_size // (level + 1))
            
            # Summarize chunks
            chunk_summaries = []
            for chunk in chunks:
                try:
                    response = self.summarizer.summarize(chunk, style=SummaryStyle.CONCISE)
                    chunk_summaries.append(response.summary)
                except Exception as e:
                    logger.error(f"Failed to summarize chunk: {e}")
                    chunk_summaries.append(chunk[:200])  # Truncate as fallback
            
            # Combine for next level
            current_text = " ".join(chunk_summaries)
        
        return current_text
    
    def _chunk_text(self, text: str, chunk_size: Optional[int] = None) -> List[str]:
        """Split text into chunks.
        
        Args:
            text: Input text
            chunk_size: Size of each chunk
            
        Returns:
            List of text chunks
        """
        chunk_size = chunk_size or self.chunk_size
        
        words = text.split()
        chunks = []
        
        i = 0
        while i < len(words):
            # Get chunk
            end = min(i + chunk_size, len(words))
            chunk = " ".join(words[i:end])
            chunks.append(chunk)
            
            # Move with overlap
            i += chunk_size - self.chunk_overlap
        
        return chunks


# Global instance getter
def get_summarizer(llm_client=None, config: Optional[Dict] = None) -> Summarizer:
    """Get summarizer instance.
    
    Args:
        llm_client: LLM client instance
        config: Configuration dictionary
        
    Returns:
        Summarizer instance
    """
    from ai_engine.llm_client import get_llm_client, LLMConfig
    
    if llm_client is None:
        llm_client = get_llm_client(LLMConfig(
            model_name="gpt-3.5-turbo",
            provider="openai"
        ))
    
    return Summarizer(llm_client, config)