# src/ai_engine/tagging.py

"""Auto-tagging functionality for documents.

Provides AI-powered tag generation, keyword extraction, and taxonomy management.
"""

import json
import time
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Union, Set
from collections import Counter
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Tag:
    """Tag for document."""
    name: str
    confidence: float
    category: Optional[str] = None
    source: str = "ai"
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class TaggingConfig:
    """Configuration for tagging."""
    max_tags: int = 10
    min_confidence: float = 0.5
    use_taxonomy: bool = True
    allow_new_tags: bool = True
    extract_keywords: bool = True
    language: str = "en"


class KeywordExtractor:
    """Keyword extractor for documents."""
    
    def __init__(self, llm_client):
        """Initialize keyword extractor.
        
        Args:
            llm_client: LLM client instance
        """
        self.llm_client = llm_client
        
        # Common stop words (simplified)
        self.stop_words = {
            'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
            'which', 'this', 'that', 'these', 'those', 'then', 'just', 'so', 'than',
            'such', 'both', 'through', 'about', 'for', 'is', 'of', 'while', 'during',
            'to', 'from', 'in', 'on', 'at', 'by', 'with', 'without', 'after', 'before'
        }
    
    def extract(self, text: str, num_keywords: int = 10) -> List[str]:
        """Extract keywords from text.
        
        Args:
            text: Input text
            num_keywords: Number of keywords to extract
            
        Returns:
            List of keywords
        """
        # Try AI-based extraction first
        try:
            return self._ai_extract(text, num_keywords)
        except Exception as e:
            logger.warning(f"AI keyword extraction failed, using statistical method: {e}")
            return self._statistical_extract(text, num_keywords)
    
    def extract_with_scores(self, text: str, num_keywords: int = 10) -> List[Tuple[str, float]]:
        """Extract keywords with confidence scores.
        
        Args:
            text: Input text
            num_keywords: Number of keywords to extract
            
        Returns:
            List of (keyword, score) tuples
        """
        try:
            return self._ai_extract_with_scores(text, num_keywords)
        except Exception as e:
            logger.warning(f"AI keyword extraction failed: {e}")
            # Fallback to simple scoring
            keywords = self._statistical_extract(text, num_keywords)
            return [(k, 0.5) for k in keywords]
    
    def extract_phrases(self, text: str, num_phrases: int = 5) -> List[str]:
        """Extract key phrases from text.
        
        Args:
            text: Input text
            num_phrases: Number of phrases to extract
            
        Returns:
            List of key phrases
        """
        prompt = f"""Extract the most important {num_phrases} key phrases from the following text. Return only the phrases as a comma-separated list, no explanations.

Text: {text}

Key phrases:"""
        
        try:
            response = self.llm_client.generate(prompt, max_tokens=200)
            phrases = [p.strip() for p in response.text.split(",")]
            return phrases[:num_phrases]
        except Exception as e:
            logger.error(f"Failed to extract phrases: {e}")
            return []
    
    def get_keyword_stats(self, text: str) -> Dict[str, Any]:
        """Get keyword statistics.
        
        Args:
            text: Input text
            
        Returns:
            Statistics dictionary
        """
        words = self._tokenize(text)
        word_freq = Counter(words)
        
        return {
            "total_words": len(words),
            "unique_words": len(word_freq),
            "top_keywords": word_freq.most_common(10),
            "vocabulary_size": len(set(words))
        }
    
    def _ai_extract(self, text: str, num_keywords: int) -> List[str]:
        """Extract keywords using AI."""
        prompt = f"""Extract the most important {num_keywords} keywords from the following text. Return only the keywords as a comma-separated list, no explanations.

Text: {text}

Keywords:"""
        
        response = self.llm_client.generate(prompt, max_tokens=200)
        keywords = [k.strip() for k in response.text.split(",")]
        return [k for k in keywords if k][:num_keywords]
    
    def _ai_extract_with_scores(self, text: str, num_keywords: int) -> List[Tuple[str, float]]:
        """Extract keywords with scores using AI."""
        prompt = f"""Extract the most important {num_keywords} keywords from the following text. For each keyword, provide a relevance score from 0 to 1. Return as JSON array with format [{{"keyword": "...", "score": 0.0}}].

Text: {text}

JSON:"""
        
        response = self.llm_client.generate(prompt, max_tokens=500)
        
        try:
            # Try to parse JSON response
            import json
            # Find JSON array in response
            match = re.search(r'\[.*\]', response.text, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return [(item["keyword"], item["score"]) for item in data[:num_keywords]]
        except:
            pass
        
        # Fallback
        return [(k, 0.5) for k in self._ai_extract(text, num_keywords)]
    
    def _statistical_extract(self, text: str, num_keywords: int) -> List[str]:
        """Extract keywords using statistical methods."""
        words = self._tokenize(text)
        
        # Remove stop words
        words = [w for w in words if w.lower() not in self.stop_words and len(w) > 2]
        
        # Count frequencies
        word_freq = Counter(words)
        
        # Return top keywords
        return [word for word, _ in word_freq.most_common(num_keywords)]
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        # Convert to lowercase and split by non-alphabetic characters
        words = re.findall(r'\b[a-z]+\b', text.lower())
        return words


class TaxonomyManager:
    """Manager for tag taxonomy."""
    
    def __init__(self, taxonomy_path: Optional[Union[str, Path]] = None):
        """Initialize taxonomy manager.
        
        Args:
            taxonomy_path: Path to taxonomy file
        """
        self.taxonomy_path = Path(taxonomy_path) if taxonomy_path else None
        self.taxonomy = {
            "categories": {},
            "tags": {}
        }
        self._load_taxonomy()
    
    def load_taxonomy(self, path: Optional[Union[str, Path]] = None) -> bool:
        """Load taxonomy from file.
        
        Args:
            path: Path to taxonomy file
            
        Returns:
            True if successful
        """
        load_path = Path(path) if path else self.taxonomy_path
        if not load_path or not load_path.exists():
            return False
        
        try:
            with open(load_path, 'r') as f:
                self.taxonomy = json.load(f)
            logger.info(f"Loaded taxonomy from {load_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load taxonomy: {e}")
            return False
    
    def save_taxonomy(self, path: Optional[Union[str, Path]] = None) -> bool:
        """Save taxonomy to file.
        
        Args:
            path: Path to save to
            
        Returns:
            True if successful
        """
        save_path = Path(path) if path else self.taxonomy_path
        if not save_path:
            return False
        
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(self.taxonomy, f, indent=2)
            logger.info(f"Saved taxonomy to {save_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save taxonomy: {e}")
            return False
    
    def add_tag(self, tag: str, category: Optional[str] = None,
                synonyms: List[str] = None) -> None:
        """Add tag to taxonomy.
        
        Args:
            tag: Tag name
            category: Tag category
            synonyms: List of synonyms
        """
        self.taxonomy["tags"][tag] = {
            "category": category,
            "synonyms": synonyms or []
        }
        
        if category and category not in self.taxonomy["categories"]:
            self.taxonomy["categories"][category] = []
        
        if category and tag not in self.taxonomy["categories"].get(category, []):
            self.taxonomy["categories"].setdefault(category, []).append(tag)
    
    def remove_tag(self, tag: str) -> bool:
        """Remove tag from taxonomy.
        
        Args:
            tag: Tag name
            
        Returns:
            True if removed
        """
        if tag in self.taxonomy["tags"]:
            # Remove from category
            category = self.taxonomy["tags"][tag].get("category")
            if category and category in self.taxonomy["categories"]:
                if tag in self.taxonomy["categories"][category]:
                    self.taxonomy["categories"][category].remove(tag)
            
            del self.taxonomy["tags"][tag]
            return True
        return False
    
    def get_categories(self) -> List[str]:
        """Get all categories.
        
        Returns:
            List of categories
        """
        return list(self.taxonomy["categories"].keys())
    
    def get_tags_by_category(self, category: str) -> List[str]:
        """Get tags in category.
        
        Returns:
            List of tags
        """
        return self.taxonomy["categories"].get(category, [])
    
    def search_tags(self, query: str) -> List[str]:
        """Search for tags matching query.
        
        Args:
            query: Search query
            
        Returns:
            List of matching tags
        """
        query = query.lower()
        results = []
        
        for tag in self.taxonomy["tags"]:
            if query in tag.lower():
                results.append(tag)
            else:
                # Check synonyms
                synonyms = self.taxonomy["tags"][tag].get("synonyms", [])
                if any(query in s.lower() for s in synonyms):
                    results.append(tag)
        
        return results
    
    def suggest_similar_tags(self, tag: str, max_suggestions: int = 5) -> List[str]:
        """Suggest similar tags.
        
        Args:
            tag: Input tag
            max_suggestions: Maximum suggestions
            
        Returns:
            List of similar tags
        """
        if tag not in self.taxonomy["tags"]:
            return []
        
        category = self.taxonomy["tags"][tag].get("category")
        if category:
            # Get other tags in same category
            tags_in_category = self.get_tags_by_category(category)
            return [t for t in tags_in_category if t != tag][:max_suggestions]
        
        return []
    
    def _load_taxonomy(self) -> None:
        """Load default taxonomy."""
        if self.taxonomy_path and self.taxonomy_path.exists():
            self.load_taxonomy()


class Tagger:
    """Main tagger for documents."""
    
    def __init__(self, llm_client, config: Optional[Union[TaggingConfig, Dict]] = None,
                 taxonomy_manager: Optional[TaxonomyManager] = None):
        """Initialize tagger.
        
        Args:
            llm_client: LLM client instance
            config: Tagging configuration
            taxonomy_manager: Taxonomy manager instance
        """
        self.llm_client = llm_client
        
        if isinstance(config, dict):
            self.config = TaggingConfig(**config)
        else:
            self.config = config or TaggingConfig()
        
        self.taxonomy_manager = taxonomy_manager or TaxonomyManager()
        self.keyword_extractor = KeywordExtractor(llm_client)
        
        # Import here to avoid circular imports
        from ai_engine.prompt_templates import get_prompt_manager
        self.prompt_manager = get_prompt_manager()
    
    def tag(self, text: str, max_tags: Optional[int] = None,
           min_confidence: Optional[float] = None) -> List[Tag]:
        """Generate tags for document.
        
        Args:
            text: Input text
            max_tags: Maximum number of tags
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of Tag objects
        """
        max_tags = max_tags or self.config.max_tags
        min_confidence = min_confidence or self.config.min_confidence
        
        # Get existing tags if using taxonomy
        existing_tags = []
        if self.config.use_taxonomy:
            existing_tags = self.taxonomy_manager.search_tags("")
        
        # Prepare prompt
        existing_tags_str = ", ".join(existing_tags[:20]) if existing_tags else "none"
        
        prompt = self.prompt_manager.render(
            "tagging",
            document=text,
            existing_tags=existing_tags_str
        )
        
        # Generate tags
        try:
            response = self.llm_client.generate(prompt, max_tokens=200)
            
            # Parse tags
            tags_text = response.text.strip()
            tag_names = [t.strip() for t in tags_text.split(",")]
            
            # Filter and score tags
            tags = []
            for tag_name in tag_names[:max_tags]:
                if not tag_name:
                    continue
                
                # Calculate confidence
                confidence = self._calculate_confidence(tag_name, text)
                
                if confidence >= min_confidence:
                    # Get category from taxonomy if available
                    category = None
                    if self.config.use_taxonomy and tag_name in self.taxonomy_manager.taxonomy["tags"]:
                        category = self.taxonomy_manager.taxonomy["tags"][tag_name].get("category")
                    
                    tag = Tag(
                        name=tag_name,
                        confidence=confidence,
                        category=category,
                        source="ai"
                    )
                    tags.append(tag)
            
            return tags
            
        except Exception as e:
            logger.error(f"Failed to generate tags: {e}")
            return []
    
    def tag_batch(self, texts: List[str], max_tags: Optional[int] = None) -> List[List[Tag]]:
        """Tag multiple documents.
        
        Args:
            texts: List of input texts
            max_tags: Maximum tags per document
            
        Returns:
            List of tag lists
        """
        results = []
        for text in texts:
            try:
                tags = self.tag(text, max_tags)
                results.append(tags)
            except Exception as e:
                logger.error(f"Failed to tag document: {e}")
                results.append([])
        return results
    
    def suggest_tags(self, text: str, existing_tags: List[str] = None) -> List[str]:
        """Suggest additional tags based on existing ones.
        
        Args:
            text: Input text
            existing_tags: Existing tags
            
        Returns:
            List of suggested tags
        """
        existing = existing_tags or []
        
        prompt = f"""Based on the following text and existing tags, suggest 3-5 additional relevant tags.

Text: {text}
Existing tags: {', '.join(existing) if existing else 'none'}

Suggested tags (as comma-separated list):"""
        
        try:
            response = self.llm_client.generate(prompt, max_tokens=150)
            suggestions = [s.strip() for s in response.text.split(",")]
            return [s for s in suggestions if s and s not in existing]
        except Exception as e:
            logger.error(f"Failed to suggest tags: {e}")
            return []
    
    def extract_keywords(self, text: str, num_keywords: int = 10) -> List[str]:
        """Extract keywords from text.
        
        Args:
            text: Input text
            num_keywords: Number of keywords
            
        Returns:
            List of keywords
        """
        if self.config.extract_keywords:
            return self.keyword_extractor.extract(text, num_keywords)
        return []
    
    def get_taxonomy(self) -> Dict[str, Any]:
        """Get current taxonomy.
        
        Returns:
            Taxonomy dictionary
        """
        return self.taxonomy_manager.taxonomy
    
    def add_to_taxonomy(self, tag: str, category: Optional[str] = None) -> None:
        """Add tag to taxonomy.
        
        Args:
            tag: Tag name
            category: Tag category
        """
        self.taxonomy_manager.add_tag(tag, category)
    
    def remove_from_taxonomy(self, tag: str) -> bool:
        """Remove tag from taxonomy.
        
        Args:
            tag: Tag name
            
        Returns:
            True if removed
        """
        return self.taxonomy_manager.remove_tag(tag)
    
    def _calculate_confidence(self, tag: str, text: str) -> float:
        """Calculate confidence score for a tag.
        
        Args:
            tag: Tag name
            text: Original text
            
        Returns:
            Confidence score (0-1)
        """
        # Simple confidence based on tag frequency in text
        tag_lower = tag.lower()
        text_lower = text.lower()
        
        # Check if tag appears in text
        if tag_lower in text_lower:
            # Count occurrences
            count = text_lower.count(tag_lower)
            # Normalize by text length
            words = len(text.split())
            base_confidence = min(1.0, count * 10 / words) if words > 0 else 0.5
        else:
            base_confidence = 0.3
        
        # Boost if tag is in taxonomy
        if self.config.use_taxonomy and tag in self.taxonomy_manager.taxonomy["tags"]:
            base_confidence += 0.2
        
        return min(1.0, base_confidence)


# Global instance getter
def get_tagger(llm_client=None, config: Optional[Union[TaggingConfig, Dict]] = None,
              taxonomy_manager: Optional[TaxonomyManager] = None) -> Tagger:
    """Get tagger instance.
    
    Args:
        llm_client: LLM client instance
        config: Tagging configuration
        taxonomy_manager: Taxonomy manager instance
        
    Returns:
        Tagger instance
    """
    from ai_engine.llm_client import get_llm_client, LLMConfig
    
    if llm_client is None:
        llm_client = get_llm_client(LLMConfig(
            model_name="gpt-3.5-turbo",
            provider="openai"
        ))
    
    return Tagger(llm_client, config, taxonomy_manager)