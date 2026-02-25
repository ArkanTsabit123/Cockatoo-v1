# src/ai_engine/conversation_memory.py

"""Conversation memory for maintaining context across interactions.

Provides storage, retrieval, and management of conversation history with
summarization capabilities for long-running conversations.
"""

import json
import time
import threading
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Union, Tuple
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Type of memory entry."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    SUMMARY = "summary"
    TOOL = "tool"


class MemoryPriority(Enum):
    """Priority for memory retention."""
    HIGH = 3
    MEDIUM = 2
    LOW = 1
    TRANSIENT = 0


@dataclass
class ConversationEntry:
    """Single conversation entry."""
    id: str
    type: MemoryType
    content: str
    timestamp: datetime
    tokens: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: MemoryPriority = MemoryPriority.MEDIUM
    embedding: Optional[List[float]] = None
    
    def __post_init__(self):
        """Generate ID if not provided."""
        if not self.id:
            content_hash = hashlib.md5(
                f"{self.content}{self.timestamp.isoformat()}".encode()
            ).hexdigest()[:12]
            self.id = f"entry_{content_hash}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["type"] = self.type.value
        data["priority"] = self.priority.value
        data["timestamp"] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationEntry':
        """Create from dictionary."""
        data = data.copy()
        data["type"] = MemoryType(data["type"])
        data["priority"] = MemoryPriority(data.get("priority", MemoryPriority.MEDIUM.value))
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class ConversationSummary:
    """Summarized conversation segment."""
    id: str
    content: str
    start_time: datetime
    end_time: datetime
    entry_count: int
    token_count: int
    topics: List[str] = field(default_factory=list)
    key_points: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["start_time"] = self.start_time.isoformat()
        data["end_time"] = self.end_time.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationSummary':
        """Create from dictionary."""
        data = data.copy()
        data["start_time"] = datetime.fromisoformat(data["start_time"])
        data["end_time"] = datetime.fromisoformat(data["end_time"])
        return cls(**data)


@dataclass
class ConversationConfig:
    """Configuration for conversation memory."""
    max_history_tokens: int = 4000
    max_history_entries: int = 100
    summary_threshold_tokens: int = 2000
    summary_trigger_count: int = 20
    enable_summarization: bool = True
    enable_embedding: bool = False
    ttl_days: int = 30
    persist_to_disk: bool = True
    memory_decay_hours: int = 24
    importance_threshold: float = 0.5


class MemoryStore:
    """Persistent storage for conversation memory."""
    
    def __init__(self, storage_path: Optional[Union[str, Path]] = None):
        """Initialize memory store.
        
        Args:
            storage_path: Path to storage directory
        """
        self.storage_path = Path(storage_path) if storage_path else Path.home() / ".cache" / "conversation_memory"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.conversations: Dict[str, Dict] = {}
        self._lock = threading.RLock()
        self._load_all()
    
    def save_conversation(self, conversation_id: str, data: Dict) -> bool:
        """Save conversation data.
        
        Args:
            conversation_id: Conversation ID
            data: Conversation data
            
        Returns:
            True if successful
        """
        with self._lock:
            try:
                file_path = self.storage_path / f"{conversation_id}.json"
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)
                
                self.conversations[conversation_id] = data
                logger.debug(f"Saved conversation {conversation_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to save conversation {conversation_id}: {e}")
                return False
    
    def load_conversation(self, conversation_id: str) -> Optional[Dict]:
        """Load conversation data.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Conversation data or None
        """
        with self._lock:
            # Check in-memory cache first
            if conversation_id in self.conversations:
                return self.conversations[conversation_id]
            
            # Load from disk
            try:
                file_path = self.storage_path / f"{conversation_id}.json"
                if file_path.exists():
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    self.conversations[conversation_id] = data
                    return data
            except Exception as e:
                logger.error(f"Failed to load conversation {conversation_id}: {e}")
            
            return None
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete conversation data.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            True if successful
        """
        with self._lock:
            try:
                file_path = self.storage_path / f"{conversation_id}.json"
                if file_path.exists():
                    file_path.unlink()
                
                if conversation_id in self.conversations:
                    del self.conversations[conversation_id]
                
                logger.info(f"Deleted conversation {conversation_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to delete conversation {conversation_id}: {e}")
                return False
    
    def list_conversations(self) -> List[str]:
        """List all conversation IDs.
        
        Returns:
            List of conversation IDs
        """
        with self._lock:
            # Get from disk
            try:
                files = self.storage_path.glob("*.json")
                return sorted([f.stem for f in files])
            except Exception as e:
                logger.error(f"Failed to list conversations: {e}")
                return []
    
    def cleanup_old(self, days: int) -> int:
        """Clean up conversations older than specified days.
        
        Args:
            days: Maximum age in days
            
        Returns:
            Number of conversations deleted
        """
        cutoff = time.time() - (days * 24 * 3600)
        deleted = 0
        
        for conversation_id in self.list_conversations():
            file_path = self.storage_path / f"{conversation_id}.json"
            try:
                if file_path.stat().st_mtime < cutoff:
                    if self.delete_conversation(conversation_id):
                        deleted += 1
            except Exception as e:
                logger.error(f"Error checking conversation {conversation_id}: {e}")
        
        return deleted
    
    def _load_all(self) -> None:
        """Load all conversations into memory."""
        for conversation_id in self.list_conversations():
            try:
                data = self.load_conversation(conversation_id)
                if data:
                    self.conversations[conversation_id] = data
            except Exception as e:
                logger.error(f"Failed to load conversation {conversation_id}: {e}")


class ConversationMemory:
    """Memory manager for conversations with summarization."""
    
    # Class variable for singleton pattern (optional)
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls, *args, **kwargs):
        """Optional singleton pattern - can be removed if not needed."""
        return super().__new__(cls)
    
    def __init__(self, config: Optional[Union[ConversationConfig, Dict]] = None,
                 llm_client=None, storage_path: Optional[Union[str, Path]] = None):
        """Initialize conversation memory.
        
        Args:
            config: Configuration
            llm_client: LLM client for summarization
            storage_path: Path for persistent storage
        """
        # Allow re-initialization for testing
        if hasattr(self, 'initialized') and self.initialized:
            return
            
        if isinstance(config, dict):
            self.config = ConversationConfig(**config)
        else:
            self.config = config or ConversationConfig()
        
        self.llm_client = llm_client
        self.storage = MemoryStore(storage_path)
        
        # In-memory storage
        self.entries: Dict[str, List[ConversationEntry]] = {}
        self.summaries: Dict[str, List[ConversationSummary]] = {}
        self.contexts: Dict[str, Dict[str, Any]] = {}
        
        # Metadata
        self.last_accessed: Dict[str, datetime] = {}
        self.access_count: Dict[str, int] = {}
        
        # Thread safety
        self._instance_lock = threading.RLock()
        
        # Mark as initialized
        self.initialized = True
        
        logger.info(f"ConversationMemory initialized with config: {self.config}")
    
    def add_entry(self, conversation_id: str, content: str,
                 entry_type: Union[str, MemoryType],
                 metadata: Optional[Dict] = None,
                 priority: Union[str, MemoryPriority] = MemoryPriority.MEDIUM,
                 tokens: Optional[int] = None) -> ConversationEntry:
        """Add an entry to conversation history.
        
        Args:
            conversation_id: Conversation ID
            content: Message content
            entry_type: Type of entry
            metadata: Additional metadata
            priority: Priority for retention
            tokens: Token count (auto-calculated if not provided)
            
        Returns:
            Created entry
        """
        if isinstance(entry_type, str):
            entry_type = MemoryType(entry_type)
        
        if isinstance(priority, str):
            priority = MemoryPriority[priority.upper()]
        
        # Calculate tokens if not provided
        if tokens is None:
            tokens = self._count_tokens(content)
        
        entry = ConversationEntry(
            id="",  # Will be generated in __post_init__
            type=entry_type,
            content=content,
            timestamp=datetime.now(),
            tokens=tokens,
            metadata=metadata or {},
            priority=priority
        )
        
        with self._instance_lock:
            if conversation_id not in self.entries:
                self.entries[conversation_id] = []
                self.summaries[conversation_id] = []
                self.contexts[conversation_id] = {}
            
            self.entries[conversation_id].append(entry)
            self.last_accessed[conversation_id] = datetime.now()
            self.access_count[conversation_id] = self.access_count.get(conversation_id, 0) + 1
            
            # Check if summarization is needed
            if self.config.enable_summarization and self.llm_client:
                self._check_summarization(conversation_id)
            
            # Persist if enabled
            if self.config.persist_to_disk:
                self._persist_conversation(conversation_id)
        
        logger.debug(f"Added {entry_type.value} entry to {conversation_id}")
        return entry
    
    def get_history(self, conversation_id: str, 
                   max_tokens: Optional[int] = None,
                   max_entries: Optional[int] = None,
                   include_summaries: bool = True) -> List[ConversationEntry]:
        """Get conversation history.
        
        Args:
            conversation_id: Conversation ID
            max_tokens: Maximum tokens to return
            max_entries: Maximum number of entries
            include_summaries: Whether to include summaries
            
        Returns:
            List of conversation entries
        """
        with self._instance_lock:
            if conversation_id not in self.entries:
                return []
            
            entries = self.entries[conversation_id].copy()
            self.last_accessed[conversation_id] = datetime.now()
            
            if not entries:
                return entries
            
            # Apply max entries limit
            if max_entries and len(entries) > max_entries:
                entries = entries[-max_entries:]
            
            # Apply token limit
            if max_tokens:
                entries = self._truncate_by_tokens(entries, max_tokens)
            
            # Include summaries if available and requested
            if include_summaries and conversation_id in self.summaries:
                summaries = self.summaries[conversation_id]
                if summaries:
                    # Add summary as context at beginning
                    latest_summary = summaries[-1]
                    summary_entry = ConversationEntry(
                        id=f"summary_{latest_summary.id}",
                        type=MemoryType.SUMMARY,
                        content=latest_summary.content,
                        timestamp=latest_summary.end_time,
                        metadata={"summary": True, "topics": latest_summary.topics}
                    )
                    entries.insert(0, summary_entry)
            
            return entries
    
    def get_recent(self, conversation_id: str, n: int = 10) -> List[ConversationEntry]:
        """Get n most recent entries.
        
        Args:
            conversation_id: Conversation ID
            n: Number of entries
            
        Returns:
            List of recent entries
        """
        return self.get_history(conversation_id, max_entries=n)
    
    def search(self, conversation_id: str, query: str,
              max_results: int = 5) -> List[Tuple[ConversationEntry, float]]:
        """Search conversation history.
        
        Args:
            conversation_id: Conversation ID
            query: Search query
            max_results: Maximum results
            
        Returns:
            List of (entry, relevance_score) tuples
        """
        with self._instance_lock:
            if conversation_id not in self.entries:
                return []
            
            results = []
            query_lower = query.lower()
            
            for entry in self.entries[conversation_id]:
                score = 0.0
                
                # Simple keyword matching
                content_lower = entry.content.lower()
                if query_lower in content_lower:
                    # Calculate score based on term frequency and position
                    words = content_lower.split()
                    query_words = query_lower.split()
                    
                    # Count occurrences
                    matches = sum(1 for word in words if any(qw in word for qw in query_words))
                    score = matches / max(1, len(words))
                    
                    # Boost for exact phrase matches
                    if query_lower in content_lower:
                        score *= 1.5
                
                # Boost for high priority
                if entry.priority == MemoryPriority.HIGH:
                    score *= 1.2
                elif entry.priority == MemoryPriority.LOW:
                    score *= 0.8
                
                if score > 0:
                    results.append((entry, min(1.0, score)))
            
            # Sort by score descending
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:max_results]
    
    def summarize_conversation(self, conversation_id: str) -> Optional[ConversationSummary]:
        """Generate summary of conversation.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Conversation summary or None
        """
        if not self.llm_client:
            logger.warning("No LLM client available for summarization")
            return None
        
        with self._instance_lock:
            if conversation_id not in self.entries:
                logger.debug(f"Conversation {conversation_id} not found")
                return None
            
            entries = self.entries[conversation_id]
            if len(entries) < 5:
                logger.debug(f"Conversation {conversation_id} too short to summarize (only {len(entries)} entries)")
                return None  # Too short to summarize
            
            # Prepare conversation text
            conv_text = self._format_conversation(entries[-20:])  # Last 20 messages
            
            # Generate summary
            prompt = f"""Summarize the following conversation, identifying key topics and main points:

{conv_text}

Provide a concise summary with:
1. Main topics discussed (comma-separated list)
2. Key points (bullet points)
3. Overall summary

Format:
Topics: [comma-separated topics]
Key Points:
- [point 1]
- [point 2]
Summary: [concise summary]"""
            
            try:
                response = self.llm_client.generate(prompt, max_tokens=300)
                
                # Parse response with better error handling
                topics = []
                key_points = []
                summary_text = ""
                
                lines = response.text.split('\n')
                current_section = None
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    if line.startswith('Topics:'):
                        topics_str = line.replace('Topics:', '').strip()
                        topics = [t.strip() for t in topics_str.split(',') if t.strip()]
                    elif line.startswith('Key Points:'):
                        current_section = 'key_points'
                    elif line.startswith('Summary:'):
                        current_section = 'summary'
                        summary_text = line.replace('Summary:', '').strip()
                    elif current_section == 'key_points':
                        if line.startswith('-') or line.startswith('•'):
                            key_points.append(line.lstrip('-• ').strip())
                    elif current_section == 'summary':
                        summary_text += ' ' + line
                
                # If parsing failed, use the whole response as summary
                if not summary_text:
                    summary_text = response.text[:500] + "..." if len(response.text) > 500 else response.text
                    logger.debug(f"Using fallback summary for {conversation_id}")
                
                # Calculate tokens
                token_count = self._count_tokens(summary_text)
                
                summary = ConversationSummary(
                    id=f"summary_{int(time.time())}",
                    content=summary_text,
                    start_time=entries[0].timestamp,
                    end_time=entries[-1].timestamp,
                    entry_count=len(entries),
                    token_count=token_count,
                    topics=topics,
                    key_points=key_points
                )
                
                # Initialize summaries list if needed
                if conversation_id not in self.summaries:
                    self.summaries[conversation_id] = []
                
                self.summaries[conversation_id].append(summary)
                
                # Persist
                if self.config.persist_to_disk:
                    self._persist_conversation(conversation_id)
                
                logger.info(f"Generated summary for {conversation_id} with {len(topics)} topics and {len(key_points)} key points")
                return summary
                
            except Exception as e:
                logger.error(f"Failed to generate summary for {conversation_id}: {e}")
                return None
    
    def get_context(self, conversation_id: str, 
                   additional_context: Optional[Dict] = None) -> Dict[str, Any]:
        """Get conversation context.
        
        Args:
            conversation_id: Conversation ID
            additional_context: Additional context to include
            
        Returns:
            Context dictionary
        """
        with self._instance_lock:
            context = self.contexts.get(conversation_id, {}).copy()
            
            # Add conversation stats
            if conversation_id in self.entries:
                entries = self.entries[conversation_id]
                context.update({
                    "conversation_id": conversation_id,
                    "total_messages": len(entries),
                    "total_tokens": sum(e.tokens for e in entries),
                    "last_accessed": self.last_accessed.get(conversation_id),
                    "access_count": self.access_count.get(conversation_id, 0)
                })
            
            # Add additional context
            if additional_context:
                context.update(additional_context)
            
            return context
    
    def update_context(self, conversation_id: str, 
                      context_updates: Dict[str, Any]) -> None:
        """Update conversation context.
        
        Args:
            conversation_id: Conversation ID
            context_updates: Context updates
        """
        with self._instance_lock:
            if conversation_id not in self.contexts:
                self.contexts[conversation_id] = {}
            
            self.contexts[conversation_id].update(context_updates)
            
            if self.config.persist_to_disk:
                self._persist_conversation(conversation_id)
    
    def clear_conversation(self, conversation_id: str) -> bool:
        """Clear conversation history.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            True if successful
        """
        with self._instance_lock:
            if conversation_id in self.entries:
                del self.entries[conversation_id]
            
            if conversation_id in self.summaries:
                del self.summaries[conversation_id]
            
            if conversation_id in self.contexts:
                del self.contexts[conversation_id]
            
            if conversation_id in self.last_accessed:
                del self.last_accessed[conversation_id]
            
            if conversation_id in self.access_count:
                del self.access_count[conversation_id]
            
            if self.config.persist_to_disk:
                self.storage.delete_conversation(conversation_id)
            
            logger.info(f"Cleared conversation {conversation_id}")
            return True
    
    def get_all_conversations(self) -> List[str]:
        """Get all conversation IDs.
        
        Returns:
            List of conversation IDs
        """
        with self._instance_lock:
            return list(self.entries.keys())
    
    def get_stats(self, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """Get memory statistics.
        
        Args:
            conversation_id: Optional specific conversation
            
        Returns:
            Statistics dictionary
        """
        with self._instance_lock:
            if conversation_id:
                if conversation_id not in self.entries:
                    return {}
                
                entries = self.entries[conversation_id]
                summaries = self.summaries.get(conversation_id, [])
                
                return {
                    "conversation_id": conversation_id,
                    "total_entries": len(entries),
                    "total_tokens": sum(e.tokens for e in entries),
                    "total_summaries": len(summaries),
                    "last_accessed": self.last_accessed.get(conversation_id),
                    "access_count": self.access_count.get(conversation_id, 0),
                    "entries_by_type": self._count_by_type(entries)
                }
            else:
                total_entries = sum(len(e) for e in self.entries.values())
                total_tokens = sum(sum(e.tokens for e in entries) for entries in self.entries.values())
                
                return {
                    "total_conversations": len(self.entries),
                    "total_entries": total_entries,
                    "total_tokens": total_tokens,
                    "total_summaries": sum(len(s) for s in self.summaries.values()),
                    "average_entries_per_conversation": total_entries / max(1, len(self.entries))
                }
    
    def prune_old(self, days: Optional[int] = None) -> int:
        """Prune conversations older than specified days.
        
        Args:
            days: Maximum age in days (uses config if not provided)
            
        Returns:
            Number of conversations pruned
        """
        days = days or self.config.ttl_days
        cutoff = datetime.now() - timedelta(days=days)
        pruned = 0
        
        with self._instance_lock:
            for conv_id in list(self.entries.keys()):
                last_accessed = self.last_accessed.get(conv_id)
                if last_accessed and last_accessed < cutoff:
                    if self.clear_conversation(conv_id):
                        pruned += 1
        
        logger.info(f"Pruned {pruned} conversations older than {days} days")
        return pruned
    
    def _check_summarization(self, conversation_id: str) -> None:
        """Check if conversation needs summarization.
        
        Args:
            conversation_id: Conversation ID
        """
        if conversation_id not in self.entries:
            return
        
        entries = self.entries[conversation_id]
        
        # Check token threshold
        total_tokens = sum(e.tokens for e in entries)
        if total_tokens > self.config.summary_threshold_tokens:
            logger.debug(f"Token threshold triggered for {conversation_id} ({total_tokens} tokens)")
            self.summarize_conversation(conversation_id)
        
        # Check entry count threshold
        elif len(entries) > self.config.summary_trigger_count:
            logger.debug(f"Entry count threshold triggered for {conversation_id} ({len(entries)} entries)")
            self.summarize_conversation(conversation_id)
    
    def _truncate_by_tokens(self, entries: List[ConversationEntry],
                           max_tokens: int) -> List[ConversationEntry]:
        """Truncate entries by token count (keep most recent).
        
        Args:
            entries: List of entries
            max_tokens: Maximum tokens
            
        Returns:
            Truncated list
        """
        total = 0
        result = []
        
        for entry in reversed(entries):
            if total + entry.tokens <= max_tokens:
                result.insert(0, entry)
                total += entry.tokens
            else:
                break
        
        return result
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text.
        
        Args:
            text: Input text
            
        Returns:
            Token count
        """
        # Simple approximation (4 chars per token on average)
        return max(1, (len(text) + 3) // 4)
    
    def _format_conversation(self, entries: List[ConversationEntry]) -> str:
        """Format conversation entries as text.
        
        Args:
            entries: List of entries
            
        Returns:
            Formatted conversation
        """
        lines = []
        for entry in entries:
            role = entry.type.value.capitalize()
            lines.append(f"{role}: {entry.content}")
        return "\n".join(lines)
    
    def _count_by_type(self, entries: List[ConversationEntry]) -> Dict[str, int]:
        """Count entries by type.
        
        Args:
            entries: List of entries
            
        Returns:
            Dictionary of counts by type
        """
        counts = {}
        for entry in entries:
            type_str = entry.type.value
            counts[type_str] = counts.get(type_str, 0) + 1
        return counts
    
    def _persist_conversation(self, conversation_id: str) -> None:
        """Persist conversation to disk.
        
        Args:
            conversation_id: Conversation ID
        """
        if conversation_id not in self.entries:
            return
        
        try:
            data = {
                "entries": [e.to_dict() for e in self.entries[conversation_id]],
                "summaries": [s.to_dict() for s in self.summaries.get(conversation_id, [])],
                "context": self.contexts.get(conversation_id, {}),
                "last_accessed": self.last_accessed.get(conversation_id).isoformat() if conversation_id in self.last_accessed else None,
                "access_count": self.access_count.get(conversation_id, 0),
                "updated_at": datetime.now().isoformat()
            }
            
            self.storage.save_conversation(conversation_id, data)
        except Exception as e:
            logger.error(f"Failed to persist conversation {conversation_id}: {e}")
    
    def load_conversation(self, conversation_id: str) -> bool:
        """Load conversation from disk.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            True if loaded
        """
        try:
            data = self.storage.load_conversation(conversation_id)
            if not data:
                return False
            
            with self._instance_lock:
                # Load entries
                self.entries[conversation_id] = [
                    ConversationEntry.from_dict(e) for e in data.get("entries", [])
                ]
                
                # Load summaries
                self.summaries[conversation_id] = [
                    ConversationSummary.from_dict(s) for s in data.get("summaries", [])
                ]
                
                # Load context
                self.contexts[conversation_id] = data.get("context", {})
                
                # Load metadata
                if data.get("last_accessed"):
                    self.last_accessed[conversation_id] = datetime.fromisoformat(data["last_accessed"])
                if data.get("access_count"):
                    self.access_count[conversation_id] = data["access_count"]
            
            logger.info(f"Loaded conversation {conversation_id} with {len(self.entries[conversation_id])} entries")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load conversation {conversation_id}: {e}")
            return False


# Global instance getter
def get_conversation_memory(config: Optional[Union[ConversationConfig, Dict]] = None,
                           llm_client=None,
                           storage_path: Optional[Union[str, Path]] = None) -> ConversationMemory:
    """Get conversation memory instance.
    
    Args:
        config: Configuration
        llm_client: LLM client for summarization
        storage_path: Path for persistent storage
        
    Returns:
        ConversationMemory instance
    """
    return ConversationMemory(config, llm_client, storage_path)