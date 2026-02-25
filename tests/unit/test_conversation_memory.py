# tests/unit/test_conversation_memory.py

"""Unit tests for conversation memory system.

Tests cover conversation storage, retrieval, context management,
summarization, pruning, search, export, and memory optimization.
"""

import time
import json
import threading
import tempfile
import os
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, Mock
from typing import Dict, List, Any, Optional, Generator

import pytest
import numpy as np

from src.ai_engine.conversation_memory import (
    ConversationMemory,
    ConversationConfig,
    ConversationEntry,
    ConversationSummary,
    MemoryType,
    MemoryPriority,
    MemoryStore
)


@pytest.fixture
def memory_config(tmp_path):
    """Create memory configuration."""
    return ConversationConfig(
        max_history_tokens=4000,
        max_history_entries=100,
        summary_threshold_tokens=2000,
        summary_trigger_count=20,
        enable_summarization=True,
        enable_embedding=False,
        ttl_days=30,
        persist_to_disk=False,
        memory_decay_hours=24,
        importance_threshold=0.5
    )


@pytest.fixture
def conversation_memory(memory_config):
    """Create conversation memory instance."""
    # Reset singleton if it exists
    if hasattr(ConversationMemory, '_instance'):
        ConversationMemory._instance = None
    memory = ConversationMemory(config=memory_config)
    return memory


@pytest.fixture
def mock_llm():
    """Create mock LLM client for summarization."""
    mock = MagicMock()
    
    # Create a response that matches the expected format
    class MockResponse:
        def __init__(self, text):
            self.text = text
    
    mock.generate.return_value = MockResponse(
        "Topics: test, conversation, machine learning\n"
        "Key Points:\n"
        "- The user needs help with a document\n"
        "- The document is a PDF about machine learning\n"
        "- The user wants a summary of main topics\n"
        "Summary: The conversation involves a user seeking help with a machine learning PDF document, "
        "with the assistant offering to help extract information and summarize topics."
    )
    return mock


@pytest.fixture
def sample_entries():
    """Create sample conversation entries."""
    now = datetime.now()
    return [
        ConversationEntry(
            id="entry_1",
            type=MemoryType.USER,
            content="Hello, I need help with my document.",
            timestamp=now - timedelta(minutes=10),
            tokens=8,
            metadata={"source": "test", "importance": 0.8},
            priority=MemoryPriority.HIGH
        ),
        ConversationEntry(
            id="entry_2",
            type=MemoryType.ASSISTANT,
            content="Of course! What kind of document are you working with?",
            timestamp=now - timedelta(minutes=9),
            tokens=10,
            metadata={"source": "test", "importance": 0.6},
            priority=MemoryPriority.MEDIUM
        ),
        ConversationEntry(
            id="entry_3",
            type=MemoryType.USER,
            content="It's a PDF file about machine learning.",
            timestamp=now - timedelta(minutes=5),
            tokens=7,
            metadata={"source": "test", "importance": 0.9},
            priority=MemoryPriority.HIGH
        ),
        ConversationEntry(
            id="entry_4",
            type=MemoryType.ASSISTANT,
            content="I can help you extract information from that PDF. What would you like to know?",
            timestamp=now - timedelta(minutes=4),
            tokens=12,
            metadata={"source": "test", "importance": 0.7},
            priority=MemoryPriority.MEDIUM
        ),
        ConversationEntry(
            id="entry_5",
            type=MemoryType.USER,
            content="Can you summarize the main topics?",
            timestamp=now - timedelta(minutes=2),
            tokens=5,
            metadata={"source": "test", "importance": 0.85},
            priority=MemoryPriority.HIGH
        )
    ]


@pytest.fixture
def populated_memory(conversation_memory, sample_entries):
    """Create memory with pre-populated conversations."""
    conv_id = "test_conv_001"
    for entry in sample_entries:
        conversation_memory.add_entry(
            conversation_id=conv_id,
            content=entry.content,
            entry_type=entry.type,
            metadata=entry.metadata,
            priority=entry.priority,
            tokens=entry.tokens
        )
    
    # Add a few more conversations - creating 4 total conversations
    for i in range(1, 4):  # This creates 3 additional conversations (002, 003, 004)
        conv_id = f"test_conv_{i+1:03d}"  # Start from 002
        for j in range(3):
            conversation_memory.add_entry(
                conversation_id=conv_id,
                content=f"Test message {j} in conversation {i+1}",
                entry_type=MemoryType.USER if j % 2 == 0 else MemoryType.ASSISTANT,
                metadata={"test": True}
            )
    
    return conversation_memory


# ==================== TEST CLASSES ====================

class TestConversationConfig:
    """Test conversation configuration."""

    def test_default_configuration(self):
        """Test default configuration values."""
        config = ConversationConfig()
        assert config.max_history_tokens == 4000
        assert config.max_history_entries == 100
        assert config.summary_threshold_tokens == 2000
        assert config.summary_trigger_count == 20
        assert config.enable_summarization is True
        assert config.ttl_days == 30
        assert config.persist_to_disk is True

    def test_custom_configuration(self, memory_config):
        """Test custom configuration values."""
        assert memory_config.max_history_tokens == 4000
        assert memory_config.max_history_entries == 100
        assert memory_config.summary_threshold_tokens == 2000
        assert memory_config.summary_trigger_count == 20
        assert memory_config.enable_summarization is True
        assert memory_config.ttl_days == 30


class TestConversationEntry:
    """Test ConversationEntry model."""

    def test_entry_creation(self):
        """Test creating a conversation entry."""
        now = datetime.now()
        entry = ConversationEntry(
            id="test_id",
            type=MemoryType.USER,
            content="Hello world",
            timestamp=now,
            tokens=10,
            metadata={"source": "test"},
            priority=MemoryPriority.HIGH
        )

        assert entry.id == "test_id"
        assert entry.type == MemoryType.USER
        assert entry.content == "Hello world"
        assert entry.timestamp == now
        assert entry.tokens == 10
        assert entry.metadata["source"] == "test"
        assert entry.priority == MemoryPriority.HIGH

    def test_auto_id_generation(self):
        """Test automatic ID generation when not provided."""
        entry = ConversationEntry(
            id="",
            type=MemoryType.USER,
            content="Test content",
            timestamp=datetime.now()
        )
        
        assert entry.id.startswith("entry_")
        assert len(entry.id) > 5

    def test_to_dict_from_dict(self):
        """Test dictionary conversion."""
        now = datetime.now()
        original = ConversationEntry(
            id="test_id",
            type=MemoryType.ASSISTANT,
            content="Test response",
            timestamp=now,
            tokens=5,
            metadata={"test": True},
            priority=MemoryPriority.HIGH
        )
        
        dict_data = original.to_dict()
        restored = ConversationEntry.from_dict(dict_data)
        
        assert restored.id == original.id
        assert restored.type == original.type
        assert restored.content == original.content
        assert restored.timestamp.isoformat() == now.isoformat()
        assert restored.tokens == original.tokens
        assert restored.metadata == original.metadata
        assert restored.priority == original.priority


class TestConversationSummary:
    """Test ConversationSummary model."""

    def test_summary_creation(self):
        """Test creating conversation summary."""
        start = datetime.now() - timedelta(hours=1)
        end = datetime.now()
        
        summary = ConversationSummary(
            id="summary_1",
            content="Test summary content",
            start_time=start,
            end_time=end,
            entry_count=10,
            token_count=100,
            topics=["topic1", "topic2"],
            key_points=["point1", "point2"]
        )
        
        assert summary.id == "summary_1"
        assert summary.content == "Test summary content"
        assert summary.start_time == start
        assert summary.end_time == end
        assert summary.entry_count == 10
        assert summary.token_count == 100
        assert summary.topics == ["topic1", "topic2"]
        assert summary.key_points == ["point1", "point2"]

    def test_to_dict_from_dict(self):
        """Test dictionary conversion."""
        start = datetime.now() - timedelta(hours=1)
        end = datetime.now()
        
        original = ConversationSummary(
            id="summary_1",
            content="Test",
            start_time=start,
            end_time=end,
            entry_count=5,
            token_count=50,
            topics=["test"],
            key_points=["point"]
        )
        
        dict_data = original.to_dict()
        restored = ConversationSummary.from_dict(dict_data)
        
        assert restored.id == original.id
        assert restored.content == original.content
        assert restored.start_time.isoformat() == start.isoformat()
        assert restored.end_time.isoformat() == end.isoformat()
        assert restored.entry_count == original.entry_count
        assert restored.token_count == original.token_count
        assert restored.topics == original.topics
        assert restored.key_points == original.key_points


class TestConversationMemoryInitialization:
    """Test conversation memory initialization."""

    def test_default_initialization(self, conversation_memory):
        """Test default initialization."""
        assert conversation_memory.config.max_history_entries == 100
        assert conversation_memory.entries == {}
        assert conversation_memory.summaries == {}
        assert conversation_memory.contexts == {}

    def test_with_llm_client(self, memory_config, mock_llm):
        """Test initialization with LLM client."""
        memory = ConversationMemory(config=memory_config, llm_client=mock_llm)
        assert memory.llm_client == mock_llm

    def test_with_storage_path(self, tmp_path):
        """Test initialization with storage path."""
        storage_path = tmp_path / "memory_storage"
        memory = ConversationMemory(storage_path=storage_path)
        assert memory.storage.storage_path == storage_path


class TestMemoryStore:
    """Test MemoryStore functionality."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_save_load_conversation(self, temp_storage):
        """Test saving and loading conversation."""
        store = MemoryStore(storage_path=temp_storage)
        
        conv_id = "test_conv"
        data = {
            "id": conv_id,
            "entries": [],
            "updated_at": datetime.now().isoformat()
        }
        
        # Save
        assert store.save_conversation(conv_id, data) is True
        
        # Load
        loaded = store.load_conversation(conv_id)
        assert loaded is not None
        assert loaded["id"] == conv_id

    def test_delete_conversation(self, temp_storage):
        """Test deleting conversation."""
        store = MemoryStore(storage_path=temp_storage)
        
        conv_id = "test_conv"
        data = {"id": conv_id}
        
        store.save_conversation(conv_id, data)
        assert store.load_conversation(conv_id) is not None
        
        store.delete_conversation(conv_id)
        assert store.load_conversation(conv_id) is None

    def test_list_conversations(self, temp_storage):
        """Test listing conversations."""
        store = MemoryStore(storage_path=temp_storage)
        
        # Save multiple conversations
        for i in range(3):
            store.save_conversation(f"conv_{i}", {"id": f"conv_{i}"})
        
        conversations = store.list_conversations()
        assert len(conversations) == 3
        assert all(f"conv_{i}" in conversations for i in range(3))

    def test_cleanup_old(self, temp_storage):
        """Test cleaning up old conversations."""
        store = MemoryStore(storage_path=temp_storage)
        
        # Save conversations
        for i in range(3):
            conv_id = f"conv_{i}"
            store.save_conversation(conv_id, {"id": conv_id})
            
            # Manually modify file timestamps for older conversations
            if i == 0:
                file_path = temp_storage / f"{conv_id}.json"
                old_time = time.time() - (10 * 24 * 3600)  # 10 days ago
                os.utime(file_path, (old_time, old_time))
        
        # Cleanup older than 5 days
        deleted = store.cleanup_old(days=5)
        assert deleted == 1


class TestConversationOperations:
    """Test conversation CRUD operations."""

    def test_add_entry(self, conversation_memory):
        """Test adding an entry."""
        entry = conversation_memory.add_entry(
            conversation_id="test_conv",
            content="Test message",
            entry_type=MemoryType.USER,
            metadata={"source": "test"}
        )

        assert entry.id.startswith("entry_")
        assert entry.content == "Test message"
        assert entry.type == MemoryType.USER
        assert "test_conv" in conversation_memory.entries
        assert len(conversation_memory.entries["test_conv"]) == 1

    def test_add_entry_with_string_types(self, conversation_memory):
        """Test adding entry with string type and priority."""
        entry = conversation_memory.add_entry(
            conversation_id="test_conv",
            content="Test message",
            entry_type="user",
            priority="high"
        )

        assert entry.type == MemoryType.USER
        assert entry.priority == MemoryPriority.HIGH

    def test_get_history(self, populated_memory):
        """Test getting conversation history."""
        conv_id = "test_conv_001"
        history = populated_memory.get_history(conv_id)
        
        assert len(history) >= 5
        assert all(isinstance(e, ConversationEntry) for e in history)

    def test_get_history_with_limit(self, populated_memory):
        """Test history with entry limit."""
        conv_id = "test_conv_001"
        history = populated_memory.get_history(conv_id, max_entries=3)
        
        assert len(history) == 3

    def test_get_history_with_token_limit(self, populated_memory):
        """Test history with token limit."""
        conv_id = "test_conv_001"
        history = populated_memory.get_history(conv_id, max_tokens=30)
        
        total_tokens = sum(e.tokens for e in history)
        assert total_tokens <= 30

    def test_get_recent(self, populated_memory):
        """Test getting recent entries."""
        conv_id = "test_conv_001"
        recent = populated_memory.get_recent(conv_id, n=2)
        
        assert len(recent) == 2
        # Should be most recent (by timestamp)
        assert recent[0].timestamp >= recent[1].timestamp

    def test_search(self, populated_memory):
        """Test searching entries."""
        conv_id = "test_conv_001"
        results = populated_memory.search(conv_id, "machine learning")
        
        assert len(results) > 0
        for entry, score in results:
            assert "machine learning" in entry.content.lower()
            assert 0 <= score <= 1

    def test_search_with_priority_boost(self, populated_memory, sample_entries):
        """Test search with priority boosting scores."""
        conv_id = "test_conv_001"
        results = populated_memory.search(conv_id, "document")
        
        # High priority entries should appear first
        if len(results) > 1:
            assert results[0][0].priority.value >= results[1][0].priority.value

    def test_get_nonexistent_conversation(self, conversation_memory):
        """Test getting history for non-existent conversation."""
        history = conversation_memory.get_history("nonexistent")
        assert history == []


class TestContextManagement:
    """Test context management functionality."""

    def test_get_context(self, populated_memory):
        """Test getting conversation context."""
        conv_id = "test_conv_001"
        context = populated_memory.get_context(conv_id)
        
        assert context["conversation_id"] == conv_id
        assert context["total_messages"] >= 5
        assert "last_accessed" in context

    def test_update_context(self, populated_memory):
        """Test updating context."""
        conv_id = "test_conv_001"
        
        populated_memory.update_context(conv_id, {"key": "value"})
        context = populated_memory.get_context(conv_id)
        assert context["key"] == "value"

    def test_context_with_additional_data(self, populated_memory):
        """Test context with additional data."""
        conv_id = "test_conv_001"
        context = populated_memory.get_context(conv_id, {"extra": "data"})
        
        assert context["extra"] == "data"
        assert context["conversation_id"] == conv_id


class TestSummarization:
    """Test conversation summarization."""

    def test_summarize_conversation(self, conversation_memory, sample_entries, mock_llm):
        """Test generating conversation summary."""
        conversation_memory.llm_client = mock_llm
        conv_id = "test_conv"
        
        for entry in sample_entries:
            if conv_id not in conversation_memory.entries:
                conversation_memory.entries[conv_id] = []
            conversation_memory.entries[conv_id].append(entry)

        summary = conversation_memory.summarize_conversation(conv_id)
        
        assert summary is not None
        assert isinstance(summary, ConversationSummary)
        assert "machine learning" in summary.content or "conversation" in summary.content
        assert len(summary.topics) > 0
        assert len(summary.key_points) > 0
        assert summary.entry_count == len(sample_entries)

    def test_summarize_short_conversation(self, conversation_memory, mock_llm):
        """Test summarizing a conversation that's too short."""
        conversation_memory.llm_client = mock_llm
        conv_id = "test_conv"
        
        # Add only 3 entries
        for i in range(3):
            entry = ConversationEntry(
                id=f"entry_{i}",
                type=MemoryType.USER,
                content=f"Message {i}",
                timestamp=datetime.now(),
                tokens=5
            )
            if conv_id not in conversation_memory.entries:
                conversation_memory.entries[conv_id] = []
            conversation_memory.entries[conv_id].append(entry)
        
        summary = conversation_memory.summarize_conversation(conv_id)
        assert summary is None  # Too short to summarize

    def test_summarize_without_llm(self, conversation_memory, sample_entries):
        """Test summarization when no LLM client is available."""
        conv_id = "test_conv"
        conversation_memory.entries[conv_id] = sample_entries
        
        summary = conversation_memory.summarize_conversation(conv_id)
        assert summary is None

    def test_auto_summarization_trigger_by_tokens(self, conversation_memory, sample_entries, mock_llm):
        """Test auto-summarization triggered by token threshold."""
        conversation_memory.llm_client = mock_llm
        conversation_memory.config.summary_threshold_tokens = 30
        conv_id = "test_conv"
        
        with patch.object(conversation_memory, 'summarize_conversation') as mock_summarize:
            for entry in sample_entries:
                conversation_memory.add_entry(
                    conv_id,
                    entry.content,
                    entry.type,
                    tokens=entry.tokens
                )
            
            # Should have triggered summarization when token threshold was crossed
            assert mock_summarize.called

    def test_auto_summarization_trigger_by_count(self, conversation_memory, mock_llm):
        """Test auto-summarization triggered by entry count threshold."""
        conversation_memory.llm_client = mock_llm
        conversation_memory.config.summary_trigger_count = 3
        conv_id = "test_conv"
        
        with patch.object(conversation_memory, 'summarize_conversation') as mock_summarize:
            for i in range(5):
                conversation_memory.add_entry(
                    conv_id,
                    f"Message {i}",
                    MemoryType.USER,
                    tokens=5
                )
            
            # Should have triggered summarization when count threshold was crossed
            assert mock_summarize.called


class TestPruning:
    """Test conversation pruning functionality."""

    def test_prune_old_conversations(self, conversation_memory, sample_entries):
        """Test pruning conversations older than threshold."""
        conv_id_old = "old_conv"
        conv_id_new = "new_conv"
        
        # Add old conversation
        conversation_memory.entries[conv_id_old] = []
        for entry in sample_entries:
            # Make timestamps older
            old_entry = ConversationEntry(
                id=entry.id,
                type=entry.type,
                content=entry.content,
                timestamp=datetime.now() - timedelta(days=40),
                tokens=entry.tokens
            )
            conversation_memory.entries[conv_id_old].append(old_entry)
        
        # Add new conversation
        conversation_memory.entries[conv_id_new] = sample_entries.copy()
        
        conversation_memory.last_accessed[conv_id_old] = datetime.now() - timedelta(days=40)
        conversation_memory.last_accessed[conv_id_new] = datetime.now()
        
        # Prune older than 30 days
        pruned = conversation_memory.prune_old(days=30)
        
        assert pruned == 1
        assert conv_id_new in conversation_memory.entries
        assert conv_id_old not in conversation_memory.entries

    def test_prune_with_custom_days(self, conversation_memory):
        """Test pruning with custom days parameter."""
        conv_id = "test_conv"
        conversation_memory.entries[conv_id] = []
        conversation_memory.last_accessed[conv_id] = datetime.now() - timedelta(days=10)
        
        # Prune with 5 days threshold
        pruned = conversation_memory.prune_old(days=5)
        assert pruned == 1
        
        # Should not prune with 15 days threshold
        conversation_memory.entries[conv_id] = []
        conversation_memory.last_accessed[conv_id] = datetime.now() - timedelta(days=10)
        pruned = conversation_memory.prune_old(days=15)
        assert pruned == 0


class TestStatistics:
    """Test memory statistics functionality."""

    def test_get_stats_single_conversation(self, populated_memory):
        """Test getting stats for a single conversation."""
        conv_id = "test_conv_001"
        stats = populated_memory.get_stats(conv_id)
        
        assert stats["conversation_id"] == conv_id
        assert stats["total_entries"] >= 5
        assert "total_tokens" in stats
        assert "entries_by_type" in stats
        assert "user" in stats["entries_by_type"]
        assert "assistant" in stats["entries_by_type"]

    def test_get_stats_global(self, populated_memory):
        """Test getting global statistics."""
        stats = populated_memory.get_stats()
        
        assert stats["total_conversations"] >= 4  # Now expecting 4 conversations
        assert stats["total_entries"] >= 14  # 5 + 3*3
        assert "total_summaries" in stats
        assert "average_entries_per_conversation" in stats

    def test_stats_for_nonexistent_conversation(self, conversation_memory):
        """Test stats for non-existent conversation."""
        stats = conversation_memory.get_stats("nonexistent")
        assert stats == {}


class TestPersistence:
    """Test persistence functionality."""

    def test_persist_and_load_conversation(self, tmp_path):
        """Test persisting and loading a conversation."""
        storage_path = tmp_path / "memory"
        
        # Create config with persist_to_disk=True
        config = ConversationConfig(persist_to_disk=True)
        memory = ConversationMemory(config=config, storage_path=storage_path)
        
        conv_id = "test_conv"
        memory.add_entry(conv_id, "Hello", MemoryType.USER)
        memory.add_entry(conv_id, "Hi there", MemoryType.ASSISTANT)
        
        # Create new memory instance and load
        memory2 = ConversationMemory(config=config, storage_path=storage_path)
        loaded = memory2.load_conversation(conv_id)
        
        assert loaded is True
        history = memory2.get_history(conv_id)
        assert len(history) == 2
        assert history[0].content == "Hello"
        assert history[1].content == "Hi there"

    def test_clear_conversation(self, populated_memory):
        """Test clearing a conversation."""
        conv_id = "test_conv_001"
        
        assert populated_memory.clear_conversation(conv_id) is True
        assert conv_id not in populated_memory.entries
        assert conv_id not in populated_memory.contexts
        assert conv_id not in populated_memory.last_accessed


class TestMemoryOptimization:
    """Test memory optimization features."""

    def test_token_counting(self, conversation_memory):
        """Test token counting approximation."""
        text = "This is a test sentence."
        tokens = conversation_memory._count_tokens(text)
        assert tokens > 0
        assert tokens == (len(text) + 3) // 4

    def test_truncate_by_tokens(self, conversation_memory, sample_entries):
        """Test truncating entries by token count."""
        truncated = conversation_memory._truncate_by_tokens(sample_entries, max_tokens=20)
        
        total_tokens = sum(e.tokens for e in truncated)
        assert total_tokens <= 20
        # Should keep most recent
        assert truncated[-1].id == sample_entries[-1].id

    def test_format_conversation(self, conversation_memory, sample_entries):
        """Test formatting conversation as text."""
        formatted = conversation_memory._format_conversation(sample_entries)
        
        assert "User: Hello" in formatted
        assert "Assistant: Of course!" in formatted
        lines = formatted.split('\n')
        assert len(lines) == len(sample_entries)


class TestConcurrency:
    """Test concurrent access to conversation memory."""

    def test_concurrent_add_entries(self, conversation_memory):
        """Test adding entries from multiple threads."""
        errors = []
        conv_id = "concurrent_test"
        
        def add_entries(thread_id):
            try:
                for i in range(20):
                    conversation_memory.add_entry(
                        conv_id,
                        f"Thread {thread_id} message {i}",
                        MemoryType.USER
                    )
            except Exception as e:
                errors.append(str(e))
        
        threads = [threading.Thread(target=add_entries, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        assert len(conversation_memory.entries[conv_id]) == 60

    def test_concurrent_read_write(self, populated_memory):
        """Test concurrent read and write operations."""
        errors = []
        conv_id = "test_conv_001"
        
        def reader():
            try:
                for _ in range(20):
                    populated_memory.get_history(conv_id)
                    populated_memory.get_context(conv_id)
            except Exception as e:
                errors.append(str(e))
        
        def writer():
            try:
                for i in range(10):
                    populated_memory.add_entry(
                        conv_id,
                        f"New message {i}",
                        MemoryType.USER
                    )
            except Exception as e:
                errors.append(str(e))
        
        threads = []
        for _ in range(2):
            threads.append(threading.Thread(target=reader))
            threads.append(threading.Thread(target=writer))
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_add_entry_with_invalid_type(self, conversation_memory):
        """Test adding entry with invalid type string."""
        with pytest.raises(ValueError):
            conversation_memory.add_entry(
                "test_conv",
                "test",
                entry_type="invalid_type"
            )

    def test_add_entry_with_invalid_priority(self, conversation_memory):
        """Test adding entry with invalid priority string."""
        with pytest.raises(KeyError):
            conversation_memory.add_entry(
                "test_conv",
                "test",
                entry_type="user",
                priority="invalid_priority"
            )

    def test_persist_without_storage(self, conversation_memory):
        """Test persisting when storage is not available."""
        conversation_memory.config.persist_to_disk = False
        # Should not raise exception
        conversation_memory._persist_conversation("test_conv")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])