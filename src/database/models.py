# cockatoo_v1/src/database/models.py

"""
SQLAlchemy ORM models for Cockatoo database.

Defines database schema including Document, Chunk, Conversation, Message, Tag,
and Setting models with relationships, indexes, constraints, and utility methods.
Provides database initialization, session management, and JSON serialization
for structured data storage.
"""

import json
import logging
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Generator

from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Boolean, ForeignKey, create_engine,
    Table, func, Index, CheckConstraint, event
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship, scoped_session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.engine import Engine

# Setup logging
logger = logging.getLogger(__name__)

# SQLAlchemy base class
Base = declarative_base()

# Create engine variable
_engine = None
_session_factory = None


def get_engine(database_url: str = "sqlite:///cockatoo.db", echo: bool = False):
    """Create SQLAlchemy engine with proper configuration."""
    global _engine
    if _engine is None:
        _engine = create_engine(
            database_url,
            echo=echo,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=3600,
            connect_args={
                'timeout': 30,
                'check_same_thread': False
            }
        )
        
        # Enable foreign keys for SQLite
        @event.listens_for(Engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA temp_store=MEMORY")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA cache_size=-2000")
            cursor.close()
    
    return _engine


def get_session_local(engine=None):
    """Create session local factory."""
    global _session_factory
    if _session_factory is None:
        engine = engine or get_engine()
        _session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return _session_factory


def create_tables(engine=None) -> None:
    """Create all tables."""
    try:
        engine = engine or get_engine()
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create tables: {e}")
        raise


def init_database(database_url: str = "sqlite:///cockatoo.db", echo: bool = False):
    """Initialize database."""
    try:
        engine = get_engine(database_url, echo)
        create_tables(engine)
        get_session_local(engine)  # Initialize session factory
        return engine
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


# Junction table for document-tag relationship
document_tags = Table(
    'document_tags',
    Base.metadata,
    Column('document_id', String(255), ForeignKey('documents.id', ondelete='CASCADE'), primary_key=True),
    Column('tag_id', String(255), ForeignKey('tags.id', ondelete='CASCADE'), primary_key=True),
    Column('added_at', DateTime, default=datetime.utcnow),
    Index('idx_document_tags_document', 'document_id'),
    Index('idx_document_tags_tag', 'tag_id')
)

# Junction table for conversation-tag relationship
conversation_tags = Table(
    'conversation_tags',
    Base.metadata,
    Column('conversation_id', String(255), ForeignKey('conversations.id', ondelete='CASCADE'), primary_key=True),
    Column('tag_id', String(255), ForeignKey('tags.id', ondelete='CASCADE'), primary_key=True),
    Column('added_at', DateTime, default=datetime.utcnow),
    Index('idx_conversation_tags_conversation', 'conversation_id'),
    Index('idx_conversation_tags_tag', 'tag_id')
)


class Document(Base):
    """Document model."""
    __tablename__ = "documents"  # FIXED: Pastikan 'documents' bukan 'documments'
    
    id = Column(String(255), primary_key=True)
    file_path = Column(Text, nullable=False)
    file_name = Column(String(255), nullable=False)
    file_type = Column(String(50), nullable=False)
    file_size = Column(Integer)
    upload_date = Column(DateTime, default=datetime.utcnow)
    processing_status = Column(String(50), default="pending")
    processing_error = Column(Text)
    metadata_json = Column(Text)
    vector_ids_json = Column(Text)
    chunk_count = Column(Integer, default=0)
    word_count = Column(Integer, default=0)
    language = Column(String(10))
    tags_json = Column(Text)
    summary = Column(Text)
    is_indexed = Column(Boolean, default=False)
    indexed_at = Column(DateTime)
    last_accessed = Column(DateTime)
    access_count = Column(Integer, default=0)
    
    # Relationships
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan", lazy="dynamic")
    tags = relationship("Tag", secondary=document_tags, back_populates="documents", lazy="dynamic")
    
    # Indexes
    __table_args__ = (
        Index('idx_documents_status', 'processing_status'),
        Index('idx_documents_type', 'file_type'),
        Index('idx_documents_upload_date', 'upload_date'),
        Index('idx_documents_last_accessed', 'last_accessed'),
        Index('idx_documents_file_name', 'file_name'),
        CheckConstraint('file_size >= 0', name='check_file_size_positive'),
    )
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata as dictionary."""
        try:
            return json.loads(self.metadata_json) if self.metadata_json else {}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse metadata JSON: {e}")
            return {}
    
    def set_metadata(self, value: Dict[str, Any]) -> None:
        """Set metadata from dictionary."""
        self.metadata_json = json.dumps(value) if value else None
    
    def get_vector_ids(self) -> List[str]:
        """Get vector IDs as list."""
        try:
            return json.loads(self.vector_ids_json) if self.vector_ids_json else []
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse vector IDs JSON: {e}")
            return []
    
    def set_vector_ids(self, value: List[str]) -> None:
        """Set vector IDs from list."""
        self.vector_ids_json = json.dumps(value) if value else None
    
    def get_tags_list(self) -> List[str]:
        """Get tags as list."""
        try:
            return json.loads(self.tags_json) if self.tags_json else []
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse tags JSON: {e}")
            return []
    
    def set_tags_list(self, value: List[str]) -> None:
        """Set tags from list."""
        self.tags_json = json.dumps(value) if value else None
    
    def update_last_accessed(self) -> None:
        """Update last accessed timestamp."""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1
    
    def __repr__(self) -> str:
        return f"<Document(id={self.id}, name={self.file_name}, type={self.file_type})>"


class Chunk(Base):
    """Chunk model."""
    __tablename__ = "chunks"
    
    id = Column(String(255), primary_key=True)
    document_id = Column(String(255), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    text_content = Column(Text, nullable=False)
    cleaned_text = Column(Text, nullable=False)
    token_count = Column(Integer)
    embedding_model = Column(String(100))
    vector_id = Column(String(255), nullable=False)
    metadata_json = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    document = relationship("Document", back_populates="chunks")
    
    # Indexes
    __table_args__ = (
        Index('idx_chunks_document', 'document_id'),
        Index('idx_chunks_index', 'chunk_index'),
        Index('idx_chunks_vector', 'vector_id'),
        Index('idx_chunks_created', 'created_at'),
        CheckConstraint('chunk_index >= 0', name='check_chunk_index_positive'),
        CheckConstraint('token_count >= 0', name='check_token_count_positive'),
    )
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata as dictionary."""
        try:
            return json.loads(self.metadata_json) if self.metadata_json else {}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse metadata JSON: {e}")
            return {}
    
    def set_metadata(self, value: Dict[str, Any]) -> None:
        """Set metadata from dictionary."""
        self.metadata_json = json.dumps(value) if value else None
    
    def __repr__(self) -> str:
        return f"<Chunk(id={self.id}, document={self.document_id}, index={self.chunk_index})>"


class Conversation(Base):
    """Conversation model."""
    __tablename__ = "conversations"
    
    id = Column(String(255), primary_key=True)
    title = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    message_count = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    tags_json = Column(Text)
    is_archived = Column(Boolean, default=False)
    export_path = Column(Text)
    
    # Relationships
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan", lazy="dynamic")
    tags = relationship("Tag", secondary=conversation_tags, back_populates="conversations", lazy="dynamic")
    
    # Indexes
    __table_args__ = (
        Index('idx_conversations_updated', 'updated_at'),
        Index('idx_conversations_created', 'created_at'),
        Index('idx_conversations_archived', 'is_archived'),
        CheckConstraint('message_count >= 0', name='check_message_count_positive'),
        CheckConstraint('total_tokens >= 0', name='check_total_tokens_positive'),
    )
    
    def get_tags_list(self) -> List[str]:
        """Get tags as list."""
        try:
            return json.loads(self.tags_json) if self.tags_json else []
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse tags JSON: {e}")
            return []
    
    def set_tags_list(self, value: List[str]) -> None:
        """Set tags from list."""
        self.tags_json = json.dumps(value) if value else None
    
    def update_message_count(self) -> None:
        """Update message count."""
        self.message_count = len(self.messages)
        self.updated_at = datetime.utcnow()
    
    def __repr__(self) -> str:
        return f"<Conversation(id={self.id}, title={self.title}, messages={self.message_count})>"


class Message(Base):
    """Message model."""
    __tablename__ = "messages"
    
    id = Column(String(255), primary_key=True)
    conversation_id = Column(String(255), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False)
    role = Column(String(50), nullable=False)
    content = Column(Text, nullable=False)
    tokens = Column(Integer)
    model_used = Column(String(100))
    sources_json = Column(Text)
    processing_time_ms = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
    
    # Indexes
    __table_args__ = (
        Index('idx_messages_conversation', 'conversation_id'),
        Index('idx_messages_created', 'created_at'),
        Index('idx_messages_role', 'role'),
        CheckConstraint("role IN ('system', 'user', 'assistant')", name='check_valid_role'),
        CheckConstraint('tokens >= 0', name='check_tokens_positive'),
        CheckConstraint('processing_time_ms >= 0', name='check_processing_time_positive'),
    )
    
    def get_sources(self) -> List[Dict[str, Any]]:
        """Get sources as list."""
        try:
            return json.loads(self.sources_json) if self.sources_json else []
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse sources JSON: {e}")
            return []
    
    def set_sources(self, value: List[Dict[str, Any]]) -> None:
        """Set sources from list."""
        self.sources_json = json.dumps(value) if value else None
    
    def __repr__(self) -> str:
        return f"<Message(id={self.id}, role={self.role}, conversation={self.conversation_id})>"


class Tag(Base):
    """Tag model."""
    __tablename__ = "tags"
    
    id = Column(String(255), primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    color = Column(String(50))
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    usage_count = Column(Integer, default=0)
    
    # Relationships
    documents = relationship("Document", secondary=document_tags, back_populates="tags", lazy="dynamic")
    conversations = relationship("Conversation", secondary=conversation_tags, back_populates="tags", lazy="dynamic")
    
    # Indexes
    __table_args__ = (
        Index('idx_tags_name', 'name'),
        Index('idx_tags_usage', 'usage_count'),
        CheckConstraint('usage_count >= 0', name='check_usage_count_positive'),
    )
    
    def __repr__(self) -> str:
        return f"<Tag(id={self.id}, name={self.name})>"


class Setting(Base):
    """Setting model."""
    __tablename__ = "settings"
    
    key = Column(String(255), primary_key=True)
    value = Column(Text)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index('idx_settings_key', 'key'),
    )
    
    def get_value(self) -> Any:
        """Get parsed value."""
        try:
            if self.value and (self.value.startswith('{') or self.value.startswith('[')):
                return json.loads(self.value)
            return self.value
        except (json.JSONDecodeError, AttributeError):
            return self.value
    
    def set_value(self, value: Any) -> None:
        """Set value with automatic JSON serialization."""
        if isinstance(value, (dict, list)):
            self.value = json.dumps(value)
        else:
            self.value = str(value)
    
    def __repr__(self) -> str:
        return f"<Setting(key={self.key})>"


@contextmanager
def session_scope(engine=None) -> Generator[Session, None, None]:
    """Provide a transactional scope around a series of operations."""
    session_factory = get_session_local(engine)
    session = session_factory()
    try:
        yield session
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        session.close()