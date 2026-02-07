# cockatoo_v1/src/database/initial_migration.py

"""
Alembic migration file for initial database schema creation.

Defines the complete Cockatoo database schema including tables for documents,
chunks, conversations, messages, tags, and settings with proper relationships
and indexes. Provides upgrade and downgrade functions for schema migration.
"""

import logging
from datetime import datetime

from alembic import op
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Boolean, ForeignKey, Table
)

# Setup logging
logger = logging.getLogger(__name__)


def upgrade() -> None:
    """Apply database upgrades."""
    try:
        # Create documents table (matching models.py)
        op.create_table(
            'documents',
            Column('id', String(255), primary_key=True),
            Column('file_path', Text, nullable=False),
            Column('file_name', String(255), nullable=False),
            Column('file_type', String(50), nullable=False),
            Column('file_size', Integer),
            Column('upload_date', DateTime, default=datetime.utcnow),
            Column('processing_status', String(50), default='pending'),
            Column('processing_error', Text),
            Column('metadata_json', Text),
            Column('vector_ids_json', Text),
            Column('chunk_count', Integer, default=0),
            Column('word_count', Integer, default=0),
            Column('language', String(10)),
            Column('tags_json', Text),
            Column('summary', Text),
            Column('is_indexed', Boolean, default=False),
            Column('indexed_at', DateTime),
            Column('last_accessed', DateTime),
            Column('access_count', Integer, default=0)
        )
        
        # Create chunks table
        op.create_table(
            'chunks',
            Column('id', String(255), primary_key=True),
            Column('document_id', String(255), ForeignKey('documents.id', ondelete='CASCADE'), nullable=False),
            Column('chunk_index', Integer, nullable=False),
            Column('text_content', Text, nullable=False),
            Column('cleaned_text', Text, nullable=False),
            Column('token_count', Integer),
            Column('embedding_model', String(100)),
            Column('vector_id', String(255), nullable=False),
            Column('metadata_json', Text),
            Column('created_at', DateTime, default=datetime.utcnow)
        )
        
        # Create conversations table
        op.create_table(
            'conversations',
            Column('id', String(255), primary_key=True),
            Column('title', String(255)),
            Column('created_at', DateTime, default=datetime.utcnow),
            Column('updated_at', DateTime, default=datetime.utcnow),
            Column('message_count', Integer, default=0),
            Column('total_tokens', Integer, default=0),
            Column('tags_json', Text),
            Column('is_archived', Boolean, default=False),
            Column('export_path', Text)
        )
        
        # Create messages table
        op.create_table(
            'messages',
            Column('id', String(255), primary_key=True),
            Column('conversation_id', String(255), ForeignKey('conversations.id', ondelete='CASCADE'), nullable=False),
            Column('role', String(50), nullable=False),
            Column('content', Text, nullable=False),
            Column('tokens', Integer),
            Column('model_used', String(100)),
            Column('sources_json', Text),
            Column('processing_time_ms', Integer),
            Column('created_at', DateTime, default=datetime.utcnow)
        )
        
        # Create tags table
        op.create_table(
            'tags',
            Column('id', String(255), primary_key=True),
            Column('name', String(100), unique=True, nullable=False),
            Column('color', String(50)),
            Column('description', Text),
            Column('created_at', DateTime, default=datetime.utcnow),
            Column('usage_count', Integer, default=0)
        )
        
        # Create document_tags junction table
        op.create_table(
            'document_tags',
            Column('document_id', String(255), ForeignKey('documents.id', ondelete='CASCADE'), nullable=False),
            Column('tag_id', String(255), ForeignKey('tags.id', ondelete='CASCADE'), nullable=False),
            Column('added_at', DateTime, default=datetime.utcnow)
        )
        
        # Create conversation_tags junction table
        op.create_table(
            'conversation_tags',
            Column('conversation_id', String(255), ForeignKey('conversations.id', ondelete='CASCADE'), nullable=False),
            Column('tag_id', String(255), ForeignKey('tags.id', ondelete='CASCADE'), nullable=False),
            Column('added_at', DateTime, default=datetime.utcnow)
        )
        
        # Create settings table
        op.create_table(
            'settings',
            Column('key', String(255), primary_key=True),
            Column('value', Text),
            Column('updated_at', DateTime, default=datetime.utcnow)
        )
        
        # Create indexes for performance
        op.create_index('idx_documents_status', 'documents', ['processing_status'])
        op.create_index('idx_documents_type', 'documents', ['file_type'])
        op.create_index('idx_chunks_document', 'chunks', ['document_id'])
        op.create_index('idx_messages_conversation', 'messages', ['conversation_id'])
        op.create_index('idx_messages_created', 'messages', ['created_at'])
        op.create_index('idx_documents_upload_date', 'documents', ['upload_date'])
        
        logger.info("Database tables created successfully")
        
    except Exception as e:
        logger.error(f"Failed to apply upgrade: {e}")
        raise


def downgrade() -> None:
    """Rollback database changes."""
    try:
        # Drop tables in reverse order
        op.drop_table('document_tags')
        op.drop_table('conversation_tags')
        op.drop_table('settings')
        op.drop_table('messages')
        op.drop_table('chunks')
        op.drop_table('conversations')
        op.drop_table('tags')
        op.drop_table('documents')
        
        # Drop indexes
        op.drop_index('idx_documents_status')
        op.drop_index('idx_documents_type')
        op.drop_index('idx_chunks_document')
        op.drop_index('idx_messages_conversation')
        op.drop_index('idx_messages_created')
        op.drop_index('idx_documents_upload_date')
        
        logger.info("Database tables dropped successfully")
        
    except Exception as e:
        logger.error(f"Failed to apply downgrade: {e}")
        raise