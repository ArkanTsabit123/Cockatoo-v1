# cockatoo_v1/src/database/queries.py

"""
Advanced database query functions for Cockatoo.

Provides ORM-based query operations for complex searches, statistics, and data
aggregation. Includes document search, conversation management, tagging, and
analytical queries with proper parameterization and SQLAlchemy best practices.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import asc, desc, func, or_, and_
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, Query, joinedload

from .models import Document, Conversation, Message, Tag, Setting, Chunk, session_scope

# Setup logging
logger = logging.getLogger(__name__)

# Query constants
DEFAULT_PAGE_SIZE = 20
DEFAULT_MAX_RESULTS = 100
SORT_ASC = "asc"
SORT_DESC = "desc"


def get_document_by_id(session: Session, document_id: str) -> Optional[Document]:
    """Get document by ID with parameterized query."""
    try:
        # Using SQLAlchemy ORM - automatically parameterized
        return session.query(Document).filter(Document.id == document_id).first()
    except SQLAlchemyError as e:
        logger.error(f"Failed to get document by ID {document_id}: {e}")
        return None


def get_recent_documents(
    session: Session,
    limit: int = 10,
    offset: int = 0
) -> List[Document]:
    """Get recent documents."""
    try:
        # Using SQLAlchemy ORM - automatically parameterized
        return session.query(Document)\
            .order_by(Document.upload_date.desc())\
            .offset(offset)\
            .limit(limit)\
            .all()
    except SQLAlchemyError as e:
        logger.error(f"Failed to get recent documents: {e}")
        return []


def search_documents(
    session: Session,
    query: str,
    file_type: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
    sort_by: str = "upload_date",
    sort_order: str = SORT_DESC
) -> Tuple[List[Document], int]:
    """Search documents using SQLAlchemy ORM (parameterized)."""
    try:
        # Build query with SQLAlchemy ORM
        db_query: Query = session.query(Document)
        
        # Apply search term
        if query:
            db_query = db_query.filter(
                or_(
                    Document.file_name.ilike(f"%{query}%"),
                    Document.summary.ilike(f"%{query}%")
                )
            )
        
        # Apply filters
        if file_type:
            db_query = db_query.filter(Document.file_type == file_type)
        
        if status:
            db_query = db_query.filter(Document.processing_status == status)
        
        # Get total count
        total_count = db_query.count()
        
        # Apply sorting
        sort_column = getattr(Document, sort_by, Document.upload_date)
        if sort_order == SORT_ASC:
            db_query = db_query.order_by(asc(sort_column))
        else:
            db_query = db_query.order_by(desc(sort_column))
        
        # Apply pagination
        documents = db_query.offset(offset).limit(limit).all()
        
        return documents, total_count
        
    except SQLAlchemyError as e:
        logger.error(f"Failed to search documents: {e}")
        return [], 0


def get_conversation_history(
    session: Session,
    conversation_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """Get conversation history using SQLAlchemy ORM."""
    try:
        query = session.query(Conversation)
        
        if conversation_id:
            query = query.filter(Conversation.id == conversation_id)
        
        conversations = query\
            .order_by(Conversation.updated_at.desc())\
            .offset(offset)\
            .limit(limit)\
            .all()
        
        # Convert to dictionaries
        result = []
        for conv in conversations:
            result.append({
                'id': conv.id,
                'title': conv.title,
                'message_count': conv.message_count,
                'created_at': conv.created_at.isoformat() if conv.created_at else None,
                'updated_at': conv.updated_at.isoformat() if conv.updated_at else None
            })
        
        return result
        
    except SQLAlchemyError as e:
        logger.error(f"Failed to get conversation history: {e}")
        return []


def update_document_status(
    session: Session,
    document_id: str,
    status: str,
    error_message: Optional[str] = None
) -> bool:
    """Update document status using SQLAlchemy ORM."""
    try:
        document = session.query(Document).filter(Document.id == document_id).first()
        if document:
            document.processing_status = status
            document.processing_error = error_message
            session.commit()
            return True
        return False
        
    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"Failed to update document status: {e}")
        return False


def add_document_tag(
    session: Session,
    document_id: str,
    tag_name: str
) -> bool:
    """Add tag to document using SQLAlchemy ORM."""
    try:
        # Check if tag exists
        tag = session.query(Tag).filter(Tag.name == tag_name).first()
        if not tag:
            # Create new tag
            tag = Tag(
                id=f"tag_{int(datetime.now().timestamp())}",
                name=tag_name
            )
            session.add(tag)
        
        # Check if document exists
        document = session.query(Document).filter(Document.id == document_id).first()
        if not document:
            logger.error(f"Document {document_id} not found")
            return False
        
        # Add relationship
        if tag not in document.tags:
            document.tags.append(tag)
        
        session.commit()
        return True
        
    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"Failed to add document tag: {e}")
        return False


def get_statistics(session: Session) -> Dict[str, Any]:
    """Get database statistics using SQLAlchemy ORM."""
    try:
        # Use SQLAlchemy functions for safe queries
        total_documents = session.query(func.count(Document.id)).scalar() or 0
        total_chunks = session.query(func.count(Chunk.id)).scalar() or 0
        total_conversations = session.query(func.count(Conversation.id)).scalar() or 0
        total_messages = session.query(func.count(Message.id)).scalar() or 0
        
        # Get document type distribution
        type_distribution = {}
        type_counts = session.query(
            Document.file_type,
            func.count(Document.id)
        ).group_by(Document.file_type).all()
        
        for file_type, count in type_counts:
            type_distribution[file_type] = count
        
        # Get recent activity
        recent_activity = session.query(Document)\
            .order_by(Document.upload_date.desc())\
            .limit(5)\
            .all()
        
        # Get status distribution
        status_distribution = {}
        status_counts = session.query(
            Document.processing_status,
            func.count(Document.id)
        ).group_by(Document.processing_status).all()
        
        for status, count in status_counts:
            status_distribution[status] = count
        
        # Get top tags
        top_tags = session.query(
            Tag.name,
            Tag.usage_count
        ).order_by(Tag.usage_count.desc())\
         .limit(10)\
         .all()
        
        return {
            'total_documents': total_documents,
            'total_chunks': total_chunks,
            'total_conversations': total_conversations,
            'total_messages': total_messages,
            'type_distribution': type_distribution,
            'status_distribution': status_distribution,
            'top_tags': [{'name': name, 'count': count} for name, count in top_tags],
            'recent_activity': [
                {
                    'id': doc.id,
                    'name': doc.file_name,
                    'type': doc.file_type,
                    'upload_date': doc.upload_date.isoformat() if doc.upload_date else None,
                    'status': doc.processing_status
                }
                for doc in recent_activity
            ]
        }
        
    except SQLAlchemyError as e:
        logger.error(f"Failed to get statistics: {e}")
        return {}


def search_documents_advanced(
    session: Session,
    filters: Dict[str, Any],
    page: int = 1,
    page_size: int = DEFAULT_PAGE_SIZE
) -> Tuple[List[Document], int]:
    """Advanced document search with multiple filters."""
    try:
        query = session.query(Document)
        
        # Apply filters
        if 'query' in filters and filters['query']:
            search_term = f"%{filters['query']}%"
            query = query.filter(
                or_(
                    Document.file_name.ilike(search_term),
                    Document.summary.ilike(search_term),
                    Document.tags_json.ilike(search_term)
                )
            )
        
        if 'file_type' in filters and filters['file_type']:
            query = query.filter(Document.file_type == filters['file_type'])
        
        if 'status' in filters and filters['status']:
            query = query.filter(Document.processing_status == filters['status'])
        
        if 'language' in filters and filters['language']:
            query = query.filter(Document.language == filters['language'])
        
        if 'min_size' in filters and filters['min_size']:
            query = query.filter(Document.file_size >= filters['min_size'])
        
        if 'max_size' in filters and filters['max_size']:
            query = query.filter(Document.file_size <= filters['max_size'])
        
        if 'start_date' in filters and filters['start_date']:
            query = query.filter(Document.upload_date >= filters['start_date'])
        
        if 'end_date' in filters and filters['end_date']:
            query = query.filter(Document.upload_date <= filters['end_date'])
        
        # Get total count
        total = query.count()
        
        # Apply pagination
        offset = (page - 1) * page_size
        documents = query.order_by(Document.upload_date.desc())\
                        .offset(offset)\
                        .limit(page_size)\
                        .all()
        
        return documents, total
        
    except SQLAlchemyError as e:
        logger.error(f"Failed to perform advanced search: {e}")
        return [], 0