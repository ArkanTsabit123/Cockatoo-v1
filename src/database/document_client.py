# cockatoo_v1/src/database/document_client.py

"""
Document management database operations module.

Handles document CRUD operations with SQLite database including metadata management,
search, bulk operations, and statistics. Provides secure parameterized queries
and JSON serialization for document metadata, tags, and vector IDs.
"""
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .sqlite_client import SQLiteClient

logger = logging.getLogger(__name__)


class DocumentClient:
    """Document-specific database operations."""
    
    def __init__(self, sqlite_client: SQLiteClient) -> None:
        self.client = sqlite_client
        logger.info("Document client initialized")
    
    def add_document(self, document_data: Dict[str, Any]) -> Optional[str]:
        """Add a new document with parameterized query."""
        try:
            document_id = document_data.get('id') or f"doc_{uuid.uuid4().hex[:16]}"
            
            # Parameterized query - FIXED
            query = """
            INSERT INTO documents (
                id, file_path, file_name, file_type, file_size,
                upload_date, processing_status, metadata_json,
                vector_ids_json, chunk_count, word_count, language,
                tags_json, summary, is_indexed, last_accessed, access_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            params = (
                document_id,
                document_data.get('file_path', ''),
                document_data.get('file_name', ''),
                document_data.get('file_type', ''),
                document_data.get('file_size'),
                document_data.get('upload_date', datetime.now()),
                document_data.get('processing_status', 'pending'),
                json.dumps(document_data.get('metadata', {})),
                json.dumps(document_data.get('vector_ids', [])),
                document_data.get('chunk_count', 0),
                document_data.get('word_count', 0),
                document_data.get('language', 'en'),
                json.dumps(document_data.get('tags', [])),
                document_data.get('summary', ''),
                document_data.get('is_indexed', False),
                document_data.get('last_accessed', datetime.now()),
                document_data.get('access_count', 0)
            )
            
            self.client.execute_query(query, params)
            return document_id
            
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            return None
    
    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID with parameterized query."""
        try:
            query = "SELECT * FROM documents WHERE id = ?"
            results = self.client.execute_query(query, (document_id,))
            
            if not results:
                return None
            
            doc = results[0]
            return self._process_document_row(doc)
            
        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {e}")
            return None
    
    def update_document(self, document_id: str, updates: Dict[str, Any]) -> bool:
        """Update document with parameterized query - FIXED."""
        try:
            # Build parameterized update query
            set_clauses = []
            params = []
            
            for key, value in updates.items():
                if key in ['metadata', 'vector_ids', 'tags']:
                    value = json.dumps(value) if value else None
                set_clauses.append(f"{key} = ?")
                params.append(value)
            
            params.append(document_id)
            set_str = ", ".join(set_clauses)
            
            # FIXED: Gunakan parameterized query untuk seluruh statement
            query = "UPDATE documents SET " + set_str + ", last_accessed = ? WHERE id = ?"
            params.append(datetime.now())
            params.append(document_id)
            
            self.client.execute_query(query, tuple(params))
            return True
            
        except Exception as e:
            logger.error(f"Failed to update document {document_id}: {e}")
            return False
    
    def search_documents(self, filters: Dict[str, Any], page: int = 1, page_size: int = 20) -> Tuple[List[Dict[str, Any]], int]:
        """Search documents with parameterized queries - FIXED."""
        try:
            # Build parameterized where clause
            where_clauses = []
            params = []
            
            if 'query' in filters and filters['query']:
                # Parameterized search
                where_clauses.append("(file_name LIKE ? OR summary LIKE ? OR tags_json LIKE ?)")
                search_term = f"%{filters['query']}%"
                params.extend([search_term, search_term, search_term])
            
            if 'file_type' in filters and filters['file_type']:
                where_clauses.append("file_type = ?")
                params.append(filters['file_type'])
            
            if 'status' in filters and filters['status']:
                where_clauses.append("processing_status = ?")
                params.append(filters['status'])
            
            if 'tags' in filters and filters['tags']:
                # Search for tags in JSON array
                tags = filters['tags']
                if isinstance(tags, list):
                    for tag in tags:
                        where_clauses.append("tags_json LIKE ?")
                        params.append(f'%"{tag}"%')
            
            # Build query - FIXED: Hindari string interpolation untuk SQL
            where_str = " AND ".join(where_clauses) if where_clauses else "1=1"
            
            # Count total with parameterized query
            count_query = f"SELECT COUNT(*) as total FROM documents WHERE {where_str}"
            count_result = self.client.execute_query(count_query, tuple(params))
            total = count_result[0]['total'] if count_result else 0
            
            # Get paginated results with parameterized query
            offset = (page - 1) * page_size
            query = f"""
            SELECT * FROM documents 
            WHERE {where_str}
            ORDER BY upload_date DESC
            LIMIT ? OFFSET ?
            """
            
            params_with_pagination = params + [page_size, offset]
            results = self.client.execute_query(query, tuple(params_with_pagination))
            
            # Process results
            documents = [self._process_document_row(row) for row in results]
            
            return documents, total
            
        except Exception as e:
            logger.error(f"Failed to search documents: {e}")
            return [], 0
    
    def delete_document(self, document_id: str) -> bool:
        """Delete document with parameterized query."""
        try:
            query = "DELETE FROM documents WHERE id = ?"
            self.client.execute_query(query, (document_id,))
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False
    
    def get_document_statistics(self) -> Dict[str, Any]:
        """Get document statistics with parameterized queries."""
        try:
            stats = {}
            
            # Total documents with parameterized query
            total_query = "SELECT COUNT(*) as total FROM documents"
            total_result = self.client.execute_query(total_query, ())
            stats['total_documents'] = total_result[0]['total'] if total_result else 0
            
            # By status with parameterized query
            status_query = """
            SELECT processing_status, COUNT(*) as count 
            FROM documents 
            GROUP BY processing_status
            """
            status_results = self.client.execute_query(status_query, ())
            stats['by_status'] = {row['processing_status']: row['count'] for row in status_results}
            
            # By file type with parameterized query
            type_query = """
            SELECT file_type, COUNT(*) as count 
            FROM documents 
            GROUP BY file_type
            """
            type_results = self.client.execute_query(type_query, ())
            stats['by_type'] = {row['file_type']: row['count'] for row in type_results}
            
            # Recent activity with parameterized query
            recent_query = """
            SELECT file_name, file_type, upload_date, processing_status
            FROM documents 
            ORDER BY upload_date DESC 
            LIMIT 10
            """
            recent_results = self.client.execute_query(recent_query, ())
            stats['recent_activity'] = [
                {
                    'name': row['file_name'],
                    'type': row['file_type'],
                    'upload_date': row['upload_date'],
                    'status': row['processing_status']
                }
                for row in recent_results
            ]
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get document statistics: {e}")
            return {}
    
    def _process_document_row(self, row: Dict) -> Dict[str, Any]:
        """Process database row into document dictionary."""
        try:
            doc = dict(row)
            
            # Parse JSON fields
            for json_field in ['metadata_json', 'vector_ids_json', 'tags_json']:
                if doc.get(json_field):
                    try:
                        key = json_field.replace('_json', '')
                        doc[key] = json.loads(doc[json_field])
                    except json.JSONDecodeError:
                        doc[key] = [] if key in ['vector_ids', 'tags'] else {}
                else:
                    key = json_field.replace('_json', '')
                    doc[key] = [] if key in ['vector_ids', 'tags'] else {}
            
            # Remove JSON fields
            for json_field in ['metadata_json', 'vector_ids_json', 'tags_json']:
                doc.pop(json_field, None)
            
            return doc
            
        except Exception as e:
            logger.error(f"Failed to process document row: {e}")
            return dict(row)
    
    def bulk_insert_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Bulk insert documents with parameterized queries."""
        try:
            query = """
            INSERT INTO documents (
                id, file_path, file_name, file_type, file_size,
                upload_date, processing_status, metadata_json,
                vector_ids_json, chunk_count, word_count, language,
                tags_json, summary, is_indexed, last_accessed, access_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            params_list = []
            document_ids = []
            
            for doc_data in documents:
                document_id = doc_data.get('id') or f"doc_{uuid.uuid4().hex[:16]}"
                document_ids.append(document_id)
                
                params = (
                    document_id,
                    doc_data.get('file_path', ''),
                    doc_data.get('file_name', ''),
                    doc_data.get('file_type', ''),
                    doc_data.get('file_size'),
                    doc_data.get('upload_date', datetime.now()),
                    doc_data.get('processing_status', 'pending'),
                    json.dumps(doc_data.get('metadata', {})),
                    json.dumps(doc_data.get('vector_ids', [])),
                    doc_data.get('chunk_count', 0),
                    doc_data.get('word_count', 0),
                    doc_data.get('language', 'en'),
                    json.dumps(doc_data.get('tags', [])),
                    doc_data.get('summary', ''),
                    doc_data.get('is_indexed', False),
                    doc_data.get('last_accessed', datetime.now()),
                    doc_data.get('access_count', 0)
                )
                params_list.append(params)
            
            success = self.client.execute_many(query, params_list)
            
            if success:
                return document_ids
            else:
                return []
            
        except Exception as e:
            logger.error(f"Failed to bulk insert documents: {e}")
            return []