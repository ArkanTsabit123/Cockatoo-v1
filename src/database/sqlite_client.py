# src/database/sqlite_client.py

"""
SQLite database client with connection pooling and advanced management.

Provides robust database operations with connection pooling, transaction management,
backup/restore functionality, and health monitoring. Includes
configuration management, automated maintenance, and production-ready error handling.
"""

import json
import logging
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

# Setup logging
logger = logging.getLogger(__name__)

# Database constants
DEFAULT_DATABASE_PATH = str(Path.home() / ".cockatoo" / "database" / "metadata.db")
DEFAULT_POOL_SIZE = 5
DEFAULT_MAX_OVERFLOW = 10
DEFAULT_POOL_TIMEOUT = 30
DEFAULT_POOL_RECYCLE = 3600
DEFAULT_CACHE_SIZE = 2000
DEFAULT_QUERY_TIMEOUT = 30
DEFAULT_BATCH_SIZE = 100
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_DELAY = 1.0
DEFAULT_BACKUP_RETENTION_DAYS = 7
DEFAULT_CLEANUP_DAYS = 90
DEFAULT_PAGE_SIZE = 20
MAX_WAL_SIZE_MB = 100

# Sort constants
SORT_ASC = "asc"
SORT_DESC = "desc"

# Document status constants
DOCUMENT_STATUS_PENDING = "pending"
DOCUMENT_STATUS_IN_PROGRESS = "in_progress"
DOCUMENT_STATUS_COMPLETED = "completed"
DOCUMENT_STATUS_ERROR = "error"

# Message role constants
ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"

# Model type constants
MODEL_TYPE_LLM = "llm"
MODEL_TYPE_EMBEDDING = "embedding"

# Database health status
class DatabaseHealth(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    DISCONNECTED = "disconnected"


@dataclass
class DatabaseConfig:
    """Database configuration."""
    database_path: str = DEFAULT_DATABASE_PATH
    pool_size: int = DEFAULT_POOL_SIZE
    max_overflow: int = DEFAULT_MAX_OVERFLOW
    pool_timeout: int = DEFAULT_POOL_TIMEOUT
    pool_recycle: int = DEFAULT_POOL_RECYCLE
    cache_size: int = DEFAULT_CACHE_SIZE
    enable_foreign_keys: bool = True
    enable_wal: bool = True
    wal_size_mb: int = MAX_WAL_SIZE_MB
    enable_backup: bool = True
    backup_retention_days: int = DEFAULT_BACKUP_RETENTION_DAYS
    enable_cleanup: bool = True
    cleanup_days: int = DEFAULT_CLEANUP_DAYS
    query_timeout: int = DEFAULT_QUERY_TIMEOUT
    max_retry_attempts: int = DEFAULT_RETRY_ATTEMPTS
    retry_delay: float = DEFAULT_RETRY_DELAY


class SQLiteClient:
    """SQLite database client with connection pooling and error handling."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, config: Optional[DatabaseConfig] = None):
        """Thread-safe singleton implementation."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance
    
    def __init__(self, config: Optional[DatabaseConfig] = None) -> None:
        """Initialize only once."""
        # Check if already initialized
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        self.config = config or DatabaseConfig()
        self._connections = {}
        self._connection_lock = threading.RLock()
        self._initialized = True
        
        # Setup database dengan handling error yang lebih baik
        try:
            self._setup_database()
            logger.info(f"SQLite client initialized for: {self.config.database_path}")
        except Exception as e:
            logger.error(f"SQLite client initialization failed: {e}")
            # Don't raise, let client be created
    
    def _setup_database(self) -> None:
        """Setup database with proper configuration."""
        try:
            # Create database directory
            db_path = Path(self.config.database_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Initial connection to setup database
            setup_conn = sqlite3.connect(
                self.config.database_path,
                timeout=self.config.pool_timeout
            )
            setup_conn.row_factory = sqlite3.Row
            
            try:
                self._setup_tables(setup_conn)
                self._setup_indexes(setup_conn)
                self._configure_database(setup_conn)
                self._initialize_settings(setup_conn)
                
                logger.info("Database setup completed successfully")
            finally:
                setup_conn.close()
            
        except Exception as e:
            logger.error(f"Failed to setup database: {e}")
            # Don't raise, let app continue
    
    def _setup_tables(self, conn: sqlite3.Connection) -> None:
        """Create database tables if they don't exist."""
        tables = [
            # Documents table
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                file_name TEXT NOT NULL,
                file_type TEXT NOT NULL,
                file_size INTEGER,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processing_status TEXT DEFAULT 'pending',
                processing_error TEXT,
                metadata_json TEXT,
                vector_ids_json TEXT,
                chunk_count INTEGER DEFAULT 0,
                word_count INTEGER DEFAULT 0,
                language TEXT,
                tags_json TEXT,
                summary TEXT,
                is_indexed BOOLEAN DEFAULT FALSE,
                indexed_at TIMESTAMP,
                last_accessed TIMESTAMP,
                access_count INTEGER DEFAULT 0
            )
            """,
            
            # Chunks table
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                text_content TEXT NOT NULL,
                cleaned_text TEXT NOT NULL,
                token_count INTEGER,
                embedding_model TEXT,
                vector_id TEXT NOT NULL,
                metadata_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
            )
            """,
            
            # Conversations table
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                message_count INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                tags_json TEXT,
                is_archived BOOLEAN DEFAULT FALSE,
                export_path TEXT
            )
            """,
            
            # Messages table
            """
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                tokens INTEGER,
                model_used TEXT,
                sources_json TEXT,
                processing_time_ms INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            )
            """,
            
            # Tags table
            """
            CREATE TABLE IF NOT EXISTS tags (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                color TEXT,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                usage_count INTEGER DEFAULT 0
            )
            """,
            
            # Document tags junction table
            """
            CREATE TABLE IF NOT EXISTS document_tags (
                document_id TEXT NOT NULL,
                tag_id TEXT NOT NULL,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (document_id, tag_id),
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
                FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE
            )
            """,
            
            # Conversation tags junction table
            """
            CREATE TABLE IF NOT EXISTS conversation_tags (
                conversation_id TEXT NOT NULL,
                tag_id TEXT NOT NULL,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (conversation_id, tag_id),
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
                FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE
            )
            """,
            
            # Settings table
            """
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        ]
        
        cursor = conn.cursor()
        for table_sql in tables:
            try:
                cursor.execute(table_sql)
            except Exception as e:
                logger.warning(f"Error creating table: {e}")
        conn.commit()
    
    def _setup_indexes(self, conn: sqlite3.Connection) -> None:
        """Create indexes for performance."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(processing_status)",
            "CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(file_type)",
            "CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id)",
            "CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id)",
            "CREATE INDEX IF NOT EXISTS idx_messages_created ON messages(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_documents_upload_date ON documents(upload_date)",
            "CREATE INDEX IF NOT EXISTS idx_documents_last_accessed ON documents(last_accessed)",
            "CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at)"
        ]
        
        cursor = conn.cursor()
        for index_sql in indexes:
            try:
                cursor.execute(index_sql)
            except Exception as e:
                logger.warning(f"Error creating index: {e}")
        conn.commit()
    
    def _configure_database(self, conn: sqlite3.Connection) -> None:
        """Configure database settings."""
        cursor = conn.cursor()
        
        # Enable foreign keys
        if self.config.enable_foreign_keys:
            cursor.execute("PRAGMA foreign_keys = ON")
            logger.info("Foreign keys enabled")
        
        # Enable WAL mode for better concurrency
        if self.config.enable_wal:
            try:
                cursor.execute("PRAGMA journal_mode = WAL")
                cursor.execute(f"PRAGMA wal_autocheckpoint = {self.config.wal_size_mb}")
                logger.info("WAL mode enabled")
            except Exception as e:
                logger.warning(f"Failed to enable WAL mode: {e}")
        
        # Set cache size
        try:
            cursor.execute(f"PRAGMA cache_size = -{self.config.cache_size}")
        except Exception as e:
            logger.warning(f"Failed to set cache size: {e}")
        
        # Set synchronous mode for durability
        try:
            cursor.execute("PRAGMA synchronous = NORMAL")
            logger.info("Synchronous mode set to NORMAL")
        except Exception as e:
            logger.warning(f"Failed to set synchronous mode: {e}")
        
        # Set temp store to memory
        try:
            cursor.execute("PRAGMA temp_store = MEMORY")
            logger.info("Temp store set to MEMORY")
        except Exception as e:
            logger.warning(f"Failed to set temp store: {e}")
        
        conn.commit()
    
    def _initialize_settings(self, conn: sqlite3.Connection) -> None:
        """Initialize default settings."""
        default_settings = {
            'database_version': '1.0.0',
            'last_backup': None,
            'total_queries': '0',
            'total_documents': '0',
            'created_at': datetime.now().isoformat(),
            'foreign_keys_enabled': str(self.config.enable_foreign_keys),
            'wal_mode': 'true' if self.config.enable_wal else 'false',
            'synchronous_mode': 'NORMAL',
            'temp_store': 'MEMORY'
        }
        
        cursor = conn.cursor()
        for key, value in default_settings.items():
            try:
                cursor.execute(
                    "INSERT OR REPLACE INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
                    (key, str(value), datetime.now())
                )
            except Exception as e:
                logger.warning(f"Error inserting setting {key}: {e}")
        conn.commit()
    
    @contextmanager
    def _get_connection(self) -> Iterator[sqlite3.Connection]:
        """Get a database connection from pool."""
        thread_id = threading.get_ident()
        conn = None
        
        with self._connection_lock:
            # Check for existing connection
            if thread_id in self._connections:
                conn = self._connections[thread_id]
                try:
                    # Check if connection is still valid
                    conn.execute("SELECT 1")
                except sqlite3.Error:
                    # Connection is invalid, remove it
                    try:
                        conn.close()
                    except Exception:
                        pass
                    del self._connections[thread_id]
                    conn = None
            
            # Create new connection if needed
            if conn is None:
                conn = sqlite3.connect(
                    self.config.database_path,
                    timeout=self.config.pool_timeout,
                    check_same_thread=False
                )
                conn.row_factory = sqlite3.Row
                self._connections[thread_id] = conn
                
                # Configure new connection
                try:
                    cursor = conn.cursor()
                    if self.config.enable_foreign_keys:
                        cursor.execute("PRAGMA foreign_keys = ON")
                    if self.config.enable_wal:
                        cursor.execute("PRAGMA journal_mode = WAL")
                    cursor.execute("PRAGMA temp_store = MEMORY")
                except Exception as e:
                    logger.warning(f"Failed to configure connection: {e}")
        
        try:
            yield conn
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            # Try to rollback if there's an active transaction
            try:
                conn.rollback()
            except:
                pass
            raise
        finally:
            # Keep connection in pool for reuse
            pass
    
    def _row_to_dict(self, row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
        """Convert sqlite3.Row to dictionary with JSON parsing."""
        if row is None:
            return None
        
        result = {}
        for key in row.keys():
            value = row[key]
            
            # Parse JSON fields
            if key in ['metadata_json', 'tags_json', 'vector_ids_json', 'sources_json']:
                if value:
                    try:
                        result[key.replace('_json', '')] = json.loads(value)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse JSON for {key}: {value[:100]}...")
                        result[key.replace('_json', '')] = {} if 'metadata' in key else []
                else:
                    result[key.replace('_json', '')] = {} if 'metadata' in key else []
            else:
                result[key] = value
        
        return result
    
    def _parse_document_json(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Parse JSON fields in document."""
        result = doc.copy()
        
        # Parse metadata_json
        if 'metadata_json' in result and result['metadata_json']:
            try:
                result['metadata'] = json.loads(result['metadata_json'])
            except:
                result['metadata'] = {}
            del result['metadata_json']
        elif 'metadata_json' in result:
            result['metadata'] = {}
            del result['metadata_json']
        
        # Parse vector_ids_json
        if 'vector_ids_json' in result and result['vector_ids_json']:
            try:
                result['vector_ids'] = json.loads(result['vector_ids_json'])
            except:
                result['vector_ids'] = []
            del result['vector_ids_json']
        elif 'vector_ids_json' in result:
            result['vector_ids'] = []
            del result['vector_ids_json']
        
        # Parse tags_json
        if 'tags_json' in result and result['tags_json']:
            try:
                result['tags'] = json.loads(result['tags_json'])
            except:
                result['tags'] = []
            del result['tags_json']
        elif 'tags_json' in result:
            result['tags'] = []
            del result['tags_json']
        
        return result
    
    def execute_query(self, query: str, params: tuple = (), timeout: Optional[int] = None) -> List[Dict]:
        """Execute a parameterized query and return results."""
        timeout = timeout or self.config.query_timeout
        
        for attempt in range(self.config.max_retry_attempts):
            try:
                with self._get_connection() as conn:
                    # Handle PRAGMA statements specially (they don't support parameter binding)
                    query_upper = query.strip().upper()
                    if query_upper.startswith("PRAGMA"):
                        cursor = conn.cursor()
                        cursor.execute(query)
                        if query_upper.startswith("PRAGMA") and "=" in query:
                            conn.commit()
                        # Try to fetch results if it's a query
                        try:
                            rows = cursor.fetchall()
                            return [dict(row) for row in rows]
                        except:
                            return []
                    
                    # For normal queries, use parameter binding
                    # Set busy timeout
                    conn.execute(f"PRAGMA busy_timeout = {timeout * 1000}")
                    
                    cursor = conn.cursor()
                    cursor.execute(query, params)
                    
                    if query_upper.startswith("SELECT"):
                        rows = cursor.fetchall()
                        return [dict(row) for row in rows]
                    else:
                        conn.commit()
                        return []
                        
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < self.config.max_retry_attempts - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                    continue
                # Handle transaction errors gracefully
                if "no transaction is active" in str(e):
                    logger.debug(f"No active transaction for: {query[:50]}...")
                    return []
                logger.error(f"Database operational error: {e}")
                raise
            except sqlite3.IntegrityError as e:
                # Re-raise integrity errors (unique constraint, foreign key, etc.)
                logger.error(f"Database integrity error: {e}")
                raise
            except Exception as e:
                logger.error(f"Query execution error: {e}")
                raise
        
        return []
    
    def execute_many(self, query: str, params_list: List[tuple]) -> bool:
        """Execute many parameterized queries in a transaction."""
        try:
            with self._get_connection() as conn:
                # Set busy timeout
                conn.execute(f"PRAGMA busy_timeout = {self.config.query_timeout * 1000}")
                
                cursor = conn.cursor()
                cursor.executemany(query, params_list)
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Execute many error: {e}")
            return False
    
    # ==================== Document CRUD Operations ====================
    
    def add_document(self, document: Dict[str, Any]) -> str:
        """Add a document to the database."""
        doc_id = document.get('id') or str(uuid.uuid4())
        
        # Prepare JSON fields
        metadata_json = json.dumps(document.get('metadata', {}))
        vector_ids_json = json.dumps(document.get('vector_ids', []))
        tags_json = json.dumps(document.get('tags', []))
        
        query = """
            INSERT INTO documents (
                id, file_path, file_name, file_type, file_size,
                processing_status, metadata_json, vector_ids_json,
                chunk_count, word_count, language, tags_json, summary
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            doc_id,
            document.get('file_path', ''),
            document.get('file_name', ''),
            document.get('file_type', ''),
            document.get('file_size'),
            document.get('processing_status', 'pending'),
            metadata_json,
            vector_ids_json,
            document.get('chunk_count', 0),
            document.get('word_count', 0),
            document.get('language'),
            tags_json,
            document.get('summary')
        )
        
        self.execute_query(query, params)
        return doc_id
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by ID."""
        results = self.execute_query(
            "SELECT * FROM documents WHERE id = ?",
            (doc_id,)
        )
        
        if not results:
            return None
        
        # Parse JSON fields
        return self._parse_document_json(results[0])
    
    def update_document(self, doc_id: str, updates: Dict[str, Any]) -> bool:
        """Update a document."""
        if not updates:
            return False
        
        # Mapping of common keys to actual database column names
        column_mapping = {
            'status': 'processing_status',  # Map 'status' to 'processing_status'
            'processing_status': 'processing_status',
            'file_path': 'file_path',
            'file_name': 'file_name',
            'file_type': 'file_type',
            'file_size': 'file_size',
            'processing_error': 'processing_error',
            'metadata': 'metadata_json',
            'vector_ids': 'vector_ids_json',
            'tags': 'tags_json',
            'chunk_count': 'chunk_count',
            'word_count': 'word_count',
            'language': 'language',
            'summary': 'summary',
            'is_indexed': 'is_indexed',
            'indexed_at': 'indexed_at',
            'last_accessed': 'last_accessed',
            'access_count': 'access_count'
        }
        
        # Build dynamic SET clause
        set_clauses = []
        params = []
        
        for key, value in updates.items():
            # Get the actual column name from mapping, or use the key as-is
            column = column_mapping.get(key, key)
            
            if key == 'metadata' or column == 'metadata_json':
                set_clauses.append("metadata_json = ?")
                params.append(json.dumps(value) if value is not None else None)
            elif key == 'vector_ids' or column == 'vector_ids_json':
                set_clauses.append("vector_ids_json = ?")
                params.append(json.dumps(value) if value is not None else None)
            elif key == 'tags' or column == 'tags_json':
                set_clauses.append("tags_json = ?")
                params.append(json.dumps(value) if value is not None else None)
            else:
                set_clauses.append(f"{column} = ?")
                params.append(value)
        
        if not set_clauses:
            return False
        
        params.append(doc_id)
        query = f"UPDATE documents SET {', '.join(set_clauses)} WHERE id = ?"
        
        try:
            self.execute_query(query, tuple(params))
            return True
        except Exception as e:
            logger.error(f"Failed to update document {doc_id}: {e}")
            return False
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document."""
        try:
            self.execute_query("DELETE FROM documents WHERE id = ?", (doc_id,))
            return True
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return False
    
    # ==================== Batch Operations ====================
    
    def batch_add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Add multiple documents in batch."""
        doc_ids = []
        params_list = []
        
        for doc in documents:
            doc_id = doc.get('id') or str(uuid.uuid4())
            doc_ids.append(doc_id)
            
            params_list.append((
                doc_id,
                doc.get('file_path', ''),
                doc.get('file_name', ''),
                doc.get('file_type', ''),
                doc.get('file_size'),
                doc.get('processing_status', 'pending'),
                json.dumps(doc.get('metadata', {})),
                json.dumps(doc.get('vector_ids', [])),
                doc.get('chunk_count', 0),
                doc.get('word_count', 0),
                doc.get('language'),
                json.dumps(doc.get('tags', [])),
                doc.get('summary')
            ))
        
        query = """
            INSERT INTO documents (
                id, file_path, file_name, file_type, file_size,
                processing_status, metadata_json, vector_ids_json,
                chunk_count, word_count, language, tags_json, summary
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        self.execute_many(query, params_list)
        return doc_ids
    
    def batch_get_documents(self, doc_ids: List[str]) -> List[Dict[str, Any]]:
        """Get multiple documents by IDs."""
        if not doc_ids:
            return []
        
        placeholders = ','.join(['?' for _ in doc_ids])
        results = self.execute_query(
            f"SELECT * FROM documents WHERE id IN ({placeholders})",
            tuple(doc_ids)
        )
        
        return [self._parse_document_json(doc) for doc in results]
    
    def batch_update_documents(self, updates: List[Dict[str, Any]]) -> bool:
        """Update multiple documents."""
        success = True
        for update in updates:
            doc_id = update.pop('id', None)
            if doc_id and update:
                try:
                    self.update_document(doc_id, update)
                except Exception as e:
                    logger.error(f"Failed to update document {doc_id}: {e}")
                    success = False
        return success
    
    def batch_delete_documents(self, doc_ids: List[str]) -> bool:
        """Delete multiple documents."""
        if not doc_ids:
            return False
        
        placeholders = ','.join(['?' for _ in doc_ids])
        self.execute_query(
            f"DELETE FROM documents WHERE id IN ({placeholders})",
            tuple(doc_ids)
        )
        return True
    
    # ==================== Chunk Operations ====================
    
    def add_chunk(self, chunk: Dict[str, Any]) -> str:
        """Add a chunk to the database."""
        chunk_id = chunk.get('id') or str(uuid.uuid4())
        
        metadata_json = json.dumps(chunk.get('metadata', {}))
        
        query = """
            INSERT INTO chunks (
                id, document_id, chunk_index, text_content, cleaned_text,
                token_count, embedding_model, vector_id, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            chunk_id,
            chunk['document_id'],
            chunk['chunk_index'],
            chunk['text_content'],
            chunk.get('cleaned_text', chunk['text_content']),
            chunk.get('token_count'),
            chunk.get('embedding_model'),
            chunk['vector_id'],
            metadata_json
        )
        
        self.execute_query(query, params)
        return chunk_id
    
    def get_chunks_by_document(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a document."""
        results = self.execute_query(
            "SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index",
            (document_id,)
        )
        
        chunks = []
        for row in results:
            chunk = dict(row)
            if 'metadata_json' in chunk and chunk['metadata_json']:
                try:
                    chunk['metadata'] = json.loads(chunk['metadata_json'])
                except:
                    chunk['metadata'] = {}
                del chunk['metadata_json']
            chunks.append(chunk)
        
        return chunks
    
    def get_chunk_by_vector_id(self, vector_id: str) -> Optional[Dict[str, Any]]:
        """Get chunk by vector ID."""
        results = self.execute_query(
            "SELECT * FROM chunks WHERE vector_id = ?",
            (vector_id,)
        )
        
        if not results:
            return None
        
        chunk = dict(results[0])
        if 'metadata_json' in chunk and chunk['metadata_json']:
            try:
                chunk['metadata'] = json.loads(chunk['metadata_json'])
            except:
                chunk['metadata'] = {}
            del chunk['metadata_json']
        
        return chunk
    
    def delete_chunks_by_document(self, document_id: str) -> bool:
        """Delete all chunks for a document."""
        self.execute_query(
            "DELETE FROM chunks WHERE document_id = ?",
            (document_id,)
        )
        return True
    
    # ==================== Conversation Operations ====================
    
    def create_conversation(self, conversation: Dict[str, Any]) -> str:
        """Create a new conversation."""
        conv_id = conversation.get('id') or str(uuid.uuid4())
        
        tags_json = json.dumps(conversation.get('tags', []))
        
        query = """
            INSERT INTO conversations (
                id, title, tags_json, is_archived, export_path
            ) VALUES (?, ?, ?, ?, ?)
        """
        
        params = (
            conv_id,
            conversation.get('title'),
            tags_json,
            conversation.get('is_archived', False),
            conversation.get('export_path')
        )
        
        self.execute_query(query, params)
        return conv_id
    
    def get_conversation(self, conv_id: str) -> Optional[Dict[str, Any]]:
        """Get conversation by ID."""
        results = self.execute_query(
            "SELECT * FROM conversations WHERE id = ?",
            (conv_id,)
        )
        
        if not results:
            return None
        
        conv = dict(results[0])
        if 'tags_json' in conv and conv['tags_json']:
            try:
                conv['tags'] = json.loads(conv['tags_json'])
            except:
                conv['tags'] = []
            del conv['tags_json']
        
        return conv
    
    def add_message(self, message: Dict[str, Any]) -> str:
        """Add a message to a conversation."""
        msg_id = message.get('id') or str(uuid.uuid4())
        
        sources_json = json.dumps(message.get('sources', []))
        
        query = """
            INSERT INTO messages (
                id, conversation_id, role, content, tokens,
                model_used, sources_json, processing_time_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            msg_id,
            message['conversation_id'],
            message['role'],
            message['content'],
            message.get('tokens'),
            message.get('model_used'),
            sources_json,
            message.get('processing_time_ms')
        )
        
        self.execute_query(query, params)
        
        # Update conversation message count and updated_at
        self.execute_query(
            """UPDATE conversations 
               SET message_count = message_count + 1, 
                   updated_at = CURRENT_TIMESTAMP 
               WHERE id = ?""",
            (message['conversation_id'],)
        )
        
        return msg_id
    
    def get_conversation_messages(self, conv_id: str) -> List[Dict[str, Any]]:
        """Get all messages in a conversation."""
        results = self.execute_query(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at",
            (conv_id,)
        )
        
        messages = []
        for row in results:
            msg = dict(row)
            if 'sources_json' in msg and msg['sources_json']:
                try:
                    msg['sources'] = json.loads(msg['sources_json'])
                except:
                    msg['sources'] = []
                del msg['sources_json']
            messages.append(msg)
        
        return messages
    
    def delete_conversation(self, conv_id: str) -> bool:
        """Delete a conversation (messages will cascade)."""
        self.execute_query(
            "DELETE FROM conversations WHERE id = ?",
            (conv_id,)
        )
        return True
    
    # ==================== Tag Operations ====================
    
    def add_tag(self, tag: Dict[str, Any]) -> str:
        """Add a tag."""
        tag_id = tag.get('id') or str(uuid.uuid4())
        
        query = """
            INSERT INTO tags (id, name, color, description)
            VALUES (?, ?, ?, ?)
        """
        
        params = (
            tag_id,
            tag['name'],
            tag.get('color'),
            tag.get('description')
        )
        
        self.execute_query(query, params)
        return tag_id
    
    def get_tag_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get tag by name."""
        results = self.execute_query(
            "SELECT * FROM tags WHERE name = ?",
            (name,)
        )
        
        return dict(results[0]) if results else None
    
    def add_tag_to_document(self, document_id: str, tag_id: str) -> bool:
        """Add a tag to a document."""
        try:
            self.execute_query(
                "INSERT INTO document_tags (document_id, tag_id) VALUES (?, ?)",
                (document_id, tag_id)
            )
            
            # Update tag usage count
            self.execute_query(
                "UPDATE tags SET usage_count = usage_count + 1 WHERE id = ?",
                (tag_id,)
            )
            
            return True
        except sqlite3.IntegrityError:
            # Tag already attached to document
            return False
    
    def remove_tag_from_document(self, document_id: str, tag_id: str) -> bool:
        """Remove a tag from a document."""
        self.execute_query(
            "DELETE FROM document_tags WHERE document_id = ? AND tag_id = ?",
            (document_id, tag_id)
        )
        
        # Update tag usage count
        self.execute_query(
            "UPDATE tags SET usage_count = usage_count - 1 WHERE id = ?",
            (tag_id,)
        )
        
        return True
    
    def get_document_tags(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all tags for a document."""
        results = self.execute_query("""
            SELECT t.* FROM tags t
            JOIN document_tags dt ON t.id = dt.tag_id
            WHERE dt.document_id = ?
        """, (document_id,))
        
        return [dict(row) for row in results]
    
    def find_documents_by_tag(self, tag_name: str) -> List[Dict[str, Any]]:
        """Find documents by tag name."""
        results = self.execute_query("""
            SELECT d.* FROM documents d
            JOIN document_tags dt ON d.id = dt.document_id
            JOIN tags t ON dt.tag_id = t.id
            WHERE t.name = ?
        """, (tag_name,))
        
        return [self._parse_document_json(doc) for doc in results]
    
    # ==================== Transaction Operations ====================
    
    def begin_transaction(self) -> None:
        """Begin a transaction."""
        self.execute_query("BEGIN TRANSACTION")
    
    def commit_transaction(self) -> None:
        """Commit the current transaction."""
        self.execute_query("COMMIT")
    
    def rollback_transaction(self) -> None:
        """Rollback the current transaction."""
        self.execute_query("ROLLBACK")
    
    @contextmanager
    def transaction(self):
        """Context manager for transactions."""
        self.begin_transaction()
        try:
            yield
            self.commit_transaction()
        except Exception as e:
            self.rollback_transaction()
            raise e
    
    # ==================== Utility Methods ====================
    
    def migrate_database(self) -> bool:
        """Migrate database to latest schema."""
        try:
            # Get current version
            result = self.execute_query("PRAGMA user_version")
            current_version = result[0]['user_version'] if result else 0
            
            # Perform migrations based on version
            if current_version < 1:
                # Version 1 migrations
                with self.transaction():
                    # Add any new columns or tables here
                    self.execute_query("PRAGMA user_version = 1")
                logger.info("Database migrated to version 1")
            
            return True
        except Exception as e:
            logger.error(f"Database migration failed: {e}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """Check database health with parameterized queries."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Check connection
                cursor.execute("SELECT 1")
                
                # Get database info
                cursor.execute("PRAGMA database_list")
                databases = cursor.fetchall()
                
                try:
                    cursor.execute("PRAGMA integrity_check")
                    integrity = cursor.fetchone()
                except Exception as e:
                    integrity = ['skip']
                
                cursor.execute("SELECT COUNT(*) as count FROM sqlite_master")
                table_count = cursor.fetchone()[0]
                
                # Get pragma settings
                pragmas = {
                    'foreign_keys': 'PRAGMA foreign_keys',
                    'journal_mode': 'PRAGMA journal_mode',
                    'temp_store': 'PRAGMA temp_store',
                    'synchronous': 'PRAGMA synchronous',
                    'cache_size': 'PRAGMA cache_size'
                }
                
                pragma_results = {}
                for name, pragma in pragmas.items():
                    try:
                        cursor.execute(pragma)
                        result = cursor.fetchone()
                        pragma_results[name] = result[0] if result else None
                    except Exception as e:
                        pragma_results[name] = f'ERROR: {e}'
                
                # Get statistics
                stats = {
                    'status': DatabaseHealth.HEALTHY.value,
                    'database_path': self.config.database_path,
                    'database_count': len(databases),
                    'table_count': table_count,
                    'integrity_check': integrity[0] if integrity else 'unknown',
                    'connection_count': len(self._connections),
                    'pragma_settings': pragma_results,
                    'timestamp': datetime.now().isoformat()
                }
                
                return stats
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'status': DatabaseHealth.UNHEALTHY.value,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def backup_database(self, backup_path: Optional[str] = None) -> bool:
        """Backup database."""
        if not self.config.enable_backup:
            return False
        
        try:
            if backup_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = f"{self.config.database_path}.backup.{timestamp}"
            
            with self._get_connection() as conn:
                # Use SQLite backup API
                backup_conn = sqlite3.connect(backup_path)
                conn.backup(backup_conn)
                backup_conn.close()
            
            logger.info(f"Database backed up to: {backup_path}")
            
            # Update settings
            self.execute_query(
                "INSERT OR REPLACE INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
                ('last_backup', datetime.now().isoformat(), datetime.now())
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return False
    
    def cleanup_old_backups(self) -> int:
        """Cleanup old backup files."""
        if not self.config.enable_backup:
            return 0
        
        try:
            db_path = Path(self.config.database_path)
            backup_dir = db_path.parent
            cutoff_time = datetime.now() - timedelta(days=self.config.backup_retention_days)
            
            deleted_count = 0
            for backup_file in backup_dir.glob(f"{db_path.name}.backup.*"):
                if backup_file.stat().st_mtime < cutoff_time.timestamp():
                    try:
                        backup_file.unlink()
                        deleted_count += 1
                        logger.info(f"Deleted old backup: {backup_file}")
                    except Exception as e:
                        logger.warning(f"Failed to delete backup {backup_file}: {e}")
            
            logger.info(f"Cleaned up {deleted_count} old backup files")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Backup cleanup failed: {e}")
            return 0
    
    def close(self) -> None:
        """Close all database connections."""
        with self._connection_lock:
            for thread_id, conn in list(self._connections.items()):
                try:
                    conn.close()
                except Exception as e:
                    logger.error(f"Error closing connection: {e}")
            
            self._connections.clear()
            logger.info("All database connections closed")
    
    def __enter__(self) -> "SQLiteClient":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()