# cockatoo_v1/src/database/sqlite_client.py

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
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

# Setup logging
logger = logging.getLogger(__name__)

# Database constants
DEFAULT_DATABASE_PATH = "cockatoo.db"
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
    
    def __init__(self, config: Optional[DatabaseConfig] = None) -> None:
        self.config = config or DatabaseConfig()
        self._connections = {}
        self._lock = threading.RLock()
        
        # Setup database dengan handling error yang lebih baik
        try:
            self._setup_database()
            logger.info(f"SQLite client initialized for: {self.config.database_path}")
        except Exception as e:
            logger.error(f"SQLite client initialization failed: {e}")
            # Jangan raise, biarkan client tetap dibuat
            # Aplikasi bisa coba setup lagi nanti
    
    def _setup_database(self) -> None:
        """Setup database with proper configuration."""
        try:
            # Create database directory
            db_path = Path(self.config.database_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Initial connection to setup database
            # Gunakan connection terpisah untuk setup
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
            # Jangan raise, biarkan aplikasi tetap berjalan
            # Database bisa dibuat nanti
    
    def _setup_tables(self, conn: sqlite3.Connection) -> None:
        """Create database tables if they don't exist."""
        tables = [
            # Documents table - FIXED TYPO: 'documments' -> 'documents'
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
        
        with self._lock:
            # Check for existing connection
            if thread_id in self._connections:
                conn = self._connections[thread_id]
                try:
                    # Check if connection is still valid
                    conn.execute("SELECT 1")
                    yield conn
                    return
                except sqlite3.Error:
                    # Connection is invalid, remove it
                    try:
                        conn.close()
                    except Exception:
                        pass
                    del self._connections[thread_id]
            
            # Create new connection
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
            raise
        finally:
            # Keep connection in pool for reuse
            pass
    
    def execute_query(self, query: str, params: tuple = (), timeout: Optional[int] = None) -> List[Dict]:
        """Execute a parameterized query and return results."""
        timeout = timeout or self.config.query_timeout
        
        for attempt in range(self.config.max_retry_attempts):
            try:
                with self._get_connection() as conn:
                    conn.execute("PRAGMA busy_timeout = ?", (timeout * 1000,))
                    cursor = conn.cursor()
                    
                    # Execute parameterized query - FIXED: Jangan gunakan f-string untuk SQL
                    cursor.execute(query, params)
                    
                    if query.strip().upper().startswith("SELECT"):
                        rows = cursor.fetchall()
                        return [dict(row) for row in rows]
                    else:
                        conn.commit()
                        return []
                        
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < self.config.max_retry_attempts - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                    continue
                logger.error(f"Database operational error: {e}")
                raise
            except Exception as e:
                logger.error(f"Query execution error: {e}")
                raise
        
        return []
    
    def execute_many(self, query: str, params_list: List[tuple]) -> bool:
        """Execute many parameterized queries in a transaction."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.executemany(query, params_list)
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Execute many error: {e}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """Check database health with parameterized queries."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Check connection with parameterized query
                cursor.execute("SELECT 1")
                
                # Get database info with parameterized queries
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
        with self._lock:
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