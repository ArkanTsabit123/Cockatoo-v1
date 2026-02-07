# cockatoo_v1/src/database/__init__.py

"""
Database module initialization.
Provides migration utilities, configuration, and database setup with enhanced security and error handling.
"""

import logging
import sys
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

# Fix import path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Initialize logger first to catch import errors
logger = logging.getLogger(__name__)

# Database constants
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "llama2:7b"

# Database connection constants
DEFAULT_DATABASE_PATH = "cockatoo.db"
DEFAULT_POOL_SIZE = 5
DEFAULT_MAX_OVERFLOW = 10
DEFAULT_POOL_TIMEOUT = 30

# Document status constants
DOCUMENT_STATUS_PENDING = "pending"
DOCUMENT_STATUS_PROCESSING = "processing"
DOCUMENT_STATUS_COMPLETED = "completed"
DOCUMENT_STATUS_ERROR = "error"

# Query constants
DEFAULT_QUERY_TIMEOUT = 30
DEFAULT_MAX_RESULTS = 100

# String constants
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
CONNECTION_STRING_PREFIX = "sqlite:///"
INIT_SUCCESS_MESSAGE = "Database initialized at: {path}"
MODELS_IMPORT_ERROR = "Failed to import database models: {error}"
CLIENT_IMPORT_ERROR = "Failed to import SQLite client: {error}"
QUERIES_IMPORT_ERROR = "Failed to import queries: {error}"
SECURITY_IMPORT_ERROR = "Failed to import security utilities: {error}"
SPECIALIZED_CLIENTS_WARNING = "Specialized clients not available: {error}"
TABLES_CREATED_SUCCESS = "SQLAlchemy tables created successfully"
SQLITE_CLIENT_INIT_SUCCESS = "SQLite client initialized"
SPECIALIZED_CLIENTS_INIT_SUCCESS = "Specialized clients initialized"
DATABASE_CONNECTIONS_CLOSED = "Database connections closed"
FAILED_INIT_CLIENTS = "Failed to initialize specialized clients: {client_error}"
FAILED_IMPORT_MODULES = "Failed to import database modules: {import_error}"
FAILED_CREATE_DIR = "Failed to create database directory: {os_error}"
UNEXPECTED_INIT_ERROR = "Unexpected error initializing database: {unexpected_error}"
ERROR_ACCESSING_CLIENT = "Error accessing client for closing: {attr_error}"
UNEXPECTED_CLOSE_ERROR = "Unexpected error closing database connections: {unexpected_error}"
SQLALCHEMY_MODELS_UNAVAILABLE = "SQLAlchemy models not available or not initialized"
FAILED_INIT_DB_CLIENT = "Failed to initialize database client"
FAILED_INIT_DOC_CLIENT = "Failed to initialize document client"
FAILED_INIT_CONV_CLIENT = "Failed to initialize conversation client"
FAILED_INIT_DB_MANAGER = "Failed to initialize database manager"
FAILED_INIT_DB_CONFIG = "Failed to initialize database with provided configuration"
ALREADY_INITIALIZED = "Database already initialized"

# Type checking imports to avoid circular imports
if TYPE_CHECKING:
    from sqlalchemy.orm import Session
    from .sqlite_client import SQLiteClient
    from .document_client import DocumentClient
    from .conversation_client import ConversationClient

# Import database models and clients with better error handling
try:
    from .models import (
        Base, get_engine, get_session_local,
        Document, Chunk, Conversation, Message, Tag, Setting,
        session_scope, init_database as init_models_database
    )
    MODELS_AVAILABLE = True
except ImportError as import_error:
    logger.error(MODELS_IMPORT_ERROR.format(error=import_error))
    MODELS_AVAILABLE = False
    # Create placeholder classes
    class BasePlaceholder:
        metadata = type('Metadata', (), {})()
        def __init__(self):
            pass
    Base = BasePlaceholder
    get_engine = None
    get_session_local = None
    session_scope = None
    
    class DocumentPlaceholder: pass
    class ChunkPlaceholder: pass
    class ConversationPlaceholder: pass
    class MessagePlaceholder: pass
    class TagPlaceholder: pass
    class SettingPlaceholder: pass
    
    Document = DocumentPlaceholder
    Chunk = ChunkPlaceholder
    Conversation = ConversationPlaceholder
    Message = MessagePlaceholder
    Tag = TagPlaceholder
    Setting = SettingPlaceholder

try:
    from .sqlite_client import SQLiteClient, DatabaseConfig
    SQLITE_CLIENT_AVAILABLE = True
except ImportError as import_error:
    logger.error(CLIENT_IMPORT_ERROR.format(error=import_error))
    SQLITE_CLIENT_AVAILABLE = False
    SQLiteClient = type("SQLiteClientPlaceholder", (), {})
    DatabaseConfig = None

try:
    from .queries import (
        get_document_by_id, get_recent_documents,
        search_documents, get_conversation_history,
        update_document_status, add_document_tag, get_statistics
    )
    QUERIES_AVAILABLE = True
except ImportError as import_error:
    logger.error(QUERIES_IMPORT_ERROR.format(error=import_error))
    QUERIES_AVAILABLE = False
    get_document_by_id = None
    get_recent_documents = None
    search_documents = None
    get_conversation_history = None
    update_document_status = None
    add_document_tag = None
    get_statistics = None

try:
    from .security import encrypt_data, decrypt_data, hash_password, verify_password
    SECURITY_AVAILABLE = True
except ImportError as import_error:
    logger.error(SECURITY_IMPORT_ERROR.format(error=import_error))
    SECURITY_AVAILABLE = False
    encrypt_data = None
    decrypt_data = None
    hash_password = None
    verify_password = None

# Import specialized clients (optional)
try:
    from .document_client import DocumentClient
    from .conversation_client import ConversationClient
    SPECIALIZED_CLIENTS_AVAILABLE = True
except ImportError as import_error:
    logger.warning(SPECIALIZED_CLIENTS_WARNING.format(error=import_error))
    SPECIALIZED_CLIENTS_AVAILABLE = False
    DocumentClient = type("DocumentClientPlaceholder", (), {})
    ConversationClient = type("ConversationClientPlaceholder", (), {})

__all__ = [
    # Core models
    "Base",
    "get_engine",
    "get_session_local",
    "session_scope",
    "Document",
    "Chunk",
    "Conversation",
    "Message",
    "Tag",
    "Setting",
    
    # Clients
    "SQLiteClient",
    "DatabaseConfig",
    "DocumentClient",
    "ConversationClient",
    
    # Query functions
    "get_document_by_id",
    "get_recent_documents",
    "search_documents",
    "get_conversation_history",
    "update_document_status",
    "add_document_tag",
    "get_statistics",
    
    # Security
    "encrypt_data",
    "decrypt_data",
    "hash_password",
    "verify_password",
    
    # Constants
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_CHUNK_OVERLAP",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_LLM_MODEL",
    "DOCUMENT_STATUS_PENDING",
    "DOCUMENT_STATUS_PROCESSING",
    "DOCUMENT_STATUS_COMPLETED",
    "DOCUMENT_STATUS_ERROR",
    "DEFAULT_QUERY_TIMEOUT",
    "DEFAULT_MAX_RESULTS",
    
    # Database manager
    "DatabaseManager",
    "get_database_manager",
    "init_database",
]

# Create DatabaseConfig if not imported
if DatabaseConfig is None:
    @dataclass
    class DatabaseConfig:
        """Database configuration settings."""
        database_path: str = DEFAULT_DATABASE_PATH
        pool_size: int = DEFAULT_POOL_SIZE
        max_overflow: int = DEFAULT_MAX_OVERFLOW
        pool_timeout: int = DEFAULT_POOL_TIMEOUT
        echo: bool = False
        enable_foreign_keys: bool = True
        
        def get_connection_string(self) -> str:
            """Get SQLAlchemy connection string."""
            # Use string concatenation instead of f-string to avoid false SQL injection warnings
            return CONNECTION_STRING_PREFIX + str(self.database_path)


class DatabaseManager:
    """Main database manager class with enhanced features."""
    
    def __init__(self, config: Optional[DatabaseConfig] = None) -> None:
        self.config = config or DatabaseConfig()
        self.client: Optional[SQLiteClient] = None
        self.document_client: Optional[DocumentClient] = None
        self.conversation_client: Optional[ConversationClient] = None
        self._engine = None
        self._session_local = None
        self._initialized = False
        self._lock = False
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Setup database logging."""
        db_logger = logging.getLogger("database")
        db_logger.setLevel(logging.INFO)
        
        if not db_logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(LOG_FORMAT)
            handler.setFormatter(formatter)
            db_logger.addHandler(handler)
    
    def initialize(self) -> bool:
        """Initialize database connection."""
        # Check if already initialized
        if self._initialized:
            logger.info(ALREADY_INITIALIZED)
            return True
            
        # Prevent multiple concurrent initializations
        if self._lock:
            logger.warning("Database initialization already in progress")
            return False
            
        self._lock = True
        try:
            # Create database directory if it doesn't exist
            db_path = Path(self.config.database_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Initialize engine and create tables if models are available
            if MODELS_AVAILABLE and get_engine and Base:
                connection_string = self.config.get_connection_string()
                self._engine = get_engine(connection_string, echo=self.config.echo)
                
                # Create tables
                if hasattr(Base, 'metadata'):
                    Base.metadata.create_all(bind=self._engine)
                    logger.info(TABLES_CREATED_SUCCESS)
                else:
                    logger.warning("Base.metadata not available, skipping table creation")
                
                # Create session factory
                if get_session_local:
                    self._session_local = get_session_local(self._engine)
            else:
                logger.warning("SQLAlchemy models not available, continuing with SQLite client only")
            
            # Create database client if available
            if SQLITE_CLIENT_AVAILABLE and SQLiteClient:
                self.client = SQLiteClient(self.config)
                logger.info(SQLITE_CLIENT_INIT_SUCCESS)
                
                # Force foreign keys to be enabled
                try:
                    if self.client and hasattr(self.client, 'execute_query'):
                        self.client.execute_query("PRAGMA foreign_keys = ON", ())
                        logger.info("Foreign keys explicitly enabled")
                except Exception as e:
                    logger.warning(f"Failed to enable foreign keys: {e}")
            else:
                logger.warning("SQLite client not available")
            
            # Create specialized clients if available and client exists
            if SPECIALIZED_CLIENTS_AVAILABLE and self.client:
                try:
                    self.document_client = DocumentClient(self.client)
                    self.conversation_client = ConversationClient(self.client)
                    logger.info(SPECIALIZED_CLIENTS_INIT_SUCCESS)
                except Exception as client_error:
                    logger.warning(FAILED_INIT_CLIENTS.format(client_error=client_error))
            elif SPECIALIZED_CLIENTS_AVAILABLE and not self.client:
                logger.warning("Specialized clients available but SQLite client not initialized")
            
            logger.info(INIT_SUCCESS_MESSAGE.format(path=self.config.database_path))
            self._initialized = True
            return True
            
        except ImportError as import_error:
            logger.error(FAILED_IMPORT_MODULES.format(import_error=import_error))
            return False
        except OSError as os_error:
            logger.error(FAILED_CREATE_DIR.format(os_error=os_error))
            return False
        except Exception as unexpected_error:
            logger.error(UNEXPECTED_INIT_ERROR.format(unexpected_error=unexpected_error))
            return False
        finally:
            self._lock = False
    
    def get_session(self) -> Any:
        """Get database session (SQLAlchemy)."""
        if not self._initialized:
            if not self.initialize():
                raise RuntimeError(SQLALCHEMY_MODELS_UNAVAILABLE)
        
        if self._session_local is not None:
            return self._session_local()
        elif MODELS_AVAILABLE and get_session_local and self._engine:
            # Fallback: create session factory on the fly
            self._session_local = get_session_local(self._engine)
            return self._session_local()
        else:
            raise RuntimeError(SQLALCHEMY_MODELS_UNAVAILABLE)
    
    def get_client(self) -> SQLiteClient:
        """Get SQLite client."""
        if self.client is None:
            if not self.initialize():
                raise RuntimeError(FAILED_INIT_DB_CLIENT)
        return self.client
    
    def get_document_client(self) -> DocumentClient:
        """Get document client."""
        if self.document_client is None:
            client = self.get_client()
            if SPECIALIZED_CLIENTS_AVAILABLE and DocumentClient and client:
                self.document_client = DocumentClient(client)
            else:
                raise RuntimeError(FAILED_INIT_DOC_CLIENT)
        return self.document_client
    
    def get_conversation_client(self) -> ConversationClient:
        """Get conversation client."""
        if self.conversation_client is None:
            client = self.get_client()
            if SPECIALIZED_CLIENTS_AVAILABLE and ConversationClient and client:
                self.conversation_client = ConversationClient(client)
            else:
                raise RuntimeError(FAILED_INIT_CONV_CLIENT)
        return self.conversation_client
    
    def close(self) -> None:
        """Close database connections."""
        try:
            if self.client:
                self.client.close()
                logger.info(DATABASE_CONNECTIONS_CLOSED)
        except AttributeError as attr_error:
            logger.error(ERROR_ACCESSING_CLIENT.format(attr_error=attr_error))
        except Exception as unexpected_error:
            logger.error(UNEXPECTED_CLOSE_ERROR.format(unexpected_error=unexpected_error))
        finally:
            # Reset state even if there was an error
            self.client = None
            self.document_client = None
            self.conversation_client = None
            self._engine = None
            self._session_local = None
            self._initialized = False
    
    def is_initialized(self) -> bool:
        """Check if database is initialized."""
        return self._initialized
    
    def get_engine(self):
        """Get SQLAlchemy engine (if available)."""
        if not self._initialized:
            self.initialize()
        return self._engine
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on database."""
        try:
            if self.client:
                return self.client.health_check()
            else:
                return {"status": "uninitialized", "error": "Client not initialized"}
        except Exception as e:
            return {"status": "error", "error": str(e)}


# Create default database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    """Get or create database manager instance."""
    global _db_manager
    if _db_manager is None or not _db_manager.is_initialized():
        _db_manager = DatabaseManager()
        success = _db_manager.initialize()
        if not success:
            raise RuntimeError(FAILED_INIT_DB_MANAGER)
    return _db_manager


def init_database(config: Optional[DatabaseConfig] = None) -> DatabaseManager:
    """Initialize database with optional configuration."""
    global _db_manager
    if _db_manager is not None and _db_manager.is_initialized():
        logger.warning("Database already initialized, returning existing manager")
        return _db_manager
    
    _db_manager = DatabaseManager(config)
    success = _db_manager.initialize()
    if not success:
        raise RuntimeError(FAILED_INIT_DB_CONFIG)
    return _db_manager