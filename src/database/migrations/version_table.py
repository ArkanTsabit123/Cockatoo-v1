# version_table.py

"""
Version table operations for migration tracking.

Manages the migration version table that tracks applied migrations,
their status, execution history, and provides validation capabilities.
"""

import sqlite3
import hashlib
import threading
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Set
import logging
from enum import Enum
import json
import os
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class MigrationStatus(Enum):
    """Enumeration of migration statuses."""
    PENDING = "pending"
    APPLIED = "applied"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


class SchemaVersions:
    """Schema versions for the version table itself."""
    V1_BASIC = 1  # Original schema
    V2_WITH_LOCK = 2  # Added lock_version
    V3_WITH_BATCH = 3  # Added batch_id and file_path
    CURRENT = V3_WITH_BATCH


class ConcurrencyControl:
    """Manages concurrent access to migration operations with deadlock prevention."""
    
    def __init__(self, timeout_seconds: int = 30):
        self._global_lock = threading.RLock()
        self._version_locks: Dict[str, threading.RLock] = {}
        self._version_locks_lock = threading.RLock()
        self._lock_timeout = timeout_seconds
        self._lock_acquire_timeout = 5  # seconds
        self._deadlock_detection = set()
    
    @contextmanager
    def version_lock(self, version: str):
        """Context manager for version-specific locking with timeout."""
        lock = self._get_version_lock(version)
        acquired = False
        
        try:
            acquired = lock.acquire(timeout=self._lock_acquire_timeout)
            if not acquired:
                raise TimeoutError(f"Failed to acquire lock for version {version} after {self._lock_acquire_timeout}s")
            
            # Deadlock detection
            with self._version_locks_lock:
                if version in self._deadlock_detection:
                    raise RuntimeError(f"Potential deadlock detected for version {version}")
                self._deadlock_detection.add(version)
            
            yield
        finally:
            if acquired:
                with self._version_locks_lock:
                    self._deadlock_detection.discard(version)
                lock.release()
    
    @contextmanager
    def global_lock(self):
        """Context manager for global table-level locking."""
        acquired = self._global_lock.acquire(timeout=self._lock_acquire_timeout)
        if not acquired:
            raise TimeoutError(f"Failed to acquire global lock after {self._lock_acquire_timeout}s")
        
        try:
            yield
        finally:
            self._global_lock.release()
    
    def _get_version_lock(self, version: str) -> threading.RLock:
        """Get or create a version-specific lock."""
        with self._version_locks_lock:
            if version not in self._version_locks:
                self._version_locks[version] = threading.RLock()
            return self._version_locks[version]
    
    def cleanup_locks(self):
        """Clean up unused version locks."""
        with self._version_locks_lock:
            to_remove = []
            for version, lock in self._version_locks.items():
                if not lock._is_owned():  # Check if lock is not in use
                    to_remove.append(version)
            
            for version in to_remove:
                del self._version_locks[version]


class AuditLogger:
    """Handles audit logging for migration operations with batch support."""
    
    AUDIT_TABLE_NAME = "schema_migrations_audit"
    
    def __init__(self, connection: sqlite3.Connection, retention_days: int = 365):
        self.connection = connection
        self.retention_days = retention_days
        self._pending_audits: List[Dict] = []
        self._batch_size = 100
        self._ensure_audit_table_exists()
    
    def _ensure_audit_table_exists(self) -> None:
        """Create audit table if it doesn't exist."""
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.AUDIT_TABLE_NAME} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            migration_id INTEGER NOT NULL,
            version TEXT NOT NULL,
            operation_type TEXT NOT NULL,
            old_status TEXT,
            new_status TEXT,
            old_checksum TEXT,
            new_checksum TEXT,
            old_metadata TEXT,
            new_metadata TEXT,
            performed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            performed_by TEXT NOT NULL,
            ip_address TEXT,
            user_agent TEXT,
            batch_id TEXT,
            FOREIGN KEY (migration_id) REFERENCES schema_migrations(id) ON DELETE CASCADE
        )
        """
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(create_table_sql)
            self._create_audit_indexes(cursor)
            self.connection.commit()
            logger.info(f"Audit table '{self.AUDIT_TABLE_NAME}' ensured")
        except sqlite3.Error as e:
            logger.error(f"Failed to create audit table: {e}")
            raise
    
    def _create_audit_indexes(self, cursor: sqlite3.Cursor) -> None:
        """Create indexes for audit table."""
        try:
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_audit_migration_id 
                ON {self.AUDIT_TABLE_NAME}(migration_id)
            """)
            
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_audit_performed_at 
                ON {self.AUDIT_TABLE_NAME}(performed_at)
            """)
            
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_audit_version 
                ON {self.AUDIT_TABLE_NAME}(version)
            """)
            
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_audit_batch 
                ON {self.AUDIT_TABLE_NAME}(batch_id)
            """)
        except sqlite3.Error as e:
            logger.warning(f"Failed to create some audit indexes: {e}")
    
    def log_operation(self, migration_id: int, version: str, operation_type: str,
                     old_values: Dict[str, Any], new_values: Dict[str, Any],
                     performed_by: str, ip_address: Optional[str] = None,
                     user_agent: Optional[str] = None, batch_id: Optional[str] = None) -> bool:
        """Log an audit trail entry with batch support."""
        audit_entry = {
            'migration_id': migration_id,
            'version': version,
            'operation_type': operation_type,
            'old_status': old_values.get('status'),
            'new_status': new_values.get('status'),
            'old_checksum': old_values.get('checksum'),
            'new_checksum': new_values.get('checksum'),
            'old_metadata': json.dumps(old_values.get('metadata'), default=str) if old_values.get('metadata') else None,
            'new_metadata': json.dumps(new_values.get('metadata'), default=str) if new_values.get('metadata') else None,
            'performed_at': datetime.utcnow(),
            'performed_by': performed_by,
            'ip_address': ip_address,
            'user_agent': user_agent,
            'batch_id': batch_id
        }
        
        self._pending_audits.append(audit_entry)
        
        if len(self._pending_audits) >= self._batch_size:
            return self._flush_pending_audits()
        
        return True
    
    def _flush_pending_audits(self) -> bool:
        """Flush pending audit entries to database in batch."""
        if not self._pending_audits:
            return True
        
        try:
            cursor = self.connection.cursor()
            
            for audit in self._pending_audits:
                cursor.execute(
                    f"""
                    INSERT INTO {self.AUDIT_TABLE_NAME} 
                    (migration_id, version, operation_type, old_status, new_status,
                     old_checksum, new_checksum, old_metadata, new_metadata,
                     performed_at, performed_by, ip_address, user_agent, batch_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        audit['migration_id'],
                        audit['version'],
                        audit['operation_type'],
                        audit['old_status'],
                        audit['new_status'],
                        audit['old_checksum'],
                        audit['new_checksum'],
                        audit['old_metadata'],
                        audit['new_metadata'],
                        audit['performed_at'],
                        audit['performed_by'],
                        audit['ip_address'],
                        audit['user_agent'],
                        audit['batch_id']
                    )
                )
            
            self._pending_audits.clear()
            self.connection.commit()
            return True
            
        except sqlite3.Error as e:
            logger.error(f"Failed to flush audit entries: {e}")
            return False
    
    def apply_retention_policy(self) -> int:
        """Apply retention policy to audit table - FIXED VERSION."""
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                f"""
                DELETE FROM {self.AUDIT_TABLE_NAME}
                WHERE performed_at < datetime('now', ?)
                """,
                (f"-{self.retention_days} days",)  # Fixed parameter binding
            )
            
            deleted_count = cursor.rowcount
            self.connection.commit()
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old audit records")
            
            return deleted_count
            
        except sqlite3.Error as e:
            logger.error(f"Failed to apply audit retention policy: {e}")
            return 0


class VersionTable:
    """
    Manages migration version tracking in database with full concurrency control,
    audit trail, and schema versioning.
    """
    
    TABLE_NAME = "schema_migrations"
    
    def __init__(self, connection: sqlite3.Connection, 
                 data_retention_days: int = 365,
                 max_records: int = 1000,
                 batch_size: int = 50) -> None:
        """
        Initialize version table manager.
        
        Args:
            connection: SQLite database connection
            data_retention_days: Days to retain audit records
            max_records: Maximum number of migration records to keep
            batch_size: Batch size for paginated queries
        """
        self.connection = connection
        self.concurrency = ConcurrencyControl()
        self.audit_logger = AuditLogger(connection, data_retention_days)
        self.data_retention_days = data_retention_days
        self.max_records = max_records
        self.batch_size = batch_size
        self._ensure_table_exists()
        self._migrate_schema_if_needed()
        # Ensure audit table is ready
        self.audit_logger._flush_pending_audits()
    
    def _ensure_table_exists(self) -> None:
        """Create version table with current schema."""
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.TABLE_NAME} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            version TEXT NOT NULL UNIQUE,
            migration_name TEXT NOT NULL,
            applied_at TIMESTAMP NOT NULL,
            applied_by TEXT NOT NULL DEFAULT 'system',
            status TEXT NOT NULL DEFAULT 'pending',
            checksum TEXT,
            file_path TEXT,
            duration_ms INTEGER,
            rollback_duration_ms INTEGER,
            error_message TEXT,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            lock_version INTEGER DEFAULT 0,
            batch_id TEXT,
            schema_version INTEGER DEFAULT {SchemaVersions.CURRENT}
        )
        """
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(create_table_sql)
            self._create_table_indexes(cursor)
            self.connection.commit()
            logger.info(f"Version table '{self.TABLE_NAME}' ensured")
        except sqlite3.Error as e:
            logger.error(f"Failed to create version table: {e}")
            raise
    
    def _create_table_indexes(self, cursor: sqlite3.Cursor) -> None:
        """Create indexes for version table."""
        try:
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.TABLE_NAME}_version 
                ON {self.TABLE_NAME}(version)
            """)
            
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.TABLE_NAME}_status 
                ON {self.TABLE_NAME}(status)
            """)
            
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.TABLE_NAME}_applied_at 
                ON {self.TABLE_NAME}(applied_at DESC)
            """)
            
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.TABLE_NAME}_batch_id 
                ON {self.TABLE_NAME}(batch_id)
            """)
            
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.TABLE_NAME}_checksum 
                ON {self.TABLE_NAME}(checksum) WHERE checksum IS NOT NULL
            """)
            
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.TABLE_NAME}_status_applied 
                ON {self.TABLE_NAME}(status, applied_at DESC)
            """)
            
        except sqlite3.Error as e:
            logger.warning(f"Failed to create some indexes: {e}")
    
    def _migrate_schema_if_needed(self):
        """Migrate table schema if needed - FIXED VERSION."""
        current_version = self._get_current_schema_version()
        
        if current_version < SchemaVersions.CURRENT:
            logger.info(f"Current schema version: {current_version}, migrating to {SchemaVersions.CURRENT}")
            
            # Migrate step by step only for older versions
            if current_version < SchemaVersions.V2_WITH_LOCK:
                self._migrate_to_v2()
            
            if current_version < SchemaVersions.V3_WITH_BATCH:
                self._migrate_to_v3()
        else:
            logger.debug(f"Schema already at current version: {current_version}")
    
    def _get_current_schema_version(self) -> int:
        """Get current schema version of the version table - FIXED VERSION."""
        try:
            cursor = self.connection.cursor()
            
            # First check if the table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='schema_migrations'")
            if not cursor.fetchone():
                # Table doesn't exist yet, will be created with current schema
                return SchemaVersions.CURRENT
            
            # Check if schema_version column exists
            cursor.execute("PRAGMA table_info(schema_migrations)")
            columns = {col[1]: col for col in cursor.fetchall()}
            
            if 'schema_version' not in columns:
                # If schema_version column doesn't exist, it's V1
                return SchemaVersions.V1_BASIC
            
            # Get the schema version from an existing row or return default
            cursor.execute("SELECT schema_version FROM schema_migrations LIMIT 1")
            result = cursor.fetchone()
            if result:
                return result[0] 
            else:
                # Check if we have any rows at all
                cursor.execute("SELECT COUNT(*) FROM schema_migrations")
                count = cursor.fetchone()[0]
                if count > 0:
                    # Table has rows but no schema_version, assume V1
                    return SchemaVersions.V1_BASIC
                else:
                    # Empty table, could be freshly created with current schema
                    return SchemaVersions.CURRENT
        except sqlite3.Error as e:
            logger.warning(f"Error getting schema version: {e}")
            return SchemaVersions.V1_BASIC
    
    def _migrate_to_v2(self):
        """Migrate from V1 to V2 (add lock_version) - FIXED VERSION."""
        try:
            cursor = self.connection.cursor()
            
            # First check if lock_version column already exists
            cursor.execute("PRAGMA table_info(schema_migrations)")
            columns = [col[1] for col in cursor.fetchall()]
            
            if 'lock_version' not in columns:
                cursor.execute("""
                    ALTER TABLE schema_migrations 
                    ADD COLUMN lock_version INTEGER DEFAULT 0
                """)
                logger.info("Added lock_version column")
            else:
                logger.debug("lock_version column already exists")
            
            # Update schema version for existing rows
            cursor.execute("""
                UPDATE schema_migrations 
                SET schema_version = ?
                WHERE schema_version IS NULL OR schema_version < ?
            """, (SchemaVersions.V2_WITH_LOCK, SchemaVersions.V2_WITH_LOCK))
            
            self.connection.commit()
            logger.info("Migrated version table schema to V2")
        except sqlite3.Error as e:
            logger.error(f"Failed to migrate schema to V2: {e}")
            self.connection.rollback()
            raise
    
    def _migrate_to_v3(self):
        """Migrate from V2 to V3 (add batch_id and file_path) - FIXED VERSION."""
        try:
            cursor = self.connection.cursor()
            
            # Check if columns already exist
            cursor.execute("PRAGMA table_info(schema_migrations)")
            columns = [col[1] for col in cursor.fetchall()]
            
            # Add batch_id if it doesn't exist
            if 'batch_id' not in columns:
                cursor.execute("""
                    ALTER TABLE schema_migrations 
                    ADD COLUMN batch_id TEXT
                """)
                logger.info("Added batch_id column")
            else:
                logger.debug("batch_id column already exists")
            
            # Add file_path if it doesn't exist
            if 'file_path' not in columns:
                cursor.execute("""
                    ALTER TABLE schema_migrations 
                    ADD COLUMN file_path TEXT
                """)
                logger.info("Added file_path column")
            else:
                logger.debug("file_path column already exists")
            
            # Update schema version
            cursor.execute("""
                UPDATE schema_migrations 
                SET schema_version = ?
            """, (SchemaVersions.V3_WITH_BATCH,))
            
            self.connection.commit()
            logger.info("Migrated version table schema to V3")
        except sqlite3.Error as e:
            logger.error(f"Failed to migrate schema to V3: {e}")
            self.connection.rollback()
            raise
    
    def _execute_with_retry(self, operation_func, max_retries=3, delay=0.1):
        """
        Execute operation with retry on lock contention.
        
        Args:
            operation_func: Function to execute
            max_retries: Maximum number of retry attempts
            delay: Initial delay between retries (seconds)
        """
        for attempt in range(max_retries):
            try:
                return operation_func()
            except sqlite3.OperationalError as e:
                if 'database is locked' in str(e) and attempt < max_retries - 1:
                    sleep_time = delay * (2 ** attempt)
                    time.sleep(sleep_time)
                    logger.debug(f"Retry {attempt + 1}/{max_retries} after lock contention")
                    continue
                raise
            except TimeoutError as e:
                if attempt < max_retries - 1:
                    sleep_time = delay * (2 ** attempt)
                    time.sleep(sleep_time)
                    logger.debug(f"Retry {attempt + 1}/{max_retries} after timeout")
                    continue
                raise
    
    def _get_current_user(self) -> str:
        """Get current user/system identifier."""
        import getpass
        import platform
        
        try:
            user = getpass.getuser()
            host = platform.node()
            return f"{user}@{host}"
        except Exception:
            return "system"
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate SHA-256 checksum of a migration file."""
        try:
            with open(file_path, 'rb') as f:
                file_hash = hashlib.sha256()
                chunk = f.read(8192)
                while chunk:
                    file_hash.update(chunk)
                    chunk = f.read(8192)
                return file_hash.hexdigest()
        except FileNotFoundError:
            logger.error(f"Migration file not found: {file_path}")
            raise
    
    def _resolve_file_path(self, version: str, checksum: str, 
                          search_paths: List[str]) -> Optional[str]:
        """
        Resolve migration file path by checksum.
        
        Args:
            version: Migration version
            checksum: Expected checksum
            search_paths: Directories to search
            
        Returns:
            Path to file with matching checksum, or None
        """
        for search_path in search_paths:
            if not os.path.isdir(search_path):
                continue
                
            for root, _, files in os.walk(search_path):
                for file in files:
                    if not file.endswith(('.sql', '.py')):
                        continue
                    
                    file_path = os.path.join(root, file)
                    try:
                        file_checksum = self._calculate_checksum(file_path)
                        if file_checksum == checksum:
                            return file_path
                    except Exception:
                        continue
        
        return None
    
    def _get_migration_record_for_update(self, version: str) -> Optional[Dict[str, Any]]:
        """
        Get migration record with row-level locking for update.
        
        Args:
            version: Migration version string
            
        Returns:
            Migration record or None if not found
        """
        def operation():
            cursor = self.connection.cursor()
            cursor.execute(
                f"""
                SELECT 
                    id, version, migration_name, applied_at, applied_by,
                    status, checksum, file_path, duration_ms, rollback_duration_ms,
                    error_message, metadata, created_at, updated_at, lock_version,
                    batch_id, schema_version
                FROM {self.TABLE_NAME} 
                WHERE version = ?
                """,
                (version,)
            )
            
            row = cursor.fetchone()
            if not row:
                return None
            
            columns = [col[0] for col in cursor.description]
            migration = dict(zip(columns, row))
            
            if migration.get('metadata'):
                try:
                    migration['metadata'] = json.loads(migration['metadata'])
                except json.JSONDecodeError:
                    migration['metadata'] = {}
            else:
                migration['metadata'] = {}
            
            return migration
        
        return self._execute_with_retry(operation)
    
    # ============================================================================
    # NEW METHODS TO FIX TEST ISSUES
    # ============================================================================
    
    def get_migration_status(self, version: str) -> Optional[str]:
        """
        Get the status of a specific migration.
        
        Args:
            version: Migration version string
            
        Returns:
            Migration status or None if not found
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                f"""
                SELECT status FROM {self.TABLE_NAME} 
                WHERE version = ?
                """,
                (version,)
            )
            
            result = cursor.fetchone()
            return result[0] if result else None
            
        except sqlite3.Error as e:
            logger.error(f"Failed to get migration status for {version}: {e}")
            return None
    
    def get_migration_by_version(self, version: str) -> Optional[Dict[str, Any]]:
        """
        Get full migration record by version.
        
        Args:
            version: Migration version string
            
        Returns:
            Migration record or None if not found
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                f"""
                SELECT 
                    id, version, migration_name, applied_at, applied_by,
                    status, checksum, file_path, duration_ms, rollback_duration_ms,
                    error_message, metadata, created_at, updated_at, lock_version,
                    batch_id, schema_version
                FROM {self.TABLE_NAME} 
                WHERE version = ?
                """,
                (version,)
            )
            
            row = cursor.fetchone()
            if not row:
                return None
            
            columns = [col[0] for col in cursor.description]
            migration = dict(zip(columns, row))
            
            if migration.get('metadata'):
                try:
                    migration['metadata'] = json.loads(migration['metadata'])
                except json.JSONDecodeError:
                    migration['metadata'] = {}
            else:
                migration['metadata'] = {}
            
            return migration
            
        except sqlite3.Error as e:
            logger.error(f"Failed to get migration by version {version}: {e}")
            return None
    
    def get_all_migrations(self, page: int = 1, page_size: Optional[int] = None) -> Tuple[List[Dict[str, Any]], int]:
        """
        Get all migrations with pagination.
        
        Args:
            page: Page number (1-indexed)
            page_size: Records per page
            
        Returns:
            Tuple of (migrations list, total count)
        """
        page_size = page_size or self.batch_size
        offset = (page - 1) * page_size
        
        try:
            cursor = self.connection.cursor()
            
            cursor.execute(f"SELECT COUNT(*) FROM {self.TABLE_NAME}")
            total = cursor.fetchone()[0]
            
            cursor.execute(
                f"""
                SELECT 
                    id, version, migration_name, applied_at, applied_by,
                    status, checksum, file_path, duration_ms, rollback_duration_ms,
                    error_message, metadata, created_at, updated_at, lock_version,
                    batch_id, schema_version
                FROM {self.TABLE_NAME} 
                ORDER BY applied_at DESC
                LIMIT ? OFFSET ?
                """,
                (page_size, offset)
            )
            
            columns = [col[0] for col in cursor.description]
            rows = cursor.fetchall()
            
            migrations = []
            for row in rows:
                migration = dict(zip(columns, row))
                
                if migration.get('metadata'):
                    try:
                        migration['metadata'] = json.loads(migration['metadata'])
                    except json.JSONDecodeError:
                        migration['metadata'] = {}
                else:
                    migration['metadata'] = {}
                
                migrations.append(migration)
            
            return migrations, total
            
        except sqlite3.Error as e:
            logger.error(f"Failed to get all migrations: {e}")
            return [], 0
    
    def get_failed_migrations(self) -> List[Dict[str, Any]]:
        """
        Get all failed migrations.
        
        Returns:
            List of failed migration records
        """
        try:
            cursor = self.connection.cursor()
            
            cursor.execute(
                f"""
                SELECT 
                    id, version, migration_name, applied_at, applied_by,
                    status, checksum, file_path, duration_ms, rollback_duration_ms,
                    error_message, metadata, created_at, updated_at, lock_version,
                    batch_id, schema_version
                FROM {self.TABLE_NAME} 
                WHERE status = ?
                ORDER BY applied_at DESC
                """,
                (MigrationStatus.FAILED.value,)
            )
            
            columns = [col[0] for col in cursor.description]
            rows = cursor.fetchall()
            
            migrations = []
            for row in rows:
                migration = dict(zip(columns, row))
                
                if migration.get('metadata'):
                    try:
                        migration['metadata'] = json.loads(migration['metadata'])
                    except json.JSONDecodeError:
                        migration['metadata'] = {}
                else:
                    migration['metadata'] = {}
                
                migrations.append(migration)
            
            return migrations
            
        except sqlite3.Error as e:
            logger.error(f"Failed to get failed migrations: {e}")
            return []
    
    def update_migration_file_path(self, version: str, file_path: str) -> bool:
        """
        Update the file path for a migration.
        
        Args:
            version: Migration version
            file_path: New file path
            
        Returns:
            True if updated successfully
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                f"""
                UPDATE {self.TABLE_NAME} 
                SET file_path = ?, updated_at = ?
                WHERE version = ?
                """,
                (file_path, datetime.utcnow(), version)
            )
            
            updated = cursor.rowcount > 0
            self.connection.commit()
            
            if updated:
                logger.info(f"Updated file path for migration {version}: {file_path}")
            
            return updated
            
        except sqlite3.Error as e:
            logger.error(f"Failed to update file path for migration {version}: {e}")
            return False
    
    def get_last_migration(self) -> Optional[Dict[str, Any]]:
        """
        Get the most recently applied migration.
        
        Returns:
            Last migration record or None
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                f"""
                SELECT 
                    id, version, migration_name, applied_at, applied_by,
                    status, checksum, file_path, duration_ms, rollback_duration_ms,
                    error_message, metadata, created_at, updated_at, lock_version,
                    batch_id, schema_version
                FROM {self.TABLE_NAME} 
                WHERE status = ?
                ORDER BY applied_at DESC
                LIMIT 1
                """,
                (MigrationStatus.APPLIED.value,)
            )
            
            row = cursor.fetchone()
            if not row:
                return None
            
            columns = [col[0] for col in cursor.description]
            migration = dict(zip(columns, row))
            
            if migration.get('metadata'):
                try:
                    migration['metadata'] = json.loads(migration['metadata'])
                except json.JSONDecodeError:
                    migration['metadata'] = {}
            else:
                migration['metadata'] = {}
            
            return migration
            
        except sqlite3.Error as e:
            logger.error(f"Failed to get last migration: {e}")
            return None
    
    # ============================================================================
    # EXISTING METHODS
    # ============================================================================
    
    def create_version_table(self) -> bool:
        """Explicitly create version table."""
        try:
            self._ensure_table_exists()
            return True
        except Exception as e:
            logger.error(f"Failed to create version table: {e}")
            return False
    
    def record_migration_start(self, version: str, migration_name: str, 
                              migration_path: Optional[str] = None,
                              search_paths: Optional[List[str]] = None,
                              batch_id: Optional[str] = None) -> bool:
        """
        Record migration start in version table with full concurrency control.
        """
        def operation():
            with self.concurrency.version_lock(version):
                with self.connection:
                    cursor = self.connection.cursor()
                    
                    checksum = None
                    if migration_path:
                        try:
                            checksum = self._calculate_checksum(migration_path)
                        except Exception as e:
                            logger.warning(f"Could not calculate checksum: {e}")
                    
                    cursor.execute(
                        f"SELECT id, status, lock_version FROM {self.TABLE_NAME} WHERE version = ?",
                        (version,)
                    )
                    existing = cursor.fetchone()
                    
                    if existing:
                        migration_id, status, lock_version = existing
                        if status == MigrationStatus.PENDING.value:
                            logger.warning(f"Migration {version} is already pending")
                            return False
                        
                        cursor.execute(
                            f"""
                            UPDATE {self.TABLE_NAME} 
                            SET status = ?, checksum = ?, file_path = ?,
                                updated_at = ?, lock_version = lock_version + 1,
                                batch_id = ?
                            WHERE version = ? AND lock_version = ?
                            """,
                            (
                                MigrationStatus.PENDING.value,
                                checksum,
                                migration_path,
                                datetime.utcnow(),
                                batch_id,
                                version,
                                lock_version
                            )
                        )
                        
                        if cursor.rowcount == 0:
                            logger.error(f"Concurrent modification detected for version {version}")
                            return False
                        
                        old_values = {'status': status, 'checksum': None}
                        new_values = {'status': MigrationStatus.PENDING.value, 'checksum': checksum}
                        self.audit_logger.log_operation(
                            migration_id, version, 'migration_restart',
                            old_values, new_values, self._get_current_user(),
                            batch_id=batch_id
                        )
                    else:
                        cursor.execute(
                            f"""
                            INSERT INTO {self.TABLE_NAME} 
                            (version, migration_name, applied_at, applied_by, 
                             status, checksum, file_path, batch_id)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                version,
                                migration_name,
                                datetime.utcnow(),
                                self._get_current_user(),
                                MigrationStatus.PENDING.value,
                                checksum,
                                migration_path,
                                batch_id
                            )
                        )
                        
                        migration_id = cursor.lastrowid
                        
                        new_values = {
                            'version': version,
                            'status': MigrationStatus.PENDING.value,
                            'checksum': checksum
                        }
                        self.audit_logger.log_operation(
                            migration_id, version, 'migration_start',
                            {}, new_values, self._get_current_user(),
                            batch_id=batch_id
                        )
                    
                    logger.info(f"Recorded migration start: {version} ({migration_name})")
                    return True
        
        return self._execute_with_retry(operation)
    
    def record_migration_complete(self, version: str, duration_ms: int,
                                 metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Record migration completion with optimistic locking."""
        def operation():
            with self.concurrency.version_lock(version):
                with self.connection:
                    cursor = self.connection.cursor()
                    cursor.execute(
                        f"""
                        SELECT id, status, lock_version FROM {self.TABLE_NAME} 
                        WHERE version = ? AND status = ?
                        """,
                        (version, MigrationStatus.PENDING.value)
                    )
                    
                    record = cursor.fetchone()
                    if not record:
                        logger.error(f"No pending migration found for version {version}")
                        return False
                    
                    migration_id, old_status, lock_version = record
                    
                    metadata_str = json.dumps(metadata, default=str) if metadata else None
                    
                    cursor.execute(
                        f"""
                        UPDATE {self.TABLE_NAME} 
                        SET status = ?, duration_ms = ?, metadata = ?, 
                            updated_at = ?, lock_version = lock_version + 1
                        WHERE version = ? AND status = ? AND lock_version = ?
                        """,
                        (
                            MigrationStatus.APPLIED.value,
                            duration_ms,
                            metadata_str,
                            datetime.utcnow(),
                            version,
                            MigrationStatus.PENDING.value,
                            lock_version
                        )
                    )
                    
                    if cursor.rowcount == 0:
                        logger.error(f"Concurrent modification detected for version {version}")
                        return False
                    
                    old_values = {'status': old_status}
                    new_values = {
                        'status': MigrationStatus.APPLIED.value,
                        'duration_ms': duration_ms,
                        'metadata': metadata
                    }
                    self.audit_logger.log_operation(
                        migration_id, version, 'migration_complete',
                        old_values, new_values, self._get_current_user()
                    )
                    
                    logger.info(f"Recorded migration completion: {version} ({duration_ms}ms)")
                    return True
        
        return self._execute_with_retry(operation)
    
    def record_migration_failed(self, version: str, error_message: str,
                               duration_ms: Optional[int] = None) -> bool:
        """Record migration failure with optimistic locking."""
        def operation():
            with self.concurrency.version_lock(version):
                with self.connection:
                    cursor = self.connection.cursor()
                    cursor.execute(
                        f"""
                        SELECT id, status, lock_version FROM {self.TABLE_NAME} 
                        WHERE version = ? AND status = ?
                        """,
                        (version, MigrationStatus.PENDING.value)
                    )
                    
                    record = cursor.fetchone()
                    if not record:
                        logger.error(f"No pending migration found for version {version}")
                        return False
                    
                    migration_id, old_status, lock_version = record
                    
                    cursor.execute(
                        f"""
                        UPDATE {self.TABLE_NAME} 
                        SET status = ?, error_message = ?, duration_ms = ?, 
                            updated_at = ?, lock_version = lock_version + 1
                        WHERE version = ? AND status = ? AND lock_version = ?
                        """,
                        (
                            MigrationStatus.FAILED.value,
                            error_message[:500],
                            duration_ms,
                            datetime.utcnow(),
                            version,
                            MigrationStatus.PENDING.value,
                            lock_version
                        )
                    )
                    
                    if cursor.rowcount == 0:
                        logger.error(f"Concurrent modification detected for version {version}")
                        return False
                    
                    old_values = {'status': old_status}
                    new_values = {
                        'status': MigrationStatus.FAILED.value,
                        'error_message': error_message[:500],
                        'duration_ms': duration_ms
                    }
                    self.audit_logger.log_operation(
                        migration_id, version, 'migration_failed',
                        old_values, new_values, self._get_current_user()
                    )
                    
                    logger.error(f"Recorded migration failure: {version} - {error_message}")
                    return True
        
        return self._execute_with_retry(operation)
    
    def get_applied_migrations(self, include_failed: bool = False, 
                              page: int = 1, page_size: Optional[int] = None) -> Tuple[List[Dict[str, Any]], int]:
        """
        Get migrations with pagination for large datasets.
        
        Args:
            include_failed: Whether to include failed migrations
            page: Page number (1-indexed)
            page_size: Records per page
            
        Returns:
            Tuple of (migrations list, total count)
        """
        page_size = page_size or self.batch_size
        offset = (page - 1) * page_size
        
        try:
            cursor = self.connection.cursor()
            
            if include_failed:
                cursor.execute(
                    f"""
                    SELECT COUNT(*) FROM {self.TABLE_NAME} 
                    WHERE status IN (?, ?)
                    """,
                    (MigrationStatus.APPLIED.value, MigrationStatus.FAILED.value)
                )
            else:
                cursor.execute(
                    f"SELECT COUNT(*) FROM {self.TABLE_NAME} WHERE status = ?",
                    (MigrationStatus.APPLIED.value,)
                )
            
            total = cursor.fetchone()[0]
            
            if include_failed:
                cursor.execute(
                    f"""
                    SELECT 
                        id, version, migration_name, applied_at, applied_by,
                        status, checksum, file_path, duration_ms, rollback_duration_ms,
                        error_message, metadata, created_at, updated_at, lock_version,
                        batch_id, schema_version
                    FROM {self.TABLE_NAME} 
                    WHERE status IN (?, ?)
                    ORDER BY applied_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    (MigrationStatus.APPLIED.value, MigrationStatus.FAILED.value,
                     page_size, offset)
                )
            else:
                cursor.execute(
                    f"""
                    SELECT 
                        id, version, migration_name, applied_at, applied_by,
                        status, checksum, file_path, duration_ms, rollback_duration_ms,
                        error_message, metadata, created_at, updated_at, lock_version,
                        batch_id, schema_version
                    FROM {self.TABLE_NAME} 
                    WHERE status = ?
                    ORDER BY applied_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    (MigrationStatus.APPLIED.value, page_size, offset)
                )
            
            columns = [col[0] for col in cursor.description]
            rows = cursor.fetchall()
            
            migrations = []
            for row in rows:
                migration = dict(zip(columns, row))
                
                if migration.get('metadata'):
                    try:
                        migration['metadata'] = json.loads(migration['metadata'])
                    except json.JSONDecodeError:
                        migration['metadata'] = {}
                else:
                    migration['metadata'] = {}
                
                migrations.append(migration)
            
            return migrations, total
            
        except sqlite3.Error as e:
            logger.error(f"Failed to get applied migrations: {e}")
            return [], 0
    
    def validate_checksums(self, migration_files: Dict[str, str], 
                          search_paths: Optional[List[str]] = None) -> Tuple[bool, List[Dict[str, str]]]:
        """
        Validate migration scripts with file relocation support.
        """
        try:
            cursor = self.connection.cursor()
            
            cursor.execute(
                f"""
                SELECT version, checksum, file_path FROM {self.TABLE_NAME} 
                WHERE status = ? AND checksum IS NOT NULL
                """,
                (MigrationStatus.APPLIED.value,)
            )
            
            applied_migrations = cursor.fetchall()
            mismatches = []
            updated_paths = []
            
            for version, stored_checksum, stored_path in applied_migrations:
                file_path = migration_files.get(version)
                
                if not file_path or not os.path.exists(file_path):
                    if search_paths and stored_checksum:
                        found_path = self._resolve_file_path(version, stored_checksum, search_paths)
                        if found_path:
                            file_path = found_path
                            updated_paths.append((file_path, version))
                
                if not file_path or not os.path.exists(file_path):
                    mismatches.append({
                        'version': version,
                        'error': 'File not found',
                        'expected_checksum': stored_checksum,
                        'expected_path': stored_path
                    })
                    continue
                
                try:
                    current_checksum = self._calculate_checksum(file_path)
                    
                    if current_checksum != stored_checksum:
                        mismatches.append({
                            'version': version,
                            'expected_checksum': stored_checksum,
                            'actual_checksum': current_checksum,
                            'file_path': file_path,
                            'expected_path': stored_path
                        })
                        logger.error(
                            f"Checksum mismatch for migration {version}: "
                            f"expected {stored_checksum[:8]}..., "
                            f"got {current_checksum[:8]}..."
                        )
                    
                except Exception as e:
                    logger.error(f"Error validating checksum for {version}: {e}")
                    mismatches.append({
                        'version': version,
                        'error': str(e),
                        'file_path': file_path
                    })
            
            if updated_paths:
                with self.connection:
                    for file_path, version in updated_paths:
                        cursor.execute(
                            f"""
                            UPDATE {self.TABLE_NAME} 
                            SET file_path = ?
                            WHERE version = ?
                            """,
                            (file_path, version)
                        )
            
            is_valid = len(mismatches) == 0
            
            if is_valid:
                logger.info(f"Checksum validation passed for {len(applied_migrations)} migrations")
            else:
                logger.error(f"Checksum validation failed: {len(mismatches)} mismatches")
            
            return is_valid, mismatches
            
        except sqlite3.Error as e:
            logger.error(f"Failed to validate checksums: {e}")
            return False, [{'error': str(e)}]
    
    def get_migration_stats(self) -> Dict[str, Any]:
        """Get comprehensive migration statistics."""
        try:
            cursor = self.connection.cursor()
            
            cursor.execute(f"""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN status = ? THEN 1 ELSE 0 END) as applied,
                    SUM(CASE WHEN status = ? THEN 1 ELSE 0 END) as failed,
                    SUM(CASE WHEN status = ? THEN 1 ELSE 0 END) as rolled_back,
                    SUM(CASE WHEN status = ? THEN 1 ELSE 0 END) as pending,
                    AVG(duration_ms) as avg_duration,
                    MAX(applied_at) as last_migration,
                    COUNT(DISTINCT batch_id) as batch_count
                FROM {self.TABLE_NAME}
            """, (
                MigrationStatus.APPLIED.value,
                MigrationStatus.FAILED.value,
                MigrationStatus.ROLLED_BACK.value,
                MigrationStatus.PENDING.value
            ))
            
            row = cursor.fetchone()
            if row:
                total, applied, failed, rolled_back, pending, avg_duration, last_migration, batch_count = row
                
                return {
                    'total_migrations': total or 0,
                    'applied': applied or 0,
                    'failed': failed or 0,
                    'rolled_back': rolled_back or 0,
                    'pending': pending or 0,
                    'average_duration_ms': round(avg_duration or 0, 2),
                    'last_migration_at': last_migration,
                    'batch_count': batch_count or 0,
                    'success_rate': (applied / total * 100) if total and total > 0 else 0,
                    'table_schema_version': self._get_current_schema_version()
                }
            
            return {}
            
        except sqlite3.Error as e:
            logger.error(f"Failed to get migration stats: {e}")
            return {}
    
    def apply_data_retention_policy(self, custom_retention_days: Optional[int] = None) -> Dict[str, int]:
        """
        Apply comprehensive data retention policy - FIXED VERSION.
        
        Args:
            custom_retention_days: Optional override for retention days
            
        Returns:
            Dictionary with deletion counts
        """
        retention_days = custom_retention_days or self.data_retention_days
        
        def operation():
            with self.concurrency.global_lock():
                with self.connection:
                    cursor = self.connection.cursor()
                    results = {}
                    
                    cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
                    
                    cursor.execute(
                        f"""
                        SELECT id FROM {self.TABLE_NAME} 
                        WHERE applied_at < ? 
                        AND status IN (?, ?)
                        ORDER BY applied_at ASC
                        """,
                        (cutoff_date, MigrationStatus.APPLIED.value, MigrationStatus.ROLLED_BACK.value)
                    )
                    
                    old_migration_ids = [row[0] for row in cursor.fetchall()]
                    
                    audit_deleted = self.audit_logger.apply_retention_policy()
                    results['audit_records_deleted'] = audit_deleted
                    
                    if old_migration_ids:
                        placeholders = ','.join('?' * len(old_migration_ids))
                        cursor.execute(
                            f"DELETE FROM {self.TABLE_NAME} WHERE id IN ({placeholders})",
                            old_migration_ids
                        )
                        results['migration_records_deleted'] = cursor.rowcount
                    
                    cursor.execute(f"SELECT COUNT(*) FROM {self.TABLE_NAME}")
                    total_count = cursor.fetchone()[0]
                    
                    if total_count > self.max_records:
                        cursor.execute(
                            f"""
                            SELECT id FROM {self.TABLE_NAME} 
                            ORDER BY applied_at DESC 
                            LIMIT -1 OFFSET ?
                            """,
                            (self.max_records,)
                        )
                        
                        excess_ids = [row[0] for row in cursor.fetchall()]
                        
                        if excess_ids:
                            placeholders = ','.join('?' * len(excess_ids))
                            cursor.execute(
                                f"DELETE FROM {self.TABLE_NAME} WHERE id IN ({placeholders})",
                                excess_ids
                            )
                            results['max_records_enforced'] = cursor.rowcount
                    
                    self.concurrency.cleanup_locks()
                    
                    self.audit_logger._flush_pending_audits()
                    
                    logger.info(f"Applied retention policy: {results}")
                    return results
        
        return self._execute_with_retry(operation) or {}
    
    def get_audit_trail(self, migration_id: Optional[int] = None,
                       version: Optional[str] = None,
                       start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None,
                       page: int = 1, page_size: int = 50) -> Tuple[List[Dict[str, Any]], int]:
        """
        Get audit trail with pagination - FIXED VERSION.
        
        Returns:
            Tuple of (audit records, total count)
        """
        try:
            cursor = self.connection.cursor()
            
            query = f"SELECT * FROM {self.audit_logger.AUDIT_TABLE_NAME} WHERE 1=1"
            count_query = f"SELECT COUNT(*) FROM {self.audit_logger.AUDIT_TABLE_NAME} WHERE 1=1"
            params = []
            
            if migration_id is not None:
                query += " AND migration_id = ?"
                count_query += " AND migration_id = ?"
                params.append(migration_id)
            
            if version is not None:
                query += " AND version = ?"
                count_query += " AND version = ?"
                params.append(version)
            
            if start_date is not None:
                query += " AND performed_at >= ?"
                count_query += " AND performed_at >= ?"
                params.append(start_date)
            
            if end_date is not None:
                query += " AND performed_at <= ?"
                count_query += " AND performed_at <= ?"
                params.append(end_date)
            
            cursor.execute(count_query, params)
            total = cursor.fetchone()[0]
            
            offset = (page - 1) * page_size
            query += " ORDER BY performed_at DESC LIMIT ? OFFSET ?"
            params.extend([page_size, offset])
            
            cursor.execute(query, params)
            
            columns = [col[0] for col in cursor.description]
            rows = cursor.fetchall()
            
            audit_records = []
            for row in rows:
                record = dict(zip(columns, row))
                
                for field in ['old_metadata', 'new_metadata']:
                    if record.get(field):
                        try:
                            record[field] = json.loads(record[field])
                        except json.JSONDecodeError:
                            record[field] = {}
                
                audit_records.append(record)
            
            return audit_records, total
            
        except sqlite3.Error as e:
            logger.error(f"Failed to get audit trail: {e}")
            return [], 0
    
    def get_batch_statistics(self) -> List[Dict[str, Any]]:
        """Get migration statistics grouped by batch."""
        try:
            cursor = self.connection.cursor()
            
            cursor.execute(f"""
                SELECT 
                    batch_id,
                    COUNT(*) as migration_count,
                    SUM(CASE WHEN status = ? THEN 1 ELSE 0 END) as applied_count,
                    SUM(CASE WHEN status = ? THEN 1 ELSE 0 END) as failed_count,
                    MIN(applied_at) as batch_start,
                    MAX(applied_at) as batch_end,
                    AVG(duration_ms) as avg_duration,
                    SUM(duration_ms) as total_duration
                FROM {self.TABLE_NAME}
                WHERE batch_id IS NOT NULL
                GROUP BY batch_id
                ORDER BY batch_start DESC
            """, (MigrationStatus.APPLIED.value, MigrationStatus.FAILED.value))
            
            columns = [col[0] for col in cursor.description]
            rows = cursor.fetchall()
            
            batches = []
            for row in rows:
                batch = dict(zip(columns, row))
                
                if batch['batch_start'] and batch['batch_end']:
                    try:
                        start = datetime.fromisoformat(batch['batch_start'].replace('Z', '+00:00'))
                        end = datetime.fromisoformat(batch['batch_end'].replace('Z', '+00:00'))
                        batch['batch_duration_seconds'] = (end - start).total_seconds()
                    except (ValueError, AttributeError):
                        batch['batch_duration_seconds'] = None
                
                batch['success_rate'] = (
                    (batch['applied_count'] / batch['migration_count'] * 100)
                    if batch['migration_count'] > 0 else 0
                )
                
                batches.append(batch)
            
            return batches
            
        except sqlite3.Error as e:
            logger.error(f"Failed to get batch statistics: {e}")
            return []
    
    def reset_migration_table(self, preserve_audit: bool = False) -> bool:
        """
        Reset migration table with option to preserve audit trail.
        
        Args:
            preserve_audit: Whether to preserve audit records
            
        Returns:
            True if reset successfully
        """
        def operation():
            with self.concurrency.global_lock():
                with self.connection:
                    cursor = self.connection.cursor()
                    
                    if not preserve_audit:
                        cursor.execute(f"DROP TABLE IF EXISTS {self.audit_logger.AUDIT_TABLE_NAME}")
                    
                    cursor.execute(f"DROP TABLE IF EXISTS {self.TABLE_NAME}")
                    
                    self._ensure_table_exists()
                    if not preserve_audit:
                        self.audit_logger._ensure_audit_table_exists()
                    
                    logger.warning("Migration table reset completed")
                    return True
        
        return self._execute_with_retry(operation, max_retries=1)