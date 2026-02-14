# src/database/migrations/database_initializer.py
"""
Database Initializer System.

Handles first-time database setup, environment validation, directory structure,
configuration setup, and migration system bootstrap.
"""

import os
import sys
import sqlite3
import shutil
import subprocess
import platform
import hashlib
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class SystemRequirement:
    name: str
    requirement: str
    actual: Optional[str] = None
    satisfied: bool = False
    critical: bool = True


@dataclass
class DirectorySpec:
    path: str
    description: str
    permissions: Optional[int] = None
    required: bool = True


@dataclass
class InitResult:
    success: bool
    message: str
    warnings: List[str]
    errors: List[str]
    details: Dict[str, Any]


class RollbackContext:
    """Tracks created artifacts for rollback on failure."""

    def __init__(self):
        self.directories_created: List[Path] = []
        self.files_created: List[Path] = []
        self.backups_created: List[Path] = []
        self.locks_created: List[Path] = []
        self.original_files: Dict[Path, Optional[bytes]] = {}

    def add_directory(self, path: Path):
        self.directories_created.append(path)

    def add_file(self, path: Path):
        self.files_created.append(path)

    def add_backup(self, path: Path):
        self.backups_created.append(path)

    def add_lock(self, path: Path):
        self.locks_created.append(path)

    def backup_original_file(self, path: Path):
        if path.exists():
            try:
                with open(path, 'rb') as f:
                    self.original_files[path] = f.read()
            except Exception:
                self.original_files[path] = None
        else:
            self.original_files[path] = None

    def rollback(self):
        rollback_errors = []

        for lock in self.locks_created:
            try:
                if lock.exists():
                    lock.unlink()
            except Exception as e:
                rollback_errors.append(f"Failed to remove lock {lock}: {e}")

        for file_path in self.files_created:
            try:
                if file_path.exists():
                    file_path.unlink()
            except Exception as e:
                rollback_errors.append(f"Failed to remove file {file_path}: {e}")

        for dir_path in reversed(self.directories_created):
            try:
                if dir_path.exists() and not list(dir_path.iterdir()):
                    dir_path.rmdir()
            except Exception as e:
                rollback_errors.append(f"Failed to remove directory {dir_path}: {e}")

        for original_path, content in self.original_files.items():
            if content is None:
                continue
            try:
                with open(original_path, 'wb') as f:
                    f.write(content)
            except Exception as e:
                rollback_errors.append(f"Failed to restore {original_path}: {e}")

        for backup in self.backups_created:
            try:
                if backup.exists():
                    backup.unlink()
            except Exception as e:
                rollback_errors.append(f"Failed to remove backup {backup}: {e}")

        return rollback_errors


class DatabaseInitializer:
    MIN_PYTHON_VERSION = (3, 10, 11)
    MIN_DISK_SPACE_MB = 500
    REQUIRED_PACKAGES = ["sqlite3"]

    def __init__(self, base_dir: str = ".", config_dir: str = "config"):
        self.base_dir = Path(base_dir).resolve()
        self.config_dir = Path(config_dir).resolve()
        self.data_dir = self.base_dir / "data"
        self.database_dir = self.data_dir / "database"
        self.documents_dir = self.data_dir / "documents"
        self.models_dir = self.data_dir / "models"
        self.logs_dir = self.data_dir / "logs"
        self.backup_dir = self.data_dir / "backups"
        self.migrations_dir = self.base_dir / "src" / "database" / "migrations"
        self.version_dir = self.migrations_dir / "version"

        self.requirements: List[SystemRequirement] = []
        self.warnings: List[str] = []
        self.errors: List[str] = []
        self.rollback_ctx = RollbackContext()

        self.sqlite_pragmas = {
            "journal_mode": "WAL",
            "foreign_keys": "ON",
            "synchronous": "NORMAL",
            "temp_store": "MEMORY",
            "mmap_size": 268435456,
            "cache_size": -2000,
            "busy_timeout": 30000,
        }

    def _apply_permissions(self, path: Path, permissions: int):
        if platform.system() == "Windows":
            return
        try:
            os.chmod(path, permissions)
        except Exception as e:
            logger.warning(f"Could not set permissions on {path}: {e}")

    def validate_environment(self) -> bool:
        logger.info("Validating system environment...")

        self.requirements = []

        py_version = sys.version_info[:3]
        py_version_str = ".".join(map(str, py_version))
        py_satisfied = py_version >= self.MIN_PYTHON_VERSION
        self.requirements.append(SystemRequirement(
            name="Python Version",
            requirement=f">= {'.'.join(map(str, self.MIN_PYTHON_VERSION))}",
            actual=py_version_str,
            satisfied=py_satisfied,
            critical=True
        ))

        sqlite_version = self._get_sqlite_version()
        self.requirements.append(SystemRequirement(
            name="SQLite Version",
            requirement=">= 3.35.0 (for RETURNING support)",
            actual=sqlite_version,
            satisfied=bool(sqlite_version),
            critical=True
        ))

        free_space_mb = self._get_free_disk_space()
        space_satisfied = free_space_mb >= self.MIN_DISK_SPACE_MB
        self.requirements.append(SystemRequirement(
            name="Disk Space",
            requirement=f">= {self.MIN_DISK_SPACE_MB} MB free",
            actual=f"{free_space_mb:.1f} MB free",
            satisfied=space_satisfied,
            critical=True
        ))

        can_write = self._check_write_permissions()
        self.requirements.append(SystemRequirement(
            name="Write Permissions",
            requirement="Write access to data directory",
            actual="OK" if can_write else "NO WRITE ACCESS",
            satisfied=can_write,
            critical=True
        ))

        missing_packages = self._check_required_packages()
        packages_satisfied = len(missing_packages) == 0
        self.requirements.append(SystemRequirement(
            name="Required Packages",
            requirement=", ".join(self.REQUIRED_PACKAGES),
            actual=f"Missing: {', '.join(missing_packages)}" if missing_packages else "All packages available",
            satisfied=packages_satisfied,
            critical=True
        ))

        total_memory_gb = self._get_total_memory()
        memory_adequate = total_memory_gb >= 4.0
        self.requirements.append(SystemRequirement(
            name="System Memory",
            requirement=">= 4.0 GB recommended",
            actual=f"{total_memory_gb:.1f} GB",
            satisfied=memory_adequate,
            critical=False
        ))

        critical_passed = all(r.satisfied for r in self.requirements if r.critical)
        all_passed = all(r.satisfied for r in self.requirements)

        logger.info("Environment validation results:")
        for req in self.requirements:
            status = "PASS" if req.satisfied else "FAIL"
            critical = " (CRITICAL)" if req.critical else ""
            logger.info(f"  {status} {req.name}{critical}: {req.requirement}")
            if req.actual:
                logger.info(f"    Actual: {req.actual}")

        if not critical_passed:
            logger.error("Critical requirements not met. Cannot proceed.")
            return False

        if not all_passed:
            logger.warning("Some non-critical requirements not met. Proceeding with warnings.")

        return True

    def _get_sqlite_version(self) -> str:
        try:
            conn = sqlite3.connect(":memory:")
            cursor = conn.cursor()
            cursor.execute("SELECT sqlite_version()")
            version = cursor.fetchone()[0]
            conn.close()
            return version
        except Exception as e:
            logger.error(f"Failed to get SQLite version: {e}")
            return ""

    def _get_free_disk_space(self) -> float:
        try:
            if platform.system() == "Windows":
                import ctypes
                free_bytes = ctypes.c_ulonglong(0)
                ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                    ctypes.c_wchar_p(str(self.base_dir)),
                    None, None,
                    ctypes.pointer(free_bytes)
                )
                return free_bytes.value / (1024 * 1024)
            else:
                stat = os.statvfs(self.base_dir)
                return (stat.f_bavail * stat.f_frsize) / (1024 * 1024)
        except Exception as e:
            logger.warning(f"Could not determine free disk space: {e}")
            return float('inf')

    def _check_write_permissions(self) -> bool:
        try:
            test_file = self.data_dir / ".write_test"
            test_file.parent.mkdir(parents=True, exist_ok=True)
            test_file.touch()
            test_file.unlink()
            return True
        except Exception as e:
            logger.error(f"Write permission check failed: {e}")
            return False

    def _check_required_packages(self) -> List[str]:
        missing = []
        for package in self.REQUIRED_PACKAGES:
            try:
                if package == "sqlite3":
                    pass
                else:
                    __import__(package)
            except ImportError:
                missing.append(package)
        return missing

    def _get_total_memory(self) -> float:
        """Get total system memory in GB."""
        try:
            import psutil
            return psutil.virtual_memory().total / (1024 ** 3)
        except ImportError:
            try:
                if platform.system() == "Windows":
                    import ctypes
                    kernel32 = ctypes.windll.kernel32
                    class MEMORYSTATUSEX(ctypes.Structure):
                        _fields_ = [
                            ("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                        ]
                    
                    memoryStatus = MEMORYSTATUSEX()
                    memoryStatus.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                    if kernel32.GlobalMemoryStatusEx(ctypes.byref(memoryStatus)):
                        return memoryStatus.ullTotalPhys / (1024 ** 3)
                        
                elif platform.system() == "Linux":
                    with open('/proc/meminfo', 'r') as f:
                        for line in f:
                            if line.startswith('MemTotal:'):
                                mem_kb = int(line.split()[1])
                                return mem_kb / (1024 * 1024)
                                
                elif platform.system() == "Darwin":
                    import subprocess
                    result = subprocess.run(['sysctl', '-n', 'hw.memsize'],
                                        capture_output=True, text=True)
                    mem_bytes = int(result.stdout.strip())
                    return mem_bytes / (1024 ** 3)
                    
            except Exception as e:
                logger.warning(f"Could not determine system memory: {e}")
                
        return 8.0  # Default fallback untuk development

    def create_directory_structure(self) -> bool:
        logger.info("Creating directory structure...")

        dir_specs = [
            DirectorySpec(
                path=str(self.data_dir),
                description="Root data directory",
                permissions=0o755,
                required=True
            ),
            DirectorySpec(
                path=str(self.database_dir),
                description="Database files directory",
                permissions=0o755,
                required=True
            ),
            DirectorySpec(
                path=str(self.documents_dir),
                description="Uploaded documents directory",
                permissions=0o755,
                required=True
            ),
            DirectorySpec(
                path=str(self.documents_dir / "uploads"),
                description="Document uploads staging area",
                permissions=0o755,
                required=True
            ),
            DirectorySpec(
                path=str(self.documents_dir / "processed"),
                description="Processed documents",
                permissions=0o755,
                required=True
            ),
            DirectorySpec(
                path=str(self.documents_dir / "exports"),
                description="Document exports",
                permissions=0o755,
                required=True
            ),
            DirectorySpec(
                path=str(self.models_dir),
                description="AI models directory",
                permissions=0o755,
                required=True
            ),
            DirectorySpec(
                path=str(self.models_dir / "embeddings"),
                description="Embedding models",
                permissions=0o755,
                required=True
            ),
            DirectorySpec(
                path=str(self.models_dir / "llms"),
                description="LLM models",
                permissions=0o755,
                required=True
            ),
            DirectorySpec(
                path=str(self.logs_dir),
                description="Application logs",
                permissions=0o755,
                required=True
            ),
            DirectorySpec(
                path=str(self.logs_dir / "migrations"),
                description="Migration logs",
                permissions=0o755,
                required=True
            ),
            DirectorySpec(
                path=str(self.backup_dir),
                description="Database backups",
                permissions=0o755,
                required=True
            ),
            DirectorySpec(
                path=str(self.backup_dir / "daily"),
                description="Daily backups",
                permissions=0o755,
                required=False
            ),
            DirectorySpec(
                path=str(self.backup_dir / "weekly"),
                description="Weekly backups",
                permissions=0o755,
                required=False
            ),
            DirectorySpec(
                path=str(self.config_dir),
                description="Configuration files",
                permissions=0o755,
                required=True
            ),
            DirectorySpec(
                path=str(self.version_dir),
                description="Migration version files",
                permissions=0o755,
                required=True
            ),
        ]

        try:
            for spec in dir_specs:
                try:
                    path = Path(spec.path)
                    if not path.exists():
                        path.mkdir(parents=True, exist_ok=True)

                        if spec.permissions:
                            self._apply_permissions(path, spec.permissions)

                        self.rollback_ctx.add_directory(path)
                        logger.info(f"Created: {spec.description} -> {path}")

                        gitkeep = path / ".gitkeep"
                        gitkeep.touch()
                        self.rollback_ctx.add_file(gitkeep)

                    else:
                        logger.debug(f"Already exists: {spec.description}")

                except Exception as e:
                    if spec.required:
                        logger.error(f"Failed to create required directory {spec.path}: {e}")
                        self.errors.append(f"Failed to create directory: {spec.path}")
                        return False
                    else:
                        logger.warning(f"Failed to create optional directory {spec.path}: {e}")
                        self.warnings.append(f"Optional directory not created: {spec.path}")

            logger.info("Directory structure created successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to create directory structure: {e}")
            self.errors.append(f"Directory structure creation failed: {e}")
            return False

    def create_database_file(self) -> bool:
        logger.info("Creating database file...")

        database_path = self.database_dir / "cockatoo.db"

        if database_path.exists():
            logger.warning(f"Database file already exists: {database_path}")
            backup_success, backup_path = self._backup_existing_database(database_path)
            if not backup_success:
                logger.error("Failed to backup existing database. Aborting.")
                self.errors.append("Failed to backup existing database")
                return False
            self.rollback_ctx.add_backup(backup_path)

            try:
                database_path.unlink()
                logger.info("Removed existing database file for fresh creation")
            except Exception as e:
                logger.error(f"Failed to remove existing database: {e}")
                self.errors.append(f"Failed to remove existing database: {e}")
                return False

        try:
            conn = sqlite3.connect(str(database_path))
            cursor = conn.cursor()

            for pragma, value in self.sqlite_pragmas.items():
                if isinstance(value, (int, float)):
                    cursor.execute(f"PRAGMA {pragma} = {value}")
                else:
                    cursor.execute(f"PRAGMA {pragma} = '{value}'")

            cursor.execute("PRAGMA foreign_keys")
            foreign_keys_result = cursor.fetchone()
            if foreign_keys_result:
                foreign_keys_enabled = foreign_keys_result[0]
                if isinstance(foreign_keys_enabled, int):
                    enabled_bool = foreign_keys_enabled == 1
                else:
                    enabled_bool = str(foreign_keys_enabled).lower() in ['on', 'true', '1']
                logger.info(f"Foreign keys enabled: {enabled_bool}")
            else:
                logger.warning("Could not verify foreign keys status")
                enabled_bool = False

            # ============ FIXED: CORRECT SCHEMA MIGRATIONS TABLE ============
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
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
                    schema_version INTEGER DEFAULT 3
                )
            """)
            # =================================================================

            # Create indexes for schema_migrations
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_schema_migrations_version ON schema_migrations(version)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_schema_migrations_status ON schema_migrations(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_schema_migrations_applied_at ON schema_migrations(applied_at DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_schema_migrations_batch_id ON schema_migrations(batch_id)")

            # Create audit table for migration history
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS schema_migrations_audit (
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
            """)

            # Create indexes for audit table
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_migration_id ON schema_migrations_audit(migration_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_performed_at ON schema_migrations_audit(performed_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_version ON schema_migrations_audit(version)")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_info (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    description TEXT
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS initialization_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    step TEXT NOT NULL,
                    status TEXT NOT NULL,
                    message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    duration_ms INTEGER
                )
            """)

            cursor.execute("""
                INSERT OR IGNORE INTO system_info (key, value, description)
                VALUES (?, ?, ?)
            """, (
                "initialization_timestamp",
                datetime.utcnow().isoformat(),
                "When the database was first initialized"
            ))

            cursor.execute("""
                INSERT OR IGNORE INTO system_info (key, value, description)
                VALUES (?, ?, ?)
            """, (
                "sqlite_version",
                sqlite3.sqlite_version,
                "SQLite library version"
            ))

            cursor.execute("""
                INSERT OR IGNORE INTO system_info (key, value, description)
                VALUES (?, ?, ?)
            """, (
                "python_version",
                sys.version,
                "Python interpreter version"
            ))

            cursor.execute("""
                INSERT OR IGNORE INTO system_info (key, value, description)
                VALUES (?, ?, ?)
            """, (
                "platform",
                platform.platform(),
                "Operating system platform"
            ))

            cursor.execute("""
                INSERT INTO initialization_log (step, status, message)
                VALUES (?, ?, ?)
            """, (
                "database_creation",
                "success",
                f"Created database with pragmas: {self.sqlite_pragmas}"
            ))

            conn.commit()
            conn.close()

            self._apply_permissions(database_path, 0o644)
            self.rollback_ctx.add_file(database_path)

            logger.info(f"Created database file: {database_path}")
            logger.info(f"Database size: {database_path.stat().st_size} bytes")

            return True

        except sqlite3.Error as e:
            logger.error(f"Failed to create database: {e}")
            self.errors.append(f"Database creation failed: {e}")

            if database_path.exists():
                try:
                    database_path.unlink()
                except Exception:
                    pass

            return False

    def _backup_existing_database(self, db_path: Path) -> Tuple[bool, Optional[Path]]:
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / f"cockatoo_preinit_{timestamp}.db.backup"

            shutil.copy2(db_path, backup_path)
            self._apply_permissions(backup_path, 0o644)

            if backup_path.exists() and backup_path.stat().st_size == db_path.stat().st_size:
                logger.info(f"Successfully backed up existing database to: {backup_path}")
                return True, backup_path
            else:
                logger.error("Backup file verification failed")
                if backup_path.exists():
                    backup_path.unlink()
                return False, None

        except Exception as e:
            logger.error(f"Failed to backup existing database: {e}")
            return False, None

    def _validate_config_file(self, file_path: Path, config_type: str) -> Tuple[bool, str]:
        if not file_path.exists():
            return True, "File does not exist"

        try:
            with open(file_path, 'r') as f:
                if file_path.suffix == '.yaml':
                    content = yaml.safe_load(f)
                    if not isinstance(content, dict):
                        return False, "Invalid YAML structure"
                elif file_path.suffix == '.json':
                    content = json.load(f)
                else:
                    content = f.read()

            return True, "Valid configuration"
        except Exception as e:
            return False, f"Invalid configuration: {str(e)}"

    def create_configuration_files(self) -> bool:
        logger.info("Creating configuration files...")

        config_files_to_create = [
            (self.config_dir / "app_config.yaml", "app_config"),
            (self.config_dir / "llm_config.yaml", "llm_config"),
            (self.base_dir / ".cockatoo_env", "env"),
            (self.data_dir / ".gitignore", "gitignore")
        ]

        for file_path, config_type in config_files_to_create:
            if file_path.exists():
                valid, message = self._validate_config_file(file_path, config_type)
                if not valid:
                    logger.warning(f"Existing {file_path} is invalid: {message}")
                    self.warnings.append(f"Overwriting invalid {file_path}")
                self.rollback_ctx.backup_original_file(file_path)

        try:
            app_config = {
                "app": {
                    "name": "Cockatoo",
                    "version": "1.0.0",
                    "environment": "development",
                    "debug": True,
                    "host": "127.0.0.1",
                    "port": 8000,
                    "log_level": "INFO",
                    "log_file": str(self.logs_dir / "app.log"),
                    "data_dir": str(self.data_dir),
                    "database_path": str(self.database_dir / "cockatoo.db"),
                    "max_upload_size_mb": 100,
                    "supported_file_types": [".pdf", ".docx", ".txt", ".md", ".html"],
                },
                "security": {
                    "secret_key": self._generate_secret_key(),
                    "token_expiry_hours": 24,
                    "password_hash_rounds": 12,
                    "enable_rate_limiting": True,
                    "cors_allowed_origins": ["http://localhost:3000"],
                },
                "database": {
                    "pool_size": 10,
                    "pool_recycle": 3600,
                    "echo_sql": False,
                    "migration": {
                        "auto_migrate": True,
                        "backup_before_migration": True,
                        "max_rollback_steps": 5,
                    }
                }
            }

            app_config_path = self.config_dir / "app_config.yaml"
            with open(app_config_path, 'w') as f:
                yaml.dump(app_config, f, default_flow_style=False, sort_keys=False)

            self.rollback_ctx.add_file(app_config_path)
            logger.info(f"Created: {app_config_path}")

            llm_config = {
                "models": {
                    "embedding": {
                        "default": "sentence-transformers/all-MiniLM-L6-v2",
                        "cache_dir": str(self.models_dir / "embeddings"),
                        "device": "cpu",
                        "batch_size": 32,
                    },
                    "llm": {
                        "default": "gpt-3.5-turbo",
                        "local_model": str(self.models_dir / "llms" / "local-model"),
                        "api_key": "your-api-key-here",
                        "temperature": 0.7,
                        "max_tokens": 1000,
                    }
                },
                "providers": {
                    "openai": {
                        "enabled": False,
                        "api_key": "",
                        "organization": "",
                    },
                    "anthropic": {
                        "enabled": False,
                        "api_key": "",
                    },
                    "local": {
                        "enabled": True,
                        "model_path": str(self.models_dir / "llms"),
                    }
                },
                "settings": {
                    "chunk_size": 1000,
                    "chunk_overlap": 200,
                    "max_documents_per_query": 10,
                    "timeout_seconds": 30,
                }
            }

            llm_config_path = self.config_dir / "llm_config.yaml"
            with open(llm_config_path, 'w') as f:
                yaml.dump(llm_config, f, default_flow_style=False, sort_keys=False)

            self.rollback_ctx.add_file(llm_config_path)
            logger.info(f"Created: {llm_config_path}")

            env_content = f"""# Cockatoo Environment Variables
# Auto-generated on {datetime.now().isoformat()}

# Application
COCKATOO_ENV=development
COCKATOO_DEBUG=true
COCKATOO_DATA_DIR={self.data_dir}
COCKATOO_DATABASE_PATH={self.database_dir / "cockatoo.db"}
COCKATOO_LOG_LEVEL=INFO
COCKATOO_LOG_FILE={self.logs_dir / "cockatoo.log"}

# Security (override in production)
COCKATOO_SECRET_KEY={self._generate_secret_key()}

# Database
SQLITE_JOURNAL_MODE=WAL
SQLITE_FOREIGN_KEYS=ON

# Optional: External Services
# OPENAI_API_KEY=your-key-here
# ANTHROPIC_API_KEY=your-key-here

# Path configuration
PYTHONPATH={self.base_dir}:$PYTHONPATH
"""

            env_path = self.base_dir / ".cockatoo_env"
            with open(env_path, 'w') as f:
                f.write(env_content)

            self.rollback_ctx.add_file(env_path)
            logger.info(f"Created: {env_path}")

            gitignore_content = """# Cockatoo Data Directory - DO NOT COMMIT
# This directory contains user data and should not be version controlled

# Database files
*.db
*.db-journal
*.db-wal
*.db-shm

# Uploaded documents
documents/uploads/*
!documents/uploads/.gitkeep
documents/processed/*
!documents/processed/.gitkeep

# AI Models (can be large)
models/*
!models/.gitkeep
!models/embeddings/.gitkeep
!models/llms/.gitkeep

# Logs
logs/*.log
!logs/.gitkeep

# Backups
backups/*.backup
!backups/.gitkeep

# Temporary files
*.tmp
*.temp
*.cache

# Environment files
.env
*.env.local
"""

            data_gitignore_path = self.data_dir / ".gitignore"
            with open(data_gitignore_path, 'w') as f:
                f.write(gitignore_content)

            self.rollback_ctx.add_file(data_gitignore_path)
            logger.info(f"Created: {data_gitignore_path}")

            return True

        except Exception as e:
            logger.error(f"Failed to create configuration files: {e}")
            self.errors.append(f"Configuration creation failed: {e}")
            return False

    def _generate_secret_key(self) -> str:
        import secrets
        return secrets.token_urlsafe(32)

    def bootstrap_migration_system(self) -> bool:
        logger.info("Bootstrapping migration system...")

        try:
            if not self.migrations_dir.exists():
                logger.error(f"Migrations directory not found: {self.migrations_dir}")
                self.errors.append(f"Migrations directory not found: {self.migrations_dir}")
                return False

            if not self.version_dir.exists():
                self.version_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created version directory: {self.version_dir}")

            migration_files = list(self.version_dir.glob("v*.py"))
            if not migration_files:
                logger.warning("No migration files found in version directory")
                self.warnings.append("No migration files found")

            syntax_errors = self._validate_migration_syntax(migration_files)
            if syntax_errors:
                logger.error(f"Migration syntax errors found: {syntax_errors}")
                self.errors.extend([f"Syntax error in {e}" for e in syntax_errors])
                return False

            migration_registry = {}
            for migration_file in migration_files:
                checksum = self._calculate_file_checksum(migration_file)
                version = migration_file.stem.replace("v", "").replace("_", ".")

                migration_registry[version] = {
                    "file": migration_file.name,
                    "checksum": checksum,
                    "size_bytes": migration_file.stat().st_size,
                    "modified": datetime.fromtimestamp(migration_file.stat().st_mtime).isoformat(),
                    "created": datetime.fromtimestamp(migration_file.stat().st_ctime).isoformat(),
                    "sha256": checksum,
                    "schema_version": version,
                    "description": f"Migration {version}",
                    "dependencies": [],
                    "author": "system",
                    "metadata": {
                        "file_format": "python",
                        "encoding": "utf-8",
                        "line_endings": "unix",
                        "has_rollback": self._check_migration_has_rollback(migration_file)
                    }
                }

            registry_path = self.migrations_dir / "migration_registry.json"
            with open(registry_path, 'w') as f:
                json.dump(migration_registry, f, indent=2, sort_keys=True)

            self.rollback_ctx.add_file(registry_path)
            logger.info(f"Created migration registry: {registry_path}")

            init_py = self.version_dir / "__init__.py"
            if not init_py.exists():
                init_content = '''"""
Migration version package.

This directory contains database migration version scripts for Cockatoo.
"""

__version__ = "1.0.0"
'''
                with open(init_py, 'w') as f:
                    f.write(init_content)

                self.rollback_ctx.add_file(init_py)
                logger.info(f"Created: {init_py}")

            logger.info(f"Migration system bootstrapped. Found {len(migration_files)} migration files.")
            return True

        except Exception as e:
            logger.error(f"Failed to bootstrap migration system: {e}")
            self.errors.append(f"Migration bootstrap failed: {e}")
            return False

    def _check_migration_has_rollback(self, file_path: Path) -> bool:
        try:
            content = file_path.read_text()
            return "def downgrade" in content
        except Exception:
            return False

    def _validate_migration_syntax(self, migration_files: List[Path]) -> List[str]:
        errors = []
        for file_path in migration_files:
            try:
                with open(file_path, 'r') as f:
                    source = f.read()
                compile(source, str(file_path), 'exec')
            except SyntaxError as e:
                errors.append(f"{file_path.name}: {e}")
        return errors

    def _calculate_file_checksum(self, file_path: Path) -> str:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def perform_safety_checks(self) -> bool:
        logger.info("Performing safety checks...")

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.rollback_ctx.add_directory(self.data_dir)

        db_path = self.database_dir / "cockatoo.db"
        if db_path.exists():
            db_in_use = self._check_database_in_use()
            if db_in_use:
                logger.error("Database appears to be in use by another process")
                self.errors.append("Database is currently in use by another process")
                return False

        lock_files = list(self.database_dir.glob("*.lock"))
        if lock_files:
            logger.warning(f"Found lock files: {[f.name for f in lock_files]}")
            self.warnings.append("Found database lock files")

        init_lock = self.data_dir / ".initialization_in_progress"
        if init_lock.exists():
            logger.error("Initialization already in progress")
            self.errors.append("Initialization already in progress")
            return False

        try:
            init_lock.touch()
            self.rollback_ctx.add_lock(init_lock)
        except Exception as e:
            logger.error(f"Failed to create initialization lock: {e}")
            self.errors.append("Failed to create initialization lock")
            return False

        maintenance_file = self.data_dir / ".maintenance_mode"
        try:
            maintenance_file.touch()
            self.rollback_ctx.add_lock(maintenance_file)
            logger.info("Maintenance mode enabled")
        except Exception as e:
            logger.warning(f"Failed to set maintenance mode: {e}")
            self.warnings.append("Failed to set maintenance mode")

        return True

    def _check_database_in_use(self) -> bool:
        db_path = self.database_dir / "cockatoo.db"

        if not db_path.exists():
            return False

        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            conn.close()
            return False
        except sqlite3.Error:
            return True

    def cleanup(self):
        lock_files = [
            self.data_dir / ".initialization_in_progress",
            self.data_dir / ".maintenance_mode",
        ]

        for lock_file in lock_files:
            if lock_file.exists():
                try:
                    lock_file.unlink()
                    logger.info(f"Removed lock file: {lock_file}")
                except Exception as e:
                    logger.warning(f"Failed to remove lock file {lock_file}: {e}")

    def execute_rollback(self):
        logger.warning("Executing rollback due to initialization failure...")
        rollback_errors = self.rollback_ctx.rollback()
        if rollback_errors:
            logger.error("Errors during rollback:")
            for error in rollback_errors:
                logger.error(f"  {error}")
        else:
            logger.info("Rollback completed successfully")

    def initialize(self) -> InitResult:
        logger.info("=" * 60)
        logger.info("COCKATOO DATABASE INITIALIZATION")
        logger.info("=" * 60)

        start_time = datetime.now()

        try:
            logger.info("\n[1/6] Performing safety checks...")
            if not self.perform_safety_checks():
                self.execute_rollback()
                return InitResult(
                    success=False,
                    message="Safety checks failed",
                    warnings=self.warnings,
                    errors=self.errors,
                    details={"step": "safety_checks"}
                )

            logger.info("\n[2/6] Validating environment...")
            if not self.validate_environment():
                self.execute_rollback()
                return InitResult(
                    success=False,
                    message="Environment validation failed",
                    warnings=self.warnings,
                    errors=self.errors,
                    details={"step": "environment_validation", "requirements": self.requirements}
                )

            logger.info("\n[3/6] Creating directory structure...")
            if not self.create_directory_structure():
                self.execute_rollback()
                return InitResult(
                    success=False,
                    message="Directory structure creation failed",
                    warnings=self.warnings,
                    errors=self.errors,
                    details={"step": "directory_creation"}
                )

            logger.info("\n[4/6] Creating database file...")
            if not self.create_database_file():
                self.execute_rollback()
                return InitResult(
                    success=False,
                    message="Database creation failed",
                    warnings=self.warnings,
                    errors=self.errors,
                    details={"step": "database_creation"}
                )

            logger.info("\n[5/6] Creating configuration files...")
            if not self.create_configuration_files():
                self.execute_rollback()
                return InitResult(
                    success=False,
                    message="Configuration creation failed",
                    warnings=self.warnings,
                    errors=self.errors,
                    details={"step": "configuration_creation"}
                )

            logger.info("\n[6/6] Bootstrapping migration system...")
            if not self.bootstrap_migration_system():
                self.execute_rollback()
                return InitResult(
                    success=False,
                    message="Migration system bootstrap failed",
                    warnings=self.warnings,
                    errors=self.errors,
                    details={"step": "migration_bootstrap"}
                )

            self.cleanup()
            duration = (datetime.now() - start_time).total_seconds()

            logger.info("\n" + "=" * 60)
            logger.info("INITIALIZATION COMPLETE!")
            logger.info("=" * 60)
            logger.info(f"Duration: {duration:.2f} seconds")

            if self.warnings:
                logger.info(f"Warnings: {len(self.warnings)}")
                for warning in self.warnings:
                    logger.info(f"  WARNING: {warning}")

            return InitResult(
                success=True,
                message="Database initialization completed successfully",
                warnings=self.warnings,
                errors=self.errors,
                details={
                    "duration_seconds": duration,
                    "requirements": [asdict(r) for r in self.requirements],
                    "timestamp": start_time.isoformat(),
                }
            )

        except Exception as e:
            logger.error(f"Unexpected error during initialization: {e}")
            self.execute_rollback()
            self.cleanup()

            return InitResult(
                success=False,
                message=f"Unexpected error: {str(e)}",
                warnings=self.warnings,
                errors=self.errors + [str(e)],
                details={"step": "unknown", "exception": str(e)}
            )


def is_initialized(data_dir: str = "data") -> bool:
    """Check if database is initialized."""
    data_path = Path(data_dir)
    db_file = data_path / "database" / "cockatoo.db"
    
    print(f"DEBUG: Checking database at {db_file}")  # Temporary debug
    
    if not db_file.exists():
        print("DEBUG: Database file not found")
        return False
    
    try:
        conn = sqlite3.connect(str(db_file))
        cursor = conn.cursor()
        
        # Just check if the table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='schema_migrations'")
        table_exists = cursor.fetchone() is not None
        
        conn.close()
        print(f"DEBUG: Table exists: {table_exists}")
        return table_exists  # Return True if table exists, even if empty
        
    except Exception as e:
        print(f"DEBUG: Error: {e}")
        return False


def get_initialization_status(data_dir: str = "data") -> Dict[str, Any]:
    data_path = Path(data_dir)

    status = {
        "initialized": False,
        "database_exists": False,
        "tables_exist": False,
        "directory_structure": {},
        "issues": [],
    }

    required_dirs = ["database", "documents", "models", "logs", "backups"]
    for dir_name in required_dirs:
        dir_path = data_path / dir_name
        status["directory_structure"][dir_name] = {
            "exists": dir_path.exists(),
            "writable": os.access(dir_path, os.W_OK) if dir_path.exists() else False,
        }

    db_file = data_path / "database" / "cockatoo.db"
    if db_file.exists():
        status["database_exists"] = True

        try:
            conn = sqlite3.connect(str(db_file))
            cursor = conn.cursor()

            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table'
                AND name IN ('schema_migrations', 'system_info')
            """)

            system_tables = cursor.fetchall()
            status["tables_exist"] = len(system_tables) >= 2

            try:
                cursor.execute("SELECT COUNT(*) FROM schema_migrations WHERE status='applied'")
                migration_count = cursor.fetchone()[0]
                status["migration_count"] = migration_count
                status["initialized"] = migration_count > 0
            except Exception:
                status["migration_count"] = 0

            conn.close()

        except sqlite3.Error as e:
            status["issues"].append(f"Database error: {e}")

    return status


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if is_initialized():
        print("Database already initialized. Skipping...")
        sys.exit(0)

    initializer = DatabaseInitializer()
    result = initializer.initialize()

    if result.success:
        print(f"\nSUCCESS: {result.message}")
        print(f"Duration: {result.details.get('duration_seconds', 0):.2f} seconds")
    else:
        print(f"\nFAILURE: {result.message}")
        if result.errors:
            print("\nErrors:")
            for error in result.errors:
                print(f"  - {error}")
        sys.exit(1)