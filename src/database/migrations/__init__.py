# src/database/migrations/__init__.py
"""
Database migrations package.
Handles database schema versioning and migrations.
"""

from .migration_manager import MigrationManager, MigrationProgress
from .migration_config import MigrationConfig
from .version_table import VersionTable
from .base_migration import Migration 
from .migration_utils import (
    backup_database,
    validate_database_integrity,
    load_migration_class,
    topological_sort
)

# For backward compatibility
from .database_initializer import DatabaseInitializer, InitResult, is_initialized

__all__ = [
    'Migration', 
    'MigrationManager',
    'MigrationProgress',
    'MigrationConfig',
    'VersionTable',
    'backup_database',
    'validate_database_integrity',
    'load_migration_class',
    'topological_sort',
    'DatabaseInitializer',
    'InitResult',
    'is_initialized',
]