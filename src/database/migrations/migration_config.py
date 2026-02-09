# src/database/migrations/migration_config.py

"""
Migration configuration settings.

Defines configuration parameters for the migration system including
paths, settings, operational parameters, and validation levels.
"""

import os
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class ValidationLevel(Enum):
    """Enumeration of validation levels for migration operations."""
    NONE = "none"
    BASIC = "basic"
    STRICT = "strict"


class LogLevel(Enum):
    """Enumeration of logging levels for migration operations."""
    DEBUG = "debug"
    INFO = "info"
    WARN = "warn"
    ERROR = "error"


@dataclass
class MigrationConfig:
    """
    Complete configuration for migration system.
    
    Provides comprehensive control over migration behavior, safety features,
    and operational parameters.
    """
    
    # === CORE SETTINGS ===
    database_path: str = "cockatoo.db"
    migrations_dir: str = "src/database/migrations"
    
    # === SAFETY & BACKUP SETTINGS ===
    auto_migrate: bool = True
    create_backup: bool = True
    backup_location: str = "backups/migrations"
    max_rollback_steps: int = 5
    allow_downgrade: bool = False
    require_confirmation: bool = True
    
    # === VALIDATION & PERFORMANCE ===
    validation_level: ValidationLevel = ValidationLevel.BASIC
    timeout_seconds: int = 300
    batch_size: int = 1000
    max_retry_attempts: int = 3
    retry_delay: float = 1.0
    
    # === LOGGING & MONITORING ===
    log_level: LogLevel = LogLevel.INFO
    log_file: Optional[str] = "logs/migration.log"
    enable_metrics: bool = True
    
    # === ADVANCED SETTINGS ===
    transaction_mode: str = "auto"
    parallel_migrations: bool = False
    skip_breaking_changes: bool = False
    maintenance_mode: bool = False
    dry_run: bool = False
    
    # === INTERNAL STATE ===
    _initialized: bool = field(default=False, init=False, repr=False)
    
    def __post_init__(self) -> None:
        """
        Initialize configuration with proper paths and validation.
        
        Raises:
            ValueError: If any configuration value is invalid
            PermissionError: If directory permissions are insufficient
        """
        self._validate_config()
        self._initialize_paths()
        self._initialized = True
    
    def _validate_config(self) -> None:
        """Validate all configuration parameters."""
        if not self.database_path:
            raise ValueError("Database path cannot be empty. Provide a valid path to the SQLite database file.")
        
        if not self.migrations_dir:
            raise ValueError("Migrations directory cannot be empty. Provide a valid directory path containing migration scripts.")
        
        if not self.backup_location:
            raise ValueError("Backup location cannot be empty. Provide a valid directory path for database backups.")
        
        if self.max_rollback_steps < 0:
            raise ValueError(f"Maximum rollback steps must be non-negative. Received: {self.max_rollback_steps}")
        
        if self.timeout_seconds <= 0:
            raise ValueError(f"Timeout must be a positive integer. Received: {self.timeout_seconds}")
        
        if self.batch_size <= 0:
            raise ValueError(f"Batch size must be a positive integer. Received: {self.batch_size}")
        
        if self.max_retry_attempts < 0:
            raise ValueError(f"Maximum retry attempts must be non-negative. Received: {self.max_retry_attempts}")
        
        if self.retry_delay < 0:
            raise ValueError(f"Retry delay must be non-negative. Received: {self.retry_delay}")
        
        if self.transaction_mode not in ["auto", "always", "never"]:
            raise ValueError(
                f"Transaction mode must be 'auto', 'always', or 'never'. Received: {self.transaction_mode}"
            )
    
    def _initialize_paths(self) -> None:
        """Convert relative paths to absolute and create directories with permission validation."""
        self.database_path = os.path.abspath(self.database_path)
        self.migrations_dir = os.path.abspath(self.migrations_dir)
        self.backup_location = os.path.abspath(self.backup_location)
        
        if self.log_file:
            self.log_file = os.path.abspath(self.log_file)
        
        self._ensure_directory(self.migrations_dir, "Migrations directory")
        self._ensure_directory(self.backup_location, "Backup directory")
        
        if self.log_file:
            log_dir = os.path.dirname(self.log_file)
            self._ensure_directory(log_dir, "Log directory")
    
    def _ensure_directory(self, directory_path: str, description: str) -> None:
        """
        Ensure directory exists and is writable.
        
        Args:
            directory_path: Path to directory
            description: Human-readable description for error messages
            
        Raises:
            PermissionError: If directory cannot be created or is not writable
        """
        try:
            os.makedirs(directory_path, exist_ok=True)
            
            test_file = os.path.join(directory_path, ".permission_test")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            
        except PermissionError:
            raise PermissionError(
                f"Insufficient permissions for {description.lower()}: {directory_path}. "
                f"Ensure the process has write access to this location."
            )
        except OSError as e:
            raise OSError(
                f"Failed to create or access {description.lower()}: {directory_path}. Error: {str(e)}"
            )
    
    def get_migration_files(self) -> List[str]:
        """
        Get sorted list of migration files.
        
        Returns:
            List of migration file paths sorted by version
            
        Raises:
            FileNotFoundError: If migrations directory doesn't exist
        """
        if not os.path.exists(self.migrations_dir):
            raise FileNotFoundError(
                f"Migrations directory not found: {self.migrations_dir}. "
                f"Ensure the directory exists or update the migrations_dir configuration."
            )
        
        migration_dir = Path(self.migrations_dir)
        migration_files = []
        
        for file_path in migration_dir.glob("v*.py"):
            if file_path.name == "__init__.py":
                continue
            
            if self._is_valid_migration_file(file_path.name):
                migration_files.append(str(file_path))
        
        migration_files.sort(key=lambda x: self._extract_version(x))
        return migration_files
    
    def _is_valid_migration_file(self, filename: str) -> bool:
        """
        Check if a filename matches migration naming convention.
        
        Expected format: v1_0_0__description.py
        Alternative format: v1.0.0__description.py (will be standardized to underscores)
        
        Args:
            filename: Name of the file to check
            
        Returns:
            True if filename is valid migration file
        """
        if not filename.startswith("v") or not filename.endswith(".py"):
            return False
        
        name_without_ext = filename[:-3]
        parts = name_without_ext.split("__")
        
        if len(parts) != 2:
            return False
        
        version_part = parts[0]
        
        version_str = version_part[1:]
        version_str = version_str.replace('.', '_')
        
        version_parts = version_str.split('_')
        
        if len(version_parts) < 3:
            return False
        
        try:
            int(version_parts[0])
            int(version_parts[1])
            int(version_parts[2])
            return True
        except ValueError:
            return False
    
    def _extract_version(self, file_path: str) -> tuple:
        """
        Extract version number from migration file name.
        
        Handles both v1_0_0 and v1.0.0 formats.
        
        Args:
            file_path: Path to migration file
            
        Returns:
            Tuple of (major, minor, patch) version numbers
        """
        file_name = Path(file_path).stem
        version_part = file_name.split("__")[0]
        version_str = version_part[1:]
        
        version_str = version_str.replace('.', '_')
        version_parts = version_str.split('_')
        
        major = int(version_parts[0]) if len(version_parts) > 0 else 0
        minor = int(version_parts[1]) if len(version_parts) > 1 else 0
        patch = int(version_parts[2]) if len(version_parts) > 2 else 0
        
        return (major, minor, patch)
    
    def get_backup_path(self, migration_version: Optional[str] = None) -> str:
        """
        Generate backup file path for a migration.
        
        Args:
            migration_version: Optional migration version to include in filename
            
        Returns:
            Full path to backup file
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        db_name = Path(self.database_path).stem
        
        if migration_version:
            safe_version = migration_version.replace('.', '_').replace(os.path.sep, '_')
            filename = f"{db_name}_{safe_version}_{timestamp}.backup"
        else:
            filename = f"{db_name}_{timestamp}.backup"
        
        return os.path.join(self.backup_location, filename)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        return {
            "database_path": self.database_path,
            "migrations_dir": self.migrations_dir,
            "auto_migrate": self.auto_migrate,
            "create_backup": self.create_backup,
            "backup_location": self.backup_location,
            "max_rollback_steps": self.max_rollback_steps,
            "allow_downgrade": self.allow_downgrade,
            "require_confirmation": self.require_confirmation,
            "validation_level": self.validation_level.value,
            "timeout_seconds": self.timeout_seconds,
            "batch_size": self.batch_size,
            "max_retry_attempts": self.max_retry_attempts,
            "retry_delay": self.retry_delay,
            "log_level": self.log_level.value,
            "log_file": self.log_file,
            "enable_metrics": self.enable_metrics,
            "transaction_mode": self.transaction_mode,
            "parallel_migrations": self.parallel_migrations,
            "skip_breaking_changes": self.skip_breaking_changes,
            "maintenance_mode": self.maintenance_mode,
            "dry_run": self.dry_run,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MigrationConfig":
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Dictionary with configuration values
            
        Returns:
            MigrationConfig instance
            
        Raises:
            ValueError: If dictionary contains invalid values
        """
        processed_dict = config_dict.copy()
        
        if "validation_level" in processed_dict:
            processed_dict["validation_level"] = ValidationLevel(processed_dict["validation_level"])
        
        if "log_level" in processed_dict:
            processed_dict["log_level"] = LogLevel(processed_dict["log_level"])
        
        return cls(**processed_dict)
    
    def merge(self, other_config: "MigrationConfig") -> "MigrationConfig":
        """
        Merge this configuration with another configuration.
        
        Non-None values from the other configuration take precedence.
        
        Args:
            other_config: Configuration to merge with
            
        Returns:
            New merged configuration instance
        """
        current_dict = self.to_dict()
        other_dict = other_config.to_dict()
        
        merged_dict = {}
        for key in current_dict.keys():
            if other_dict.get(key) is not None:
                merged_dict[key] = other_dict[key]
            else:
                merged_dict[key] = current_dict[key]
        
        return self.from_dict(merged_dict)
    
    def apply_environment_overrides(self, prefix: str = "MIGRATION_") -> None:
        """
        Apply configuration overrides from environment variables.
        
        Environment variables should be in the format:
        - MIGRATION_DATABASE_PATH
        - MIGRATION_AUTO_MIGRATE
        - MIGRATION_VALIDATION_LEVEL
        
        Boolean values: "true", "false" (case-insensitive)
        Integer values: string representation of integer
        Enum values: string representation of enum value
        
        Args:
            prefix: Prefix for environment variable names
        """
        env_vars = {}
        
        for key in self.to_dict().keys():
            env_key = f"{prefix}{key.upper()}"
            env_value = os.environ.get(env_key)
            
            if env_value is not None:
                env_vars[key] = env_value
        
        if not env_vars:
            return
        
        processed_vars = {}
        for key, value in env_vars.items():
            original_value = getattr(self, key)
            
            if isinstance(original_value, bool):
                processed_vars[key] = value.lower() in ("true", "1", "yes", "on")
            elif isinstance(original_value, int):
                try:
                    processed_vars[key] = int(value)
                except ValueError:
                    raise ValueError(
                        f"Environment variable {prefix}{key.upper()} must be an integer. Received: {value}"
                    )
            elif isinstance(original_value, float):
                try:
                    processed_vars[key] = float(value)
                except ValueError:
                    raise ValueError(
                        f"Environment variable {prefix}{key.upper()} must be a float. Received: {value}"
                    )
            elif key == "validation_level":
                try:
                    processed_vars[key] = ValidationLevel(value.lower())
                except ValueError:
                    valid_levels = [e.value for e in ValidationLevel]
                    raise ValueError(
                        f"Environment variable {prefix}{key.upper()} must be one of {valid_levels}. Received: {value}"
                    )
            elif key == "log_level":
                try:
                    processed_vars[key] = LogLevel(value.lower())
                except ValueError:
                    valid_levels = [e.value for e in LogLevel]
                    raise ValueError(
                        f"Environment variable {prefix}{key.upper()} must be one of {valid_levels}. Received: {value}"
                    )
            else:
                processed_vars[key] = value
        
        for key, value in processed_vars.items():
            setattr(self, key, value)
        
        self.__post_init__()
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"MigrationConfig(database='{self.database_path}', migrations='{self.migrations_dir}')"


# === FACTORY FUNCTIONS ===

def create_production_config() -> MigrationConfig:
    """
    Create configuration optimized for production use.
    
    Returns:
        Production-ready configuration
    """
    return MigrationConfig(
        auto_migrate=True,
        create_backup=True,
        backup_location="/var/backups/cockatoo/migrations",
        max_rollback_steps=2,
        allow_downgrade=False,
        require_confirmation=False,
        validation_level=ValidationLevel.STRICT,
        timeout_seconds=600,
        batch_size=500,
        max_retry_attempts=2,
        log_level=LogLevel.INFO,
        log_file="/var/log/cockatoo/migration.log",
        enable_metrics=True,
        transaction_mode="always",
        parallel_migrations=False,
        skip_breaking_changes=False,
        maintenance_mode=True,
        dry_run=False,
    )


def create_development_config() -> MigrationConfig:
    """
    Create configuration optimized for development use.
    
    Returns:
        Development-friendly configuration
    """
    return MigrationConfig(
        auto_migrate=True,
        create_backup=True,
        backup_location="backups/migrations",
        max_rollback_steps=10,
        allow_downgrade=True,
        require_confirmation=False,
        validation_level=ValidationLevel.BASIC,
        timeout_seconds=60,
        batch_size=1000,
        max_retry_attempts=3,
        log_level=LogLevel.DEBUG,
        log_file=None,
        enable_metrics=False,
        transaction_mode="auto",
        parallel_migrations=True,
        skip_breaking_changes=False,
        maintenance_mode=False,
        dry_run=False,
    )


def create_test_config() -> MigrationConfig:
    """
    Create configuration optimized for testing.
    
    Returns:
        Test-optimized configuration
    """
    return MigrationConfig(
        auto_migrate=True,
        create_backup=False,
        backup_location="/tmp/cockatoo_test_backups",
        max_rollback_steps=999,
        allow_downgrade=True,
        require_confirmation=False,
        validation_level=ValidationLevel.NONE,
        timeout_seconds=30,
        batch_size=100,
        max_retry_attempts=1,
        log_level=LogLevel.WARN,
        log_file=None,
        enable_metrics=False,
        transaction_mode="never",
        parallel_migrations=True,
        skip_breaking_changes=False,
        maintenance_mode=False,
        dry_run=False,
    )


def create_config_from_sources(
    base_config: Optional[MigrationConfig] = None,
    config_file: Optional[str] = None,
    environment_prefix: str = "MIGRATION_"
) -> MigrationConfig:
    """
    Create configuration by merging multiple sources.
    
    Priority order (highest to lowest):
    1. Environment variables
    2. Configuration file
    3. Base configuration
    
    Args:
        base_config: Base configuration (defaults to default_config if None)
        config_file: Optional path to configuration file (JSON or YAML)
        environment_prefix: Prefix for environment variables
    
    Returns:
        Merged configuration instance
    """
    import json
    import yaml
    
    if base_config is None:
        config = MigrationConfig()
    else:
        config = base_config
    
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            if config_file.endswith('.json'):
                file_config = json.load(f)
            elif config_file.endswith(('.yaml', '.yml')):
                file_config = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_file}")
        
        file_migration_config = MigrationConfig.from_dict(file_config)
        config = config.merge(file_migration_config)
    
    config.apply_environment_overrides(prefix=environment_prefix)
    
    return config


# Default configuration instances
default_config = MigrationConfig()
production_config = create_production_config()
development_config = create_development_config()
test_config = create_test_config()