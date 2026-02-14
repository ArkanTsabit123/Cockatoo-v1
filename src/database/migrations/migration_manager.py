# src/database/migrations/migration_manager.py

"""
Migration manager for Cockatoo database migrations.

Main manager class that coordinates migration discovery, execution,
rollback, and status tracking with atomic operations, progress reporting,
and comprehensive error recovery.
"""

import sqlite3
import time
import json
import traceback
import threading
import random
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable, Union
import logging

try:
    from .migration_config import MigrationConfig
    from .version_table import VersionTable
    from .migration_utils import (
        backup_database, validate_database_integrity,
        table_exists, log_migration_step, log_data_stats,
        format_duration, create_temp_table, swap_tables,
        execute_sql_script, load_migration_class, topological_sort
    )
except ImportError:
    import sys
    from pathlib import Path
    
    current_dir = Path(__file__).parent
    src_dir = current_dir.parent.parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    
    from src.database.migrations.migration_config import MigrationConfig
    from src.database.migrations.version_table import VersionTable
    from src.database.migrations.migration_utils import (
        backup_database, validate_database_integrity,
        table_exists, log_migration_step, log_data_stats,
        format_duration, create_temp_table, swap_tables,
        execute_sql_script, load_migration_class, topological_sort
    )

logger = logging.getLogger(__name__)


class MigrationProgress:
    """Represents migration progress for reporting."""
    
    def __init__(self, total_steps: int = 0):
        self.total_steps = total_steps
        self.current_step = 0
        self.current_description = ""
        self.start_time = time.time()
        self.errors = []
        self.warnings = []
        self.lock = threading.Lock()

    def update(self, step: int, description: str) -> None:
        """Update progress."""
        with self.lock:
            self.current_step = step
            self.current_description = description
    
    def add_error(self, error: str) -> None:
        """Add error."""
        with self.lock:
            self.errors.append(error)
    
    def add_warning(self, warning: str) -> None:
        """Add warning."""
        with self.lock:
            self.warnings.append(warning)
    
    def get_percentage(self) -> float:
        """Get completion percentage."""
        with self.lock:
            if self.total_steps == 0:
                return 100.0
            return (self.current_step / self.total_steps) * 100
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        with self.lock:
            return {
                'total_steps': self.total_steps,
                'current_step': self.current_step,
                'current_description': self.current_description,
                'percentage': self.get_percentage(),
                'elapsed_seconds': self.get_elapsed_time(),
                'errors': self.errors.copy(),
                'warnings': self.warnings.copy(),
                'is_complete': self.current_step >= self.total_steps
            }


class MigrationManager:
    """
    Manages database migrations with atomic operations and progress reporting.
    """
    
    def __init__(self, config: Optional[MigrationConfig] = None) -> None:
        """
        Initialize migration manager.
        
        Args:
            config: Migration configuration
        """
        self.config = config or MigrationConfig()
        self.connection: Optional[sqlite3.Connection] = None
        self.version_table: Optional[VersionTable] = None
        self.progress_callback: Optional[Callable[[MigrationProgress], None]] = None
        self.connection_lock = threading.RLock()
        self.progress_lock = threading.Lock()
        
        log_level = getattr(logging, self.config.log_level.value.upper())
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def __enter__(self):
        """Context manager entry - automatically connect."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically disconnect."""
        self.disconnect()
        return False
    
    def set_progress_callback(self, callback: Callable[[MigrationProgress], None]) -> None:
        """
        Set callback for progress reporting.
        
        Args:
            callback: Function to call with progress updates
        """
        with self.progress_lock:
            self.progress_callback = callback
    
    def _report_progress(self, progress: MigrationProgress) -> None:
        """Report progress if callback is set."""
        with self.progress_lock:
            if self.progress_callback:
                try:
                    self.progress_callback(progress)
                except Exception as e:
                    logger.error(f"Progress callback error: {e}")
    
    def connect(self) -> bool:
        """
        Establish database connection.
        
        Returns:
            True if connection successful, False otherwise
        """
        with self.connection_lock:
            try:
                if self.connection:
                    return True
                    
                log_migration_step(
                    "Establishing database connection",
                    "INFO",
                    {'database_path': self.config.database_path}
                )
                
                self.connection = sqlite3.connect(
                    self.config.database_path,
                    timeout=self.config.timeout_seconds
                )
                
                self.connection.row_factory = sqlite3.Row
                self.connection.execute("PRAGMA foreign_keys = ON")
                
                self.version_table = VersionTable(self.connection)
                
                log_migration_step(
                    "Database connection established",
                    "INFO",
                    {'database_path': self.config.database_path}
                )
                
                return True
                
            except sqlite3.Error as e:
                log_migration_step(
                    "Failed to connect to database",
                    "ERROR",
                    {'database_path': self.config.database_path, 'error': str(e)}
                )
                self.connection = None
                self.version_table = None
                return False
    
    def disconnect(self) -> None:
        """Close database connection safely."""
        with self.connection_lock:
            if self.connection:
                try:
                    self.connection.close()
                    log_migration_step("Database connection closed", "INFO")
                except Exception as e:
                    logger.error(f"Error closing connection: {e}")
                finally:
                    self.connection = None
                    self.version_table = None
    
    def _ensure_connection(self) -> bool:
        """
        Ensure database connection is established.
        
        Returns:
            True if connection is available, False otherwise
        """
        if not self.connection:
            return self.connect()
        return True
    
    def _extract_version_from_migration_record(self, record: Optional[Union[Dict, sqlite3.Row]]) -> Optional[str]:
        """
        Safely extract version from migration record.
        
        Args:
            record: Migration record from version table
            
        Returns:
            Version string or None if not found
        """
        if record is None:
            return None
        
        if isinstance(record, dict):
            return record.get('version')
        
        if isinstance(record, sqlite3.Row):
            try:
                return record['version']
            except (KeyError, IndexError):
                pass
        
        return None
    
    def _convert_migration_record_to_dict(self, record: Optional[Union[Dict, sqlite3.Row]]) -> Dict[str, Any]:
        """
        Convert migration record to dictionary format.
        
        Args:
            record: Migration record from version table
            
        Returns:
            Dictionary with migration information
        """
        if record is None:
            return {}
        
        if isinstance(record, dict):
            return record.copy()
        
        if isinstance(record, sqlite3.Row):
            try:
                return {key: record[key] for key in record.keys()}
            except Exception:
                return {}
        
        return {}
    
    def get_current_version(self) -> Optional[str]:
        """
        Get current database version.
        
        Returns:
            Current version string or None if no migrations applied
        """
        if not self._ensure_connection():
            return None
        
        try:
            last_migration = self.version_table.get_last_migration()
            return self._extract_version_from_migration_record(last_migration)
            
        except Exception as e:
            logger.error(f"Failed to get current version: {e}")
            return None
    
    def _parse_version(self, version_str: Optional[str]) -> tuple:
        """
        Parse version string into tuple for comparison.
        
        Args:
            version_str: Version string in format "x_y_z"
            
        Returns:
            Tuple of integers (major, minor, patch)
        """
        if not version_str or version_str == 'unknown':
            return (0, 0, 0)
        
        try:
            parts = version_str.split("_")
            
            if len(parts) >= 3:
                return (int(parts[0]), int(parts[1]), int(parts[2]))
            elif len(parts) == 2:
                return (int(parts[0]), int(parts[1]), 0)
            else:
                return (int(parts[0]), 0, 0)
        except (ValueError, IndexError):
            logger.warning(f"Failed to parse version string: {version_str}")
            return (0, 0, 0)
    
    def _extract_version_from_file(self, file_path: str) -> Optional[str]:
        """Extract version from migration file name."""
        try:
            file_name = Path(file_path).name
            if file_name.startswith("v") and "__" in file_name:
                version_part = file_name.split("__")[0]
                return version_part[1:]
            return None
        except Exception:
            return None
    
    def get_available_migrations(self) -> List[Dict[str, Any]]:
        """
        Get list of available migrations from filesystem.
        
        Returns:
            List of migration dictionaries
        """
        try:
            migration_dir = Path(self.config.migrations_dir)
            
            version_dir = migration_dir / "version"
            if version_dir.exists():
                search_dir = version_dir
            else:
                search_dir = migration_dir
                logger.warning(f"Version directory not found, using {migration_dir} directly")
            
            migration_files = []
            for file_path in search_dir.glob("v*.py"):
                if file_path.name == "__init__.py":
                    continue
                migration_files.append(str(file_path))
            
            migration_files.sort()
            migrations = []
            
            for file_path in migration_files:
                try:
                    version = self._extract_version_from_file(file_path)
                    if not version:
                        continue
                    
                    migration_class = load_migration_class(file_path, "Migration")
                    
                    if migration_class:
                        migration = migration_class()
                        migrations.append({
                            'version': version,
                            'file_path': file_path,
                            'description': migration.get_description(),
                            'dependencies': migration.get_dependencies(),
                            'is_breaking': migration.is_breaking_change(),
                            'has_down_method': hasattr(migration, 'downgrade'),
                            'estimated_duration': migration.estimate_duration()
                        })
                    else:
                        migrations.append({
                            'version': version,
                            'file_path': file_path,
                            'description': 'Unable to load migration',
                            'dependencies': [],
                            'is_breaking': False,
                            'has_down_method': False,
                            'error': 'Failed to load migration class'
                        })
                        
                except Exception as e:
                    logger.error(f"Failed to process migration file {file_path}: {e}")
                    continue
            
            migrations.sort(key=lambda x: self._parse_version(x['version']))
            logger.info(f"Found {len(migrations)} available migrations")
            return migrations
            
        except Exception as e:
            logger.error(f"Failed to get available migrations: {e}")
            return []
    
    def _get_applied_migrations_as_dicts(self) -> List[Dict[str, Any]]:
        """
        Get applied migrations and convert them to dictionary format.
        
        Returns:
            List of applied migration dictionaries
        """
        try:
            applied = self.version_table.get_applied_migrations()
            
            if not applied:
                return []
            
            result = []
            for record in applied:
                migration_dict = self._convert_migration_record_to_dict(record)
                if migration_dict and migration_dict.get('version'):
                    result.append(migration_dict)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get applied migrations: {e}")
            return []
    
    def _get_applied_versions(self) -> List[str]:
        """
        Get list of applied migration versions.
        
        Returns:
            List of version strings
        """
        applied_migrations = self._get_applied_migrations_as_dicts()
        versions = []
        
        for migration in applied_migrations:
            version = migration.get('version')
            if version:
                versions.append(version)
        
        return versions
    
    def _refresh_applied_versions(self) -> List[str]:
        """
        Refresh the list of applied versions directly from database.
        
        Returns:
            List of applied version strings
        """
        try:
            if not self.connection:
                return []
            
            cursor = self.connection.cursor()
            cursor.execute("SELECT version FROM schema_migrations WHERE status='applied'")
            versions = [row[0] for row in cursor.fetchall()]
            logger.debug(f"Refreshed applied versions from DB: {versions}")
            return versions
        except Exception as e:
            logger.error(f"Failed to refresh applied versions: {e}")
            return []
    
    def check_needs_migration(self) -> Dict[str, Any]:
        """
        Check if migration is needed.
        
        Returns:
            Dictionary with migration needs analysis
        """
        if not self._ensure_connection():
            return {'needs_migration': False, 'error': 'Database connection failed'}
        
        try:
            available_migrations = self.get_available_migrations()
            applied_versions = self._get_applied_versions()
            
            available_versions = [m['version'] for m in available_migrations]
            
            pending_versions = [v for v in available_versions if v not in applied_versions]
            pending_migrations = [m for m in available_migrations if m['version'] in pending_versions]
            
            breaking_changes = [m for m in pending_migrations if m.get('is_breaking', False)]
            
            dependency_issues = []
            for migration in pending_migrations:
                for dep in migration.get('dependencies', []):
                    if dep not in applied_versions and dep not in pending_versions:
                        dependency_issues.append({
                            'migration': migration['version'],
                            'missing_dependency': dep
                        })
            
            total_estimated_time = sum(m.get('estimated_duration', 30) for m in pending_migrations)
            
            result = {
                'needs_migration': len(pending_versions) > 0,
                'current_version': self.get_current_version(),
                'pending_count': len(pending_versions),
                'pending_versions': pending_versions,
                'breaking_changes_count': len(breaking_changes),
                'breaking_changes': [m['version'] for m in breaking_changes],
                'dependency_issues': dependency_issues,
                'total_estimated_time_seconds': total_estimated_time,
                'has_dependency_issues': len(dependency_issues) > 0,
                'recommendation': 'Migration required' if pending_versions else 'Up to date'
            }
            
            if breaking_changes:
                result['recommendation'] += f" ({len(breaking_changes)} breaking changes)"
            
            log_migration_step("Migration needs check completed", "INFO", result)
            return result
            
        except Exception as e:
            error_result = {
                'needs_migration': False,
                'error': str(e),
                'recommendation': 'Unable to determine migration needs'
            }
            log_migration_step("Migration needs check failed", "ERROR", error_result)
            return error_result
    
    def validate_migration_chain(self) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Validate the entire migration chain for consistency.
        
        Returns:
            Tuple of (is_valid, issues)
        """
        issues = []
        
        try:
            available_migrations = self.get_available_migrations()
            applied_migrations = self._get_applied_migrations_as_dicts()
            
            applied_versions = [m['version'] for m in applied_migrations if 'version' in m]
            available_versions = [m['version'] for m in available_migrations if 'version' in m]
            
            missing_versions = [v for v in applied_versions if v not in available_versions]
            for version in missing_versions:
                issues.append({
                    'type': 'missing_file',
                    'version': version,
                    'message': f'Migration file for version {version} is missing'
                })
            
            for migration in available_migrations:
                version = migration['version']
                dependencies = migration.get('dependencies', [])
                
                for dep in dependencies:
                    if dep not in available_versions:
                        issues.append({
                            'type': 'missing_dependency',
                            'version': version,
                            'dependency': dep,
                            'message': f'Migration {version} depends on non-existent migration {dep}'
                        })
                
                if version in dependencies:
                    issues.append({
                        'type': 'circular_dependency',
                        'version': version,
                        'message': f'Migration {version} depends on itself'
                    })
            
            available_versions_sorted = sorted(available_versions, key=self._parse_version)
            for i in range(1, len(available_versions_sorted)):
                prev_version = available_versions_sorted[i-1]
                curr_version = available_versions_sorted[i]
                
                prev_parts = self._parse_version(prev_version)
                curr_parts = self._parse_version(curr_version)
                
                if curr_parts[2] - prev_parts[2] > 1 and curr_parts[0] == prev_parts[0] and curr_parts[1] == prev_parts[1]:
                    issues.append({
                        'type': 'version_gap',
                        'prev_version': prev_version,
                        'curr_version': curr_version,
                        'message': f'Possible version gap between {prev_version} and {curr_version}'
                    })
            
            if self.config.validation_level.value != 'none':
                migration_files_dict = {}
                for migration in available_migrations:
                    if 'file_path' in migration and 'version' in migration:
                        migration_files_dict[migration['version']] = migration['file_path']
                
                try:
                    is_valid, checksum_issues = self.version_table.validate_checksums(migration_files_dict)
                    if not is_valid:
                        issues.extend([
                            {
                                'type': 'checksum_mismatch',
                                'version': issue.get('version'),
                                'message': f'Checksum mismatch for migration {issue.get("version")}'
                            }
                            for issue in checksum_issues
                        ])
                except Exception as e:
                    logger.warning(f"Checksum validation failed: {e}")
            
            is_valid = len(issues) == 0
            
            log_migration_step(
                "Migration chain validation completed",
                "INFO" if is_valid else "WARNING",
                {'is_valid': is_valid, 'issue_count': len(issues), 'issues': issues[:5]}
            )
            
            return is_valid, issues
            
        except Exception as e:
            log_migration_step(
                "Migration chain validation failed",
                "ERROR",
                {'error': str(e)}
            )
            return False, [{'type': 'validation_error', 'message': str(e)}]
    
    def _resolve_migration_order(self, migrations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Resolve migration order using topological sort.
        
        Args:
            migrations: List of migration dictionaries
            
        Returns:
            Topologically sorted list of migrations (dependencies first)
        """
        if not migrations:
            return []
        
        from collections import deque
        
        # Build dependency graph
        graph = {}
        migration_map = {}
        
        for migration in migrations:
            version = migration['version']
            migration_map[version] = migration
            graph[version] = migration.get('dependencies', [])
        
        logger.debug(f"Dependency graph: {graph}")
        
        # Kahn's topological sort algorithm
        in_degree = {v: 0 for v in graph}
        for v, deps in graph.items():
            for dep in deps:
                if dep in in_degree:
                    in_degree[dep] += 1
        
        # Queue nodes with no dependencies
        queue = deque([v for v in graph if in_degree[v] == 0])
        result = []
        
        while queue:
            v = queue.popleft()
            result.append(v)
            
            # Reduce in-degree of dependents
            for node, deps in graph.items():
                if v in deps:
                    in_degree[node] -= 1
                    if in_degree[node] == 0 and node not in result:
                        queue.append(node)
        
        # Check if all nodes were processed
        if len(result) != len(graph):
            logger.error(f"Circular dependency detected. Processed: {result}, Total: {list(graph.keys())}")
            migrations.sort(key=lambda x: self._parse_version(x['version']))
            logger.warning(f"Using fallback sorting: {[m['version'] for m in migrations]}")
            return migrations
        
        logger.info(f"Resolved migration order: {result}")
        return [migration_map[v] for v in result]
    
    def _get_pending_migrations(self, target_version: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get list of pending migrations up to target version with dependency validation.
        
        Args:
            target_version: Target version to migrate to
            
        Returns:
            List of pending migrations
        """
        try:
            available_migrations = self.get_available_migrations()
            applied_versions = self._get_applied_versions()
            
            logger.debug(f"Applied versions: {applied_versions}")
            
            pending = []
            target_parsed = self._parse_version(target_version) if target_version else None
            
            for migration in available_migrations:
                version = migration['version']
                version_parsed = self._parse_version(version)
                
                if version in applied_versions:
                    logger.debug(f"Skipping {version} - already applied")
                    continue
                    
                if target_parsed and version_parsed > target_parsed:
                    continue
                
                pending.append(migration)
            
            logger.debug(f"Pending before ordering: {[m['version'] for m in pending]}")
            
            ordered_pending = self._resolve_migration_order(pending)
            
            all_pending_versions = [m['version'] for m in ordered_pending]
            for migration in ordered_pending:
                for dep in migration.get('dependencies', []):
                    if dep not in applied_versions and dep not in all_pending_versions:
                        logger.error(f"Dependency {dep} for migration {migration['version']} not satisfied")
                        return []
            
            return ordered_pending
            
        except Exception as e:
            logger.error(f"Failed to get pending migrations: {e}")
            return []
    
    def migrate_to(self, target_version: Optional[str] = None, 
                  dry_run: bool = False) -> Tuple[bool, str, MigrationProgress]:
        """
        Migrate to specific version with atomic operations.
        
        Args:
            target_version: Target version to migrate to (None for latest)
            dry_run: If True, only simulate migration
            
        Returns:
            Tuple of (success, message, progress)
        """
        progress = MigrationProgress()
        
        try:
            if not self._ensure_connection():
                progress.total_steps = 1
                progress.update(1, "Database connection failed")
                self._report_progress(progress)
                return False, "Database connection failed", progress
            
            needs_check = self.check_needs_migration()
            if not needs_check.get('needs_migration', False):
                progress.total_steps = 1
                progress.update(1, "No migrations needed")
                self._report_progress(progress)
                return True, "Database is already up to date", progress
            
            pending_migrations = self._get_pending_migrations(target_version)
            
            if not pending_migrations:
                progress.total_steps = 1
                progress.update(1, "No migrations to apply")
                self._report_progress(progress)
                return True, "No migrations to apply", progress
            
            breaking_changes = [m for m in pending_migrations if m.get('is_breaking', False)]
            if breaking_changes and self.config.require_confirmation and not dry_run:
                log_migration_step(
                    "Breaking changes detected",
                    "WARNING",
                    {'breaking_versions': [m['version'] for m in breaking_changes]}
                )
            
            progress.total_steps = len(pending_migrations) + 3
            
            progress.update(1, "Starting migration process")
            self._report_progress(progress)
            
            if dry_run:
                return self._execute_dry_run(pending_migrations, progress)
            
            return self._execute_migration_atomic(pending_migrations, progress)
            
        except Exception as e:
            progress.add_error(f"Migration failed: {e}")
            progress.update(progress.total_steps, f"Failed: {e}")
            self._report_progress(progress)
            
            log_migration_step(
                "Migration failed",
                "ERROR",
                {'error': str(e), 'traceback': traceback.format_exc()}
            )
            
            return False, f"Migration failed: {e}", progress
    
    def _execute_dry_run(self, pending_migrations: List[Dict[str, Any]], 
                        progress: MigrationProgress) -> Tuple[bool, str, MigrationProgress]:
        """Execute dry run simulation."""
        try:
            for i, migration in enumerate(pending_migrations, 2):
                progress.update(i, f"Would apply: {migration['version']} - {migration.get('description', '')}")
                self._report_progress(progress)
                time.sleep(0.1)
            
            progress.update(progress.total_steps, "Dry run completed")
            self._report_progress(progress)
            
            message = f"Dry run: Would apply {len(pending_migrations)} migrations"
            for migration in pending_migrations:
                message += f"\n  - {migration['version']}: {migration.get('description', '')}"
            
            log_migration_step("Dry run completed", "INFO", {'migration_count': len(pending_migrations)})
            return True, message, progress
            
        except Exception as e:
            progress.add_error(f"Dry run failed: {e}")
            progress.update(progress.total_steps, f"Dry run failed: {e}")
            self._report_progress(progress)
            return False, f"Dry run failed: {e}", progress
    
    def _execute_migration_atomic(self, pending_migrations: List[Dict[str, Any]],
                                 progress: MigrationProgress) -> Tuple[bool, str, MigrationProgress]:
        """Execute migrations with atomic guarantee using savepoints."""
        backup_path = None
        
        try:
            with self.connection_lock:
                if not self._ensure_connection():
                    return False, "Database connection failed", progress
                
                savepoint_name = f"migration_batch_{int(time.time())}_{random.randint(1000, 9999)}"
                logger.debug(f"Creating savepoint: {savepoint_name}")
                self.connection.execute(f"SAVEPOINT {savepoint_name}")
                
                progress.update(2, "Creating backup")
                self._report_progress(progress)
                
                if self.config.create_backup:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup_filename = f"migration_backup_{timestamp}.db"
                    backup_path = Path(self.config.backup_location) / backup_filename
                    
                    backup_dir = str(backup_path.parent)
                    if not backup_database(self.config.database_path, backup_dir, backup_filename):
                        progress.add_warning("Backup creation failed, proceeding without backup")
                        log_migration_step("Backup creation failed", "WARNING")
                    else:
                        log_migration_step(
                            "Backup created",
                            "INFO",
                            {'backup_path': str(backup_path)}
                        )
                
                progress.update(3, "Preparing migration environment")
                self._report_progress(progress)
                
                applied_in_batch = set()
                
                for i, migration in enumerate(pending_migrations, 4):
                    version = migration['version']
                    description = migration.get('description', 'Unknown')
                    
                    # Refresh applied versions from database before each migration
                    try:
                        cursor = self.connection.cursor()
                        cursor.execute("SELECT version FROM schema_migrations WHERE status='applied'")
                        fresh_applied = [row[0] for row in cursor.fetchall()]
                        applied_versions = set(fresh_applied)
                        logger.debug(f"Fresh applied versions before {version}: {applied_versions}")
                    except Exception as e:
                        logger.warning(f"Failed to refresh applied versions: {e}")
                        applied_versions = set(self._get_applied_versions())
                    
                    all_applied = applied_versions.union(applied_in_batch)
                    logger.debug(f"All applied before {version}: {all_applied}")
                    
                    if version in all_applied:
                        logger.info(f"Migration {version} already applied, skipping")
                        progress.update(i, f"Skipping migration: {version} - {description} (already applied)")
                        self._report_progress(progress)
                        applied_in_batch.add(version)
                        continue
                    
                    for dep in migration.get('dependencies', []):
                        if dep not in all_applied:
                            error_msg = f"Dependency {dep} not satisfied for migration {version}"
                            logger.error(error_msg)
                            progress.add_error(error_msg)
                            
                            try:
                                logger.debug(f"Rolling back to savepoint: {savepoint_name}")
                                self.connection.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                                self.connection.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                            except Exception as sp_e:
                                logger.error(f"Error during rollback: {sp_e}")
                            
                            progress.update(progress.total_steps, f"Migration failed at version {version}")
                            self._report_progress(progress)
                            
                            return False, f"Migration failed at version {version}: {error_msg}", progress
                    
                    progress.update(i, f"Applying migration: {version} - {description}")
                    self._report_progress(progress)
                    
                    success, message = self._apply_single_migration_with_savepoint(migration, savepoint_name)
                    if not success:
                        progress.add_error(f"Migration {version} failed: {message}")
                        
                        try:
                            logger.debug(f"Rolling back to savepoint: {savepoint_name}")
                            self.connection.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                            self.connection.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                        except Exception as sp_e:
                            logger.error(f"Error during rollback: {sp_e}")
                        
                        progress.update(progress.total_steps, f"Migration failed at version {version}")
                        self._report_progress(progress)
                        
                        return False, f"Migration failed at version {version}: {message}", progress
                    
                    applied_in_batch.add(version)
                
                progress.update(progress.total_steps - 2, "Finalizing migration")
                self._report_progress(progress)
                
                progress.update(progress.total_steps - 1, "Validating migration results")
                self._report_progress(progress)
                
                if self.config.validation_level.value != 'none':
                    self._validate_post_migration()
                
                # Safely release savepoint with error handling
                try:
                    logger.debug(f"Releasing savepoint: {savepoint_name}")
                    self.connection.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                except sqlite3.OperationalError as e:
                    if "no such savepoint" in str(e):
                        logger.debug(f"Savepoint {savepoint_name} already released")
                    else:
                        raise
                
                self.connection.commit()
                
                progress.update(progress.total_steps, "Migration completed successfully")
                self._report_progress(progress)
                
                log_data_stats(self.connection)
                
                elapsed_time = format_duration(progress.get_elapsed_time())
                message = f"Successfully applied {len(pending_migrations)} migrations in {elapsed_time}"
                
                log_migration_step(
                    "Migration completed successfully",
                    "INFO",
                    {
                        'applied_count': len(pending_migrations),
                        'elapsed_time': elapsed_time,
                        'backup_path': str(backup_path) if backup_path else None
                    }
                )
                
                return True, message, progress
                
        except Exception as e:
            with self.connection_lock:
                if self.connection:
                    try:
                        self.connection.execute("ROLLBACK")
                    except:
                        pass
            
            progress.add_error(f"Atomic migration failed: {e}")
            
            log_migration_step("Migration failed", "ERROR", {'error': str(e)})
            
            progress.update(progress.total_steps, f"Migration failed: {e}")
            self._report_progress(progress)
            
            return False, f"Migration failed: {e}", progress
    
    def _apply_single_migration_with_savepoint(self, migration_info: Dict[str, Any], 
                                              batch_savepoint: str) -> Tuple[bool, str]:
        """Apply a single migration with nested savepoint for rollback."""
        try:
            file_path = migration_info['file_path']
            version = migration_info['version']
            
            migration_class = load_migration_class(file_path, "Migration")
            if not migration_class:
                return False, f"Failed to load migration class for version {version}"
            
            migration = migration_class()
            
            # Query applied versions directly from database for accurate dependency check
            try:
                cursor = self.connection.cursor()
                cursor.execute("SELECT version FROM schema_migrations WHERE status='applied'")
                applied_versions = {row[0] for row in cursor.fetchall()}
                logger.debug(f"Applied versions from DB for dependency check: {applied_versions}")
            except Exception as e:
                logger.warning(f"Failed to query applied versions: {e}")
                applied_versions = set(self._get_applied_versions())
            
            for dep in migration_info.get('dependencies', []):
                if dep not in applied_versions:
                    logger.error(f"Dependency check failed: {dep} not in {applied_versions}")
                    return False, f"Dependency {dep} not satisfied"
            
            existing = self.version_table.get_migration_by_version(version)
            if existing and existing.get('status') == 'applied':
                logger.info(f"Migration {version} already applied, skipping")
                return True, f"Migration {version} already applied"
            
            if not self.version_table.record_migration_start(
                version, migration_info.get('description', ''),
                file_path
            ):
                return False, f"Failed to record migration start for version {version}"
            
            start_time = time.time()
            
            try:
                migration_savepoint = f"migration_{version}_{int(time.time())}_{random.randint(1000, 9999)}"
                logger.debug(f"Creating migration savepoint: {migration_savepoint}")
                self.connection.execute(f"SAVEPOINT {migration_savepoint}")
                
                if hasattr(migration, 'pre_upgrade') and callable(migration.pre_upgrade):
                    if not migration.pre_upgrade(self.connection):
                        self.connection.execute(f"ROLLBACK TO SAVEPOINT {migration_savepoint}")
                        self.connection.execute(f"RELEASE SAVEPOINT {migration_savepoint}")
                        return False, "Pre-upgrade checks failed"
                
                if not migration.upgrade(self.connection):
                    self.connection.execute(f"ROLLBACK TO SAVEPOINT {migration_savepoint}")
                    self.connection.execute(f"RELEASE SAVEPOINT {migration_savepoint}")
                    return False, "Migration execution failed"
                
                if hasattr(migration, 'post_upgrade') and callable(migration.post_upgrade):
                    if not migration.post_upgrade(self.connection):
                        self.connection.execute(f"ROLLBACK TO SAVEPOINT {migration_savepoint}")
                        self.connection.execute(f"RELEASE SAVEPOINT {migration_savepoint}")
                        return False, "Post-upgrade validation failed"
                
                self.connection.execute(f"RELEASE SAVEPOINT {migration_savepoint}")
                
                duration_ms = int((time.time() - start_time) * 1000)
                
                if not self.version_table.record_migration_complete(version, duration_ms):
                    return False, "Failed to record migration completion"
                
                log_migration_step(
                    f"Migration {version} applied successfully",
                    "INFO",
                    {'version': version, 'duration_ms': duration_ms}
                )
                
                return True, f"Migration {version} applied successfully"
                
            except Exception as e:
                error_str = str(e).lower()
                
                if "already exists" in error_str and version == "1_0_0":
                    logger.warning(f"Migration {version} tables already exist, marking as applied")
                    
                    try:
                        self.connection.execute(f"RELEASE SAVEPOINT {migration_savepoint}")
                    except:
                        pass
                    
                    self.connection.commit()
                    
                    duration_ms = int((time.time() - start_time) * 1000)
                    if not self.version_table.record_migration_complete(version, duration_ms):
                        logger.error(f"Failed to record migration {version} as complete")
                    
                    return True, f"Migration {version} already applied (tables exist)"
                
                try:
                    self.connection.execute(f"ROLLBACK TO SAVEPOINT {migration_savepoint}")
                    self.connection.execute(f"RELEASE SAVEPOINT {migration_savepoint}")
                except:
                    pass
                
                try:
                    self.version_table.record_migration_failed(version, str(e), int((time.time() - start_time) * 1000))
                except:
                    pass
                
                return False, f"Migration execution error: {e}"
            
        except Exception as e:
            return False, f"Migration setup error: {e}"
    
    def _validate_post_migration(self) -> None:
        """Validate database after migration."""
        try:
            log_migration_step("Starting post-migration validation", "INFO")
            
            is_valid, message = validate_database_integrity(self.connection)
            if not is_valid:
                raise Exception(f"Database integrity check failed: {message}")
            
            log_migration_step("Post-migration validation completed successfully", "INFO")
            
        except Exception as e:
            log_migration_step(f"Post-migration validation failed: {e}", "ERROR")
            raise
    
    def rollback_to(self, target_version: str, steps: Optional[int] = None,
                   dry_run: bool = False) -> Tuple[bool, str, MigrationProgress]:
        """
        Rollback to specific version.
        
        Args:
            target_version: Target version to rollback to
            steps: Number of steps to rollback (alternative to target_version)
            dry_run: If True, only simulate rollback
            
        Returns:
            Tuple of (success, message, progress)
        """
        progress = MigrationProgress()
        
        try:
            if not self._ensure_connection():
                progress.total_steps = 1
                progress.update(1, "Database connection failed")
                self._report_progress(progress)
                return False, "Database connection failed", progress
            
            applied_migrations = self._get_applied_migrations_as_dicts()
            available_migrations = self.get_available_migrations()
            
            if not applied_migrations:
                progress.total_steps = 1
                progress.update(1, "No migrations to rollback")
                self._report_progress(progress)
                return True, "No migrations to rollback", progress
            
            applied_versions = [m['version'] for m in applied_migrations if 'version' in m]
            
            target_version_parsed = self._parse_version(target_version)
            migration_map = {m['version']: m for m in available_migrations}
            
            migrations_to_rollback = []
            
            if steps:
                if steps > len(applied_migrations):
                    steps = len(applied_migrations)
                migrations_to_rollback = applied_migrations[-steps:]
            else:
                for migration in reversed(applied_migrations):
                    version = migration.get('version')
                    if version and self._parse_version(version) > target_version_parsed:
                        migrations_to_rollback.append(migration)
                    else:
                        break
            
            if not migrations_to_rollback:
                progress.total_steps = 1
                progress.update(1, "No migrations to rollback")
                self._report_progress(progress)
                return True, "No migrations to rollback", progress
            
            progress.total_steps = len(migrations_to_rollback) + 3
            
            progress.update(1, "Starting rollback process")
            self._report_progress(progress)
            
            if dry_run:
                return self._execute_rollback_dry_run(migrations_to_rollback, migration_map, progress)
            
            return self._execute_rollback_atomic(migrations_to_rollback, migration_map, progress)
            
        except Exception as e:
            progress.add_error(f"Rollback failed: {e}")
            progress.update(progress.total_steps, f"Failed: {e}")
            self._report_progress(progress)
            return False, f"Rollback failed: {e}", progress
    
    def _execute_rollback_dry_run(self, migrations_to_rollback: List[Dict[str, Any]],
                                 migration_map: Dict[str, Dict[str, Any]],
                                 progress: MigrationProgress) -> Tuple[bool, str, MigrationProgress]:
        """Execute dry run simulation for rollback."""
        try:
            for i, migration in enumerate(migrations_to_rollback, 2):
                version = migration.get('version')
                migration_info = migration_map.get(version, {})
                has_down_method = migration_info.get('has_down_method', False)
                
                if has_down_method:
                    progress.update(i, f"Would rollback: {version} - {migration_info.get('description', '')}")
                else:
                    progress.update(i, f"Would mark as failed: {version} (no down method)")
                
                self._report_progress(progress)
                time.sleep(0.1)
            
            progress.update(progress.total_steps, "Rollback dry run completed")
            self._report_progress(progress)
            
            message = f"Dry run: Would rollback {len(migrations_to_rollback)} migrations"
            for migration in migrations_to_rollback:
                version = migration.get('version')
                migration_info = migration_map.get(version, {})
                has_down_method = migration_info.get('has_down_method', False)
                
                if has_down_method:
                    message += f"\n  - {version}: {migration_info.get('description', '')} (rollback)"
                else:
                    message += f"\n  - {version}: {migration_info.get('description', '')} (mark as failed)"
            
            log_migration_step("Rollback dry run completed", "INFO", {'rollback_count': len(migrations_to_rollback)})
            return True, message, progress
            
        except Exception as e:
            progress.add_error(f"Rollback dry run failed: {e}")
            progress.update(progress.total_steps, f"Rollback dry run failed: {e}")
            self._report_progress(progress)
            return False, f"Rollback dry run failed: {e}", progress
    
    def _execute_rollback_atomic(self, migrations_to_rollback: List[Dict[str, Any]],
                                migration_map: Dict[str, Dict[str, Any]],
                                progress: MigrationProgress) -> Tuple[bool, str, MigrationProgress]:
        """Execute rollback with atomic guarantee using savepoints."""
        backup_path = None
        
        try:
            with self.connection_lock:
                if not self._ensure_connection():
                    return False, "Database connection failed", progress
                
                savepoint_name = f"rollback_batch_{int(time.time())}_{random.randint(1000, 9999)}"
                self.connection.execute(f"SAVEPOINT {savepoint_name}")
                
                progress.update(2, "Creating backup")
                self._report_progress(progress)
                
                if self.config.create_backup:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup_filename = f"rollback_backup_{timestamp}.db"
                    backup_path = Path(self.config.backup_location) / backup_filename
                    
                    backup_dir = str(backup_path.parent)
                    if not backup_database(self.config.database_path, backup_dir, backup_filename):
                        progress.add_warning("Backup creation failed, proceeding without backup")
                        log_migration_step("Backup creation failed", "WARNING")
                
                progress.update(3, "Preparing rollback environment")
                self._report_progress(progress)
                
                for i, migration in enumerate(migrations_to_rollback, 4):
                    version = migration.get('version')
                    migration_info = migration_map.get(version, {})
                    
                    progress.update(i, f"Rolling back: {version} - {migration_info.get('description', '')}")
                    self._report_progress(progress)
                    
                    success, message = self._apply_single_rollback_with_savepoint(version, migration_info, savepoint_name)
                    if not success:
                        progress.add_error(f"Rollback {version} failed: {message}")
                        
                        log_migration_step(f"Rollback {version} failed, rolling back batch", "ERROR")
                        self.connection.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                        self.connection.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                        
                        progress.update(progress.total_steps, f"Rollback failed at version {version}")
                        self._report_progress(progress)
                        
                        return False, f"Rollback failed at version {version}: {message}", progress
                
                progress.update(progress.total_steps - 1, "Validating rollback results")
                self._report_progress(progress)
                
                if self.config.validation_level.value != 'none':
                    try:
                        self._validate_post_migration()
                    except Exception as e:
                        logger.warning(f"Post-rollback validation failed: {e}")
                
                self.connection.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                self.connection.commit()
                
                progress.update(progress.total_steps, "Rollback completed successfully")
                self._report_progress(progress)
                
                elapsed_time = format_duration(progress.get_elapsed_time())
                message = f"Successfully rolled back {len(migrations_to_rollback)} migrations in {elapsed_time}"
                
                log_migration_step(
                    "Rollback completed successfully",
                    "INFO",
                    {
                        'rollback_count': len(migrations_to_rollback),
                        'elapsed_time': elapsed_time,
                        'backup_path': str(backup_path) if backup_path else None
                    }
                )
                
                return True, message, progress
                
        except Exception as e:
            with self.connection_lock:
                if self.connection:
                    try:
                        self.connection.execute("ROLLBACK")
                    except:
                        pass
            
            progress.add_error(f"Atomic rollback failed: {e}")
            progress.update(progress.total_steps, f"Rollback failed: {e}")
            self._report_progress(progress)
            
            return False, f"Rollback failed: {e}", progress
    
    def _apply_single_rollback_with_savepoint(self, version: str, migration_info: Dict[str, Any], 
                                            batch_savepoint: str) -> Tuple[bool, str]:
        """Apply a single rollback with nested savepoint."""
        try:
            file_path = migration_info.get('file_path')
            has_down_method = migration_info.get('has_down_method', False)
            
            if not has_down_method or not file_path:
                log_migration_step(
                    f"Migration {version} has no down method, marking as failed",
                    "WARNING"
                )
                try:
                    self.version_table.mark_migration_as_failed(version, "Rolled back - no down method available")
                except:
                    pass
                return True, f"Migration {version} marked as failed"
            
            migration_class = load_migration_class(file_path, "Migration")
            if not migration_class:
                return False, f"Failed to load migration class for version {version}"
            
            migration = migration_class()
            
            if not hasattr(migration, 'downgrade') or not callable(migration.downgrade):
                return False, f"Migration {version} has no downgrade method"
            
            start_time = time.time()
            
            try:
                rollback_savepoint = f"rollback_{version}_{int(time.time())}_{random.randint(1000, 9999)}"
                self.connection.execute(f"SAVEPOINT {rollback_savepoint}")
                
                if hasattr(migration, 'pre_downgrade') and callable(migration.pre_downgrade):
                    if not migration.pre_downgrade(self.connection):
                        self.connection.execute(f"ROLLBACK TO SAVEPOINT {rollback_savepoint}")
                        self.connection.execute(f"RELEASE SAVEPOINT {rollback_savepoint}")
                        return False, "Pre-downgrade checks failed"
                
                if not migration.downgrade(self.connection):
                    self.connection.execute(f"ROLLBACK TO SAVEPOINT {rollback_savepoint}")
                    self.connection.execute(f"RELEASE SAVEPOINT {rollback_savepoint}")
                    return False, "Rollback execution failed"
                
                if hasattr(migration, 'post_downgrade') and callable(migration.post_downgrade):
                    if not migration.post_downgrade(self.connection):
                        self.connection.execute(f"ROLLBACK TO SAVEPOINT {rollback_savepoint}")
                        self.connection.execute(f"RELEASE SAVEPOINT {rollback_savepoint}")
                        return False, "Post-downgrade validation failed"
                
                self.connection.execute(f"RELEASE SAVEPOINT {rollback_savepoint}")
                
                duration_ms = int((time.time() - start_time) * 1000)
                
                try:
                    self.version_table.mark_migration_as_failed(version, "Rolled back successfully", duration_ms)
                except:
                    pass
                
                log_migration_step(
                    f"Migration {version} rolled back successfully",
                    "INFO",
                    {'version': version, 'duration_ms': duration_ms}
                )
                
                return True, f"Migration {version} rolled back successfully"
                
            except Exception as e:
                try:
                    self.connection.execute(f"ROLLBACK TO SAVEPOINT {rollback_savepoint}")
                    self.connection.execute(f"RELEASE SAVEPOINT {rollback_savepoint}")
                except:
                    pass
                return False, f"Rollback execution error: {e}"
            
        except Exception as e:
            return False, f"Rollback setup error: {e}"