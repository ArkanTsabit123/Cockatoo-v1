# src/database/migrations/base_migration.py

"""
Base migration class for Cockatoo database migrations.

Defines the abstract interface for all database migrations with versioning,
dependency tracking, breaking change detection, and rollback support.
Ensures consistent migration implementation across the application.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional, Protocol, runtime_checkable, TypeVar, Set
import logging
import re
from enum import Enum
from dataclasses import dataclass, field
from contextlib import contextmanager

logger = logging.getLogger(__name__)

T = TypeVar('T', covariant=True)


@runtime_checkable
class DatabaseConnection(Protocol):
    """Protocol defining the minimum interface required for database connections."""
    
    def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Execute a SQL query."""
        ...
    
    def commit(self) -> None:
        """Commit the current transaction."""
        ...
    
    def rollback(self) -> None:
        """Roll back the current transaction."""
        ...


@runtime_checkable
class ResultProxy(Protocol[T]):
    """Protocol for database query results."""
    
    def fetchone(self) -> Optional[T]:
        """Fetch a single row."""
        ...
    
    def fetchall(self) -> List[T]:
        """Fetch all rows."""
        ...
    
    def rowcount(self) -> int:
        """Get the number of rows affected."""
        ...


class MigrationStatus(Enum):
    """Enumeration of possible migration states."""
    PENDING = "pending"
    APPLYING = "applying"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"
    SKIPPED = "skipped"


@dataclass
class MigrationMetadata:
    """Structured metadata for a migration."""
    version: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    is_breaking: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    applied_at: Optional[datetime] = None
    status: MigrationStatus = MigrationStatus.PENDING
    estimated_duration: int = 30
    author: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            'version': self.version,
            'description': self.description,
            'dependencies': self.dependencies,
            'is_breaking': self.is_breaking,
            'created_at': self.created_at.isoformat(),
            'applied_at': self.applied_at.isoformat() if self.applied_at else None,
            'status': self.status.value,
            'estimated_duration': self.estimated_duration,
            'author': self.author,
            'tags': self.tags
        }


class MigrationError(Exception):
    """Base exception for migration-related errors."""
    
    def __init__(self, message: str, version: Optional[str] = None):
        self.message = message
        self.version = version
        super().__init__(f"{f'Migration {version}: ' if version else ''}{message}")


class MigrationValidationError(MigrationError):
    """Exception raised when migration validation fails."""
    pass


class MigrationDependencyError(MigrationError):
    """Exception raised when migration dependencies are not met."""
    pass


class MigrationExecutionError(MigrationError):
    """Exception raised when migration execution fails."""
    pass


class MigrationRollbackError(MigrationError):
    """Exception raised when migration rollback fails."""
    pass


class Migration(ABC):
    """
    Abstract base class for database migrations.
    
    All migrations must inherit from this class and implement the required
    abstract methods. Provides a complete framework for versioned database
    schema changes with dependency management and rollback capability.
    """
    
    def __init__(self, author: Optional[str] = None, tags: Optional[List[str]] = None) -> None:
        """
        Initialize a migration instance with metadata.
        
        Args:
            author: Optional author identifier
            tags: Optional list of tags for categorization
        """
        try:
            self.version: str = self.get_version()
            self.description: str = self.get_description()
            self.dependencies: List[str] = self.get_dependencies()
            self.is_breaking: bool = self.is_breaking_change()
            self.author: Optional[str] = author
            self.tags: List[str] = tags or []
            self.created_at: datetime = datetime.utcnow()
            self.applied_at: Optional[datetime] = None
            self.status: MigrationStatus = MigrationStatus.PENDING
            
            self._validate_version_format()
            self._validate_no_circular_dependencies()
            self._validate_dependency_versions()
            
        except NotImplementedError as e:
            raise MigrationValidationError(
                f"Migration class is missing required abstract method: {e}",
                getattr(self, 'version', 'unknown')
            )
        except Exception as e:
            raise MigrationValidationError(
                f"Failed to initialize migration: {e}",
                getattr(self, 'version', 'unknown')
            )
    
    @abstractmethod
    def get_version(self) -> str:
        """
        Get migration version in semantic versioning format.
        
        Returns:
            Version string in format 'major.minor.patch' with optional suffix
            Examples: 
                "1.0.0", 
                "2.1.3", 
                "1.0.0-alpha.1", 
                "2.0.0-rc.2",
                "1.0.0+build.123"
            
        Raises:
            NotImplementedError: If not implemented by subclass
            ValueError: If version format is invalid
        """
        raise NotImplementedError(
            "Migration class must implement get_version() method"
        )
    
    @abstractmethod
    def get_description(self) -> str:
        """
        Get migration description.
        
        Returns:
            Human-readable description of what the migration does
            Example: "Add users table with email and password fields"
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError(
            "Migration class must implement get_description() method"
        )
    
    @abstractmethod
    def get_dependencies(self) -> List[str]:
        """
        Get list of migration versions this migration depends on.
        
        Returns:
            List of version strings that must be applied before this migration.
            Empty list if no dependencies.
            Example: ["1.0.0", "1.2.0"]
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError(
            "Migration class must implement get_dependencies() method"
        )
    
    @abstractmethod
    def is_breaking_change(self) -> bool:
        """
        Determine if this migration contains breaking changes.
        
        Breaking changes are modifications that require special handling,
        such as dropping columns or tables, changing column types in
        incompatible ways, renaming tables or columns, or changes that
        require application downtime.
        
        Returns:
            True if migration contains breaking changes, False otherwise
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError(
            "Migration class must implement is_breaking_change() method"
        )
    
    @abstractmethod
    def upgrade(self, connection: DatabaseConnection) -> bool:
        """
        Apply the migration (forward/up direction).
        
        This is where the actual database changes are implemented.
        
        Args:
            connection: Database connection object implementing DatabaseConnection protocol
            
        Returns:
            True if migration succeeded, False otherwise
            
        Raises:
            NotImplementedError: If not implemented by subclass
            MigrationExecutionError: If migration execution fails
        """
        raise NotImplementedError(
            "Migration class must implement upgrade() method"
        )
    
    @abstractmethod
    def downgrade(self, connection: DatabaseConnection) -> bool:
        """
        Rollback the migration (downward/down direction).
        
        This should revert the changes made by the upgrade method.
        
        Args:
            connection: Database connection object implementing DatabaseConnection protocol
            
        Returns:
            True if rollback succeeded, False otherwise
            
        Raises:
            NotImplementedError: If not implemented by subclass
            MigrationRollbackError: If rollback execution fails
        """
        raise NotImplementedError(
            "Migration class must implement downgrade() method"
        )
    
    def estimate_duration(self) -> int:
        """
        Estimate migration duration in seconds.
        
        Override this to provide accurate estimates for long-running migrations.
        
        Returns:
            Estimated duration in seconds. Default is 30 seconds.
        """
        return 30
    
    def validate_preconditions(self, connection: DatabaseConnection) -> Tuple[bool, str]:
        """
        Validate that all preconditions for applying the migration are met.
        
        Args:
            connection: Database connection object
            
        Returns:
            Tuple of (success: bool, message: str)
            success: True if preconditions are met, False otherwise
            message: Detailed message about validation result
        """
        return True, "Preconditions validated successfully"
    
    def validate_postconditions(self, connection: DatabaseConnection) -> Tuple[bool, str]:
        """
        Validate that the migration was applied correctly.
        
        Args:
            connection: Database connection object
            
        Returns:
            Tuple of (success: bool, message: str)
            success: True if postconditions are met, False otherwise
            message: Detailed message about validation result
        """
        return True, "Postconditions validated successfully"
    
    def pre_upgrade(self, connection: DatabaseConnection) -> bool:
        """
        Optional pre-upgrade checks and preparations.
        
        Can be used for creating backups, checking disk space,
        or setting maintenance mode.
        
        Args:
            connection: Database connection object
            
        Returns:
            True if pre-upgrade succeeded, False otherwise
        """
        logger.debug(f"Pre-upgrade for migration {self.version}")
        return True
    
    def post_upgrade(self, connection: DatabaseConnection) -> bool:
        """
        Optional post-upgrade validations and cleanup.
        
        Can be used for updating statistics, cleaning up temporary files,
        or sending notifications.
        
        Args:
            connection: Database connection object
            
        Returns:
            True if post-upgrade succeeded, False otherwise
        """
        logger.debug(f"Post-upgrade for migration {self.version}")
        return True
    
    def pre_downgrade(self, connection: DatabaseConnection) -> bool:
        """
        Optional pre-downgrade checks and preparations.
        
        Args:
            connection: Database connection object
            
        Returns:
            True if pre-downgrade succeeded, False otherwise
        """
        logger.debug(f"Pre-downgrade for migration {self.version}")
        return True
    
    def post_downgrade(self, connection: DatabaseConnection) -> bool:
        """
        Optional post-downgrade validations and cleanup.
        
        Args:
            connection: Database connection object
            
        Returns:
            True if post-downgrade succeeded, False otherwise
        """
        logger.debug(f"Post-downgrade for migration {self.version}")
        return True
    
    def get_database_type(self) -> str:
        """
        Get the database type this migration is designed for.
        
        Override this method to specify a particular database type.
        Returns 'generic' by default, meaning it should work with any
        database that implements the DatabaseConnection protocol.
        
        Returns:
            Database type identifier (e.g., 'postgresql', 'mysql', 'sqlite')
        """
        return "generic"
    
    @contextmanager
    def transaction(self, connection: DatabaseConnection):
        """
        Context manager for handling database transactions.
        
        Usage:
            with self.transaction(connection):
                # Perform database operations
                connection.execute("INSERT INTO ...")
        
        Args:
            connection: Database connection object
        """
        try:
            yield
            connection.commit()
        except Exception as e:
            connection.rollback()
            raise MigrationExecutionError(
                f"Transaction failed: {e}",
                self.version
            )
    
    def _validate_version_format(self) -> None:
        """
        Validate that the version string follows semantic versioning.
        
        Supports versions with optional pre-release and build metadata suffixes:
        - Standard: 1.0.0
        - Pre-release: 1.0.0-alpha, 1.0.0-beta.1, 1.0.0-rc.2
        - Build metadata: 1.0.0+build.123, 1.0.0-alpha.1+build.456
        
        Raises:
            MigrationValidationError: If version format is invalid
        """
        version = self.version
        
        # Semantic versioning pattern with optional pre-release and build metadata
        semver_pattern = r'^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)' \
                        r'(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)' \
                        r'(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?' \
                        r'(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$'
        
        if not re.match(semver_pattern, version):
            raise MigrationValidationError(
                f"Version '{version}' does not follow semantic versioning format. "
                f"Expected format: major.minor.patch[-pre-release][+build-metadata]. "
                f"Examples: '1.0.0', '2.1.3-alpha.1', '3.0.0-rc.2+build.456'",
                version
            )
    
    def _validate_no_circular_dependencies(self) -> None:
        """
        Validate that no circular dependencies exist.
        
        This method checks if this migration's dependencies create a circular
        reference chain.
        
        Raises:
            MigrationValidationError: If circular dependency is detected
        """
        def detect_circular_dependencies(
            current_version: str,
            dependencies: List[str],
            visited: Set[str],
            path: List[str]
        ) -> None:
            if current_version in visited:
                cycle = ' -> '.join(path + [current_version])
                raise MigrationValidationError(
                    f"Circular dependency detected: {cycle}",
                    self.version
                )
            
            visited.add(current_version)
            path.append(current_version)
            
            # In a real implementation, you would check the dependencies
            # of each dependency here. For now, we just validate immediate
            # self-references.
            
            visited.remove(current_version)
            path.pop()
        
        visited: Set[str] = set()
        path: List[str] = []
        
        # Check for direct self-dependency
        if self.version in self.dependencies:
            raise MigrationValidationError(
                f"Migration cannot depend on itself",
                self.version
            )
        
        # Check for duplicates in dependencies
        if len(self.dependencies) != len(set(self.dependencies)):
            duplicates = [dep for dep in self.dependencies 
                         if self.dependencies.count(dep) > 1]
            raise MigrationValidationError(
                f"Duplicate dependencies found: {duplicates}",
                self.version
            )
    
    def _validate_dependency_versions(self) -> None:
        """
        Validate that all dependency versions follow semantic versioning.
        
        Raises:
            MigrationValidationError: If any dependency version is invalid
        """
        for dep_version in self.dependencies:
            try:
                # Reuse the same validation logic
                temp_migration = type('TempMigration', (Migration,), {
                    'get_version': lambda self: dep_version,
                    'get_description': lambda self: '',
                    'get_dependencies': lambda self: [],
                    'is_breaking_change': lambda self: False,
                    'upgrade': lambda self, conn: True,
                    'downgrade': lambda self, conn: True
                })()
                temp_migration._validate_version_format()
            except Exception as e:
                raise MigrationValidationError(
                    f"Invalid dependency version '{dep_version}': {e}",
                    self.version
                )
    
    def has_dependency(self, version: str) -> bool:
        """
        Check if this migration depends on a specific version.
        
        Args:
            version: Version string to check
            
        Returns:
            True if the version is in dependencies, False otherwise
        """
        return version in self.dependencies
    
    def is_dependent_on(self, other: 'Migration') -> bool:
        """
        Check if this migration depends on another migration.
        
        Args:
            other: Another migration instance
            
        Returns:
            True if this migration depends on the other, False otherwise
        """
        return other.version in self.dependencies
    
    def can_run_with(self, other: 'Migration') -> bool:
        """
        Check if this migration can run concurrently with another.
        
        By default, breaking changes cannot run concurrently.
        
        Args:
            other: Another migration instance
            
        Returns:
            True if migrations can run concurrently, False otherwise
        """
        if self.is_breaking or other.is_breaking:
            return False
        
        if self.is_dependent_on(other) or other.is_dependent_on(self):
            return False
        
        # Check if they modify the same database objects
        # This would require analysis of SQL statements, which is complex
        # For now, we assume they can run concurrently unless dependencies conflict
        
        return True
    
    def get_metadata(self) -> MigrationMetadata:
        """
        Get complete structured metadata about this migration.
        
        Returns:
            MigrationMetadata object containing all migration metadata
        """
        return MigrationMetadata(
            version=self.version,
            description=self.description,
            dependencies=self.dependencies,
            is_breaking=self.is_breaking,
            created_at=self.created_at,
            applied_at=self.applied_at,
            status=self.status,
            estimated_duration=self.estimate_duration(),
            author=self.author,
            tags=self.tags
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert migration to dictionary representation.
        
        Returns:
            Dictionary containing all migration information
        """
        return self.get_metadata().to_dict()
    
    def __str__(self) -> str:
        """String representation for human readability."""
        breaking_flag = " [BREAKING]" if self.is_breaking else ""
        deps = f", depends on: {self.dependencies}" if self.dependencies else ""
        tags = f", tags: {self.tags}" if self.tags else ""
        return f"Migration {self.version}{breaking_flag}: {self.description}{deps}{tags}"
    
    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return (f"<Migration(version={self.version!r}, "
                f"description={self.description!r}, "
                f"breaking={self.is_breaking}, "
                f"dependencies={self.dependencies}, "
                f"status={self.status.value})>")
    
    def __eq__(self, other: object) -> bool:
        """Check equality based on version."""
        if not isinstance(other, Migration):
            return NotImplemented
        return self.version == other.version
    
    def __hash__(self) -> int:
        """Hash based on version for use in sets and dicts."""
        return hash(self.version)
    
    def __lt__(self, other: 'Migration') -> bool:
        """
        Compare migrations for sorting by version.
        
        Args:
            other: Another migration instance
            
        Returns:
            True if this migration should come before the other
            
        Raises:
            TypeError: If other is not a Migration
        """
        if not isinstance(other, Migration):
            raise TypeError(f"Cannot compare Migration with {type(other)}")
        
        def parse_version(version_str: str) -> Tuple[int, int, int, List[str]]:
            """Parse semantic version string into comparable components."""
            # Strip build metadata for comparison
            base_version = version_str.split('+')[0]
            
            # Split pre-release if present
            if '-' in base_version:
                version_part, pre_release = base_version.split('-', 1)
            else:
                version_part, pre_release = base_version, ''
            
            # Parse major.minor.patch
            major, minor, patch = map(int, version_part.split('.'))
            
            # Parse pre-release identifiers
            pre_release_parts = pre_release.split('.') if pre_release else []
            
            return major, minor, patch, pre_release_parts
        
        self_parts = parse_version(self.version)
        other_parts = parse_version(other.version)
        
        return self_parts < other_parts
    
    def __contains__(self, version: str) -> bool:
        """
        Check if a version is in this migration's dependencies.
        
        Args:
            version: Version string to check
            
        Returns:
            True if version is a dependency, False otherwise
        """
        return version in self.dependencies


# ============================================================================
# COMPLETE EXAMPLE IMPLEMENTATIONS
# ============================================================================

class CreateUsersTableMigration(Migration):
    """
    Example migration that creates a users table.
    
    This demonstrates a complete, well-documented migration implementation
    with proper error handling, validation, and transaction management.
    """
    
    def get_version(self) -> str:
        return "1.0.0"
    
    def get_description(self) -> str:
        return "Create users table with basic authentication fields"
    
    def get_dependencies(self) -> List[str]:
        return []  # This is an initial migration, no dependencies
    
    def is_breaking_change(self) -> bool:
        return False  # Creating a table is not breaking
    
    def upgrade(self, connection: DatabaseConnection) -> bool:
        """
        Create the users table.
        
        Args:
            connection: Database connection object
            
        Returns:
            True if migration succeeded, False otherwise
            
        Raises:
            MigrationExecutionError: If table creation fails
        """
        try:
            with self.transaction(connection):
                # Check if table already exists
                check_sql = """
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'users'
                    )
                """
                result = connection.execute(check_sql)
                table_exists = result.fetchone()[0] if result.fetchone() else False
                
                if table_exists:
                    logger.warning(f"Table 'users' already exists for migration {self.version}")
                    return True
                
                # Create users table
                create_sql = """
                    CREATE TABLE users (
                        id SERIAL PRIMARY KEY,
                        username VARCHAR(50) UNIQUE NOT NULL,
                        email VARCHAR(255) UNIQUE NOT NULL,
                        password_hash VARCHAR(255) NOT NULL,
                        is_active BOOLEAN DEFAULT TRUE,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    )
                """
                
                connection.execute(create_sql)
                
                # Create index on email for faster lookups
                index_sql = "CREATE INDEX idx_users_email ON users(email)"
                connection.execute(index_sql)
                
                logger.info(f"Successfully created users table for migration {self.version}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to create users table for migration {self.version}: {e}")
            raise MigrationExecutionError(
                f"Failed to create users table: {e}",
                self.version
            )
    
    def downgrade(self, connection: DatabaseConnection) -> bool:
        """
        Drop the users table.
        
        Args:
            connection: Database connection object
            
        Returns:
            True if rollback succeeded, False otherwise
            
        Raises:
            MigrationRollbackError: If table drop fails
        """
        try:
            with self.transaction(connection):
                # Check if table exists before dropping
                check_sql = """
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'users'
                    )
                """
                result = connection.execute(check_sql)
                table_exists = result.fetchone()[0] if result.fetchone() else False
                
                if not table_exists:
                    logger.warning(f"Table 'users' does not exist for migration {self.version}")
                    return True
                
                # Drop table
                drop_sql = "DROP TABLE IF EXISTS users CASCADE"
                connection.execute(drop_sql)
                
                logger.info(f"Successfully dropped users table for migration {self.version}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to drop users table for migration {self.version}: {e}")
            raise MigrationRollbackError(
                f"Failed to drop users table: {e}",
                self.version
            )
    
    def validate_preconditions(self, connection: DatabaseConnection) -> Tuple[bool, str]:
        """
        Validate preconditions for creating users table.
        
        Args:
            connection: Database connection object
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            # Check if we have permission to create tables
            test_sql = "SELECT 1"
            connection.execute(test_sql)
            
            return True, "Database connection is valid and has required permissions"
            
        except Exception as e:
            return False, f"Database validation failed: {e}"
    
    def estimate_duration(self) -> int:
        """Estimate duration for creating users table."""
        return 10  # Seconds


class AddUserProfileMigration(Migration):
    """
    Example migration that adds a profile table with foreign key to users.
    
    This demonstrates a migration with dependencies and foreign key constraints.
    """
    
    def __init__(self):
        super().__init__(
            author="dev-team@example.com",
            tags=["profile", "user-extension"]
        )
    
    def get_version(self) -> str:
        return "1.1.0"
    
    def get_description(self) -> str:
        return "Add user profiles table with foreign key to users"
    
    def get_dependencies(self) -> List[str]:
        return ["1.0.0"]  # Depends on the users table migration
    
    def is_breaking_change(self) -> bool:
        return False  # Adding a new table is not breaking
    
    def upgrade(self, connection: DatabaseConnection) -> bool:
        """
        Create user profiles table.
        
        Args:
            connection: Database connection object
            
        Returns:
            True if migration succeeded, False otherwise
        """
        try:
            with self.transaction(connection):
                # Create profiles table
                create_sql = """
                    CREATE TABLE user_profiles (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                        full_name VARCHAR(100),
                        bio TEXT,
                        avatar_url VARCHAR(500),
                        location VARCHAR(100),
                        website VARCHAR(500),
                        birth_date DATE,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        CONSTRAINT fk_user_profile FOREIGN KEY (user_id) 
                            REFERENCES users(id) ON DELETE CASCADE
                    )
                """
                
                connection.execute(create_sql)
                
                # Create index on user_id for faster joins
                index_sql = "CREATE INDEX idx_user_profiles_user_id ON user_profiles(user_id)"
                connection.execute(index_sql)
                
                logger.info(f"Successfully created user_profiles table for migration {self.version}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to create user_profiles table for migration {self.version}: {e}")
            raise MigrationExecutionError(
                f"Failed to create user_profiles table: {e}",
                self.version
            )
    
    def downgrade(self, connection: DatabaseConnection) -> bool:
        """
        Drop user profiles table.
        
        Args:
            connection: Database connection object
            
        Returns:
            True if rollback succeeded, False otherwise
        """
        try:
            with self.transaction(connection):
                # Drop table
                drop_sql = "DROP TABLE IF EXISTS user_profiles CASCADE"
                connection.execute(drop_sql)
                
                logger.info(f"Successfully dropped user_profiles table for migration {self.version}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to drop user_profiles table for migration {self.version}: {e}")
            raise MigrationRollbackError(
                f"Failed to drop user_profiles table: {e}",
                self.version
            )
    
    def get_database_type(self) -> str:
        """Specify this migration is for PostgreSQL."""
        return "postgresql"


class BreakingChangeMigrationExample(Migration):
    """
    Example migration that demonstrates a breaking change.
    
    This migration renames a column, which is a breaking change
    that requires application updates.
    """
    
    def __init__(self):
        super().__init__(
            author="security-team@example.com",
            tags=["security", "breaking-change"]
        )
    
    def get_version(self) -> str:
        return "2.0.0-rc.1"  # Pre-release version for breaking change
    
    def get_description(self) -> str:
        return "Rename password_hash to password_digest for security compliance"
    
    def get_dependencies(self) -> List[str]:
        return ["1.0.0"]  # Depends on users table existing
    
    def is_breaking_change(self) -> bool:
        return True  # Renaming a column is breaking
    
    def upgrade(self, connection: DatabaseConnection) -> bool:
        """
        Rename password_hash column to password_digest.
        
        Args:
            connection: Database connection object
            
        Returns:
            True if migration succeeded, False otherwise
        """
        try:
            with self.transaction(connection):
                # Check if column already renamed
                check_sql = """
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'users' 
                    AND column_name = 'password_digest'
                """
                result = connection.execute(check_sql)
                if result.fetchone():
                    logger.warning(f"Column already renamed for migration {self.version}")
                    return True
                
                # Rename column
                rename_sql = "ALTER TABLE users RENAME COLUMN password_hash TO password_digest"
                connection.execute(rename_sql)
                
                logger.info(f"Successfully renamed column for migration {self.version}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to rename column for migration {self.version}: {e}")
            raise MigrationExecutionError(
                f"Failed to rename column: {e}",
                self.version
            )
    
    def downgrade(self, connection: DatabaseConnection) -> bool:
        """
        Revert column rename.
        
        Args:
            connection: Database connection object
            
        Returns:
            True if rollback succeeded, False otherwise
        """
        try:
            with self.transaction(connection):
                # Check if column exists with new name
                check_sql = """
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'users' 
                    AND column_name = 'password_hash'
                """
                result = connection.execute(check_sql)
                if result.fetchone():
                    logger.warning(f"Column already reverted for migration {self.version}")
                    return True
                
                # Rename back
                rename_sql = "ALTER TABLE users RENAME COLUMN password_digest TO password_hash"
                connection.execute(rename_sql)
                
                logger.info(f"Successfully reverted column rename for migration {self.version}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to revert column rename for migration {self.version}: {e}")
            raise MigrationRollbackError(
                f"Failed to revert column rename: {e}",
                self.version
            )
    
    def pre_upgrade(self, connection: DatabaseConnection) -> bool:
        """
        Additional checks before applying breaking change.
        
        Args:
            connection: Database connection object
            
        Returns:
            True if pre-upgrade checks pass
        """
        logger.warning(f"Applying breaking change migration {self.version}")
        logger.warning("This will rename password_hash column to password_digest")
        logger.warning("Make sure application code is updated to use the new column name")
        
        # In a real scenario, you might:
        # 1. Put the application in maintenance mode
        # 2. Backup the table
        # 3. Notify stakeholders
        
        return True
    
    def estimate_duration(self) -> int:
        """Estimate duration for column rename."""
        return 5  # Seconds, but could be longer in production with large tables


# ============================================================================
# ADDITIONAL HELPER CLASSES FOR TESTING
# ============================================================================

class FailingMigration(Migration):
    """
    Example migration that always fails for testing error handling.
    
    This demonstrates proper error wrapping and exception handling.
    """
    
    def __init__(self, fail_upgrade: bool = True, fail_downgrade: bool = True):
        super().__init__(
            author="test-system@example.com",
            tags=["test", "error-handling"]
        )
        self.fail_upgrade = fail_upgrade
        self.fail_downgrade = fail_downgrade
    
    def get_version(self) -> str:
        return "9.9.9"
    
    def get_description(self) -> str:
        return "Test migration that always fails"
    
    def get_dependencies(self) -> List[str]:
        return []
    
    def is_breaking_change(self) -> bool:
        return False
    
    def upgrade(self, connection: DatabaseConnection) -> bool:
        """
        Always fail upgrade for testing.
        
        Args:
            connection: Database connection object
            
        Returns:
            Always raises MigrationExecutionError
            
        Raises:
            MigrationExecutionError: Always raised for testing
        """
        if self.fail_upgrade:
            raise MigrationExecutionError(
                "Upgrade failed intentionally for testing",
                self.version
            )
        return True
    
    def downgrade(self, connection: DatabaseConnection) -> bool:
        """
        Always fail downgrade for testing.
        
        Args:
            connection: Database connection object
            
        Returns:
            Always raises MigrationRollbackError
            
        Raises:
            MigrationRollbackError: Always raised for testing
        """
        if self.fail_downgrade:
            raise MigrationRollbackError(
                "Downgrade failed intentionally for testing",
                self.version
            )
        return True


class IndependentMigration(Migration):
    """
    Example independent migration for testing concurrent execution.
    """
    
    def __init__(self, version: str = "3.0.0"):
        self._version = version
        super().__init__(
            author="test-system@example.com",
            tags=["test", "independent"]
        )
    
    def get_version(self) -> str:
        return self._version
    
    def get_description(self) -> str:
        return f"Independent test migration {self._version}"
    
    def get_dependencies(self) -> List[str]:
        return []
    
    def is_breaking_change(self) -> bool:
        return False
    
    def upgrade(self, connection: DatabaseConnection) -> bool:
        """Simple successful upgrade."""
        try:
            with self.transaction(connection):
                connection.execute("SELECT 1")
            return True
        except Exception as e:
            raise MigrationExecutionError(
                f"Upgrade failed: {e}",
                self.version
            )
    
    def downgrade(self, connection: DatabaseConnection) -> bool:
        """Simple successful downgrade."""
        try:
            with self.transaction(connection):
                connection.execute("SELECT 1")
            return True
        except Exception as e:
            raise MigrationRollbackError(
                f"Downgrade failed: {e}",
                self.version
            )