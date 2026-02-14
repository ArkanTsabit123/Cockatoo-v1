# src/database/migrations/version/v1_0_0__create_users_table.py

"""
Migration version 1.0.0

This migration creates the initial users table with basic authentication fields.
It establishes the core user management structure required for the application.
"""

from database.migrations.base_migration import Migration


class CreateUsersTableMigration(Migration):
    """
    Migration that creates the users table.
    
    This migration sets up the foundation for user authentication and management
    by creating a users table with essential fields including username, email,
    password hash, and timestamps for tracking.
    """
    
    def get_version(self) -> str:
        """
        Get the migration version identifier.
        
        Returns:
            String in semantic versioning format: major.minor.patch
        """
        return "1_0_0"
    
    def get_description(self) -> str:
        """
        Get a human-readable description of this migration.
        
        Returns:
            Description string explaining what this migration does
        """
        return "Create users table with authentication fields"
    
    def get_dependencies(self) -> list:
        """
        Get the list of migration versions this migration depends on.
        
        Returns:
            Empty list as this is the initial migration with no dependencies
        """
        return []
    
    def is_breaking_change(self) -> bool:
        """
        Determine if this migration contains breaking changes.
        
        Returns:
            False because creating a new table is not a breaking change
        """
        return False
    
    def upgrade(self, connection) -> bool:
        """
        Apply the migration to create the users table.
        
        This method creates the users table structure with all necessary columns,
        constraints, and indexes for optimal performance.
        
        Args:
            connection: Database connection object
            
        Returns:
            True if migration succeeded, False otherwise
        """
        # Create users table with core fields
        connection.execute("""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                is_active INTEGER DEFAULT 1,
                is_superuser INTEGER DEFAULT 0,
                last_login TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for frequently queried columns
        connection.execute("CREATE INDEX idx_users_username ON users(username)")
        connection.execute("CREATE INDEX idx_users_email ON users(email)")
        connection.execute("CREATE INDEX idx_users_created_at ON users(created_at)")
        
        return True
    
    def downgrade(self, connection) -> bool:
        """
        Roll back the migration by dropping the users table.
        
        Args:
            connection: Database connection object
            
        Returns:
            True if rollback succeeded, False otherwise
        """
        connection.execute("DROP TABLE IF EXISTS users")
        return True
    
    def estimate_duration(self) -> int:
        """
        Estimate the time required to apply this migration.
        
        Returns:
            Estimated duration in seconds
        """
        return 10