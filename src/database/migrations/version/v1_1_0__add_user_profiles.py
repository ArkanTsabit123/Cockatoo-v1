# src/database/migrations/version/v1_1_0__add_user_profiles.py

"""
Migration version 1.1.0

This migration adds a user profiles table to store additional user information
that is not required for core authentication but useful for personalization.
"""

from database.migrations.base_migration import Migration


class AddUserProfilesMigration(Migration):
    """
    Migration that creates the user profiles table.
    
    This migration extends the user management system by adding a profiles table
    that stores extended user information such as full name, bio, avatar, and
    other personal details. Each profile is linked to a user via foreign key.
    """
    
    def __init__(self):
        """Initialize migration with author and tag metadata."""
        super().__init__(
            author="system@cockatoo.local",
            tags=["profile", "user-extension", "personalization"]
        )
    
    def get_version(self) -> str:
        """
        Get the migration version identifier.
        
        Returns:
            String in semantic versioning format: major.minor.patch
        """
        return "1.1.0"
    
    def get_description(self) -> str:
        """
        Get a human-readable description of this migration.
        
        Returns:
            Description string explaining what this migration does
        """
        return "Add user profiles table for extended user information"
    
    def get_dependencies(self) -> list:
        """
        Get the list of migration versions this migration depends on.
        
        Returns:
            List containing version 1.0.0 as users table must exist first
        """
        return ["1_0_0"]
    
    def is_breaking_change(self) -> bool:
        """
        Determine if this migration contains breaking changes.
        
        Returns:
            False because adding a new table is not a breaking change
        """
        return False
    
    def upgrade(self, connection) -> bool:
        """
        Apply the migration to create the user profiles table.
        
        This method creates a profiles table with a foreign key relationship
        to the users table, allowing for one-to-one profile extension.
        
        Args:
            connection: Database connection object
            
        Returns:
            True if migration succeeded, False otherwise
        """
        # Create user profiles table
        connection.execute("""
            CREATE TABLE user_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL UNIQUE,
                full_name TEXT,
                display_name TEXT,
                bio TEXT,
                avatar_url TEXT,
                location TEXT,
                website TEXT,
                birth_date TEXT,
                phone_number TEXT,
                preferences TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)
        
        # Create indexes for performance
        connection.execute("CREATE INDEX idx_profiles_user_id ON user_profiles(user_id)")
        connection.execute("CREATE INDEX idx_profiles_full_name ON user_profiles(full_name)")
        
        return True
    
    def downgrade(self, connection) -> bool:
        """
        Roll back the migration by dropping the user profiles table.
        
        Args:
            connection: Database connection object
            
        Returns:
            True if rollback succeeded, False otherwise
        """
        connection.execute("DROP TABLE IF EXISTS user_profiles")
        return True
    
    def estimate_duration(self) -> int:
        """
        Estimate the time required to apply this migration.
        
        Returns:
            Estimated duration in seconds
        """
        return 15
    
    def get_database_type(self) -> str:
        """
        Specify the database type this migration is designed for.
        
        Returns:
            Database type identifier (generic for SQLite compatibility)
        """
        return "sqlite"