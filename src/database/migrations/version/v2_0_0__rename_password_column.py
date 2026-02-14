# src/database/migrations/version/v2_0_0__rename_password_column.py

"""
Migration version 2.0.0

This migration renames the password_hash column to password_digest in the users table
to align with security best practices and standard naming conventions.
"""

from database.migrations.base_migration import Migration


class RenamePasswordColumnMigration(Migration):
    """
    Migration that renames a column in the users table.
    
    This is a breaking change migration because it modifies an existing column name,
    which requires application code updates to reference the new column name.
    """
    
    def __init__(self):
        """Initialize migration with author and tag metadata."""
        super().__init__(
            author="security-team@cockatoo.local",
            tags=["security", "breaking-change", "schema-update"]
        )
    
    def get_version(self) -> str:
        """
        Get the migration version identifier.
        
        Returns:
            String in semantic versioning format with major version increment
        """
        return "2_0_0"
    
    def get_description(self) -> str:
        """
        Get a human-readable description of this migration.
        
        Returns:
            Description string explaining what this migration does
        """
        return "Rename password_hash column to password_digest in users table"
    
    def get_dependencies(self) -> list:
        """
        Get the list of migration versions this migration depends on.
        
        Returns:
            List containing the latest version before breaking change
        """
        return ["1_3_0"]
    
    def is_breaking_change(self) -> bool:
        """
        Determine if this migration contains breaking changes.
        
        Returns:
            True because renaming a column breaks existing queries
        """
        return True
    
    def upgrade(self, connection) -> bool:
        """
        Apply the migration to rename the password column.
        
        This method renames the password_hash column to password_digest.
        This operation may lock the table and requires application downtime.
        
        Args:
            connection: Database connection object
            
        Returns:
            True if migration succeeded, False otherwise
        """
        # Check if column already renamed
        cursor = connection.cursor()
        cursor.execute("PRAGMA table_info(users)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if "password_digest" in columns:
            return True  # Already renamed
        
        if "password_hash" not in columns:
            raise Exception("Column 'password_hash' not found in users table")
        
        # Rename the column
        connection.execute("ALTER TABLE users RENAME COLUMN password_hash TO password_digest")
        
        return True
    
    def downgrade(self, connection) -> bool:
        """
        Roll back the migration by reverting the column name.
        
        Args:
            connection: Database connection object
            
        Returns:
            True if rollback succeeded, False otherwise
        """
        # Check if original column exists
        cursor = connection.cursor()
        cursor.execute("PRAGMA table_info(users)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if "password_hash" in columns:
            return True  # Already reverted
        
        if "password_digest" not in columns:
            raise Exception("Column 'password_digest' not found in users table")
        
        # Rename back
        connection.execute("ALTER TABLE users RENAME COLUMN password_digest TO password_hash")
        
        return True
    
    def estimate_duration(self) -> int:
        """
        Estimate the time required to apply this migration.
        
        Returns:
            Estimated duration in seconds (longer for large tables)
        """
        return 30
    
    def pre_upgrade(self, connection) -> bool:
        """
        Perform pre-upgrade checks and warnings for breaking change.
        
        Args:
            connection: Database connection object
            
        Returns:
            True if pre-upgrade checks pass
        """
        # Log warning about breaking change
        print("WARNING: This is a breaking change migration.")
        print("The column 'password_hash' will be renamed to 'password_digest'.")
        print("Ensure application code is updated to use the new column name.")
        
        # Verify users table exists and has the column
        cursor = connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
        if not cursor.fetchone():
            raise Exception("Users table does not exist")
        
        cursor.execute("PRAGMA table_info(users)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if "password_hash" not in columns:
            raise Exception("Column 'password_hash' not found in users table")
        
        return True
    
    def post_upgrade(self, connection) -> bool:
        """
        Perform post-upgrade validation.
        
        Args:
            connection: Database connection object
            
        Returns:
            True if post-upgrade validation passes
        """
        # Verify the column was renamed correctly
        cursor = connection.cursor()
        cursor.execute("PRAGMA table_info(users)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if "password_digest" not in columns:
            raise Exception("Column rename verification failed: password_digest not found")
        
        if "password_hash" in columns:
            raise Exception("Column rename verification failed: password_hash still exists")
        
        return True