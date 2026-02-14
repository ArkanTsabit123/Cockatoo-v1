# src/database/migrations/version/v1_2_0__add_documents_table.py

"""
Migration version 1.2.0

This migration adds a documents table to store file uploads and document
metadata for the document management features of the application.
"""

from database.migrations.base_migration import Migration


class AddDocumentsTableMigration(Migration):
    """
    Migration that creates the documents table.
    
    This migration establishes the document storage system by creating a table
    to track uploaded files, their metadata, and ownership relationships with users.
    """
    
    def __init__(self):
        """Initialize migration with author and tag metadata."""
        super().__init__(
            author="system@cockatoo.local",
            tags=["documents", "files", "uploads"]
        )
    
    def get_version(self) -> str:
        """
        Get the migration version identifier.
        
        Returns:
            String in semantic versioning format: major.minor.patch
        """
        return "1_2_0"
    
    def get_description(self) -> str:
        """
        Get a human-readable description of this migration.
        
        Returns:
            Description string explaining what this migration does
        """
        return "Add documents table for file upload management"
    
    def get_dependencies(self) -> list:
        """
        Get the list of migration versions this migration depends on.
        
        Returns:
            List containing version 1.0.0 as documents belong to users
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
        Apply the migration to create the documents table.
        
        This method creates a comprehensive documents table with fields for
        file metadata, storage paths, and user ownership.
        
        Args:
            connection: Database connection object
            
        Returns:
            True if migration succeeded, False otherwise
        """
        # Create documents table
        connection.execute("""
            CREATE TABLE documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                original_filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                mime_type TEXT,
                storage_type TEXT DEFAULT 'local',
                checksum TEXT,
                description TEXT,
                is_public INTEGER DEFAULT 0,
                download_count INTEGER DEFAULT 0,
                last_accessed TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)
        
        # Create indexes for search and filtering
        connection.execute("CREATE INDEX idx_documents_user_id ON documents(user_id)")
        connection.execute("CREATE INDEX idx_documents_filename ON documents(filename)")
        connection.execute("CREATE INDEX idx_documents_created_at ON documents(created_at)")
        connection.execute("CREATE INDEX idx_documents_mime_type ON documents(mime_type)")
        
        return True
    
    def downgrade(self, connection) -> bool:
        """
        Roll back the migration by dropping the documents table.
        
        Args:
            connection: Database connection object
            
        Returns:
            True if rollback succeeded, False otherwise
        """
        connection.execute("DROP TABLE IF EXISTS documents")
        return True
    
    def estimate_duration(self) -> int:
        """
        Estimate the time required to apply this migration.
        
        Returns:
            Estimated duration in seconds
        """
        return 20
    
    def pre_upgrade(self, connection) -> bool:
        """
        Perform checks before applying the migration.
        
        Args:
            connection: Database connection object
            
        Returns:
            True if pre-upgrade checks pass
        """
        # Verify that the users table exists
        cursor = connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
        if not cursor.fetchone():
            raise Exception("Users table does not exist. Migration 1.0.0 must be applied first.")
        return True