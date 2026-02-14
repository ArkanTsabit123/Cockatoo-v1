# src/database/migrations/version//v1_3_0__add_tags_system.py

"""
Migration version 1.3.0

This migration adds a comprehensive tagging system with tags table and
junction tables for many-to-many relationships with documents and other entities.
"""

from database.migrations.base_migration import Migration


class AddTagsSystemMigration(Migration):
    """
    Migration that creates the tagging system.
    
    This migration implements a flexible tagging architecture with a tags table
    and junction tables to support tagging of documents and potentially other
    entities in the future.
    """
    
    def __init__(self):
        """Initialize migration with author and tag metadata."""
        super().__init__(
            author="system@cockatoo.local",
            tags=["tags", "categorization", "metadata"]
        )
    
    def get_version(self) -> str:
        """
        Get the migration version identifier.
        
        Returns:
            String in semantic versioning format: major.minor.patch
        """
        return "1_3_0"
    
    def get_description(self) -> str:
        """
        Get a human-readable description of this migration.
        
        Returns:
            Description string explaining what this migration does
        """
        return "Add tagging system with tags and taggables tables"
    
    def get_dependencies(self) -> list:
        """
        Get the list of migration versions this migration depends on.
        
        Returns:
            List containing versions required for this migration
        """
        return ["1_2_0"]  # Depends on documents table
    
    def is_breaking_change(self) -> bool:
        """
        Determine if this migration contains breaking changes.
        
        Returns:
            False because adding new tables is not a breaking change
        """
        return False
    
    def upgrade(self, connection) -> bool:
        """
        Apply the migration to create the tagging system.
        
        This method creates a tags table and a polymorphic junction table
        that allows tagging of various entity types.
        
        Args:
            connection: Database connection object
            
        Returns:
            True if migration succeeded, False otherwise
        """
        # Create tags table
        connection.execute("""
            CREATE TABLE tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                slug TEXT NOT NULL UNIQUE,
                description TEXT,
                color TEXT,
                usage_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create taggings junction table (polymorphic)
        connection.execute("""
            CREATE TABLE taggings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tag_id INTEGER NOT NULL,
                taggable_id INTEGER NOT NULL,
                taggable_type TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE,
                UNIQUE(tag_id, taggable_id, taggable_type)
            )
        """)
        
        # Create indexes for performance
        connection.execute("CREATE INDEX idx_tags_name ON tags(name)")
        connection.execute("CREATE INDEX idx_tags_slug ON tags(slug)")
        connection.execute("CREATE INDEX idx_taggings_tag_id ON taggings(tag_id)")
        connection.execute("CREATE INDEX idx_taggings_taggable ON taggings(taggable_id, taggable_type)")
        
        return True
    
    def downgrade(self, connection) -> bool:
        """
        Roll back the migration by dropping tagging tables.
        
        Args:
            connection: Database connection object
            
        Returns:
            True if rollback succeeded, False otherwise
        """
        connection.execute("DROP TABLE IF EXISTS taggings")
        connection.execute("DROP TABLE IF EXISTS tags")
        return True
    
    def estimate_duration(self) -> int:
        """
        Estimate the time required to apply this migration.
        
        Returns:
            Estimated duration in seconds
        """
        return 25