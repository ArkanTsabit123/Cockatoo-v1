# src/database/migrations/version/v2_1_0__add_sessions_table.py

"""
Migration version 2.1.0

This migration adds a sessions table for tracking user sessions and
managing authentication tokens with expiration and security features.
"""

from database.migrations.base_migration import Migration


class AddSessionsTableMigration(Migration):
    """
    Migration that creates the sessions table.
    
    This migration adds session management capabilities to track user logins,
    manage authentication tokens, and implement security features like
    session expiration and revocation.
    """
    
    def __init__(self):
        """Initialize migration with author and tag metadata."""
        super().__init__(
            author="security-team@cockatoo.local",
            tags=["sessions", "authentication", "security"]
        )
    
    def get_version(self) -> str:
        """
        Get the migration version identifier.
        
        Returns:
            String in semantic versioning format
        """
        return "2.1.0"
    
    def get_description(self) -> str:
        """
        Get a human-readable description of this migration.
        
        Returns:
            Description string explaining what this migration does
        """
        return "Add sessions table for user session management"
    
    def get_dependencies(self) -> list:
        """
        Get the list of migration versions this migration depends on.
        
        Returns:
            List containing version 2.0.0 (after column rename)
        """
        return ["2_0_0"]
    
    def is_breaking_change(self) -> bool:
        """
        Determine if this migration contains breaking changes.
        
        Returns:
            False because adding a new table is not a breaking change
        """
        return False
    
    def upgrade(self, connection) -> bool:
        """
        Apply the migration to create the sessions table.
        
        This method creates a comprehensive sessions table for tracking
        user sessions with security and expiration features.
        
        Args:
            connection: Database connection object
            
        Returns:
            True if migration succeeded, False otherwise
        """
        # Create sessions table
        connection.execute("""
            CREATE TABLE sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                session_token TEXT NOT NULL UNIQUE,
                refresh_token TEXT UNIQUE,
                ip_address TEXT,
                user_agent TEXT,
                device_info TEXT,
                expires_at TIMESTAMP NOT NULL,
                last_activity TIMESTAMP,
                is_active INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)
        
        # Create indexes for performance and cleanup
        connection.execute("CREATE INDEX idx_sessions_user_id ON sessions(user_id)")
        connection.execute("CREATE INDEX idx_sessions_token ON sessions(session_token)")
        connection.execute("CREATE INDEX idx_sessions_expires ON sessions(expires_at)")
        connection.execute("CREATE INDEX idx_sessions_active ON sessions(is_active)")
        
        return True
    
    def downgrade(self, connection) -> bool:
        """
        Roll back the migration by dropping the sessions table.
        
        Args:
            connection: Database connection object
            
        Returns:
            True if rollback succeeded, False otherwise
        """
        connection.execute("DROP TABLE IF EXISTS sessions")
        return True
    
    def estimate_duration(self) -> int:
        """
        Estimate the time required to apply this migration.
        
        Returns:
            Estimated duration in seconds
        """
        return 15