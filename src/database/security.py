# cockatoo_v1/src/database/security.py

"""
Database security and encryption module.

Provides encryption, password hashing, and security utilities for sensitive
database operations. Includes Fernet encryption for data at rest, PBKDF2 password
hashing, input validation, and secure token generation with industry-standard
cryptographic practices.
"""

import base64
import hashlib
import logging
import secrets
import os
from typing import Optional, Tuple

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.exceptions import InvalidKey

# Setup logging
logger = logging.getLogger(__name__)


class SecurityManager:
    """Security manager for database operations with enhanced security."""
    
    def __init__(self, secret_key: Optional[str] = None) -> None:
        self.secret_key = secret_key or self._load_or_generate_secret_key()
        self.fernet = None
        self._initialize_fernet()
    
    def _load_or_generate_secret_key(self) -> str:
        """Load existing secret key or generate a new one."""
        # Try to load from environment variable
        env_key = os.getenv("COCKATOO_SECRET_KEY")
        if env_key:
            logger.info("Using secret key from environment variable")
            return env_key
        
        # Try to load from file
        try:
            key_file = ".cockatoo_secret.key"
            if os.path.exists(key_file):
                with open(key_file, 'r') as f:
                    key = f.read().strip()
                if len(key) >= 32:
                    logger.info("Loaded secret key from file")
                    return key
        except Exception as e:
            logger.warning(f"Failed to load secret key from file: {e}")
        
        # Generate new key
        new_key = self._generate_secure_secret_key()
        logger.info("Generated new secret key")
        
        # Save to file (optional)
        try:
            with open(".cockatoo_secret.key", 'w') as f:
                f.write(new_key)
            os.chmod(".cockatoo_secret.key", 0o600)  # Secure permissions
            logger.info("Saved secret key to file with secure permissions")
        except Exception as e:
            logger.warning(f"Failed to save secret key to file: {e}")
        
        return new_key
    
    def _generate_secure_secret_key(self) -> str:
        """Generate a secure secret key."""
        try:
            # Use cryptographically secure random bytes
            random_bytes = secrets.token_bytes(32)
            return base64.urlsafe_b64encode(random_bytes).decode()
        except Exception as e:
            logger.error(f"Failed to generate secret key: {e}")
            # Fallback to a deterministic but secure key
            return "cockatoo_secure_key_" + secrets.token_hex(16)
    
    def _initialize_fernet(self) -> None:
        """Initialize Fernet encryption with enhanced security."""
        try:
            # Use a fixed salt for key derivation
            salt = b"cockatoo_salt_2024_secure_v1"
            
            # Use more iterations for better security
            iterations = 600000  # Increased from 100000
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=iterations,
            )
            key = base64.urlsafe_b64encode(kdf.derive(self.secret_key.encode()))
            self.fernet = Fernet(key)
            
            logger.info("Fernet encryption initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Fernet: {e}")
            self.fernet = None
    
    def encrypt_data(self, data: str) -> Optional[str]:
        """Encrypt data with proper error handling."""
        if not self.fernet:
            logger.error("Fernet not initialized, cannot encrypt")
            return None
        
        if not data:
            logger.warning("Attempted to encrypt empty data")
            return None
        
        try:
            encrypted = self.fernet.encrypt(data.encode('utf-8'))
            return base64.urlsafe_b64encode(encrypted).decode()
        except Exception as e:
            logger.error(f"Failed to encrypt data: {e}")
            return None
    
    def decrypt_data(self, encrypted_data: str) -> Optional[str]:
        """Decrypt data with proper error handling."""
        if not self.fernet:
            logger.error("Fernet not initialized, cannot decrypt")
            return None
        
        if not encrypted_data:
            logger.warning("Attempted to decrypt empty data")
            return None
        
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted = self.fernet.decrypt(encrypted_bytes)
            return decrypted.decode('utf-8')
        except InvalidToken:
            logger.error("Invalid token - data may be tampered with")
            return None
        except Exception as e:
            logger.error(f"Failed to decrypt data: {e}")
            return None
    
    def hash_password(self, password: str, salt: Optional[str] = None) -> Optional[str]:
        """Hash password securely with PBKDF2."""
        if not password:
            logger.warning("Attempted to hash empty password")
            return None
        
        try:
            if salt is None:
                salt = secrets.token_hex(32)  # 64 characters
            
            # Use PBKDF2 for password hashing
            iterations = 310000  # OWASP recommended minimum (2021)
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt.encode('utf-8'),
                iterations=iterations,
            )
            hashed = kdf.derive(password.encode('utf-8'))
            
            # Return salt and hash combined with version indicator
            return f"pbkdf2_sha256${iterations}${salt}${base64.b64encode(hashed).decode()}"
        except Exception as e:
            logger.error(f"Failed to hash password: {e}")
            return None
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify password against hash with version detection."""
        if not password or not hashed_password:
            return False
        
        try:
            # Parse the hashed password
            if "$" not in hashed_password:
                logger.error("Invalid hash format")
                return False
            
            parts = hashed_password.split("$")
            if len(parts) != 4 or parts[0] != "pbkdf2_sha256":
                logger.error("Unsupported hash format or algorithm")
                return False
            
            iterations = int(parts[1])
            salt = parts[2]
            stored_hash = parts[3]
            
            # Recreate the hash
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt.encode('utf-8'),
                iterations=iterations,
            )
            
            new_hash = kdf.derive(password.encode('utf-8'))
            new_hash_b64 = base64.b64encode(new_hash).decode()
            
            # Use constant-time comparison to prevent timing attacks
            return secrets.compare_digest(new_hash_b64, stored_hash)
        except Exception as e:
            logger.error(f"Failed to verify password: {e}")
            return False
    
    def generate_api_key(self) -> str:
        """Generate a secure API key."""
        try:
            # Generate URL-safe API key
            api_key = secrets.token_urlsafe(32)
            
            # Add prefix for identification
            return f"ck_{api_key}"
        except Exception as e:
            logger.error(f"Failed to generate API key: {e}")
            return "ck_" + secrets.token_hex(24)
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate a secure random token."""
        try:
            return secrets.token_hex(length)
        except Exception as e:
            logger.error(f"Failed to generate secure token: {e}")
            return secrets.token_hex(16)
    
    def validate_input(self, input_string: str, max_length: int = 1000) -> Tuple[bool, str]:
        """Validate user input to prevent injection attacks."""
        if not input_string:
            return True, ""
        
        # Check length
        if len(input_string) > max_length:
            return False, f"Input too long (max {max_length} characters)"
        
        # Check for dangerous patterns
        dangerous_patterns = [
            (";", "Semicolon not allowed"),
            ("--", "SQL comment not allowed"),
            ("/*", "SQL comment not allowed"),
            ("*/", "SQL comment not allowed"),
            ("' OR '1'='1", "SQL injection attempt"),
            ("' OR '1'='1' --", "SQL injection attempt"),
            ("<script>", "XSS attempt"),
            ("javascript:", "XSS attempt"),
            ("onload=", "XSS attempt"),
            ("onerror=", "XSS attempt"),
        ]
        
        input_lower = input_string.lower()
        for pattern, message in dangerous_patterns:
            if pattern in input_lower:
                return False, f"Potential security issue: {message}"
        
        return True, ""


# Singleton instance
_security_manager: Optional[SecurityManager] = None


def get_security_manager() -> SecurityManager:
    """Get or create security manager instance."""
    global _security_manager
    if _security_manager is None:
        _security_manager = SecurityManager()
    return _security_manager


# Convenience functions
def encrypt_data(data: str, secret_key: Optional[str] = None) -> Optional[str]:
    """Encrypt data."""
    manager = SecurityManager(secret_key) if secret_key else get_security_manager()
    return manager.encrypt_data(data)


def decrypt_data(encrypted_data: str, secret_key: Optional[str] = None) -> Optional[str]:
    """Decrypt data."""
    manager = SecurityManager(secret_key) if secret_key else get_security_manager()
    return manager.decrypt_data(encrypted_data)


def hash_password(password: str, salt: Optional[str] = None) -> Optional[str]:
    """Hash password."""
    manager = get_security_manager()
    return manager.hash_password(password, salt)


def verify_password(password: str, hashed_password: str) -> bool:
    """Verify password."""
    manager = get_security_manager()
    return manager.verify_password(password, hashed_password)


def generate_api_key() -> str:
    """Generate API key."""
    manager = get_security_manager()
    return manager.generate_api_key()


def validate_input(input_string: str, max_length: int = 1000) -> Tuple[bool, str]:
    """Validate user input."""
    manager = get_security_manager()
    return manager.validate_input(input_string, max_length)