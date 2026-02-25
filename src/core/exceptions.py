# cockatoo_v1/src/core/exceptions.py
"""
Custom exceptions for Cockatoo_V1 application.

This module provides a comprehensive hierarchy of exceptions for
the configuration management system and application components.
"""

from typing import Optional, Any, Dict, List, Union
from pathlib import Path


class CockatooError(Exception):
    """
    Base exception for all Cockatoo errors.
    
    All custom exceptions in the application should inherit from this class
    to ensure consistent error handling.
    """
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """
        Initialize CockatooError.
        
        Args:
            message: Human-readable error message
            details: Additional error details (structured data)
            cause: Original exception that caused this error
        """
        self.message = message
        self.details = details or {}
        self.cause = cause
        super().__init__(message)
        
        # If there's a cause, chain the exceptions
        if cause:
            self.__cause__ = cause
    
    def __str__(self) -> str:
        """Return string representation with details if available."""
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} [{details_str}]"
        return self.message
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
            "cause": str(self.cause) if self.cause else None,
        }


# ============================================================================
# Configuration Base Exceptions
# ============================================================================

class ConfigurationError(CockatooError):
    """
    Base exception for configuration-related errors.
    
    This exception is raised when there are issues with configuration
    loading, validation, or access.
    """
    pass


# ============================================================================
# File-Related Configuration Exceptions
# ============================================================================

class ConfigFileNotFoundError(ConfigurationError):
    """
    Raised when a configuration file cannot be found.
    
    This typically occurs when attempting to load a config file
    that doesn't exist at the specified path.
    """
    
    def __init__(self, path: Union[str, Path], **kwargs):
        """
        Initialize ConfigFileNotFoundError.
        
        Args:
            path: Path to the missing configuration file
            **kwargs: Additional details
        """
        path_str = str(path)
        message = f"Configuration file not found: {path_str}"
        details = {"path": path_str, **kwargs}
        super().__init__(message, details)


class ConfigFilePermissionError(ConfigurationError):
    """
    Raised when there are permission issues accessing a configuration file.
    
    This occurs when the application doesn't have read/write permissions
    for the config file or its directory.
    """
    
    def __init__(
        self,
        path: Union[str, Path],
        operation: str = "access",
        **kwargs
    ):
        """
        Initialize ConfigFilePermissionError.
        
        Args:
            path: Path to the configuration file
            operation: Operation that failed (read/write/access)
            **kwargs: Additional details
        """
        path_str = str(path)
        message = f"Permission denied {operation} config file: {path_str}"
        details = {"path": path_str, "operation": operation, **kwargs}
        super().__init__(message, details)


class ConfigFormatError(ConfigurationError):
    """
    Raised when a configuration file has invalid format.
    
    This occurs when the config file contains malformed YAML, JSON,
    or TOML content.
    """
    
    def __init__(
        self,
        path: Union[str, Path],
        format_type: str,
        parse_error: str,
        **kwargs
    ):
        """
        Initialize ConfigFormatError.
        
        Args:
            path: Path to the configuration file
            format_type: Expected format (yaml/json/toml)
            parse_error: Detailed parse error message
            **kwargs: Additional details
        """
        path_str = str(path)
        message = f"Invalid {format_type.upper()} format in config file: {path_str}"
        details = {
            "path": path_str,
            "format": format_type,
            "parse_error": parse_error,
            **kwargs
        }
        super().__init__(message, details)


class ConfigVersionMismatchError(ConfigurationError):
    """
    Raised when the config file version doesn't match the expected version.
    
    This occurs when trying to load a config file created with a different
    version of the application.
    """
    
    def __init__(
        self,
        expected_version: str,
        actual_version: str,
        path: Optional[Union[str, Path]] = None,
        **kwargs
    ):
        """
        Initialize ConfigVersionMismatchError.
        
        Args:
            expected_version: Expected configuration version
            actual_version: Actual version found in config file
            path: Path to the configuration file (optional)
            **kwargs: Additional details
        """
        message = f"Config version mismatch: expected {expected_version}, got {actual_version}"
        details = {
            "expected_version": expected_version,
            "actual_version": actual_version,
            **kwargs
        }
        if path:
            details["path"] = str(path)
        super().__init__(message, details)


# ============================================================================
# Validation-Related Configuration Exceptions
# ============================================================================

class ConfigValidationError(ConfigurationError):
    """
    Raised when configuration validation fails.
    
    This is the base exception for all validation-related errors.
    """
    
    def __init__(
        self,
        message: str,
        errors: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize ConfigValidationError.
        
        Args:
            message: Error message
            errors: List of specific validation errors
            **kwargs: Additional details
        """
        details = {"validation_errors": errors or [], **kwargs}
        super().__init__(message, details)


class ConfigKeyError(ConfigurationError):
    """
    Raised when accessing an invalid configuration key.
    
    This occurs when trying to get or set a configuration value
    using a key that doesn't exist.
    """
    
    def __init__(
        self,
        key: str,
        available_keys: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize ConfigKeyError.
        
        Args:
            key: The invalid key that was accessed
            available_keys: List of available keys (optional)
            **kwargs: Additional details
        """
        message = f"Invalid configuration key: {key}"
        details = {"key": key, **kwargs}
        if available_keys:
            details["available_keys"] = available_keys
        super().__init__(message, details)


class ConfigValueError(ConfigValidationError):
    """
    Raised when a configuration value is invalid.
    
    This occurs when a value doesn't meet validation criteria
    (type mismatch, out of range, invalid format, etc.).
    """
    
    def __init__(
        self,
        key: str,
        value: Any,
        reason: str,
        expected_type: Optional[str] = None,
        valid_range: Optional[tuple] = None,
        allowed_values: Optional[List[Any]] = None,
        **kwargs
    ):
        """
        Initialize ConfigValueError.
        
        Args:
            key: The configuration key
            value: The invalid value
            reason: Reason why the value is invalid
            expected_type: Expected type (optional)
            valid_range: Valid range (min, max) (optional)
            allowed_values: List of allowed values (optional)
            **kwargs: Additional details
        """
        message = f"Invalid value for {key}: {value} - {reason}"
        details = {
            "key": key,
            "value": str(value),
            "reason": reason,
            **kwargs
        }
        if expected_type:
            details["expected_type"] = expected_type
        if valid_range:
            details["min"] = valid_range[0]
            details["max"] = valid_range[1]
        if allowed_values:
            details["allowed_values"] = allowed_values
        
        super().__init__(message, errors=[message], **details)


# ============================================================================
# Path-Related Exceptions
# ============================================================================

class PathError(CockatooError):
    """
    Base exception for path-related errors.
    
    This exception is raised when there are issues with file system paths.
    """
    pass


class PathNotFoundError(PathError):
    """
    Raised when a required path doesn't exist.
    
    This occurs when trying to access a directory or file that should exist
    but doesn't.
    """
    
    def __init__(self, path: Union[str, Path], purpose: str = "", **kwargs):
        """
        Initialize PathNotFoundError.
        
        Args:
            path: The path that wasn't found
            purpose: Purpose of the path (e.g., "data directory")
            **kwargs: Additional details
        """
        path_str = str(path)
        purpose_str = f" for {purpose}" if purpose else ""
        message = f"Path not found{purpose_str}: {path_str}"
        details = {"path": path_str, "purpose": purpose, **kwargs}
        super().__init__(message, details)


class PathNotWritableError(PathError):
    """
    Raised when a path is not writable.
    
    This occurs when trying to write to a directory or file
    without proper permissions.
    """
    
    def __init__(self, path: Union[str, Path], **kwargs):
        """
        Initialize PathNotWritableError.
        
        Args:
            path: The path that is not writable
            **kwargs: Additional details
        """
        path_str = str(path)
        message = f"Path is not writable: {path_str}"
        details = {"path": path_str, **kwargs}
        super().__init__(message, details)


class PathPermissionError(PathError):
    """
    Raised when there are permission issues with a path.
    
    This is more general than PathNotWritableError and can include
    read permissions, execute permissions, etc.
    """
    
    def __init__(
        self,
        path: Union[str, Path],
        operation: str = "access",
        **kwargs
    ):
        """
        Initialize PathPermissionError.
        
        Args:
            path: The path with permission issues
            operation: The operation that failed (read/write/execute)
            **kwargs: Additional details
        """
        path_str = str(path)
        message = f"Permission denied {operation} path: {path_str}"
        details = {"path": path_str, "operation": operation, **kwargs}
        super().__init__(message, details)


# ============================================================================
# Document Processing Exceptions
# ============================================================================

class DocumentProcessingError(CockatooError):
    """
    Base exception for document processing errors.
    
    This exception is raised when there are issues during document
    loading, parsing, chunking, or extraction.
    """
    pass


class UnsupportedFormatError(DocumentProcessingError):
    """
    Raised when trying to process an unsupported file format.
    """
    
    def __init__(
        self,
        format: str,
        supported_formats: List[str],
        file_path: Optional[Union[str, Path]] = None,
        **kwargs
    ):
        """
        Initialize UnsupportedFormatError.
        
        Args:
            format: The unsupported file format/extension
            supported_formats: List of supported formats
            file_path: Path to the file (optional)
            **kwargs: Additional details
        """
        message = f"Unsupported file format: {format}"
        details = {
            "format": format,
            "supported_formats": supported_formats,
            **kwargs
        }
        if file_path:
            details["file_path"] = str(file_path)
        
        super().__init__(message, details)


class FileTooLargeError(DocumentProcessingError):
    """
    Raised when a file exceeds the maximum allowed size.
    """
    
    def __init__(
        self,
        file_size_mb: float,
        max_size_mb: float,
        file_path: Optional[Union[str, Path]] = None,
        **kwargs
    ):
        """
        Initialize FileTooLargeError.
        
        Args:
            file_size_mb: Actual file size in MB
            max_size_mb: Maximum allowed size in MB
            file_path: Path to the file (optional)
            **kwargs: Additional details
        """
        message = f"File size ({file_size_mb:.2f} MB) exceeds maximum ({max_size_mb} MB)"
        details = {
            "file_size_mb": file_size_mb,
            "max_size_mb": max_size_mb,
            **kwargs
        }
        if file_path:
            details["file_path"] = str(file_path)
        
        super().__init__(message, details)


class OCRProcessingError(DocumentProcessingError):
    """
    Raised when OCR processing fails.
    """
    
    def __init__(
        self,
        message: str,
        file_path: Optional[Union[str, Path]] = None,
        language: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize OCRProcessingError.
        
        Args:
            message: Error message
            file_path: Path to the file being processed
            language: OCR language being used
            **kwargs: Additional details
        """
        details = {"ocr_error": message, **kwargs}
        if file_path:
            details["file_path"] = str(file_path)
        if language:
            details["language"] = language
        
        super().__init__(f"OCR processing failed: {message}", details)


class ExtractionError(DocumentProcessingError):
    """
    Raised when document content extraction fails.
    """
    
    def __init__(
        self,
        message: str,
        file_path: Optional[Union[str, Path]] = None,
        extractor: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize ExtractionError.
        
        Args:
            message: Error message
            file_path: Path to the file being processed
            extractor: Name of the extractor that failed
            **kwargs: Additional details
        """
        details = {"extraction_error": message, **kwargs}
        if file_path:
            details["file_path"] = str(file_path)
        if extractor:
            details["extractor"] = extractor
        
        super().__init__(f"Document extraction failed: {message}", details)


class ChunkingError(DocumentProcessingError):
    """
    Raised when document chunking fails.
    """
    
    def __init__(
        self,
        message: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize ChunkingError.
        
        Args:
            message: Error message
            chunk_size: Chunk size being used
            chunk_overlap: Chunk overlap being used
            **kwargs: Additional details
        """
        details = {"chunking_error": message, **kwargs}
        if chunk_size:
            details["chunk_size"] = chunk_size
        if chunk_overlap:
            details["chunk_overlap"] = chunk_overlap
        
        super().__init__(f"Document chunking failed: {message}", details)


# ============================================================================
# AI/ML Exceptions
# ============================================================================

class AIError(CockatooError):
    """
    Base exception for AI-related errors.
    
    This exception is raised when there are issues with LLMs,
    embeddings, RAG, or other AI components.
    """
    pass


class LLMError(AIError):
    """
    Base exception for LLM-related errors.
    """
    pass


class LLMConnectionError(LLMError):
    """
    Raised when connection to LLM provider fails.
    """
    
    def __init__(
        self,
        provider: str,
        base_url: str,
        error: str,
        **kwargs
    ):
        """
        Initialize LLMConnectionError.
        
        Args:
            provider: LLM provider name
            base_url: Base URL that was attempted
            error: Detailed connection error
            **kwargs: Additional details
        """
        message = f"Failed to connect to {provider} at {base_url}"
        details = {
            "provider": provider,
            "base_url": base_url,
            "connection_error": error,
            **kwargs
        }
        super().__init__(message, details)


class LLMRequestError(LLMError):
    """
    Raised when an LLM request fails.
    """
    
    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        status_code: Optional[int] = None,
        response_text: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize LLMRequestError.
        
        Args:
            message: Error message
            provider: LLM provider name
            model: Model being used
            status_code: HTTP status code (if applicable)
            response_text: Response text (if available)
            **kwargs: Additional details
        """
        details = {"request_error": message, **kwargs}
        if provider:
            details["provider"] = provider
        if model:
            details["model"] = model
        if status_code:
            details["status_code"] = status_code
        if response_text:
            details["response"] = response_text[:200]  # Truncate long responses
        
        super().__init__(f"LLM request failed: {message}", details)


class LLMTimeoutError(LLMError):
    """
    Raised when an LLM request times out.
    """
    
    def __init__(
        self,
        timeout_seconds: int,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize LLMTimeoutError.
        
        Args:
            timeout_seconds: Timeout value in seconds
            provider: LLM provider name
            model: Model being used
            **kwargs: Additional details
        """
        message = f"LLM request timed out after {timeout_seconds} seconds"
        details = {
            "timeout_seconds": timeout_seconds,
            **kwargs
        }
        if provider:
            details["provider"] = provider
        if model:
            details["model"] = model
        
        super().__init__(message, details)


class LLMInvalidResponseError(LLMError):
    """
    Raised when LLM returns an invalid or unexpected response.
    """
    
    def __init__(
        self,
        message: str,
        response: Optional[Any] = None,
        expected_format: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize LLMInvalidResponseError.
        
        Args:
            message: Error message
            response: The invalid response
            expected_format: Expected response format
            **kwargs: Additional details
        """
        details = {"invalid_response_error": message, **kwargs}
        if response is not None:
            # Convert to string and truncate if too long
            response_str = str(response)
            if len(response_str) > 200:
                response_str = response_str[:200] + "..."
            details["response"] = response_str
        if expected_format:
            details["expected_format"] = expected_format
        
        super().__init__(f"Invalid LLM response: {message}", details)


class EmbeddingError(AIError):
    """
    Raised when embedding generation fails.
    """
    
    def __init__(
        self,
        message: str,
        model: Optional[str] = None,
        device: Optional[str] = None,
        text_preview: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize EmbeddingError.
        
        Args:
            message: Error message
            model: Embedding model being used
            device: Device being used (cpu/cuda/mps)
            text_preview: Preview of text being embedded
            **kwargs: Additional details
        """
        details = {"embedding_error": message, **kwargs}
        if model:
            details["model"] = model
        if device:
            details["device"] = device
        if text_preview and len(text_preview) > 50:
            details["text_preview"] = text_preview[:50] + "..."
        
        super().__init__(f"Embedding generation failed: {message}", details)


class RAGError(AIError):
    """
    Base exception for RAG (Retrieval-Augmented Generation) errors.
    """
    pass


class RAGRetrievalError(RAGError):
    """
    Raised when document retrieval fails.
    """
    
    def __init__(
        self,
        message: str,
        query: Optional[str] = None,
        top_k: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize RAGRetrievalError.
        
        Args:
            message: Error message
            query: Search query (optional)
            top_k: Number of results requested (optional)
            **kwargs: Additional details
        """
        details = {"retrieval_error": message, **kwargs}
        if query and len(query) > 50:
            details["query"] = query[:50] + "..."
        if top_k:
            details["top_k"] = top_k
        
        super().__init__(f"RAG retrieval failed: {message}", details)


class DatabaseError(AIError):
    """
    Base exception for vector database errors.
    """
    pass


class DatabaseConnectionError(DatabaseError):
    """
    Raised when connection to vector database fails.
    """
    
    def __init__(
        self,
        db_type: str,
        host: Optional[str] = None,
        port: Optional[int] = None,
        error: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize DatabaseConnectionError.
        
        Args:
            db_type: Database type (chroma/qdrant/pgvector/sqlite)
            host: Database host
            port: Database port
            error: Detailed connection error
            **kwargs: Additional details
        """
        host_str = f"{host}:{port}" if host and port else host or "localhost"
        message = f"Failed to connect to {db_type} database at {host_str}"
        details = {
            "db_type": db_type,
            "host": host,
            "port": port,
            "connection_error": error,
            **kwargs
        }
        super().__init__(message, details)


class DatabaseQueryError(DatabaseError):
    """
    Raised when a database query fails.
    """
    
    def __init__(
        self,
        message: str,
        db_type: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize DatabaseQueryError.
        
        Args:
            message: Error message
            db_type: Database type
            operation: Operation that failed (insert/query/delete/update)
            **kwargs: Additional details
        """
        details = {"query_error": message, **kwargs}
        if db_type:
            details["db_type"] = db_type
        if operation:
            details["operation"] = operation
        
        super().__init__(f"Database query failed: {message}", details)


# ============================================================================
# UI Exceptions
# ============================================================================

class UIError(CockatooError):
    """
    Base exception for UI-related errors.
    """
    pass


class ThemeError(UIError):
    """
    Raised when there are issues with UI themes.
    """
    
    def __init__(
        self,
        theme: str,
        message: str = "Invalid or unsupported theme",
        **kwargs
    ):
        """
        Initialize ThemeError.
        
        Args:
            theme: The invalid theme
            message: Error message
            **kwargs: Additional details
        """
        details = {"theme": theme, **kwargs}
        super().__init__(f"{message}: {theme}", details)


# ============================================================================
# Storage Exceptions
# ============================================================================

class StorageError(CockatooError):
    """
    Base exception for storage-related errors.
    """
    pass


class StorageWriteError(StorageError):
    """
    Raised when writing to storage fails.
    """
    
    def __init__(
        self,
        message: str,
        path: Optional[Union[str, Path]] = None,
        **kwargs
    ):
        """
        Initialize StorageWriteError.
        
        Args:
            message: Error message
            path: Path being written to
            **kwargs: Additional details
        """
        details = {"write_error": message, **kwargs}
        if path:
            details["path"] = str(path)
        
        super().__init__(f"Storage write failed: {message}", details)


class StorageReadError(StorageError):
    """
    Raised when reading from storage fails.
    """
    
    def __init__(
        self,
        message: str,
        path: Optional[Union[str, Path]] = None,
        **kwargs
    ):
        """
        Initialize StorageReadError.
        
        Args:
            message: Error message
            path: Path being read from
            **kwargs: Additional details
        """
        details = {"read_error": message, **kwargs}
        if path:
            details["path"] = str(path)
        
        super().__init__(f"Storage read failed: {message}", details)


class BackupError(StorageError):
    """
    Raised when backup operations fail.
    """
    
    def __init__(
        self,
        message: str,
        source: Optional[Union[str, Path]] = None,
        destination: Optional[Union[str, Path]] = None,
        **kwargs
    ):
        """
        Initialize BackupError.
        
        Args:
            message: Error message
            source: Source path being backed up
            destination: Destination backup path
            **kwargs: Additional details
        """
        details = {"backup_error": message, **kwargs}
        if source:
            details["source"] = str(source)
        if destination:
            details["destination"] = str(destination)
        
        super().__init__(f"Backup failed: {message}", details)


class EncryptionError(StorageError):
    """
    Raised when encryption/decryption operations fail.
    """
    
    def __init__(
        self,
        message: str,
        operation: str = "encrypt",
        **kwargs
    ):
        """
        Initialize EncryptionError.
        
        Args:
            message: Error message
            operation: Operation that failed (encrypt/decrypt)
            **kwargs: Additional details
        """
        details = {"encryption_error": message, "operation": operation, **kwargs}
        super().__init__(f"Encryption {operation} failed: {message}", details)


class CompressionError(StorageError):
    """
    Raised when compression/decompression operations fail.
    """
    
    def __init__(
        self,
        message: str,
        operation: str = "compress",
        level: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize CompressionError.
        
        Args:
            message: Error message
            operation: Operation that failed (compress/decompress)
            level: Compression level being used
            **kwargs: Additional details
        """
        details = {"compression_error": message, "operation": operation, **kwargs}
        if level is not None:
            details["level"] = level
        
        super().__init__(f"Compression {operation} failed: {message}", details)


# ============================================================================
# Performance Exceptions
# ============================================================================

class PerformanceError(CockatooError):
    """
    Base exception for performance-related errors.
    """
    pass


class CacheError(PerformanceError):
    """
    Raised when cache operations fail.
    """
    
    def __init__(
        self,
        message: str,
        operation: str = "cache",
        key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize CacheError.
        
        Args:
            message: Error message
            operation: Operation that failed (read/write/delete)
            key: Cache key being accessed
            **kwargs: Additional details
        """
        details = {"cache_error": message, "operation": operation, **kwargs}
        if key:
            details["key"] = key
        
        super().__init__(f"Cache {operation} failed: {message}", details)


class ResourceLimitError(PerformanceError):
    """
    Raised when a resource limit is exceeded.
    """
    
    def __init__(
        self,
        resource: str,
        value: float,
        limit: float,
        unit: str = "",
        **kwargs
    ):
        """
        Initialize ResourceLimitError.
        
        Args:
            resource: Resource that exceeded limit (memory, cpu, etc.)
            value: Current value
            limit: Maximum allowed value
            unit: Unit of measurement
            **kwargs: Additional details
        """
        unit_str = f" {unit}" if unit else ""
        message = f"{resource.capitalize()} limit exceeded: {value}{unit_str} > {limit}{unit_str}"
        details = {
            "resource": resource,
            "value": value,
            "limit": limit,
            "unit": unit,
            **kwargs
        }
        super().__init__(message, details)


class MemoryLimitError(ResourceLimitError):
    """
    Raised when memory limit is exceeded.
    """
    
    def __init__(
        self,
        used_mb: float,
        limit_mb: float,
        **kwargs
    ):
        """
        Initialize MemoryLimitError.
        
        Args:
            used_mb: Memory used in MB
            limit_mb: Memory limit in MB
            **kwargs: Additional details
        """
        super().__init__(
            resource="memory",
            value=used_mb,
            limit=limit_mb,
            unit="MB",
            **kwargs
        )


# ============================================================================
# Privacy Exceptions
# ============================================================================

class PrivacyError(CockatooError):
    """
    Base exception for privacy-related errors.
    """
    pass


class TelemetryError(PrivacyError):
    """
    Raised when there are issues with telemetry.
    """
    
    def __init__(
        self,
        message: str,
        operation: str = "telemetry",
        **kwargs
    ):
        """
        Initialize TelemetryError.
        
        Args:
            message: Error message
            operation: Operation that failed
            **kwargs: Additional details
        """
        details = {"telemetry_error": message, "operation": operation, **kwargs}
        super().__init__(f"Telemetry {operation} failed: {message}", details)


class ConsentError(PrivacyError):
    """
    Raised when required consent is missing.
    """
    
    def __init__(
        self,
        consent_type: str,
        message: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize ConsentError.
        
        Args:
            consent_type: Type of consent required
            message: Error message
            **kwargs: Additional details
        """
        if message is None:
            message = f"{consent_type.capitalize()} consent is required"
        details = {"consent_type": consent_type, **kwargs}
        super().__init__(message, details)


# ============================================================================
# Logging Exceptions
# ============================================================================

class LoggingError(CockatooError):
    """
    Base exception for logging-related errors.
    """
    pass


class LogFileError(LoggingError):
    """
    Raised when there are issues with log files.
    """
    
    def __init__(
        self,
        message: str,
        path: Optional[Union[str, Path]] = None,
        **kwargs
    ):
        """
        Initialize LogFileError.
        
        Args:
            message: Error message
            path: Path to log file
            **kwargs: Additional details
        """
        details = {"log_error": message, **kwargs}
        if path:
            details["path"] = str(path)
        
        super().__init__(f"Log file error: {message}", details)


# ============================================================================
# App Exceptions
# ============================================================================

class AppInitializationError(CockatooError):
    """
    Raised when application initialization fails.
    """
    
    def __init__(
        self,
        message: str,
        component: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize AppInitializationError.
        
        Args:
            message: Error message
            component: Component that failed to initialize
            **kwargs: Additional details
        """
        details = {"component": component, **kwargs} if component else kwargs
        super().__init__(f"App initialization failed: {message}", details)


class AppRuntimeError(CockatooError):
    """
    Raised when application runtime error occurs.
    """
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize AppRuntimeError.
        
        Args:
            message: Error message
            operation: Operation that failed
            **kwargs: Additional details
        """
        details = {"operation": operation, **kwargs} if operation else kwargs
        super().__init__(f"App runtime error: {message}", details)


class AppShutdownError(CockatooError):
    """
    Raised when application shutdown fails.
    """
    
    def __init__(
        self,
        message: str,
        component: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize AppShutdownError.
        
        Args:
            message: Error message
            component: Component that failed to shutdown
            **kwargs: Additional details
        """
        details = {"component": component, **kwargs} if component else kwargs
        super().__init__(f"App shutdown failed: {message}", details)


# ============================================================================
# Utility function to create appropriate exception from error data
# ============================================================================

def exception_from_dict(error_data: Dict[str, Any]) -> CockatooError:
    """
    Create an exception instance from a dictionary representation.
    
    Args:
        error_data: Dictionary with error information (from to_dict())
        
    Returns:
        CockatooError: Appropriate exception instance
        
    Raises:
        ValueError: If error type is unknown
    """
    error_type = error_data.get("error_type")
    message = error_data.get("message", "Unknown error")
    details = error_data.get("details", {}).copy()  # Copy to avoid modifying original
    cause_str = error_data.get("cause")
    
    cause = None
    if cause_str:
        cause = Exception(cause_str)
    
    # Map error types to exception classes
    exception_map = {
        # Base
        "CockatooError": CockatooError,
        
        # Configuration
        "ConfigurationError": ConfigurationError,
        "ConfigFileNotFoundError": ConfigFileNotFoundError,
        "ConfigFilePermissionError": ConfigFilePermissionError,
        "ConfigFormatError": ConfigFormatError,
        "ConfigVersionMismatchError": ConfigVersionMismatchError,
        "ConfigValidationError": ConfigValidationError,
        "ConfigKeyError": ConfigKeyError,
        "ConfigValueError": ConfigValueError,
        
        # Path
        "PathError": PathError,
        "PathNotFoundError": PathNotFoundError,
        "PathNotWritableError": PathNotWritableError,
        "PathPermissionError": PathPermissionError,
        
        # Document Processing
        "DocumentProcessingError": DocumentProcessingError,
        "UnsupportedFormatError": UnsupportedFormatError,
        "FileTooLargeError": FileTooLargeError,
        "OCRProcessingError": OCRProcessingError,
        "ExtractionError": ExtractionError,
        "ChunkingError": ChunkingError,
        
        # AI
        "AIError": AIError,
        "LLMError": LLMError,
        "LLMConnectionError": LLMConnectionError,
        "LLMRequestError": LLMRequestError,
        "LLMTimeoutError": LLMTimeoutError,
        "LLMInvalidResponseError": LLMInvalidResponseError,
        "EmbeddingError": EmbeddingError,
        "RAGError": RAGError,
        "RAGRetrievalError": RAGRetrievalError,
        "DatabaseError": DatabaseError,
        "DatabaseConnectionError": DatabaseConnectionError,
        "DatabaseQueryError": DatabaseQueryError,
        
        # UI
        "UIError": UIError,
        "ThemeError": ThemeError,
        
        # Storage
        "StorageError": StorageError,
        "StorageWriteError": StorageWriteError,
        "StorageReadError": StorageReadError,
        "BackupError": BackupError,
        "EncryptionError": EncryptionError,
        "CompressionError": CompressionError,
        
        # Performance
        "PerformanceError": PerformanceError,
        "CacheError": CacheError,
        "ResourceLimitError": ResourceLimitError,
        "MemoryLimitError": MemoryLimitError,
        
        # Privacy
        "PrivacyError": PrivacyError,
        "TelemetryError": TelemetryError,
        "ConsentError": ConsentError,
        
        # Logging
        "LoggingError": LoggingError,
        "LogFileError": LogFileError,
        
        # App
        "AppInitializationError": AppInitializationError,
        "AppRuntimeError": AppRuntimeError,
        "AppShutdownError": AppShutdownError,
    }
    
    exception_class = exception_map.get(error_type)
    if not exception_class:
        raise ValueError(f"Unknown error type: {error_type}")
    
    # Create instance with appropriate parameters - FIXED: remove path from kwargs
    if error_type == "ConfigFileNotFoundError":
        path_value = details.pop("path", None)
        return exception_class(path_value, **details)
    elif error_type == "ConfigFilePermissionError":
        path_value = details.pop("path", None)
        operation = details.pop("operation", "access")
        return exception_class(path_value, operation, **details)
    elif error_type == "ConfigFormatError":
        path_value = details.pop("path", None)
        format_type = details.pop("format", "yaml")
        parse_error = details.pop("parse_error", "")
        return exception_class(path_value, format_type, parse_error, **details)
    elif error_type == "ConfigVersionMismatchError":
        expected = details.pop("expected_version", "unknown")
        actual = details.pop("actual_version", "unknown")
        path_value = details.pop("path", None)
        return exception_class(expected, actual, path_value, **details)
    elif error_type == "ConfigKeyError":
        key_value = details.pop("key", "")
        available = details.pop("available_keys", None)
        return exception_class(key_value, available, **details)
    elif error_type == "ConfigValueError":
        key_value = details.pop("key", "")
        value = details.pop("value", "")
        reason = details.pop("reason", "")
        expected_type = details.pop("expected_type", None)
        min_val = details.pop("min", None)
        max_val = details.pop("max", None)
        allowed = details.pop("allowed_values", None)
        valid_range = (min_val, max_val) if min_val is not None and max_val is not None else None
        return exception_class(key_value, value, reason, expected_type, valid_range, allowed, **details)
    elif error_type in ["PathNotFoundError", "PathNotWritableError"]:
        path_value = details.pop("path", None)
        return exception_class(path_value, **details)
    elif error_type == "PathPermissionError":
        path_value = details.pop("path", None)
        operation = details.pop("operation", "access")
        return exception_class(path_value, operation, **details)
    elif error_type == "UnsupportedFormatError":
        format_value = details.pop("format", "")
        supported = details.pop("supported_formats", [])
        file_path = details.pop("file_path", None)
        return exception_class(format_value, supported, file_path, **details)
    elif error_type == "FileTooLargeError":
        file_size = details.pop("file_size_mb", 0)
        max_size = details.pop("max_size_mb", 0)
        file_path = details.pop("file_path", None)
        return exception_class(file_size, max_size, file_path, **details)
    elif error_type == "OCRProcessingError":
        error_msg = details.pop("ocr_error", "")
        file_path = details.pop("file_path", None)
        language = details.pop("language", None)
        return exception_class(error_msg, file_path, language, **details)
    elif error_type == "ResourceLimitError":
        resource = details.pop("resource", "unknown")
        value = details.pop("value", 0)
        limit = details.pop("limit", 0)
        unit = details.pop("unit", "")
        return exception_class(resource, value, limit, unit, **details)
    elif error_type == "MemoryLimitError":
        value = details.pop("value", 0)
        limit = details.pop("limit", 0)
        return exception_class(value, limit, **details)
    elif error_type == "ConsentError":
        consent_type = details.pop("consent_type", "unknown")
        return exception_class(consent_type, message, **details)
    else:
        # Generic creation
        return exception_class(message, details, cause)