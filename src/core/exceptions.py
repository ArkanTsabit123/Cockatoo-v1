# cockatoo_v1/src/core/exceptions.py
"""
Custom exceptions for Cockatoo_V1 application.

This module provides a comprehensive hierarchy of exceptions for
the configuration management system and application components.
"""

from typing import Optional, Any, Dict, List, Union
from pathlib import Path


class CockatooError(Exception):
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        self.message = message
        self.details = details or {}
        self.cause = cause
        super().__init__(message)
        
        if cause:
            self.__cause__ = cause
    
    def __str__(self) -> str:
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} [{details_str}]"
        return self.message
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
            "cause": str(self.cause) if self.cause else None,
        }


class ConfigurationError(CockatooError):
    pass


class ConfigFileNotFoundError(ConfigurationError):
    
    def __init__(self, path: Union[str, Path], **kwargs):
        path_str = str(path)
        message = f"Configuration file not found: {path_str}"
        details = {"path": path_str, **kwargs}
        super().__init__(message, details)


class ConfigFilePermissionError(ConfigurationError):
    
    def __init__(
        self,
        path: Union[str, Path],
        operation: str = "access",
        **kwargs
    ):
        path_str = str(path)
        message = f"Permission denied {operation} config file: {path_str}"
        details = {"path": path_str, "operation": operation, **kwargs}
        super().__init__(message, details)


class ConfigFormatError(ConfigurationError):
    
    def __init__(
        self,
        path: Union[str, Path],
        format_type: str,
        parse_error: str,
        **kwargs
    ):
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
    
    def __init__(
        self,
        expected_version: str,
        actual_version: str,
        path: Optional[Union[str, Path]] = None,
        **kwargs
    ):
        message = f"Config version mismatch: expected {expected_version}, got {actual_version}"
        details = {
            "expected_version": expected_version,
            "actual_version": actual_version,
            **kwargs
        }
        if path:
            details["path"] = str(path)
        super().__init__(message, details)


class ConfigValidationError(ConfigurationError):
    
    def __init__(
        self,
        message: str,
        errors: Optional[List[str]] = None,
        **kwargs
    ):
        details = {"validation_errors": errors or [], **kwargs}
        super().__init__(message, details)


class ConfigKeyError(ConfigurationError):
    
    def __init__(
        self,
        key: str,
        available_keys: Optional[List[str]] = None,
        **kwargs
    ):
        message = f"Invalid configuration key: {key}"
        details = {"key": key, **kwargs}
        if available_keys:
            details["available_keys"] = available_keys
        super().__init__(message, details)


class ConfigValueError(ConfigValidationError):
    
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


class PathError(CockatooError):
    pass


class PathNotFoundError(PathError):
    
    def __init__(self, path: Union[str, Path], purpose: str = "", **kwargs):
        path_str = str(path)
        purpose_str = f" for {purpose}" if purpose else ""
        message = f"Path not found{purpose_str}: {path_str}"
        details = {"path": path_str, "purpose": purpose, **kwargs}
        super().__init__(message, details)


class PathNotWritableError(PathError):
    
    def __init__(self, path: Union[str, Path], **kwargs):
        path_str = str(path)
        message = f"Path is not writable: {path_str}"
        details = {"path": path_str, **kwargs}
        super().__init__(message, details)


class PathPermissionError(PathError):
    
    def __init__(
        self,
        path: Union[str, Path],
        operation: str = "access",
        **kwargs
    ):
        path_str = str(path)
        message = f"Permission denied {operation} path: {path_str}"
        details = {"path": path_str, "operation": operation, **kwargs}
        super().__init__(message, details)


class DocumentProcessingError(CockatooError):
    pass


class UnsupportedFormatError(DocumentProcessingError):
    
    def __init__(
        self,
        format: str,
        supported_formats: List[str],
        file_path: Optional[Union[str, Path]] = None,
        **kwargs
    ):
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
    
    def __init__(
        self,
        file_size_mb: float,
        max_size_mb: float,
        file_path: Optional[Union[str, Path]] = None,
        **kwargs
    ):
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
    
    def __init__(
        self,
        message: str,
        file_path: Optional[Union[str, Path]] = None,
        language: Optional[str] = None,
        **kwargs
    ):
        details = {"ocr_error": message, **kwargs}
        if file_path:
            details["file_path"] = str(file_path)
        if language:
            details["language"] = language
        
        super().__init__(f"OCR processing failed: {message}", details)


class ExtractionError(DocumentProcessingError):
    
    def __init__(
        self,
        message: str,
        file_path: Optional[Union[str, Path]] = None,
        extractor: Optional[str] = None,
        **kwargs
    ):
        details = {"extraction_error": message, **kwargs}
        if file_path:
            details["file_path"] = str(file_path)
        if extractor:
            details["extractor"] = extractor
        
        super().__init__(f"Document extraction failed: {message}", details)


class ChunkingError(DocumentProcessingError):
    
    def __init__(
        self,
        message: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        **kwargs
    ):
        details = {"chunking_error": message, **kwargs}
        if chunk_size:
            details["chunk_size"] = chunk_size
        if chunk_overlap:
            details["chunk_overlap"] = chunk_overlap
        
        super().__init__(f"Document chunking failed: {message}", details)


class AIError(CockatooError):
    pass


class LLMError(AIError):
    pass


class LLMConnectionError(LLMError):
    
    def __init__(
        self,
        provider: str,
        base_url: str,
        error: str,
        **kwargs
    ):
        message = f"Failed to connect to {provider} at {base_url}"
        details = {
            "provider": provider,
            "base_url": base_url,
            "connection_error": error,
            **kwargs
        }
        super().__init__(message, details)


class LLMRequestError(LLMError):
    
    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        status_code: Optional[int] = None,
        response_text: Optional[str] = None,
        **kwargs
    ):
        details = {"request_error": message, **kwargs}
        if provider:
            details["provider"] = provider
        if model:
            details["model"] = model
        if status_code:
            details["status_code"] = status_code
        if response_text:
            details["response"] = response_text[:200]
        
        super().__init__(f"LLM request failed: {message}", details)


class LLMTimeoutError(LLMError):
    
    def __init__(
        self,
        timeout_seconds: int,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ):
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
    
    def __init__(
        self,
        message: str,
        response: Optional[Any] = None,
        expected_format: Optional[str] = None,
        **kwargs
    ):
        details = {"invalid_response_error": message, **kwargs}
        if response is not None:
            response_str = str(response)
            if len(response_str) > 200:
                response_str = response_str[:200] + "..."
            details["response"] = response_str
        if expected_format:
            details["expected_format"] = expected_format
        
        super().__init__(f"Invalid LLM response: {message}", details)


class EmbeddingError(AIError):
    
    def __init__(
        self,
        message: str,
        model: Optional[str] = None,
        device: Optional[str] = None,
        text_preview: Optional[str] = None,
        **kwargs
    ):
        details = {"embedding_error": message, **kwargs}
        if model:
            details["model"] = model
        if device:
            details["device"] = device
        if text_preview and len(text_preview) > 50:
            details["text_preview"] = text_preview[:50] + "..."
        
        super().__init__(f"Embedding generation failed: {message}", details)


class RAGError(AIError):
    pass


class RAGRetrievalError(RAGError):
    
    def __init__(
        self,
        message: str,
        query: Optional[str] = None,
        top_k: Optional[int] = None,
        **kwargs
    ):
        details = {"retrieval_error": message, **kwargs}
        if query and len(query) > 50:
            details["query"] = query[:50] + "..."
        if top_k:
            details["top_k"] = top_k
        
        super().__init__(f"RAG retrieval failed: {message}", details)


class DatabaseError(AIError):
    pass


class DatabaseConnectionError(DatabaseError):
    
    def __init__(
        self,
        db_type: str,
        host: Optional[str] = None,
        port: Optional[int] = None,
        error: Optional[str] = None,
        **kwargs
    ):
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
    
    def __init__(
        self,
        message: str,
        db_type: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        details = {"query_error": message, **kwargs}
        if db_type:
            details["db_type"] = db_type
        if operation:
            details["operation"] = operation
        
        super().__init__(f"Database query failed: {message}", details)


class UIError(CockatooError):
    pass


class ThemeError(UIError):
    
    def __init__(
        self,
        theme: str,
        message: str = "Invalid or unsupported theme",
        **kwargs
    ):
        details = {"theme": theme, **kwargs}
        super().__init__(f"{message}: {theme}", details)


class StorageError(CockatooError):
    pass


class StorageWriteError(StorageError):
    
    def __init__(
        self,
        message: str,
        path: Optional[Union[str, Path]] = None,
        **kwargs
    ):
        details = {"write_error": message, **kwargs}
        if path:
            details["path"] = str(path)
        
        super().__init__(f"Storage write failed: {message}", details)


class StorageReadError(StorageError):
    
    def __init__(
        self,
        message: str,
        path: Optional[Union[str, Path]] = None,
        **kwargs
    ):
        details = {"read_error": message, **kwargs}
        if path:
            details["path"] = str(path)
        
        super().__init__(f"Storage read failed: {message}", details)


class BackupError(StorageError):
    
    def __init__(
        self,
        message: str,
        source: Optional[Union[str, Path]] = None,
        destination: Optional[Union[str, Path]] = None,
        **kwargs
    ):
        details = {"backup_error": message, **kwargs}
        if source:
            details["source"] = str(source)
        if destination:
            details["destination"] = str(destination)
        
        super().__init__(f"Backup failed: {message}", details)


class EncryptionError(StorageError):
    
    def __init__(
        self,
        message: str,
        operation: str = "encrypt",
        **kwargs
    ):
        details = {"encryption_error": message, "operation": operation, **kwargs}
        super().__init__(f"Encryption {operation} failed: {message}", details)


class CompressionError(StorageError):
    
    def __init__(
        self,
        message: str,
        operation: str = "compress",
        level: Optional[int] = None,
        **kwargs
    ):
        details = {"compression_error": message, "operation": operation, **kwargs}
        if level is not None:
            details["level"] = level
        
        super().__init__(f"Compression {operation} failed: {message}", details)


class PerformanceError(CockatooError):
    pass


class CacheError(PerformanceError):
    
    def __init__(
        self,
        message: str,
        operation: str = "cache",
        key: Optional[str] = None,
        **kwargs
    ):
        details = {"cache_error": message, "operation": operation, **kwargs}
        if key:
            details["key"] = key
        
        super().__init__(f"Cache {operation} failed: {message}", details)


class ResourceLimitError(PerformanceError):
    
    def __init__(
        self,
        resource: str,
        value: float,
        limit: float,
        unit: str = "",
        **kwargs
    ):
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
    
    def __init__(
        self,
        used_mb: float,
        limit_mb: float,
        **kwargs
    ):
        super().__init__(
            resource="memory",
            value=used_mb,
            limit=limit_mb,
            unit="MB",
            **kwargs
        )


class PrivacyError(CockatooError):
    pass


class TelemetryError(PrivacyError):
    
    def __init__(
        self,
        message: str,
        operation: str = "telemetry",
        **kwargs
    ):
        details = {"telemetry_error": message, "operation": operation, **kwargs}
        super().__init__(f"Telemetry {operation} failed: {message}", details)


class ConsentError(PrivacyError):
    
    def __init__(
        self,
        consent_type: str,
        message: Optional[str] = None,
        **kwargs
    ):
        if message is None:
            message = f"{consent_type.capitalize()} consent is required"
        details = {"consent_type": consent_type, **kwargs}
        super().__init__(message, details)


class LoggingError(CockatooError):
    pass


class LogFileError(LoggingError):
    
    def __init__(
        self,
        message: str,
        path: Optional[Union[str, Path]] = None,
        **kwargs
    ):
        details = {"log_error": message, **kwargs}
        if path:
            details["path"] = str(path)
        
        super().__init__(f"Log file error: {message}", details)


class AppInitializationError(CockatooError):
    
    def __init__(
        self,
        message: str,
        component: Optional[str] = None,
        **kwargs
    ):
        details = {"component": component, **kwargs} if component else kwargs
        super().__init__(f"App initialization failed: {message}", details)


class AppRuntimeError(CockatooError):
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        **kwargs
    ):
        details = {"operation": operation, **kwargs} if operation else kwargs
        super().__init__(f"App runtime error: {message}", details)


class AppShutdownError(CockatooError):
    
    def __init__(
        self,
        message: str,
        component: Optional[str] = None,
        **kwargs
    ):
        details = {"component": component, **kwargs} if component else kwargs
        super().__init__(f"App shutdown failed: {message}", details)


def exception_from_dict(error_data: Dict[str, Any]) -> CockatooError:
    error_type = error_data.get("error_type")
    message = error_data.get("message", "Unknown error")
    details = error_data.get("details", {}).copy()
    cause_str = error_data.get("cause")
    
    cause = None
    if cause_str:
        cause = Exception(cause_str)
    
    exception_map = {
        "CockatooError": CockatooError,
        "ConfigurationError": ConfigurationError,
        "ConfigFileNotFoundError": ConfigFileNotFoundError,
        "ConfigFilePermissionError": ConfigFilePermissionError,
        "ConfigFormatError": ConfigFormatError,
        "ConfigVersionMismatchError": ConfigVersionMismatchError,
        "ConfigValidationError": ConfigValidationError,
        "ConfigKeyError": ConfigKeyError,
        "ConfigValueError": ConfigValueError,
        "PathError": PathError,
        "PathNotFoundError": PathNotFoundError,
        "PathNotWritableError": PathNotWritableError,
        "PathPermissionError": PathPermissionError,
        "DocumentProcessingError": DocumentProcessingError,
        "UnsupportedFormatError": UnsupportedFormatError,
        "FileTooLargeError": FileTooLargeError,
        "OCRProcessingError": OCRProcessingError,
        "ExtractionError": ExtractionError,
        "ChunkingError": ChunkingError,
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
        "UIError": UIError,
        "ThemeError": ThemeError,
        "StorageError": StorageError,
        "StorageWriteError": StorageWriteError,
        "StorageReadError": StorageReadError,
        "BackupError": BackupError,
        "EncryptionError": EncryptionError,
        "CompressionError": CompressionError,
        "PerformanceError": PerformanceError,
        "CacheError": CacheError,
        "ResourceLimitError": ResourceLimitError,
        "MemoryLimitError": MemoryLimitError,
        "PrivacyError": PrivacyError,
        "TelemetryError": TelemetryError,
        "ConsentError": ConsentError,
        "LoggingError": LoggingError,
        "LogFileError": LogFileError,
        "AppInitializationError": AppInitializationError,
        "AppRuntimeError": AppRuntimeError,
        "AppShutdownError": AppShutdownError,
    }
    
    exception_class = exception_map.get(error_type)
    if not exception_class:
        raise ValueError(f"Unknown error type: {error_type}")
    
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
        return exception_class(message, details, cause)