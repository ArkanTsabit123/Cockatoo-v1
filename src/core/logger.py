# cockatoo_v1/src/core/logger.py
"""
Logging configuration module for Cockatoo_V1 application.

This module provides a comprehensive logging system with:
- Structured logging (JSON format)
- Log rotation and management
- Context-aware logging with correlation IDs
- Performance tracking and profiling
- Sensitive data masking
- Multiple output destinations
"""

import os
import sys
import json
import logging
import logging.handlers
import traceback
import uuid
import time
import re
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Callable
from datetime import datetime
from enum import Enum
from functools import wraps
import threading

try:
    from pythonjsonlogger import json as jsonlogger
except ImportError:
    try:
        from pythonjsonlogger import jsonlogger
    except ImportError:
        import logging
        jsonlogger = None
        logging.warning("pythonjsonlogger not installed. JSON formatting disabled.")

try:
    from logging import NullHandler
except ImportError:
    pass

from .constants import (
    DEFAULT_LOG_LEVEL,
    DEFAULT_LOG_FORMAT,
    DEFAULT_MAX_LOG_SIZE_MB,
    DEFAULT_MAX_LOG_FILES,
    DEFAULT_LOG_FILE_ENABLED,
    DEFAULT_CONSOLE_ENABLED,
    DEFAULT_ANONYMIZE_LOGS,
    DEFAULT_LOG_SENSITIVE_DATA,
    BYTES_PER_MB,
)
from .config import LoggingConfig, AppConfig


class LogLevel(str, Enum):
    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    AUDIT = "AUDIT"
    PERFORMANCE = "PERFORMANCE"


TRACE_LEVEL = 5
SUCCESS_LEVEL = 25
AUDIT_LEVEL = 45
PERFORMANCE_LEVEL = 15

logging.addLevelName(TRACE_LEVEL, LogLevel.TRACE.value)
logging.addLevelName(SUCCESS_LEVEL, LogLevel.SUCCESS.value)
logging.addLevelName(AUDIT_LEVEL, LogLevel.AUDIT.value)
logging.addLevelName(PERFORMANCE_LEVEL, LogLevel.PERFORMANCE.value)


class CockatooLogger(logging.Logger):
    
    def __init__(self, name: str, level: int = logging.NOTSET):
        super().__init__(name, level)
        self._correlation_id: Optional[str] = None
        self._context: Dict[str, Any] = {}
        self._sensitive_patterns: List[re.Pattern] = []
        self._mask_token = "***MASKED***"
        self._log_sensitive_data = False
        
        self._init_sensitive_patterns()
    
    def _init_sensitive_patterns(self) -> None:
        patterns = [
            r'api[_-]?key["\']?\s*[:=]\s*["\']?([^"\'}\s]+)',
            r'password["\']?\s*[:=]\s*["\']?([^"\'}\s]+)',
            r'token["\']?\s*[:=]\s*["\']?([^"\'}\s]+)',
            r'secret["\']?\s*[:=]\s*["\']?([^"\'}\s]+)',
            r'auth["\']?\s*[:=]\s*["\']?([^"\'}\s]+)',
            r'credit[_-]?card["\']?\s*[:=]\s*["\']?([^"\'}\s]+)',
            r'ssn["\']?\s*[:=]\s*["\']?([^"\'}\s]+)',
            r'access[_-]?key["\']?\s*[:=]\s*["\']?([^"\'}\s]+)',
            r'private[_-]?key["\']?\s*[:=]\s*["\']?([^"\'}\s]+)',
        ]
        self._sensitive_patterns = [re.compile(p, re.IGNORECASE) for p in patterns]
    
    def set_correlation_id(self, correlation_id: Optional[str] = None) -> str:
        if correlation_id is None:
            correlation_id = str(uuid.uuid4())
        self._correlation_id = correlation_id
        return correlation_id
    
    def get_correlation_id(self) -> Optional[str]:
        return self._correlation_id
    
    def clear_correlation_id(self) -> None:
        self._correlation_id = None
    
    def set_context(self, key: str, value: Any) -> None:
        self._context[key] = value
    
    def update_context(self, context: Dict[str, Any]) -> None:
        self._context.update(context)
    
    def clear_context(self) -> None:
        self._context.clear()
    
    def get_context(self) -> Dict[str, Any]:
        return self._context.copy()
    
    def add_sensitive_pattern(self, pattern: Union[str, re.Pattern]) -> None:
        if isinstance(pattern, str):
            pattern = re.compile(pattern, re.IGNORECASE)
        self._sensitive_patterns.append(pattern)
    
    def mask_sensitive_data(self, message: str) -> str:
        if not self._sensitive_patterns:
            return message
        
        masked = message
        for pattern in self._sensitive_patterns:
            masked = pattern.sub(
                lambda m: f"{m.group(0).split(m.group(1))[0]}{self._mask_token}",
                masked
            )
        return masked
    
    def _log_with_context(
        self,
        level: int,
        msg: str,
        args: tuple,
        exc_info: Optional[Any] = None,
        extra: Optional[Dict[str, Any]] = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        **kwargs
    ) -> None:
        if extra is None:
            extra = {}
        
        if self._correlation_id:
            extra['correlation_id'] = self._correlation_id
        
        if self._context:
            extra['context'] = self._context.copy()
        
        if kwargs:
            extra['data'] = kwargs
        
        if not getattr(self, '_log_sensitive_data', False):
            msg = self.mask_sensitive_data(msg)
        
        super()._log(level, msg, args, exc_info, extra, stack_info, stacklevel)
    
    def trace(self, msg: str, *args, **kwargs) -> None:
        if self.isEnabledFor(TRACE_LEVEL):
            self._log_with_context(TRACE_LEVEL, msg, args, **kwargs)
    
    def success(self, msg: str, *args, **kwargs) -> None:
        if self.isEnabledFor(SUCCESS_LEVEL):
            self._log_with_context(SUCCESS_LEVEL, msg, args, **kwargs)
    
    def audit(self, msg: str, *args, **kwargs) -> None:
        if self.isEnabledFor(AUDIT_LEVEL):
            self._log_with_context(AUDIT_LEVEL, msg, args, **kwargs)
    
    def performance(self, msg: str, *args, **kwargs) -> None:
        if self.isEnabledFor(PERFORMANCE_LEVEL):
            self._log_with_context(PERFORMANCE_LEVEL, msg, args, **kwargs)
    
    def metric(self, name: str, value: float, unit: str = "", **kwargs) -> None:
        self.performance(
            f"Metric: {name} = {value} {unit}",
            metric_name=name,
            metric_value=value,
            metric_unit=unit,
            **kwargs
        )
    
    def exception_with_context(self, msg: str, exc_info: bool = True, **kwargs) -> None:
        self.error(msg, exc_info=exc_info, **kwargs)


class StructuredJSONFormatter:
    
    def __init__(self, *args, **kwargs):
        if jsonlogger is None:
            self.formatter = logging.Formatter(
                fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            self._json_enabled = False
        else:
            self.formatter = jsonlogger.JsonFormatter(*args, **kwargs)
            self._json_enabled = True
        
        self._reserved_keys = {
            'args', 'asctime', 'created', 'exc_info', 'exc_text', 'filename',
            'funcName', 'levelname', 'levelno', 'lineno', 'module',
            'msecs', 'message', 'msg', 'name', 'pathname', 'process',
            'processName', 'relativeCreated', 'stack_info', 'thread', 'threadName'
        }
    
    def format(self, record: logging.LogRecord) -> str:
        if self._json_enabled:
            return self.formatter.format(record)
        else:
            return self.formatter.format(record)
    
    def formatTime(self, record: logging.LogRecord, datefmt: Optional[str] = None) -> str:
        if hasattr(self.formatter, 'formatTime'):
            return self.formatter.formatTime(record, datefmt)
        return logging.Formatter.formatTime(self.formatter, record, datefmt)
    
    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]) -> None:
        if hasattr(self.formatter, 'add_fields'):
            self.formatter.add_fields(log_record, record, message_dict)
        
        if 'asctime' in log_record:
            try:
                log_record['timestamp'] = datetime.fromtimestamp(record.created).isoformat()
            except (ValueError, TypeError):
                log_record['timestamp'] = record.created
        
        log_record['level'] = record.levelname
        log_record['logger'] = record.name
        
        if record.exc_info:
            log_record['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': self.formatException(record.exc_info)
            }
        
        for key, value in list(log_record.items()):
            if not isinstance(value, (str, int, float, bool, type(None), dict, list)):
                try:
                    log_record[key] = str(value)
                except Exception:
                    del log_record[key]


def log_performance(logger: Optional[CockatooLogger] = None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = logging.getLogger(func.__module__)
            
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start_time
                
                if isinstance(logger, CockatooLogger):
                    logger.performance(
                        f"{func.__name__} completed in {elapsed:.3f}s",
                        function=func.__name__,
                        duration=elapsed,
                        args_count=len(args),
                        kwargs_count=len(kwargs)
                    )
                else:
                    logger.debug(f"{func.__name__} completed in {elapsed:.3f}s")
                
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start_time
                if isinstance(logger, CockatooLogger):
                    logger.performance(
                        f"{func.__name__} failed after {elapsed:.3f}s",
                        function=func.__name__,
                        duration=elapsed,
                        error=str(e)
                    )
                raise
        
        return wrapper
    return decorator


def log_async_performance(logger: Optional[CockatooLogger] = None):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = logging.getLogger(func.__module__)
            
            start_time = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                elapsed = time.perf_counter() - start_time
                
                if isinstance(logger, CockatooLogger):
                    logger.performance(
                        f"{func.__name__} completed in {elapsed:.3f}s",
                        function=func.__name__,
                        duration=elapsed,
                        args_count=len(args),
                        kwargs_count=len(kwargs)
                    )
                else:
                    logger.debug(f"{func.__name__} completed in {elapsed:.3f}s")
                
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start_time
                if isinstance(logger, CockatooLogger):
                    logger.performance(
                        f"{func.__name__} failed after {elapsed:.3f}s",
                        function=func.__name__,
                        duration=elapsed,
                        error=str(e)
                    )
                raise
        
        return wrapper
    return decorator


class LoggingContext:
    
    def __init__(self, logger: CockatooLogger, **kwargs):
        self.logger = logger
        self.context = kwargs
        self.saved_context = {}
    
    def __enter__(self):
        self.saved_context = self.logger.get_context()
        self.logger.update_context(self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.clear_context()
        self.logger.update_context(self.saved_context)
        
        if exc_type is not None:
            self.logger.exception_with_context(
                f"Exception in context: {exc_val}",
                exc_info=(exc_type, exc_val, exc_tb)
            )


class CorrelationId:
    
    def __init__(self, logger: CockatooLogger, correlation_id: Optional[str] = None):
        self.logger = logger
        self.correlation_id = correlation_id
        self.saved_id = None
    
    def __enter__(self):
        self.saved_id = self.logger.get_correlation_id()
        self.logger.set_correlation_id(self.correlation_id)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.saved_id:
            self.logger.set_correlation_id(self.saved_id)
        else:
            self.logger.clear_correlation_id()


def create_console_handler(
    level: Union[str, int] = logging.INFO,
    json_format: bool = False
) -> logging.Handler:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    if json_format:
        formatter = StructuredJSONFormatter(
            fmt='%(timestamp)s %(level)s %(name)s %(message)s',
            json_ensure_ascii=False
        )
    else:
        formatter = logging.Formatter(
            fmt=DEFAULT_LOG_FORMAT,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    handler.setFormatter(formatter)
    return handler


def create_file_handler(
    log_file: Path,
    level: Union[str, int] = logging.INFO,
    max_bytes: int = DEFAULT_MAX_LOG_SIZE_MB * BYTES_PER_MB,
    backup_count: int = DEFAULT_MAX_LOG_FILES,
    json_format: bool = False
) -> logging.Handler:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    handler = logging.handlers.RotatingFileHandler(
        filename=log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    handler.setLevel(level)
    
    if json_format:
        formatter = StructuredJSONFormatter(
            fmt='%(timestamp)s %(level)s %(name)s %(message)s',
            json_ensure_ascii=False
        )
    else:
        formatter = logging.Formatter(
            fmt=DEFAULT_LOG_FORMAT,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    handler.setFormatter(formatter)
    return handler


def create_daily_file_handler(
    log_dir: Path,
    level: Union[str, int] = logging.INFO,
    backup_count: int = DEFAULT_MAX_LOG_FILES,
    json_format: bool = False
) -> logging.Handler:
    log_dir.mkdir(parents=True, exist_ok=True)
    
    handler = logging.handlers.TimedRotatingFileHandler(
        filename=log_dir / "cockatoo.log",
        when='midnight',
        interval=1,
        backupCount=backup_count,
        encoding='utf-8'
    )
    handler.setLevel(level)
    
    if json_format:
        formatter = StructuredJSONFormatter(
            fmt='%(timestamp)s %(level)s %(name)s %(message)s',
            json_ensure_ascii=False
        )
    else:
        formatter = logging.Formatter(
            fmt=DEFAULT_LOG_FORMAT,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    handler.setFormatter(formatter)
    return handler


class LogManager:
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self._loggers: Dict[str, CockatooLogger] = {}
        self._handlers: List[logging.Handler] = []
        self._config: Optional[LoggingConfig] = None
        self._app_config: Optional[AppConfig] = None
        self._log_dir: Optional[Path] = None
        self._log_sensitive_data = DEFAULT_LOG_SENSITIVE_DATA
        self._anonymize_logs = DEFAULT_ANONYMIZE_LOGS
    
    def initialize(
        self,
        config: Optional[LoggingConfig] = None,
        app_config: Optional[AppConfig] = None,
        log_dir: Optional[Path] = None
    ) -> None:
        self._config = config
        self._app_config = app_config
        self._log_dir = log_dir or Path.home() / ".cockatoo" / "logs"
        
        self._configure_root_logger()
    
    def _configure_root_logger(self) -> None:
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        logging.setLoggerClass(CockatooLogger)
        
        level = logging.INFO
        if self._config:
            level = getattr(logging, self._config.level.value)
        root_logger.setLevel(level)
        
        if not self._config or self._config.console_enabled:
            console_handler = create_console_handler(
                level=level,
                json_format=False
            )
            root_logger.addHandler(console_handler)
            self._handlers.append(console_handler)
        
        if not self._config or self._config.file_enabled:
            if self._config and self._config.log_directory:
                log_dir = self._config.log_directory
            else:
                log_dir = self._log_dir
            
            log_file = log_dir / f"cockatoo_{datetime.now().strftime('%Y%m%d')}.log"
            
            file_handler = create_file_handler(
                log_file=log_file,
                level=level,
                max_bytes=(self._config.max_log_size_mb if self._config else DEFAULT_MAX_LOG_SIZE_MB) * BYTES_PER_MB,
                backup_count=self._config.max_log_files if self._config else DEFAULT_MAX_LOG_FILES,
                json_format=False
            )
            root_logger.addHandler(file_handler)
            self._handlers.append(file_handler)
    
    def get_logger(
        self,
        name: str,
        level: Optional[Union[str, int]] = None
    ) -> CockatooLogger:
        if name in self._loggers:
            return self._loggers[name]
        
        logger = logging.getLogger(name)
        
        if not isinstance(logger, CockatooLogger):
            logger.__class__ = CockatooLogger
        
        if level is not None:
            if isinstance(level, str):
                level = getattr(logging, level.upper())
            logger.setLevel(level)
        
        logger._log_sensitive_data = self._log_sensitive_data
        
        self._loggers[name] = logger
        return logger
    
    def set_log_level(self, level: Union[str, int]) -> None:
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        
        for handler in root_logger.handlers:
            handler.setLevel(level)
    
    def enable_json_logging(self, enable: bool = True) -> None:
        root_logger = logging.getLogger()
        
        for handler in root_logger.handlers:
            if enable:
                formatter = StructuredJSONFormatter(
                    fmt='%(timestamp)s %(level)s %(name)s %(message)s',
                    json_ensure_ascii=False
                )
            else:
                formatter = logging.Formatter(
                    fmt=DEFAULT_LOG_FORMAT,
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
            handler.setFormatter(formatter)
    
    def enable_sensitive_logging(self, enable: bool = True) -> None:
        self._log_sensitive_data = enable
        for logger in self._loggers.values():
            logger._log_sensitive_data = enable
    
    def add_handler(self, handler: logging.Handler) -> None:
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        self._handlers.append(handler)
    
    def remove_handler(self, handler: logging.Handler) -> None:
        root_logger = logging.getLogger()
        root_logger.removeHandler(handler)
        if handler in self._handlers:
            self._handlers.remove(handler)
    
    def get_log_files(self) -> List[Path]:
        log_files = []
        
        for handler in self._handlers:
            if isinstance(handler, (logging.handlers.RotatingFileHandler, 
                                    logging.handlers.TimedRotatingFileHandler)):
                if hasattr(handler, 'baseFilename'):
                    log_files.append(Path(handler.baseFilename))
        
        return log_files
    
    def archive_logs(self, archive_dir: Optional[Path] = None) -> Optional[Path]:
        import zipfile
        from datetime import datetime
        
        if archive_dir is None:
            archive_dir = self._log_dir / "archives"
        
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = archive_dir / f"logs_{timestamp}.zip"
        
        log_files = self.get_log_files()
        
        if not log_files:
            return None
        
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            used_names = set()
            for log_file in log_files:
                if log_file.exists():
                    base_name = log_file.name
                    arcname = base_name
                    counter = 1
                    
                    while arcname in used_names:
                        name_parts = base_name.split('.')
                        if len(name_parts) > 1:
                            arcname = f"{name_parts[0]}_{counter}.{name_parts[1]}"
                        else:
                            arcname = f"{base_name}_{counter}"
                        counter += 1
                    
                    used_names.add(arcname)
                    zipf.write(log_file, arcname)
        
        return archive_path
    
    def clear_logs(self, older_than_days: int = 30) -> int:
        import time
        
        deleted = 0
        cutoff_time = time.time() - (older_than_days * 24 * 3600)
        
        for log_file in self.get_log_files():
            if log_file.exists():
                mtime = log_file.stat().st_mtime
                if mtime < cutoff_time:
                    try:
                        log_file.unlink()
                        deleted += 1
                    except Exception:
                        pass
        
        return deleted
    
    def shutdown(self) -> None:
        logging.shutdown()


_log_manager = LogManager()


def get_logger(name: str) -> CockatooLogger:
    return _log_manager.get_logger(name)


def initialize_logging(
    config: Optional[LoggingConfig] = None,
    app_config: Optional[AppConfig] = None,
    log_dir: Optional[Path] = None
) -> None:
    _log_manager.initialize(config, app_config, log_dir)


def set_log_level(level: Union[str, int]) -> None:
    _log_manager.set_log_level(level)


def enable_json_logging(enable: bool = True) -> None:
    _log_manager.enable_json_logging(enable)


def enable_sensitive_logging(enable: bool = True) -> None:
    _log_manager.enable_sensitive_logging(enable)


def get_log_manager() -> LogManager:
    return _log_manager


def log_entry_exit(logger: Optional[CockatooLogger] = None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__module__)
            
            logger.debug(f"Entering {func.__name__}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"Exiting {func.__name__}")
                return result
            except Exception as e:
                logger.error(f"Exception in {func.__name__}: {e}")
                raise
        
        return wrapper
    return decorator


def log_error_context(logger: CockatooLogger, error: Exception, context: Dict[str, Any] = None):
    error_data = {
        'error_type': type(error).__name__,
        'error_message': str(error),
        'traceback': traceback.format_exc()
    }
    
    if context:
        error_data['context'] = context
    
    logger.error(
        f"Error occurred: {error}",
        extra={'error_data': error_data}
    )


class LogCapture:
    
    def __init__(self, logger_name: Optional[str] = None, level: int = logging.DEBUG):
        self.logger_name = logger_name
        self.level = level
        self.handler = None
        self.records = []
    
    def __enter__(self):
        logger = logging.getLogger(self.logger_name)
        
        class CapturingHandler(logging.Handler):
            def __init__(self, records):
                super().__init__()
                self.records = records
            
            def emit(self, record):
                self.records.append(record)
        
        self.handler = CapturingHandler(self.records)
        self.handler.setLevel(self.level)
        logger.addHandler(self.handler)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handler:
            logger = logging.getLogger(self.logger_name)
            logger.removeHandler(self.handler)
    
    def get_messages(self) -> List[str]:
        return [record.getMessage() for record in self.records]
    
    def get_records(self) -> List[logging.LogRecord]:
        return self.records.copy()
    
    def contains(self, text: str) -> bool:
        return any(text in msg for msg in self.get_messages())
    
    def clear(self) -> None:
        self.records.clear()


def main():
    print("Cockatoo Logging System Test")
    print("=" * 50)
    
    initialize_logging()
    
    logger = get_logger(__name__)
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.success("This is a success message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    
    logger.info(
        "User action",
        user="john_doe",
        action="login",
        ip_address="192.168.1.100"
    )
    
    logger.performance("Query executed in 0.45s", query="SELECT ...", duration=0.45)
    logger.metric("api_latency", 0.23, "seconds", endpoint="/api/documents")
    
    logger.audit(
        "User permissions changed",
        user="admin",
        target_user="john_doe",
        new_role="editor"
    )
    
    with CorrelationId(logger, "test-123"):
        logger.info("This message has correlation ID")
        
        with LoggingContext(logger, component="database", operation="query"):
            logger.info("This has context")
    
    try:
        1 / 0
    except Exception as e:
        logger.exception_with_context("Division error occurred", value=1)
    
    @log_performance(logger)
    def slow_function():
        time.sleep(0.1)
        return "done"
    
    slow_function()
    
    with LogCapture(__name__) as capture:
        logger.info("Test capture message")
        print(f"Captured: {capture.get_messages()}")
    
    print("\nLogging system test complete!")


if __name__ == "__main__":
    main()