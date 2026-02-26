# src/utilities/logger.py

"""Logging utilities with multiple levels, formats, and rotation support."""

import os
import sys
import logging
import traceback
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from datetime import datetime
from typing import Optional, Dict, Any, Union
from pathlib import Path
from enum import Enum
from functools import wraps
import json
import time


class LogLevel(Enum):
    """Log levels mapped to logging module constants."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class LogConfig:
    """Configuration for logger instances."""
    
    def __init__(
        self,
        log_dir: Union[str, Path] = "logs",
        log_level: LogLevel = LogLevel.INFO,
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5,
        format_string: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        date_format: str = "%Y-%m-%d %H:%M:%S",
        json_format: bool = False,
        console_output: bool = True,
        file_output: bool = True,
        rotation_type: str = "size",
        rotation_interval: int = 1,
        encoding: str = "utf-8"
    ):
        self.log_dir = Path(log_dir)
        self.log_level = log_level
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.format_string = format_string
        self.date_format = date_format
        self.json_format = json_format
        self.console_output = console_output
        self.file_output = file_output
        self.rotation_type = rotation_type
        self.rotation_interval = rotation_interval
        self.encoding = encoding
        
        if file_output:
            self.log_dir.mkdir(parents=True, exist_ok=True)


class JSONFormatter(logging.Formatter):
    """Formatter that outputs logs as JSON."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        if hasattr(record, 'extra'):
            log_entry["extra"] = record.extra
        
        return json.dumps(log_entry)


class LogContext:
    """Context manager for adding temporary context to logs."""
    
    def __init__(self, logger: logging.Logger, **kwargs):
        self.logger = logger
        self.context = kwargs
        self.old_context = {}
    
    def __enter__(self):
        if hasattr(self.logger, 'extra'):
            self.old_context = getattr(self.logger, 'extra', {})
            self.logger.extra = {**self.old_context, **self.context}
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self.logger, 'extra'):
            self.logger.extra = self.old_context


class Logger:
    """Wrapper around logging.Logger with enhanced functionality."""
    
    def __init__(self, name: str, config: Optional[LogConfig] = None):
        self.name = name
        self.config = config or LogConfig()
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.config.log_level.value)
        self.logger.propagate = False
        
        self.logger.handlers.clear()
        self._setup_handlers()
        self.extra = {}
    
    def _setup_handlers(self):
        if self.config.json_format:
            formatter = JSONFormatter()
        else:
            formatter = logging.Formatter(
                self.config.format_string,
                datefmt=self.config.date_format
            )
        
        if self.config.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        if self.config.file_output:
            log_file = self.config.log_dir / f"{self.name}.log"
            
            if self.config.rotation_type == "size":
                file_handler = RotatingFileHandler(
                    log_file,
                    maxBytes=self.config.max_bytes,
                    backupCount=self.config.backup_count,
                    encoding=self.config.encoding
                )
            else:
                file_handler = TimedRotatingFileHandler(
                    log_file,
                    when="midnight",
                    interval=self.config.rotation_interval,
                    backupCount=self.config.backup_count,
                    encoding=self.config.encoding
                )
            
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def _log(self, level: LogLevel, message: str, **kwargs):
        extra = {**self.extra, **kwargs}
        
        if extra:
            old_factory = logging.getLogRecordFactory()
            
            def record_factory(*args, **kwargs):
                record = old_factory(*args, **kwargs)
                record.extra = extra
                return record
            
            logging.setLogRecordFactory(record_factory)
            self.logger.log(level.value, message)
            logging.setLogRecordFactory(old_factory)
        else:
            self.logger.log(level.value, message)
    
    def debug(self, message: str, **kwargs):
        """Log a debug message."""
        self._log(LogLevel.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log an info message."""
        self._log(LogLevel.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log a warning message."""
        self._log(LogLevel.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log an error message."""
        self._log(LogLevel.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log a critical message."""
        self._log(LogLevel.CRITICAL, message, **kwargs)
    
    def exception(self, message: str, **kwargs):
        """Log an exception with traceback."""
        self._log(LogLevel.ERROR, message, exc_info=True, **kwargs)
    
    def with_context(self, **kwargs) -> 'Logger':
        """Create a new logger with added context."""
        new_logger = Logger(self.name, self.config)
        new_logger.extra = {**self.extra, **kwargs}
        return new_logger


_loggers: Dict[str, Logger] = {}
_default_config: Optional[LogConfig] = None


def get_logger(name: str, config: Optional[LogConfig] = None) -> Logger:
    """Get or create a logger instance."""
    if name not in _loggers:
        _loggers[name] = Logger(name, config or _default_config)
    return _loggers[name]


def setup_logging(config: LogConfig):
    """Set up logging with the given configuration."""
    global _default_config
    _default_config = config
    
    root_logger = get_logger("root", config)
    
    for logger in _loggers.values():
        logger.config = config
        logger._setup_handlers()


def log_execution_time(logger: Optional[Logger] = None):
    """Decorator to log function execution time."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__module__)
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.debug(f"Function {func.__name__} executed in {execution_time:.3f}s")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"Function {func.__name__} failed after {execution_time:.3f}s: {str(e)}")
                raise
        
        return wrapper
    return decorator