# src/utilities/__init__.py

"""Utilities package providing core functionality."""

from .cleanup import (
    CleanupManager, CleanupTask, CleanupPolicy,
    TempFileCleaner, CacheCleaner, cleanup_on_exit,
    get_cleanup_manager, register_temp_dir, register_temp_file,
    create_temp_dir, create_temp_file
)

from .formatter import (
    FormatType, LanguageCode, FormattingOptions,
    BaseFormatter, TextFormatter, DocumentFormatter,
    ChatMessageFormatter, ExportFormatter, FormatterFactory,
    format_text, format_document_metadata, format_chat_message,
    export_to_markdown
)

from .helpers import (
    chunk_list, flatten_list, ensure_list,
    merge_dicts, safe_get, safe_divide,
    parse_bool, format_timedelta, slugify,
    truncate_string, generate_id, Timer,
    generate_uuid, generate_hash
)

from .logger import (
    Logger, LogLevel, LogConfig, get_logger,
    setup_logging, log_execution_time, LogContext
)

from .monitor import (
    Monitor, MetricType, Metric, MonitorConfig,
    SystemMonitor, PerformanceMonitor, get_monitor
)

from .retry import (
    retry, async_retry, RetryConfig, RetryError,
    exponential_backoff, fixed_delay, linear_delay,
    random_delay, RetryStrategy
)

from .task_queue import (
    TaskQueue, Task, TaskStatus, TaskPriority,
    Worker, QueueManager, TaskResult
)

from .validator import (
    Validator, ValidationResult, ValidationRule,
    SchemaValidator, DataValidator, validate_email,
    validate_url, validate_json, validate_xml,
    validate_ip_address, validate_phone_number,
    validate_date_string
)

__version__ = "1.0.0"

__all__ = [
    # Cleanup
    'CleanupManager', 'CleanupTask', 'CleanupPolicy',
    'TempFileCleaner', 'CacheCleaner', 'cleanup_on_exit',
    'get_cleanup_manager', 'register_temp_dir', 'register_temp_file',
    'create_temp_dir', 'create_temp_file',

    # Formatter
    'FormatType', 'LanguageCode', 'FormattingOptions',
    'BaseFormatter', 'TextFormatter', 'DocumentFormatter',
    'ChatMessageFormatter', 'ExportFormatter', 'FormatterFactory',
    'format_text', 'format_document_metadata', 'format_chat_message',
    'export_to_markdown',

    # Helpers
    'chunk_list', 'flatten_list', 'ensure_list',
    'merge_dicts', 'safe_get', 'safe_divide',
    'parse_bool', 'format_timedelta', 'slugify',
    'truncate_string', 'generate_id', 'Timer',
    'generate_uuid', 'generate_hash',

    # Logger
    'Logger', 'LogLevel', 'LogConfig', 'get_logger',
    'setup_logging', 'log_execution_time', 'LogContext',

    # Monitor
    'Monitor', 'MetricType', 'Metric', 'MonitorConfig',
    'SystemMonitor', 'PerformanceMonitor', 'get_monitor',

    # Retry
    'retry', 'async_retry', 'RetryConfig', 'RetryError',
    'exponential_backoff', 'fixed_delay', 'linear_delay',
    'random_delay', 'RetryStrategy',

    # Task Queue
    'TaskQueue', 'Task', 'TaskStatus', 'TaskPriority',
    'Worker', 'QueueManager', 'TaskResult',

    # Validator
    'Validator', 'ValidationResult', 'ValidationRule',
    'SchemaValidator', 'DataValidator', 'validate_email',
    'validate_url', 'validate_json', 'validate_xml',
    'validate_ip_address', 'validate_phone_number',
    'validate_date_string'
]