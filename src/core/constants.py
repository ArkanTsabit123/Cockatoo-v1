# cockatoo_v1/src/core/constants.py
"""
Centralized constants for Cockatoo_V1 application.

This module provides all constant values used throughout the application,
including defaults, ranges, supported values, and error messages.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import re

APP_NAME: str = "Cockatoo"
APP_VERSION: str = "1.0.0"
APP_DESCRIPTION: str = "AI-powered document intelligence system"
APP_AUTHOR: str = "Cockatoo_V1 Team"
APP_LICENSE: str = "MIT"
APP_WEBSITE: str = "https://github.com/cockatoo-v1"
CONFIG_VERSION: str = "1.0.0"


DEFAULT_CONFIG_DIR_NAME: str = "config"
DEFAULT_DATA_DIR_NAME: str = "data"
DEFAULT_MODELS_DIR_NAME: str = "models"
DEFAULT_DOCUMENTS_DIR_NAME: str = "documents"
DEFAULT_DATABASE_DIR_NAME: str = "database"
DEFAULT_LOGS_DIR_NAME: str = "logs"
DEFAULT_EXPORTS_DIR_NAME: str = "exports"
DEFAULT_CACHE_DIR_NAME: str = "cache"
DEFAULT_TEMP_DIR_NAME: str = "temp"
DEFAULT_BACKUP_DIR_NAME: str = "backups"


CONFIG_FILE_NAME: str = "config"
CONFIG_FILE_EXTENSIONS: Dict[str, str] = {
    "yaml": ".yaml",
    "yml": ".yml",
    "json": ".json",
    "toml": ".toml",
}
SUPPORTED_CONFIG_FORMATS: List[str] = ["yaml", "yml", "json", "toml"]
DEFAULT_CONFIG_FORMAT: str = "yaml"


DEFAULT_CHUNK_SIZE: int = 500
DEFAULT_CHUNK_OVERLAP: int = 50
MIN_CHUNK_SIZE: int = 100
MAX_CHUNK_SIZE: int = 2000
MIN_CHUNK_OVERLAP: int = 0
MAX_CHUNK_OVERLAP: int = 500
RECOMMENDED_CHUNK_SIZE_MIN: int = 200
RECOMMENDED_CHUNK_SIZE_MAX: int = 1000
RECOMMENDED_CHUNK_OVERLAP_MIN: int = 10
RECOMMENDED_CHUNK_OVERLAP_MAX: int = 200


DEFAULT_MAX_FILE_SIZE_MB: int = 100
MAX_FILE_SIZE_MB_MIN: int = 1
MAX_FILE_SIZE_MB_MAX: int = 1024
DEFAULT_MAX_PAGES_PER_DOCUMENT: int = 1000
MAX_PAGES_PER_DOCUMENT_MIN: int = 1
MAX_PAGES_PER_DOCUMENT_MAX: int = 10000


DEFAULT_SUPPORTED_FORMATS: List[str] = [
    ".pdf", ".docx", ".txt", ".md", ".html", 
    ".epub", ".jpg", ".png", ".csv", ".pptx", ".xlsx"
]
ALL_SUPPORTED_FORMATS: List[str] = [
    ".pdf", ".docx", ".doc", ".odt", ".rtf", ".txt", ".md", ".rst",
    ".html", ".htm", ".xml", ".json", ".yaml", ".yml",
    ".epub", ".mobi", ".azw3", ".fb2",
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp",
    ".csv", ".xlsx", ".xls", ".ods",
    ".pptx", ".ppt", ".odp",
    ".py", ".js", ".java", ".cpp", ".c", ".h", ".rs", ".go",
]


DEFAULT_OCR_ENABLED: bool = True
DEFAULT_OCR_LANGUAGES: List[str] = ["eng", "ind"]
SUPPORTED_OCR_LANGUAGES: List[str] = [
    "afr", "amh", "ara", "asm", "aze", "aze_cyrl", "bel", "ben", "bod", "bos",
    "bul", "cat", "ceb", "ces", "chi_sim", "chi_tra", "chr", "cym", "dan",
    "deu", "dzo", "ell", "eng", "enm", "epo", "est", "eus", "fas", "fin",
    "fra", "frk", "frm", "gle", "glg", "grc", "guj", "hat", "heb", "hin",
    "hrv", "hun", "iku", "ind", "isl", "ita", "ita_old", "jav", "jpn",
    "kan", "kat", "kat_old", "kaz", "khm", "kir", "kor", "kur", "lao",
    "lat", "lav", "lit", "mal", "mar", "mkd", "mlt", "msa", "mya", "nep",
    "nld", "nor", "ori", "osd", "pan", "pol", "por", "pus", "ron", "rus",
    "san", "sin", "slk", "slv", "snd", "spa", "spa_old", "sqi", "srp",
    "srp_latn", "swa", "swe", "syr", "tam", "tel", "tgk", "tgl", "tha",
    "tir", "tur", "uig", "ukr", "urd", "uzb", "uzb_cyrl", "vie", "yid",
]


DEFAULT_EXTRACT_TABLES: bool = True
DEFAULT_EXTRACT_IMAGES: bool = True
DEFAULT_PRESERVE_FORMATTING: bool = True
DEFAULT_PARALLEL_PROCESSING: bool = True
DEFAULT_MAX_PARALLEL_FILES: int = 4
MAX_PARALLEL_FILES_MIN: int = 1
MAX_PARALLEL_FILES_MAX: int = 16


DEFAULT_LLM_PROVIDER: str = "ollama"
SUPPORTED_LLM_PROVIDERS: List[str] = ["ollama", "openai", "anthropic", "cohere", "huggingface", "llamacpp", "vllm"]
DEFAULT_LLM_MODEL: str = "llama2:7b"
DEFAULT_LLM_BASE_URL: str = "http://localhost:11434"


DEFAULT_TEMPERATURE: float = 0.1
DEFAULT_TOP_P: float = 0.9
DEFAULT_TOP_K: int = 40
DEFAULT_MAX_TOKENS: int = 1024
DEFAULT_CONTEXT_WINDOW: int = 4096
DEFAULT_TIMEOUT: int = 60
DEFAULT_MAX_RETRIES: int = 3
DEFAULT_STREAM: bool = False
DEFAULT_ECHO: bool = False


TEMPERATURE_MIN: float = 0.0
TEMPERATURE_MAX: float = 2.0
TOP_P_MIN: float = 0.0
TOP_P_MAX: float = 1.0
TOP_K_MIN: int = 1
TOP_K_MAX: int = 100
MAX_TOKENS_MIN: int = 1
MAX_TOKENS_MAX: int = 32768
CONTEXT_WINDOW_MIN: int = 512
CONTEXT_WINDOW_MAX: int = 131072


DEFAULT_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
SUPPORTED_EMBEDDING_MODELS: List[str] = [
    "all-MiniLM-L6-v2", "all-mpnet-base-v2", "multi-qa-mpnet-base-dot-v1",
    "text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large",
    "BAAI/bge-small-en", "BAAI/bge-base-en", "BAAI/bge-large-en",
    "intfloat/e5-small-v2", "intfloat/e5-base-v2", "intfloat/e5-large-v2",
]
DEFAULT_EMBEDDING_DIMENSIONS: int = 384
SUPPORTED_EMBEDDING_DEVICES: List[str] = ["auto", "cpu", "cuda", "mps"]
DEFAULT_EMBEDDING_DEVICE: str = "auto"
DEFAULT_EMBEDDING_CACHE_ENABLED: bool = True
DEFAULT_EMBEDDING_CACHE_SIZE_MB: int = 500
DEFAULT_EMBEDDING_BATCH_SIZE: int = 32
DEFAULT_NORMALIZE_EMBEDDINGS: bool = True


DEFAULT_RAG_TOP_K: int = 5
DEFAULT_SIMILARITY_THRESHOLD: float = 0.7
DEFAULT_MAX_CONTEXT_LENGTH: int = 3000
DEFAULT_ENABLE_HYBRID_SEARCH: bool = True
DEFAULT_BM25_WEIGHT: float = 0.5
DEFAULT_RERANK_RESULTS: bool = False
DEFAULT_RERANKER_MODEL: Optional[str] = None
DEFAULT_CHUNK_SEPARATOR: str = "\n\n"
DEFAULT_PRESERVE_METADATA: bool = True


RAG_TOP_K_MIN: int = 1
RAG_TOP_K_MAX: int = 50
SIMILARITY_THRESHOLD_MIN: float = 0.0
SIMILARITY_THRESHOLD_MAX: float = 1.0
MAX_CONTEXT_LENGTH_MIN: int = 100
MAX_CONTEXT_LENGTH_MAX: int = 10000
BM25_WEIGHT_MIN: float = 0.0
BM25_WEIGHT_MAX: float = 1.0


SUPPORTED_DATABASES: List[str] = ["chroma", "qdrant", "pgvector", "sqlite"]
DEFAULT_DATABASE_TYPE: str = "chroma"


DEFAULT_DATABASE_HOST: Optional[str] = None
DEFAULT_DATABASE_PORT: Optional[int] = None
DEFAULT_DATABASE_NAME: Optional[str] = None


CHROMA_DEFAULT_PATH: str = "chroma"
CHROMA_DEFAULT_COLLECTION: str = "documents"


QDRANT_DEFAULT_PORT: int = 6333
QDRANT_DEFAULT_GRPC_PORT: int = 6334
QDRANT_DEFAULT_COLLECTION: str = "documents"


PGVECTOR_DEFAULT_PORT: int = 5432
PGVECTOR_DEFAULT_TABLE: str = "embeddings"


SQLITE_DEFAULT_DB_FILE: str = "cockatoo.db"


DEFAULT_UI_THEME: str = "dark"
SUPPORTED_THEMES: List[str] = ["light", "dark", "auto", "system"]


DEFAULT_LANGUAGE: str = "en"
SUPPORTED_LANGUAGES: List[str] = ["en", "id", "es", "fr", "de", "zh", "ja", "ko"]


DEFAULT_FONT_FAMILY: str = "Segoe UI, Inter, -apple-system, BlinkMacSystemFont, Roboto, sans-serif"
DEFAULT_FONT_SIZE: int = 12
FONT_SIZE_MIN: int = 8
FONT_SIZE_MAX: int = 24
RECOMMENDED_FONT_SIZE_MIN: int = 10
RECOMMENDED_FONT_SIZE_MAX: int = 20
DEFAULT_LINE_HEIGHT: float = 1.5
LINE_HEIGHT_MIN: float = 1.0
LINE_HEIGHT_MAX: float = 2.5


DEFAULT_ENABLE_ANIMATIONS: bool = True
DEFAULT_ANIMATION_DURATION: int = 200
ANIMATION_DURATION_MIN: int = 0
ANIMATION_DURATION_MAX: int = 1000


DEFAULT_AUTO_SAVE: bool = True
DEFAULT_AUTO_SAVE_INTERVAL: int = 60
AUTO_SAVE_INTERVAL_MIN: int = 10
AUTO_SAVE_INTERVAL_MAX: int = 3600


DEFAULT_SHOW_TOOLTIPS: bool = True
DEFAULT_TOOLTIP_DELAY: int = 500
TOOLTIP_DELAY_MIN: int = 0
TOOLTIP_DELAY_MAX: int = 5000


DEFAULT_MAX_RECENT_FILES: int = 10
MAX_RECENT_FILES_MIN: int = 0
MAX_RECENT_FILES_MAX: int = 50


DEFAULT_CONFIRM_BEFORE_EXIT: bool = True


DEFAULT_MAX_DOCUMENTS: int = 10000
MAX_DOCUMENTS_MIN: int = 0
MAX_DOCUMENTS_MAX: int = 1000000
DEFAULT_MAX_DOCUMENT_SIZE_MB: int = 50


DEFAULT_AUTO_CLEANUP: bool = True
DEFAULT_AUTO_CLEANUP_DAYS: int = 90
AUTO_CLEANUP_DAYS_MIN: int = 1
AUTO_CLEANUP_DAYS_MAX: int = 365
DEFAULT_CLEANUP_INTERVAL_HOURS: int = 24
CLEANUP_INTERVAL_HOURS_MIN: int = 1
CLEANUP_INTERVAL_HOURS_MAX: int = 168


DEFAULT_BACKUP_ENABLED: bool = True
DEFAULT_BACKUP_INTERVAL_HOURS: int = 24
BACKUP_INTERVAL_HOURS_MIN: int = 1
BACKUP_INTERVAL_HOURS_MAX: int = 168
DEFAULT_MAX_BACKUPS: int = 10
MAX_BACKUPS_MIN: int = 1
MAX_BACKUPS_MAX: int = 100


DEFAULT_ENCRYPTION_ENABLED: bool = False
DEFAULT_ENCRYPTION_KEY: Optional[str] = None
ENCRYPTION_KEY_LENGTH: int = 32


DEFAULT_COMPRESSION_ENABLED: bool = True
DEFAULT_COMPRESSION_LEVEL: int = 6
COMPRESSION_LEVEL_MIN: int = 1
COMPRESSION_LEVEL_MAX: int = 9


DEFAULT_MAX_WORKERS: int = 4
MAX_WORKERS_MIN: int = 1
MAX_WORKERS_MAX: int = 32


DEFAULT_CACHE_ENABLED: bool = True
DEFAULT_CACHE_SIZE_MB: int = 500
CACHE_SIZE_MB_MIN: int = 10
CACHE_SIZE_MB_MAX: int = 10000
DEFAULT_CACHE_TTL_SECONDS: int = 3600
CACHE_TTL_SECONDS_MIN: int = 60
CACHE_TTL_SECONDS_MAX: int = 86400


DEFAULT_ENABLE_MONITORING: bool = True
DEFAULT_MONITOR_INTERVAL_SECONDS: int = 60
MONITOR_INTERVAL_SECONDS_MIN: int = 10
MONITOR_INTERVAL_SECONDS_MAX: int = 3600


DEFAULT_MEMORY_LIMIT_MB: Optional[int] = None
MEMORY_LIMIT_MB_MIN: int = 100
MEMORY_LIMIT_MB_MAX: int = 32768
DEFAULT_GPU_MEMORY_LIMIT_MB: Optional[int] = None
GPU_MEMORY_LIMIT_MB_MIN: int = 100
GPU_MEMORY_LIMIT_MB_MAX: int = 32768


DEFAULT_PRELOAD_MODELS: bool = False
DEFAULT_LAZY_LOADING: bool = True


DEFAULT_TELEMETRY_ENABLED: bool = False
DEFAULT_CRASH_REPORTS_ENABLED: bool = False
DEFAULT_USAGE_STATISTICS_ENABLED: bool = False


DEFAULT_AUTO_UPDATE_CHECK: bool = False
SUPPORTED_UPDATE_CHANNELS: List[str] = ["stable", "beta", "dev"]
DEFAULT_UPDATE_CHANNEL: str = "stable"


DEFAULT_DATA_COLLECTION_CONSENT: bool = False
DEFAULT_ANALYTICS_ID: Optional[str] = None


DEFAULT_LOG_SENSITIVE_DATA: bool = False
DEFAULT_ANONYMIZE_LOGS: bool = True


DEFAULT_LOG_LEVEL: str = "INFO"
SUPPORTED_LOG_LEVELS: List[str] = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


DEFAULT_LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"


DEFAULT_LOG_FILE_ENABLED: bool = True
DEFAULT_CONSOLE_ENABLED: bool = True
DEFAULT_MAX_LOG_SIZE_MB: int = 10
MAX_LOG_SIZE_MB_MIN: int = 1
MAX_LOG_SIZE_MB_MAX: int = 100
DEFAULT_MAX_LOG_FILES: int = 5
MAX_LOG_FILES_MIN: int = 1
MAX_LOG_FILES_MAX: int = 50


NUMERIC_RANGES: Dict[str, Tuple[float, float]] = {
    "chunk_size": (MIN_CHUNK_SIZE, MAX_CHUNK_SIZE),
    "chunk_overlap": (MIN_CHUNK_OVERLAP, MAX_CHUNK_OVERLAP),
    "max_file_size_mb": (MAX_FILE_SIZE_MB_MIN, MAX_FILE_SIZE_MB_MAX),
    "max_pages_per_document": (MAX_PAGES_PER_DOCUMENT_MIN, MAX_PAGES_PER_DOCUMENT_MAX),
    "max_parallel_files": (MAX_PARALLEL_FILES_MIN, MAX_PARALLEL_FILES_MAX),
    "temperature": (TEMPERATURE_MIN, TEMPERATURE_MAX),
    "top_p": (TOP_P_MIN, TOP_P_MAX),
    "top_k": (TOP_K_MIN, TOP_K_MAX),
    "max_tokens": (MAX_TOKENS_MIN, MAX_TOKENS_MAX),
    "context_window": (CONTEXT_WINDOW_MIN, CONTEXT_WINDOW_MAX),
    "rag_top_k": (RAG_TOP_K_MIN, RAG_TOP_K_MAX),
    "similarity_threshold": (SIMILARITY_THRESHOLD_MIN, SIMILARITY_THRESHOLD_MAX),
    "max_context_length": (MAX_CONTEXT_LENGTH_MIN, MAX_CONTEXT_LENGTH_MAX),
    "bm25_weight": (BM25_WEIGHT_MIN, BM25_WEIGHT_MAX),
    "font_size": (FONT_SIZE_MIN, FONT_SIZE_MAX),
    "line_height": (LINE_HEIGHT_MIN, LINE_HEIGHT_MAX),
    "animation_duration": (ANIMATION_DURATION_MIN, ANIMATION_DURATION_MAX),
    "auto_save_interval": (AUTO_SAVE_INTERVAL_MIN, AUTO_SAVE_INTERVAL_MAX),
    "tooltip_delay": (TOOLTIP_DELAY_MIN, TOOLTIP_DELAY_MAX),
    "max_recent_files": (MAX_RECENT_FILES_MIN, MAX_RECENT_FILES_MAX),
    "max_documents": (MAX_DOCUMENTS_MIN, MAX_DOCUMENTS_MAX),
    "max_document_size_mb": (MAX_FILE_SIZE_MB_MIN, MAX_FILE_SIZE_MB_MAX),
    "auto_cleanup_days": (AUTO_CLEANUP_DAYS_MIN, AUTO_CLEANUP_DAYS_MAX),
    "cleanup_interval_hours": (CLEANUP_INTERVAL_HOURS_MIN, CLEANUP_INTERVAL_HOURS_MAX),
    "backup_interval_hours": (BACKUP_INTERVAL_HOURS_MIN, BACKUP_INTERVAL_HOURS_MAX),
    "max_backups": (MAX_BACKUPS_MIN, MAX_BACKUPS_MAX),
    "compression_level": (COMPRESSION_LEVEL_MIN, COMPRESSION_LEVEL_MAX),
    "max_workers": (MAX_WORKERS_MIN, MAX_WORKERS_MAX),
    "cache_size_mb": (CACHE_SIZE_MB_MIN, CACHE_SIZE_MB_MAX),
    "cache_ttl_seconds": (CACHE_TTL_SECONDS_MIN, CACHE_TTL_SECONDS_MAX),
    "monitor_interval_seconds": (MONITOR_INTERVAL_SECONDS_MIN, MONITOR_INTERVAL_SECONDS_MAX),
    "memory_limit_mb": (MEMORY_LIMIT_MB_MIN, MEMORY_LIMIT_MB_MAX),
    "gpu_memory_limit_mb": (GPU_MEMORY_LIMIT_MB_MIN, GPU_MEMORY_LIMIT_MB_MAX),
    "max_log_size_mb": (MAX_LOG_SIZE_MB_MIN, MAX_LOG_SIZE_MB_MAX),
    "max_log_files": (MAX_LOG_FILES_MIN, MAX_LOG_FILES_MAX),
}


RECOMMENDED_RANGES: Dict[str, Tuple[float, float]] = {
    "chunk_size": (RECOMMENDED_CHUNK_SIZE_MIN, RECOMMENDED_CHUNK_SIZE_MAX),
    "chunk_overlap": (RECOMMENDED_CHUNK_OVERLAP_MIN, RECOMMENDED_CHUNK_OVERLAP_MAX),
    "font_size": (RECOMMENDED_FONT_SIZE_MIN, RECOMMENDED_FONT_SIZE_MAX),
}


ENV_VAR_PREFIX: str = "COCKATOO_"

ENV_VAR_MAPPING: Dict[str, str] = {
    f"{ENV_VAR_PREFIX}LLM_MODEL": "llm_model",
    f"{ENV_VAR_PREFIX}CHUNK_SIZE": "chunk_size",
    f"{ENV_VAR_PREFIX}CHUNK_OVERLAP": "chunk_overlap",
    f"{ENV_VAR_PREFIX}LLM_TEMPERATURE": "llm_temperature",
    f"{ENV_VAR_PREFIX}OCR_ENABLED": "ocr_enabled",
    f"{ENV_VAR_PREFIX}OCR_LANGUAGES": "ocr_languages",
    f"{ENV_VAR_PREFIX}UI_THEME": "ui_theme",
    f"{ENV_VAR_PREFIX}UI_ANIMATIONS": "ui_animations",
    f"{ENV_VAR_PREFIX}RAG_HYBRID_SEARCH": "rag_hybrid_search",
    f"{ENV_VAR_PREFIX}PRIVACY_TELEMETRY": "privacy_telemetry",
    f"{ENV_VAR_PREFIX}DATA_DIR": "data_dir",
    f"{ENV_VAR_PREFIX}LOG_LEVEL": "log_level",
    f"{ENV_VAR_PREFIX}DATABASE_TYPE": "database_type",
    f"{ENV_VAR_PREFIX}DATABASE_HOST": "database_host",
    f"{ENV_VAR_PREFIX}DATABASE_PORT": "database_port",
    f"{ENV_VAR_PREFIX}DATABASE_NAME": "database_name",
}


ERROR_MESSAGES: Dict[str, str] = {
    "config_not_found": "Configuration file not found: {path}",
    "config_permission_denied": "Permission denied accessing config file: {path}",
    "config_invalid_yaml": "Invalid YAML format in config file: {error}",
    "config_invalid_json": "Invalid JSON format in config file: {error}",
    "config_invalid_toml": "Invalid TOML format in config file: {error}",
    "config_validation_failed": "Configuration validation failed: {errors}",
    "config_key_not_found": "Configuration key not found: {key}",
    "config_version_mismatch": "Config version mismatch. Expected {expected}, got {actual}",
    "path_not_found": "Path not found: {path}",
    "path_not_writable": "Path is not writable: {path}",
    "path_permission_denied": "Permission denied for path: {path}",
    "path_not_directory": "Path is not a directory: {path}",
    "path_not_file": "Path is not a file: {path}",
    "unsupported_format": "Unsupported file format: {format}. Supported formats: {supported}",
    "file_too_large": "File size exceeds limit: {size} MB > {limit} MB",
    "ocr_failed": "OCR processing failed for file {file}: {error}",
    "extraction_failed": "Document extraction failed: {error}",
    "chunking_failed": "Document chunking failed: {error}",
    "llm_connection_failed": "Failed to connect to LLM provider {provider}: {error}",
    "llm_request_failed": "LLM request failed: {error}",
    "llm_timeout": "LLM request timed out after {timeout} seconds",
    "llm_invalid_response": "Invalid response from LLM: {error}",
    "embedding_failed": "Embedding generation failed: {error}",
    "rag_retrieval_failed": "RAG retrieval failed: {error}",
    "database_connection_failed": "Database connection failed: {error}",
    "storage_write_failed": "Failed to write to storage: {error}",
    "storage_read_failed": "Failed to read from storage: {error}",
    "backup_failed": "Backup failed: {error}",
    "encryption_failed": "Encryption failed: {error}",
    "compression_failed": "Compression failed: {error}",
    "cache_write_failed": "Failed to write to cache: {error}",
    "cache_read_failed": "Failed to read from cache: {error}",
    "resource_limit_exceeded": "Resource limit exceeded: {resource} = {value} > {limit}",
    "memory_limit_exceeded": "Memory limit exceeded: {used} MB > {limit} MB",
    "value_out_of_range": "{field} value {value} out of range [{min}, {max}]",
    "value_out_of_recommended": "{field} value {value} outside recommended range [{min}, {max}]",
    "invalid_enum_value": "Invalid {field} value: {value}. Must be one of: {allowed}",
    "required_field_missing": "Required field missing: {field}",
    "invalid_format": "Invalid format for {field}: {value}",
    "chunk_overlap_too_large": "chunk_overlap ({overlap}) must be less than chunk_size ({size})",
    "encryption_key_required": "Encryption key is required when encryption is enabled",
    "database_config_required": "Database host and name required for external database",
    "telemetry_consent_required": "Telemetry consent is required to enable telemetry",
    "data_collection_consent_required": "Data collection consent is required",
    "log_file_creation_failed": "Failed to create log file: {error}",
    "log_directory_not_writable": "Log directory is not writable: {path}",
    "app_initialization_failed": "Failed to initialize application: {error}",
    "app_runtime_error": "Application runtime error: {error}",
    "app_shutdown_failed": "Failed to shutdown application: {error}",
}


EMAIL_PATTERN: re.Pattern = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
URL_PATTERN: re.Pattern = re.compile(r"^https?://[^\s/$.?#].[^\s]*$")
FILE_EXTENSION_PATTERN: re.Pattern = re.compile(r"^\.[a-zA-Z0-9]+$")
LANGUAGE_CODE_PATTERN: re.Pattern = re.compile(r"^[a-z]{3}$")
VERSION_PATTERN: re.Pattern = re.compile(r"^\d+\.\d+\.\d+$")
PATH_PATTERN: re.Pattern = re.compile(r"^[^\0]+$")


MIME_TYPES: Dict[str, str] = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc": "application/msword",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".html": "text/html",
    ".htm": "text/html",
    ".epub": "application/epub+zip",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".csv": "text/csv",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xls": "application/vnd.ms-excel",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".ppt": "application/vnd.ms-powerpoint",
    ".json": "application/json",
    ".yaml": "application/x-yaml",
    ".yml": "application/x-yaml",
    ".xml": "application/xml",
}


BYTES_PER_MB: int = 1024 * 1024
BYTES_PER_GB: int = 1024 * 1024 * 1024
MS_PER_SECOND: int = 1000
SECONDS_PER_MINUTE: int = 60
MINUTES_PER_HOUR: int = 60
HOURS_PER_DAY: int = 24