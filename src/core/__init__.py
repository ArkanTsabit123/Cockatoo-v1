# cockatoo_v1/src/core/__init__.py
"""
Core module for Cockatoo_V1 application.

This module provides configuration management, constants, exceptions,
and core application functionality.
"""

from .config import (
    # Main configuration classes
    AppConfig,
    ConfigManager,
    
    # Configuration sections
    AppInfoConfig,
    PathsConfig,
    DocumentProcessingConfig,
    AIConfig,
    UIConfig,
    StorageConfig,
    PerformanceConfig,
    PrivacyConfig,
    LoggingConfig,
    
    # Enums
    PlatformType,
    LogLevel,
    ThemeMode,
    DatabaseType,
    
    # Global functions
    get_config,
    init_config,
    save_config,
    reload_config,
    setup_crossplatform_environment,
    
    # Constants
    TOML_WRITER_AVAILABLE,
    
    # Internal (needed for tests)
    _default_config_manager,
)

from .constants import (
    # Application constants
    APP_NAME,
    APP_VERSION,
    APP_DESCRIPTION,
    APP_AUTHOR,
    APP_LICENSE,
    APP_WEBSITE,
    CONFIG_VERSION,
    
    # Path constants
    DEFAULT_CONFIG_DIR_NAME,
    DEFAULT_DATA_DIR_NAME,
    DEFAULT_MODELS_DIR_NAME,
    DEFAULT_DOCUMENTS_DIR_NAME,
    DEFAULT_DATABASE_DIR_NAME,
    DEFAULT_LOGS_DIR_NAME,
    DEFAULT_EXPORTS_DIR_NAME,
    DEFAULT_CACHE_DIR_NAME,
    DEFAULT_TEMP_DIR_NAME,
    DEFAULT_BACKUP_DIR_NAME,
    
    # File constants
    CONFIG_FILE_NAME,
    CONFIG_FILE_EXTENSIONS,
    SUPPORTED_CONFIG_FORMATS,
    DEFAULT_CONFIG_FORMAT,
    
    # Document processing constants
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    MIN_CHUNK_SIZE,
    MAX_CHUNK_SIZE,
    MIN_CHUNK_OVERLAP,
    MAX_CHUNK_OVERLAP,
    DEFAULT_MAX_FILE_SIZE_MB,
    DEFAULT_MAX_PAGES_PER_DOCUMENT,
    DEFAULT_SUPPORTED_FORMATS,
    DEFAULT_OCR_LANGUAGES,
    SUPPORTED_OCR_LANGUAGES,
    
    # AI/ML constants
    DEFAULT_LLM_PROVIDER,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_TOP_K,
    DEFAULT_MAX_TOKENS,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_DIMENSIONS,
    DEFAULT_RAG_TOP_K,
    DEFAULT_SIMILARITY_THRESHOLD,
    SUPPORTED_LLM_PROVIDERS,
    SUPPORTED_EMBEDDING_MODELS,
    
    # Database constants
    SUPPORTED_DATABASES,
    DEFAULT_DATABASE_TYPE,
    
    # UI constants
    DEFAULT_UI_THEME,
    SUPPORTED_THEMES,
    DEFAULT_LANGUAGE,
    SUPPORTED_LANGUAGES,
    DEFAULT_FONT_SIZE,
    DEFAULT_FONT_FAMILY,
    DEFAULT_ANIMATION_DURATION,
    DEFAULT_AUTO_SAVE_INTERVAL,
    DEFAULT_MAX_RECENT_FILES,
    
    # Performance constants
    DEFAULT_MAX_WORKERS,
    DEFAULT_CACHE_SIZE_MB,
    DEFAULT_CACHE_TTL_SECONDS,
    
    # Logging constants
    DEFAULT_LOG_LEVEL,
    SUPPORTED_LOG_LEVELS,
    DEFAULT_LOG_FORMAT,
    DEFAULT_MAX_LOG_SIZE_MB,
    DEFAULT_MAX_LOG_FILES,
    
    # Validation constants
    NUMERIC_RANGES,
    RECOMMENDED_RANGES,
    
    # Environment constants
    ENV_VAR_PREFIX,
    ENV_VAR_MAPPING,
    
    # Error messages
    ERROR_MESSAGES,
    
    # MIME types
    MIME_TYPES,
)

from .exceptions import (
    # Base exceptions
    CockatooError,
    ConfigurationError,
    
    # Configuration-specific exceptions
    ConfigFileNotFoundError,
    ConfigFilePermissionError,
    ConfigValidationError,
    ConfigKeyError,
    ConfigFormatError,
    ConfigVersionMismatchError,
    ConfigValueError,
    
    # Path-related exceptions
    PathError,
    PathPermissionError,
    PathNotFoundError,
    PathNotWritableError,
    
    # Document processing exceptions
    DocumentProcessingError,
    UnsupportedFormatError,
    FileTooLargeError,
    OCRProcessingError,
    ExtractionError,
    ChunkingError,
    
    # AI/ML exceptions
    AIError,
    LLMError,
    LLMConnectionError,
    LLMRequestError,
    LLMTimeoutError,
    LLMInvalidResponseError,
    EmbeddingError,
    RAGError,
    RAGRetrievalError,
    DatabaseError,
    DatabaseConnectionError,
    DatabaseQueryError,
    
    # UI exceptions
    UIError,
    ThemeError,
    
    # Storage exceptions
    StorageError,
    StorageWriteError,
    StorageReadError,
    BackupError,
    EncryptionError,
    CompressionError,
    
    # Performance exceptions
    PerformanceError,
    CacheError,
    ResourceLimitError,
    MemoryLimitError,
    
    # Privacy exceptions
    PrivacyError,
    TelemetryError,
    ConsentError,
    
    # Logging exceptions
    LoggingError,
    LogFileError,
    
    # Utility
    exception_from_dict,
)

from .app import (
    # Main application class
    CockatooApp,
    
    # Application components
    DocumentProcessor,
    AIProcessor,
    StorageManager,
    UIManager,
    
    # Application state and data classes
    AppState,
    ProcessingStatus,
    ProcessingJob,
    DocumentInfo,
    
    # Application errors
    AppInitializationError,
    AppRuntimeError,
    AppShutdownError,
)

__all__ = [
    # From config
    'AppConfig',
    'ConfigManager',
    'AppInfoConfig',
    'PathsConfig',
    'DocumentProcessingConfig',
    'AIConfig',
    'UIConfig',
    'StorageConfig',
    'PerformanceConfig',
    'PrivacyConfig',
    'LoggingConfig',
    'PlatformType',
    'LogLevel',
    'ThemeMode',
    'DatabaseType',
    'get_config',
    'init_config',
    'save_config',
    'reload_config',
    'setup_crossplatform_environment',
    'TOML_WRITER_AVAILABLE',
    '_default_config_manager',
    
    # From constants
    'APP_NAME',
    'APP_VERSION',
    'APP_DESCRIPTION',
    'APP_AUTHOR',
    'APP_LICENSE',
    'APP_WEBSITE',
    'CONFIG_VERSION',
    'DEFAULT_CONFIG_DIR_NAME',
    'DEFAULT_DATA_DIR_NAME',
    'DEFAULT_MODELS_DIR_NAME',
    'DEFAULT_DOCUMENTS_DIR_NAME',
    'DEFAULT_DATABASE_DIR_NAME',
    'DEFAULT_LOGS_DIR_NAME',
    'DEFAULT_EXPORTS_DIR_NAME',
    'DEFAULT_CACHE_DIR_NAME',
    'DEFAULT_TEMP_DIR_NAME',
    'DEFAULT_BACKUP_DIR_NAME',
    'CONFIG_FILE_NAME',
    'CONFIG_FILE_EXTENSIONS',
    'SUPPORTED_CONFIG_FORMATS',
    'DEFAULT_CONFIG_FORMAT',
    'DEFAULT_CHUNK_SIZE',
    'DEFAULT_CHUNK_OVERLAP',
    'MIN_CHUNK_SIZE',
    'MAX_CHUNK_SIZE',
    'MIN_CHUNK_OVERLAP',
    'MAX_CHUNK_OVERLAP',
    'DEFAULT_MAX_FILE_SIZE_MB',
    'DEFAULT_MAX_PAGES_PER_DOCUMENT',
    'DEFAULT_SUPPORTED_FORMATS',
    'DEFAULT_OCR_LANGUAGES',
    'SUPPORTED_OCR_LANGUAGES',
    'DEFAULT_LLM_PROVIDER',
    'DEFAULT_LLM_MODEL',
    'DEFAULT_LLM_BASE_URL',
    'DEFAULT_TEMPERATURE',
    'DEFAULT_TOP_P',
    'DEFAULT_TOP_K',
    'DEFAULT_MAX_TOKENS',
    'DEFAULT_CONTEXT_WINDOW',
    'DEFAULT_EMBEDDING_MODEL',
    'DEFAULT_EMBEDDING_DIMENSIONS',
    'DEFAULT_RAG_TOP_K',
    'DEFAULT_SIMILARITY_THRESHOLD',
    'SUPPORTED_LLM_PROVIDERS',
    'SUPPORTED_EMBEDDING_MODELS',
    'SUPPORTED_DATABASES',
    'DEFAULT_DATABASE_TYPE',
    'DEFAULT_UI_THEME',
    'SUPPORTED_THEMES',
    'DEFAULT_LANGUAGE',
    'SUPPORTED_LANGUAGES',
    'DEFAULT_FONT_SIZE',
    'DEFAULT_FONT_FAMILY',
    'DEFAULT_ANIMATION_DURATION',
    'DEFAULT_AUTO_SAVE_INTERVAL',
    'DEFAULT_MAX_RECENT_FILES',
    'DEFAULT_MAX_WORKERS',
    'DEFAULT_CACHE_SIZE_MB',
    'DEFAULT_CACHE_TTL_SECONDS',
    'DEFAULT_LOG_LEVEL',
    'SUPPORTED_LOG_LEVELS',
    'DEFAULT_LOG_FORMAT',
    'DEFAULT_MAX_LOG_SIZE_MB',
    'DEFAULT_MAX_LOG_FILES',
    'NUMERIC_RANGES',
    'RECOMMENDED_RANGES',
    'ENV_VAR_PREFIX',
    'ENV_VAR_MAPPING',
    'ERROR_MESSAGES',
    'MIME_TYPES',
    
    # From exceptions
    'CockatooError',
    'ConfigurationError',
    'ConfigFileNotFoundError',
    'ConfigFilePermissionError',
    'ConfigValidationError',
    'ConfigKeyError',
    'ConfigFormatError',
    'ConfigVersionMismatchError',
    'ConfigValueError',
    'PathError',
    'PathPermissionError',
    'PathNotFoundError',
    'PathNotWritableError',
    'DocumentProcessingError',
    'UnsupportedFormatError',
    'FileTooLargeError',
    'OCRProcessingError',
    'ExtractionError',
    'ChunkingError',
    'AIError',
    'LLMError',
    'LLMConnectionError',
    'LLMRequestError',
    'LLMTimeoutError',
    'LLMInvalidResponseError',
    'EmbeddingError',
    'RAGError',
    'RAGRetrievalError',
    'DatabaseError',
    'DatabaseConnectionError',
    'DatabaseQueryError',
    'UIError',
    'ThemeError',
    'StorageError',
    'StorageWriteError',
    'StorageReadError',
    'BackupError',
    'EncryptionError',
    'CompressionError',
    'PerformanceError',
    'CacheError',
    'ResourceLimitError',
    'MemoryLimitError',
    'PrivacyError',
    'TelemetryError',
    'ConsentError',
    'LoggingError',
    'LogFileError',
    'exception_from_dict',
    
    # From app
    'CockatooApp',
    'DocumentProcessor',
    'AIProcessor',
    'StorageManager',
    'UIManager',
    'AppState',
    'ProcessingStatus',
    'ProcessingJob',
    'DocumentInfo',
    'AppInitializationError',
    'AppRuntimeError',
    'AppShutdownError',
]

__version__ = '1.0.0'