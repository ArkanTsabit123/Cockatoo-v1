"""
This module provides a configuration management system with:
- Platform-aware path detection
- YAML configuration file support
- Validation and type safety
- Cross-platform compatibility
- Complete type hinting
"""

import os
import platform
import logging
import logging.handlers
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union, ClassVar, Set
from enum import Enum
from datetime import datetime
import yaml
import json
import sys
import shutil
from dataclasses import dataclass, field, asdict
from pydantic import BaseModel, Field, validator, root_validator, ConfigDict
import tomllib
import tomli_w

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger: logging.Logger = logging.getLogger(__name__)


class PlatformType(str, Enum):
    """Supported operating system platforms with type-safe values."""
    WINDOWS: str = "windows"
    MACOS: str = "macos"
    LINUX: str = "linux"
    UNKNOWN: str = "unknown"


class LogLevel(str, Enum):
    """Supported logging levels."""
    DEBUG: str = "DEBUG"
    INFO: str = "INFO"
    WARNING: str = "WARNING"
    ERROR: str = "ERROR"
    CRITICAL: str = "CRITICAL"


class ThemeMode(str, Enum):
    """Supported UI theme modes."""
    LIGHT: str = "light"
    DARK: str = "dark"
    AUTO: str = "auto"
    SYSTEM: str = "system"


class FileFormat(str, Enum):
    """Supported document file formats."""
    PDF: str = ".pdf"
    DOCX: str = ".docx"
    TXT: str = ".txt"
    MD: str = ".md"
    HTML: str = ".html"
    EPUB: str = ".epub"
    JPG: str = ".jpg"
    PNG: str = ".png"
    CSV: str = ".csv"


class LLMModel(str, Enum):
    """Supported LLM models with detailed descriptions."""
    LLAMA2_7B: str = "llama2:7b"
    LLAMA2_13B: str = "llama2:13b"
    MISTRAL_7B: str = "mistral:7b"
    NEURAL_CHAT_7B: str = "neural-chat:7b"
    CODELLAMA_7B: str = "codellama:7b"
    DOLPHIN_MISTRAL_7B: str = "dolphin-mistral:7b"


class EmbeddingModel(str, Enum):
    """Supported embedding models with specifications."""
    ALL_MINILM_L6_V2: str = "all-MiniLM-L6-v2"
    ALL_MPNET_BASE_V2: str = "all-mpnet-base-v2"
    E5_BASE_V2: str = "intfloat/e5-base-v2"
    MULTILINGUAL_E5_BASE: str = "intfloat/multilingual-e5-base"


class DatabaseType(str, Enum):
    """Supported database backends."""
    CHROMA: str = "chroma"
    QDRANT: str = "qdrant"
    PGVECTOR: str = "pgvector"
    SQLITE: str = "sqlite"


class ConfigSection(BaseModel):
    """Base class for configuration sections."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class AppInfoConfig(ConfigSection):
    """Application information configuration."""
    name: str = Field(default="cockatoo_v1", description="Application name")
    version: str = Field(default="1.0.0", description="Application version")
    description: str = Field(default="AI-powered document intelligence system")
    author: str = Field(default="Cockatoo_V1 Team")
    license: str = Field(default="MIT")
    website: str = Field(default="https://github.com/cockatoo-v1")


class PathsConfig(ConfigSection):
    """Paths configuration with platform-aware defaults."""
    
    platform: PlatformType = Field(default=PlatformType.UNKNOWN, description="Current platform")
    
    @classmethod
    def get_platform_specific_path(cls, platform_type: PlatformType) -> Path:
        """Get platform-specific base directory."""
        home = Path.home()
        
        if platform_type == PlatformType.WINDOWS:
            appdata = os.environ.get('APPDATA', '')
            if appdata:
                return Path(appdata) / "cockatoo_v1"
            return home / "AppData" / "Roaming" / "cockatoo_v1"
        
        elif platform_type == PlatformType.MACOS:
            return home / "Library" / "Application Support" / "cockatoo_v1"
        
        elif platform_type == PlatformType.LINUX:
            xdg_data_home = os.environ.get('XDG_DATA_HOME', '')
            if xdg_data_home:
                return Path(xdg_data_home) / "cockatoo_v1"
            return home / ".local" / "share" / "cockatoo_v1"
        
        else:
            return home / ".cockatoo_v1"
    
    @validator('data_dir', 'models_dir', 'documents_dir', 'database_dir', 
               'logs_dir', 'exports_dir', 'config_dir', 'cache_dir', 
               'temp_dir', 'backup_dir', pre=True, always=True)
    def ensure_path_objects(cls, v, values):
        """Ensure all paths are Path objects."""
        if v is None:
            platform_type = values.get('platform', PlatformType.UNKNOWN)
            base_dir = cls.get_platform_specific_path(platform_type)
            
            # Map field names to subdirectories
            subdir_map = {
                'data_dir': base_dir,
                'models_dir': base_dir / "models",
                'documents_dir': base_dir / "documents",
                'database_dir': base_dir / "database",
                'logs_dir': base_dir / "logs",
                'exports_dir': base_dir / "exports",
                'config_dir': base_dir / "config",
                'cache_dir': base_dir / "cache",
                'temp_dir': base_dir / "temp",
                'backup_dir': base_dir / "backups",
            }
            
            # Get the field name from context (we need to handle this differently)
            # Since we can't get field name directly in Pydantic v2, we'll return a function
            # that gets the appropriate path
            return lambda field_name: subdir_map.get(field_name, base_dir)
        
        return Path(v) if isinstance(v, str) else v
    
    data_dir: Optional[Path] = Field(default=None, description="Main data directory")
    models_dir: Optional[Path] = Field(default=None, description="Models storage directory")
    documents_dir: Optional[Path] = Field(default=None, description="Documents storage directory")
    database_dir: Optional[Path] = Field(default=None, description="Database directory")
    logs_dir: Optional[Path] = Field(default=None, description="Logs directory")
    exports_dir: Optional[Path] = Field(default=None, description="Exports directory")
    config_dir: Optional[Path] = Field(default=None, description="Configuration directory")
    cache_dir: Optional[Path] = Field(default=None, description="Cache directory")
    temp_dir: Optional[Path] = Field(default=None, description="Temporary files directory")
    backup_dir: Optional[Path] = Field(default=None, description="Backup directory")
    
    @root_validator(pre=True)
    def set_default_paths(cls, values):
        """Set default paths based on platform."""
        platform_type = values.get('platform', PlatformType.UNKNOWN)
        base_dir = cls.get_platform_specific_path(platform_type)
        
        # Set default paths if not provided
        path_defaults = {
            'data_dir': base_dir,
            'models_dir': base_dir / "models",
            'documents_dir': base_dir / "documents",
            'database_dir': base_dir / "database",
            'logs_dir': base_dir / "logs",
            'exports_dir': base_dir / "exports",
            'config_dir': base_dir / "config",
            'cache_dir': base_dir / "cache",
            'temp_dir': base_dir / "temp",
            'backup_dir': base_dir / "backups",
        }
        
        for key, default_value in path_defaults.items():
            if key not in values or values[key] is None:
                values[key] = default_value
        
        return values


class DocumentProcessingConfig(ConfigSection):
    """Document processing configuration."""
    chunk_size: int = Field(default=500, ge=100, le=2000, description="Chunk size in characters")
    chunk_overlap: int = Field(default=50, ge=0, le=500, description="Chunk overlap in characters")
    max_file_size_mb: int = Field(default=100, ge=1, le=1024, description="Maximum file size in MB")
    max_pages_per_document: int = Field(default=1000, ge=1, le=10000, description="Maximum pages per document")
    
    supported_formats: List[str] = Field(
        default_factory=lambda: [
            ".pdf", ".docx", ".txt", ".md", ".html", 
            ".epub", ".jpg", ".png", ".csv", ".pptx", ".xlsx"
        ],
        description="Supported file formats"
    )
    
    ocr_enabled: bool = Field(default=True, description="Enable OCR processing")
    ocr_languages: List[str] = Field(
        default_factory=lambda: ["eng", "ind"],
        description="OCR languages"
    )
    
    extract_tables: bool = Field(default=True, description="Extract tables from documents")
    extract_images: bool = Field(default=True, description="Extract images from documents")
    preserve_formatting: bool = Field(default=True, description="Preserve document formatting")
    
    parallel_processing: bool = Field(default=True, description="Enable parallel processing")
    max_parallel_files: int = Field(default=4, ge=1, le=16, description="Maximum parallel files")


class AIConfig(ConfigSection):
    """AI and ML configuration."""
    
    class LLMConfig(ConfigSection):
        """LLM configuration."""
        provider: str = Field(default="ollama", description="LLM provider")
        model: str = Field(default="llama2:7b", description="LLM model name")
        base_url: str = Field(default="http://localhost:11434", description="LLM API base URL")
        api_key: Optional[str] = Field(default=None, description="API key (if required)")
        
        temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Temperature for generation")
        top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p sampling")
        top_k: int = Field(default=40, ge=1, le=100, description="Top-k sampling")
        max_tokens: int = Field(default=1024, ge=1, le=8192, description="Maximum tokens per response")
        context_window: int = Field(default=4096, ge=512, le=32768, description="Context window size")
        
        timeout: int = Field(default=60, description="Request timeout in seconds")
        max_retries: int = Field(default=3, description="Maximum retry attempts")
        
        stream: bool = Field(default=False, description="Enable streaming responses")
        echo: bool = Field(default=False, description="Echo prompt in response")
    
    class EmbeddingConfig(ConfigSection):
        """Embedding configuration."""
        model: str = Field(default="all-MiniLM-L6-v2", description="Embedding model")
        dimensions: int = Field(default=384, description="Embedding dimensions")
        device: str = Field(default="auto", description="Device for embedding computation")
        
        cache_enabled: bool = Field(default=True, description="Enable embedding cache")
        cache_size_mb: int = Field(default=500, ge=10, le=10000, description="Cache size in MB")
        
        batch_size: int = Field(default=32, ge=1, le=256, description="Batch size for embeddings")
        normalize_embeddings: bool = Field(default=True, description="Normalize embeddings")
        
        @validator('device')
        def validate_device(cls, v):
            """Validate device setting."""
            valid_devices = ["auto", "cpu", "cuda", "mps"]
            if v not in valid_devices:
                raise ValueError(f"Device must be one of {valid_devices}")
            return v
    
    class RAGConfig(ConfigSection):
        """RAG (Retrieval-Augmented Generation) configuration."""
        top_k: int = Field(default=5, ge=1, le=50, description="Number of chunks to retrieve")
        similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Similarity threshold")
        max_context_length: int = Field(default=3000, ge=100, le=10000, description="Maximum context length")
        
        enable_hybrid_search: bool = Field(default=True, description="Enable hybrid search")
        bm25_weight: float = Field(default=0.5, ge=0.0, le=1.0, description="BM25 weight in hybrid search")
        
        rerank_results: bool = Field(default=False, description="Enable result reranking")
        reranker_model: Optional[str] = Field(default=None, description="Reranker model")
        
        chunk_separator: str = Field(default="\n\n", description="Chunk separator")
        preserve_metadata: bool = Field(default=True, description="Preserve chunk metadata")
    
    llm: LLMConfig = Field(default_factory=LLMConfig, description="LLM configuration")
    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig, description="Embedding configuration")
    rag: RAGConfig = Field(default_factory=RAGConfig, description="RAG configuration")
    
    database_type: DatabaseType = Field(default=DatabaseType.CHROMA, description="Database type")
    database_host: Optional[str] = Field(default=None, description="Database host")
    database_port: Optional[int] = Field(default=None, description="Database port")
    database_name: Optional[str] = Field(default=None, description="Database name")


class UIConfig(ConfigSection):
    """User interface configuration."""
    theme: ThemeMode = Field(default=ThemeMode.DARK, description="UI theme")
    language: str = Field(default="en", description="UI language code")
    
    font_family: str = Field(default="Segoe UI, Inter, -apple-system", description="Font family")
    font_size: int = Field(default=12, ge=8, le=24, description="Base font size")
    line_height: float = Field(default=1.5, ge=1.0, le=2.5, description="Line height")
    
    enable_animations: bool = Field(default=True, description="Enable UI animations")
    animation_duration: int = Field(default=200, description="Animation duration in ms")
    
    auto_save: bool = Field(default=True, description="Enable auto-save")
    auto_save_interval: int = Field(default=60, ge=10, le=3600, description="Auto-save interval in seconds")
    
    show_tooltips: bool = Field(default=True, description="Show tooltips")
    tooltip_delay: int = Field(default=500, description="Tooltip delay in ms")
    
    max_recent_files: int = Field(default=10, ge=0, le=50, description="Maximum recent files")
    confirm_before_exit: bool = Field(default=True, description="Confirm before exiting")


class StorageConfig(ConfigSection):
    """Storage and persistence configuration."""
    max_documents: int = Field(default=10000, ge=0, le=1000000, description="Maximum documents")
    max_document_size_mb: int = Field(default=50, ge=1, le=1024, description="Maximum document size in MB")
    
    auto_cleanup: bool = Field(default=True, description="Enable automatic cleanup")
    auto_cleanup_days: int = Field(default=90, ge=1, le=365, description="Cleanup documents older than N days")
    cleanup_interval_hours: int = Field(default=24, ge=1, le=168, description="Cleanup interval in hours")
    
    backup_enabled: bool = Field(default=True, description="Enable backups")
    backup_interval_hours: int = Field(default=24, ge=1, le=168, description="Backup interval in hours")
    max_backups: int = Field(default=10, ge=1, le=100, description="Maximum number of backups")
    
    encryption_enabled: bool = Field(default=False, description="Enable data encryption")
    encryption_key: Optional[str] = Field(default=None, description="Encryption key")
    
    compression_enabled: bool = Field(default=True, description="Enable data compression")
    compression_level: int = Field(default=6, ge=1, le=9, description="Compression level")
    
    @validator('encryption_key')
    def validate_encryption_key(cls, v, values):
        """Validate encryption key if encryption is enabled."""
        if values.get('encryption_enabled') and not v:
            raise ValueError("Encryption key is required when encryption is enabled")
        return v


class PerformanceConfig(ConfigSection):
    """Performance and optimization configuration."""
    max_workers: int = Field(default=4, ge=1, le=16, description="Maximum worker threads")
    
    cache_enabled: bool = Field(default=True, description="Enable caching")
    cache_size_mb: int = Field(default=500, ge=10, le=10000, description="Cache size in MB")
    cache_ttl_seconds: int = Field(default=3600, ge=60, le=86400, description="Cache TTL in seconds")
    
    enable_monitoring: bool = Field(default=True, description="Enable performance monitoring")
    monitor_interval_seconds: int = Field(default=60, ge=10, le=3600, description="Monitoring interval")
    
    memory_limit_mb: Optional[int] = Field(default=None, ge=100, le=32768, description="Memory limit in MB")
    gpu_memory_limit_mb: Optional[int] = Field(default=None, ge=100, le=32768, description="GPU memory limit in MB")
    
    preload_models: bool = Field(default=False, description="Preload models on startup")
    lazy_loading: bool = Field(default=True, description="Enable lazy loading")


class PrivacyConfig(ConfigSection):
    """Privacy and telemetry configuration."""
    telemetry_enabled: bool = Field(default=False, description="Enable telemetry")
    crash_reports_enabled: bool = Field(default=False, description="Enable crash reports")
    usage_statistics_enabled: bool = Field(default=False, description="Enable usage statistics")
    
    auto_update_check: bool = Field(default=False, description="Check for updates automatically")
    update_channel: str = Field(default="stable", description="Update channel")
    
    data_collection_consent: bool = Field(default=False, description="Data collection consent")
    analytics_id: Optional[str] = Field(default=None, description="Analytics identifier")
    
    log_sensitive_data: bool = Field(default=False, description="Log sensitive data")
    anonymize_logs: bool = Field(default=True, description="Anonymize log data")


class LoggingConfig(ConfigSection):
    """Logging configuration."""
    level: LogLevel = Field(default=LogLevel.INFO, description="Logging level")
    file_enabled: bool = Field(default=True, description="Enable file logging")
    console_enabled: bool = Field(default=True, description="Enable console logging")
    
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )
    
    max_log_size_mb: int = Field(default=10, ge=1, le=100, description="Maximum log file size in MB")
    max_log_files: int = Field(default=5, ge=1, le=50, description="Maximum number of log files")
    
    log_directory: Optional[Path] = Field(default=None, description="Log directory")


class AppConfig(BaseModel):
    """
    Main application configuration class.
    configuration management for Cockatoo_V1.
    """
    
    # Configuration sections
    app_info: AppInfoConfig = Field(default_factory=AppInfoConfig, description="Application information")
    paths: PathsConfig = Field(default_factory=PathsConfig, description="Paths configuration")
    document_processing: DocumentProcessingConfig = Field(
        default_factory=DocumentProcessingConfig,
        description="Document processing configuration"
    )
    ai: AIConfig = Field(default_factory=AIConfig, description="AI configuration")
    ui: UIConfig = Field(default_factory=UIConfig, description="UI configuration")
    storage: StorageConfig = Field(default_factory=StorageConfig, description="Storage configuration")
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig, description="Performance configuration")
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig, description="Privacy configuration")
    logging: LoggingConfig = Field(default_factory=LoggingConfig, description="Logging configuration")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now, description="Configuration creation time")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update time")
    config_version: str = Field(default="1.0.0", description="Configuration schema version")
    
    # Runtime state (not saved to config file)
    config_file_path: Optional[Path] = Field(default=None, description="Configuration file path")
    is_loaded: bool = Field(default=False, description="Whether config was loaded from file")
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid"
    )
    
    def __init__(self, **data):
        """Initialize configuration with platform detection."""
        # Detect platform first
        platform_type = PlatformType.UNKNOWN
        system = platform.system().lower()
        
        if system.startswith('win'):
            platform_type = PlatformType.WINDOWS
        elif system.startswith('darwin'):
            platform_type = PlatformType.MACOS
        elif system.startswith('linux'):
            platform_type = PlatformType.LINUX
        
        # Ensure paths have platform info
        if 'paths' not in data:
            data['paths'] = {'platform': platform_type}
        elif isinstance(data['paths'], dict):
            data['paths']['platform'] = platform_type
        
        super().__init__(**data)
    
    @root_validator(pre=False)
    def update_dependent_paths(cls, values):
        """Update paths that depend on other paths after all validation."""
        paths = values.get('paths')
        logging_config = values.get('logging')
        
        if paths and logging_config and not logging_config.log_directory:
            logging_config.log_directory = paths.logs_dir
        
        return values
    
    def validate_numeric_ranges(self) -> List[str]:
        """
        Validate numeric values are within acceptable ranges.
        
        Returns:
            List[str]: List of error messages
        """
        errors = []
        
        # Validate chunk size range
        chunk_size = self.document_processing.chunk_size
        if not (100 <= chunk_size <= 2000):
            errors.append(f"Chunk size {chunk_size} out of range [100, 2000]")
        
        # Validate chunk overlap range
        chunk_overlap = self.document_processing.chunk_overlap
        if not (0 <= chunk_overlap <= 500):
            errors.append(f"Chunk overlap {chunk_overlap} out of range [0, 500]")
        
        # Validate file size limit
        max_file_size = self.document_processing.max_file_size_mb
        if not (1 <= max_file_size <= 1024):
            errors.append(f"Max file size {max_file_size}MB out of range [1, 1024]")
        
        # Validate LLM temperature
        temperature = self.ai.llm.temperature
        if not (0.0 <= temperature <= 2.0):
            errors.append(f"LLM temperature {temperature} out of range [0.0, 2.0]")
        
        # Validate top_p
        top_p = self.ai.llm.top_p
        if not (0.0 <= top_p <= 1.0):
            errors.append(f"LLM top_p {top_p} out of range [0.0, 1.0]")
        
        # Validate similarity threshold
        similarity_threshold = self.ai.rag.similarity_threshold
        if not (0.0 <= similarity_threshold <= 1.0):
            errors.append(f"Similarity threshold {similarity_threshold} out of range [0.0, 1.0]")
        
        # Validate BM25 weight
        bm25_weight = self.ai.rag.bm25_weight
        if not (0.0 <= bm25_weight <= 1.0):
            errors.append(f"BM25 weight {bm25_weight} out of range [0.0, 1.0]")
        
        # Validate compression level
        compression_level = self.storage.compression_level
        if not (1 <= compression_level <= 9):
            errors.append(f"Compression level {compression_level} out of range [1, 9]")
        
        return errors
    
    def validate_required_fields(self) -> List[str]:
        """
        Validate all required fields are present and non-empty.
        
        Returns:
            List[str]: List of error messages
        """
        errors = []
        
        # Required string fields
        required_strings = [
            (self.app_info.name, "app_info.name"),
            (self.app_info.version, "app_info.version"),
            (self.ai.llm.model, "ai.llm.model"),
            (self.ai.embeddings.model, "ai.embeddings.model"),
            (self.ui.language, "ui.language"),
        ]
        
        for value, field_name in required_strings:
            if not value or not isinstance(value, str) or value.strip() == "":
                errors.append(f"Required field missing or empty: {field_name}")
        
        # Required paths
        required_paths = [
            (self.paths.data_dir, "paths.data_dir"),
            (self.paths.models_dir, "paths.models_dir"),
            (self.paths.documents_dir, "paths.documents_dir"),
            (self.paths.database_dir, "paths.database_dir"),
        ]
        
        for path, field_name in required_paths:
            if path is None:
                errors.append(f"Required path missing: {field_name}")
        
        # Required if encryption enabled
        if self.storage.encryption_enabled and not self.storage.encryption_key:
            errors.append("Encryption key required when encryption is enabled")
        
        # Required if using external database
        if self.ai.database_type != DatabaseType.CHROMA:
            if not self.ai.database_host:
                errors.append("Database host required for external database")
            if not self.ai.database_name:
                errors.append("Database name required for external database")
        
        return errors
    
    def validate_file_paths(self) -> List[str]:
        """
        Validate file and directory paths.
        
        Returns:
            List[str]: List of error messages
        """
        errors = []
        
        # Check if paths can be created
        test_paths = [
            self.paths.data_dir,
            self.paths.models_dir,
            self.paths.documents_dir,
            self.paths.database_dir,
            self.paths.logs_dir,
            self.paths.config_dir,
        ]
        
        for path in test_paths:
            if path:
                try:
                    # Test if path is writable
                    path.mkdir(parents=True, exist_ok=True)
                    test_file = path / ".cockatoo_test"
                    test_file.write_text("test")
                    test_file.unlink()
                    
                except PermissionError:
                    errors.append(f"Permission denied for path: {path}")
                except OSError as e:
                    errors.append(f"OS error for path {path}: {e}")
                except Exception as e:
                    errors.append(f"Error with path {path}: {e}")
        
        # Check config file path if set
        if self.config_file_path:
            config_parent = self.config_file_path.parent
            try:
                config_parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create config directory {config_parent}: {e}")
        
        # Check log directory
        if self.logging.log_directory:
            try:
                self.logging.log_directory.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create log directory {self.logging.log_directory}: {e}")
        
        # Check for invalid characters in paths
        invalid_chars = ['<', '>', ':', '"', '|', '?', '*']
        for path in test_paths:
            if path:
                path_str = str(path)
                for char in invalid_chars:
                    if char in path_str:
                        errors.append(f"Invalid character '{char}' in path: {path}")
                        break
        
        return errors
    
    def validate_types_and_formats(self) -> List[str]:
        """
        Validate data types and formats.
        
        Returns:
            List[str]: List of error messages
        """
        errors = []
        
        # Validate file formats
        for fmt in self.document_processing.supported_formats:
            if not fmt.startswith('.'):
                errors.append(f"File format should start with '.': {fmt}")
        
        # Validate OCR languages
        for lang in self.document_processing.ocr_languages:
            if not isinstance(lang, str) or len(lang) != 3:
                errors.append(f"Invalid OCR language code: {lang}")
        
        # Validate LLM provider URL
        if self.ai.llm.base_url:
            url = self.ai.llm.base_url
            if not url.startswith(('http://', 'https://')):
                errors.append(f"LLM base URL should start with http:// or https://: {url}")
        
        # Validate theme
        if self.ui.theme not in [t.value for t in ThemeMode]:
            errors.append(f"Invalid theme: {self.ui.theme}")
        
        # Validate log level
        if self.logging.level not in [l.value for l in LogLevel]:
            errors.append(f"Invalid log level: {self.logging.level}")
        
        return errors
    
    # ========== REQUIRED METHODS FOR P1.2.1 ==========
    
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Returns:
            Dict[str, Any]: Loaded configuration dictionary
        """
        if not self.config_file_path or not self.config_file_path.exists():
            logger.warning(f"Config file not found: {self.config_file_path}")
            return self.to_dict()
        
        try:
            with open(self.config_file_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f) or {}
            
            logger.info(f"Configuration loaded from {self.config_file_path}")
            
            # Update self with loaded data
            if 'app' in config_data:
                loaded_config = self.from_dict(config_data['app'])
                for field_name, field_value in loaded_config.model_dump().items():
                    if hasattr(self, field_name):
                        setattr(self, field_name, field_value)
                
                self.is_loaded = True
                self.updated_at = datetime.now()
            
            return config_data.get('app', {})
            
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in config file: {e}")
            raise ValueError(f"Invalid YAML format: {e}")
        except FileNotFoundError as e:
            logger.error(f"Config file not found: {e}")
            raise e
        except PermissionError as e:
            logger.error(f"Permission denied reading config file: {e}")
            raise e
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise e
    
    def save_config(self, config_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Save configuration to YAML file.
        
        Args:
            config_data: Optional configuration data to save
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.config_file_path:
            logger.error("No config file path specified")
            return False
        
        try:
            # Use provided data or current config
            data_to_save = config_data or self.to_dict()
            
            # Ensure config directory exists
            self.config_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare YAML data
            yaml_data = {
                'app': data_to_save,
                'metadata': {
                    'created_at': self.created_at.isoformat(),
                    'updated_at': datetime.now().isoformat(),
                    'config_version': self.config_version,
                    'cockatoo_v1_version': self.app_info.version
                }
            }
            
            # Write to file
            with open(self.config_file_path, 'w', encoding='utf-8') as f:
                yaml.dump(yaml_data, f, default_flow_style=False, indent=2, sort_keys=False)
            
            self.updated_at = datetime.now()
            logger.info(f"Configuration saved to {self.config_file_path}")
            return True
            
        except PermissionError as e:
            logger.error(f"Permission denied writing config: {e}")
            return False
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate configuration values.
        
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_issues)
        """
        issues: List[str] = []
        
        # Run all validation methods
        issues.extend(self.validate_numeric_ranges())
        issues.extend(self.validate_required_fields())
        issues.extend(self.validate_file_paths())
        issues.extend(self.validate_types_and_formats())
        
        # Additional validations
        if self.document_processing.chunk_overlap >= self.document_processing.chunk_size:
            issues.append(f"chunk_overlap ({self.document_processing.chunk_overlap}) must be less than chunk_size ({self.document_processing.chunk_size})")
        
        if self.performance.max_workers < 1 or self.performance.max_workers > 32:
            issues.append(f"max_workers ({self.performance.max_workers}) must be between 1 and 32")
        
        if self.storage.max_documents < 0:
            issues.append(f"max_documents ({self.storage.max_documents}) cannot be negative")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot notation.
        
        Args:
            key: Configuration key (e.g., 'ai.llm.model')
            default: Default value if key not found
            
        Returns:
            Any: Configuration value or default
        """
        try:
            parts = key.split('.')
            value: Any = self.model_dump()
            
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return default
            
            return value
        except (KeyError, AttributeError, TypeError) as e:
            logger.debug(f"Error getting key '{key}': {e}")
            return default
    
    def update(self, key: str, value: Any) -> bool:
        """
        Update configuration value.
        
        Args:
            key: Configuration key to update
            value: New value
            
        Returns:
            bool: True if update successful, False otherwise
        """
        try:
            parts = key.split('.')
            obj = self
            
            # Navigate to the parent object
            for part in parts[:-1]:
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                elif isinstance(obj, dict) and part in obj:
                    obj = obj[part]
                else:
                    logger.warning(f"Key '{key}' not found")
                    return False
            
            # Get the last part and update
            last_part = parts[-1]
            
            if hasattr(obj, last_part):
                current_value = getattr(obj, last_part)
                
                # Type conversion if needed
                if isinstance(current_value, bool) and isinstance(value, str):
                    converted_value = value.lower() in ('true', 'yes', '1', 'on')
                elif isinstance(current_value, int) and isinstance(value, str):
                    converted_value = int(value)
                elif isinstance(current_value, float) and isinstance(value, str):
                    converted_value = float(value)
                else:
                    converted_value = value
                
                setattr(obj, last_part, converted_value)
                self.updated_at = datetime.now()
                logger.info(f"Updated config key '{key}' to '{converted_value}'")
                return True
            elif isinstance(obj, dict):
                obj[last_part] = value
                self.updated_at = datetime.now()
                return True
            else:
                logger.warning(f"Key '{key}' not found")
                return False
                
        except (ValueError, TypeError, AttributeError) as e:
            logger.error(f"Error updating config key '{key}': {e}")
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dict[str, Any]: Configuration as dictionary
        """
        config_dict = self.model_dump(exclude={'config_file_path', 'is_loaded'})
        
        # Convert Path objects to strings
        def convert_paths(obj: Any) -> Any:
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(item) for item in obj]
            elif isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, datetime):
                return obj.isoformat()
            else:
                return obj
        
        return convert_paths(config_dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AppConfig':
        """
        Create AppConfig instance from dictionary.
        
        Args:
            data: Configuration dictionary
            
        Returns:
            AppConfig: New AppConfig instance
        """
        # Convert string paths back to Path objects
        def restore_paths(obj: Any) -> Any:
            if isinstance(obj, str) and ('/' in obj or '\\' in obj):
                # Check if it looks like a path
                try:
                    return Path(obj)
                except:
                    return obj
            elif isinstance(obj, dict):
                return {k: restore_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [restore_paths(item) for item in obj]
            else:
                return obj
        
        processed_data = restore_paths(data)
        return cls(**processed_data)
    
    def get_data_dir(self) -> Path:
        """
        Get data directory path.
        
        Returns:
            Path: Data directory path
        """
        return self.paths.data_dir
    
    # ========== ADDITIONAL USEFUL METHODS ==========
    
    def ensure_directories(self) -> None:
        """Create all required application directories."""
        directories = [
            self.paths.data_dir,
            self.paths.models_dir,
            self.paths.documents_dir,
            self.paths.database_dir,
            self.paths.logs_dir,
            self.paths.exports_dir,
            self.paths.config_dir,
            self.paths.cache_dir,
            self.paths.temp_dir,
            self.paths.backup_dir,
            
            # Subdirectories
            self.paths.documents_dir / "uploads",
            self.paths.documents_dir / "processed",
            self.paths.documents_dir / "thumbnails",
            self.paths.models_dir / "sentence-transformers",
            self.paths.models_dir / "nltk_data",
            self.paths.models_dir / "ocr_tessdata",
            self.paths.database_dir / "chroma",
            self.paths.cache_dir / "embeddings",
            self.paths.cache_dir / "responses",
            self.paths.temp_dir / "processing",
        ]
        
        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Ensured directory exists: {directory}")
            except Exception as e:
                logger.error(f"Error creating directory {directory}: {e}")
    
    def get_default_config_path(self) -> Path:
        """
        Get default configuration file path.
        
        Returns:
            Path: Default config file path
        """
        return self.paths.config_dir / "app_config.yaml"
    
    def get_llm_config(self) -> Dict[str, Any]:
        """
        Get LLM configuration.
        
        Returns:
            Dict[str, Any]: LLM configuration
        """
        return {
            "provider": self.ai.llm.provider,
            "model": self.ai.llm.model,
            "temperature": self.ai.llm.temperature,
            "top_p": self.ai.llm.top_p,
            "max_tokens": self.ai.llm.max_tokens,
            "context_window": self.ai.llm.context_window,
            "timeout": self.ai.llm.timeout,
            "stream": self.ai.llm.stream
        }
    
    def get_embedding_config(self) -> Dict[str, Any]:
        """
        Get embedding configuration.
        
        Returns:
            Dict[str, Any]: Embedding configuration
        """
        return {
            "model": self.ai.embeddings.model,
            "dimensions": self.ai.embeddings.dimensions,
            "device": self.ai.embeddings.device,
            "cache_enabled": self.ai.embeddings.cache_enabled,
            "cache_size_mb": self.ai.embeddings.cache_size_mb,
            "batch_size": self.ai.embeddings.batch_size,
            "normalize_embeddings": self.ai.embeddings.normalize_embeddings
        }
    
    def get_rag_config(self) -> Dict[str, Any]:
        """
        Get RAG configuration.
        
        Returns:
            Dict[str, Any]: RAG configuration
        """
        return {
            "top_k": self.ai.rag.top_k,
            "similarity_threshold": self.ai.rag.similarity_threshold,
            "max_context_length": self.ai.rag.max_context_length,
            "enable_hybrid_search": self.ai.rag.enable_hybrid_search,
            "bm25_weight": self.ai.rag.bm25_weight,
            "rerank_results": self.ai.rag.rerank_results,
            "chunk_separator": self.ai.rag.chunk_separator
        }
    
    def setup_logging(self) -> None:
        """Setup logging based on configuration."""
        # Remove existing handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # Create formatter
        formatter = logging.Formatter(
            self.logging.log_format,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        if self.logging.console_enabled:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            console_handler.setLevel(self.logging.level.value)
            logging.root.addHandler(console_handler)
        
        # File handler
        if self.logging.file_enabled and self.logging.log_directory:
            log_file = self.logging.log_directory / f"cockatoo_v1_{datetime.now().strftime('%Y%m%d')}.log"
            
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=self.logging.max_log_size_mb * 1024 * 1024,
                backupCount=self.logging.max_log_files
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(self.logging.level.value)
            logging.root.addHandler(file_handler)
        
        # Set root logger level
        logging.root.setLevel(self.logging.level.value)
        
        logger.info("Logging system initialized")
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to defaults."""
        default_config = AppConfig()
        
        for field_name in self.model_fields:
            if field_name not in ['config_file_path', 'is_loaded', 'created_at']:
                setattr(self, field_name, getattr(default_config, field_name))
        
        self.updated_at = datetime.now()
        logger.info("Configuration reset to defaults")
    
    def export_config(self, export_path: Path, format: str = 'yaml') -> bool:
        """
        Export configuration to file.
        
        Args:
            export_path: Path to export to
            format: Export format ('yaml', 'json', 'toml')
            
        Returns:
            bool: True if successful
        """
        try:
            config_dict = self.to_dict()
            
            with open(export_path, 'w', encoding='utf-8') as f:
                if format == 'json':
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)
                elif format == 'toml':
                    tomli_w.dump(config_dict, f)
                else:  # yaml
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration exported to {export_path}")
            return True
        except Exception as e:
            logger.error(f"Error exporting configuration: {e}")
            return False
    
    def import_config(self, import_path: Path, format: str = 'yaml') -> bool:
        """
        Import configuration from file.
        
        Args:
            import_path: Path to import from
            format: Import format ('yaml', 'json', 'toml')
            
        Returns:
            bool: True if successful
        """
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                if format == 'json':
                    imported_data = json.load(f)
                elif format == 'toml':
                    imported_data = tomllib.load(f)
                else:  # yaml
                    imported_data = yaml.safe_load(f)
            
            # Update configuration
            imported_config = self.from_dict(imported_data)
            
            for field_name in self.model_fields:
                if field_name not in ['config_file_path', 'is_loaded', 'created_at']:
                    setattr(self, field_name, getattr(imported_config, field_name))
            
            self.updated_at = datetime.now()
            logger.info(f"Configuration imported from {import_path}")
            return True
        except Exception as e:
            logger.error(f"Error importing configuration: {e}")
            return False


# ========== CONFIGURATION MANAGER ==========

class ConfigManager:
    """Configuration manager for loading, saving, and managing AppConfig."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize ConfigManager.
        
        Args:
            config_path: Optional custom config file path
        """
        self.config_path: Optional[Path] = config_path
        self.config: AppConfig = AppConfig()
        
    def load(self) -> AppConfig:
        """
        Load configuration from file.
        
        Returns:
            AppConfig: Loaded configuration
        """
        try:
            # Jika config_path tidak diberikan, gunakan default dari AppConfig
            if self.config_path is None:
                self.config_path = self.config.get_default_config_path()
            
            self.config.config_file_path = self.config_path
            
            if self.config_path.exists():
                self.config.load_config()
            else:
                logger.warning(f"Config file not found, using defaults: {self.config_path}")
            
            # Ensure directories exist
            self.config.ensure_directories()
            
            # Setup logging
            self.config.setup_logging()
            
            logger.info("Configuration loaded successfully")
            return self.config
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return self.config
    
    def save(self) -> bool:
        """
        Save configuration to file.
        
        Returns:
            bool: True if successful
        """
        return self.config.save_config()
    
    def reload(self) -> AppConfig:
        """
        Reload configuration from file.
        
        Returns:
            AppConfig: Reloaded configuration
        """
        return self.load()
    
    def get_config_path(self) -> Path:
        """
        Get configuration file path.
        
        Returns:
            Path: Config file path
        """
        return self.config.config_file_path if self.config.config_file_path else Path()
    
    def set_config_path(self, path: Path) -> None:
        """
        Set configuration file path.
        
        Args:
            path: New config file path
        """
        self.config_path = path
        self.config.config_file_path = path
    
    def validate_config(self) -> Tuple[bool, List[str]]:
        """
        Validate current configuration.
        
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_issues)
        """
        return self.config.validate()


# ========== GLOBAL CONFIGURATION INSTANCE ==========

_default_config_manager: Optional[ConfigManager] = None


def get_config() -> AppConfig:
    """
    Get global configuration instance.
    
    Returns:
        AppConfig: Global configuration
    """
    global _default_config_manager
    
    if _default_config_manager is None:
        _default_config_manager = ConfigManager()
    
    return _default_config_manager.load()


def init_config(config_path: Optional[Path] = None) -> AppConfig:
    """
    Initialize configuration system.
    
    Args:
        config_path: Optional custom config path
        
    Returns:
        AppConfig: Initialized configuration
    """
    global _default_config_manager
    
    _default_config_manager = ConfigManager(config_path)
    config = _default_config_manager.load()
    
    # Save default config if it doesn't exist
    if config_path is None or not config_path.exists():
        _default_config_manager.save()
    
    logger.info("Configuration system initialized")
    return config


def save_config() -> bool:
    """
    Save current configuration.
    
    Returns:
        bool: True if successful
    """
    global _default_config_manager
    
    if _default_config_manager is None:
        logger.error("Configuration not initialized")
        return False
    
    return _default_config_manager.save()


def reload_config() -> AppConfig:
    """
    Reload configuration from file.
    
    Returns:
        AppConfig: Reloaded configuration
    """
    global _default_config_manager
    
    if _default_config_manager is None:
        return init_config()
    
    return _default_config_manager.reload()


def setup_crossplatform_environment() -> Dict[str, Path]:
    """
    Setup complete cross-platform environment.
    
    Returns:
        Dict[str, Path]: Created directories
    """
    config = get_config()
    config.ensure_directories()
    
    directories = {
        'data_dir': config.paths.data_dir,
        'models_dir': config.paths.models_dir,
        'documents_dir': config.paths.documents_dir,
        'database_dir': config.paths.database_dir,
        'logs_dir': config.paths.logs_dir,
        'exports_dir': config.paths.exports_dir,
        'config_dir': config.paths.config_dir,
        'cache_dir': config.paths.cache_dir,
        'temp_dir': config.paths.temp_dir,
        'backup_dir': config.paths.backup_dir,
    }
    
    logger.info("Cross-platform environment setup complete")
    return directories


# ========== MAIN EXECUTION ==========

def main():
    """Main function for testing configuration system."""
    print("Cockatoo_V1 Configuration System Test")
    print("=" * 50)
    
    # Initialize configuration
    config = init_config()
    
    # Display configuration
    print(f"App Name: {config.app_info.name}")
    print(f"Version: {config.app_info.version}")
    print(f"Platform: {config.paths.platform}")
    print(f"Data Directory: {config.paths.data_dir}")
    
    # Validate configuration
    is_valid, issues = config.validate()
    if is_valid:
        print("✓ Configuration is valid")
    else:
        print("✗ Configuration issues:")
        for issue in issues:
            print(f"  - {issue}")
    
    # Test get method
    llm_model = config.get('ai.llm.model')
    print(f"LLM Model: {llm_model}")
    
    # Test update method
    config.update('ui.theme', 'light')
    print(f"Updated theme: {config.ui.theme}")
    
    # Save configuration
    if save_config():
        print("✓ Configuration saved successfully")
    else:
        print("✗ Failed to save configuration")
    
    print("\nConfiguration system test complete!")


if __name__ == "__main__":
    main()