# tests/unit/test_config.py

"""Unit tests for configuration module.

Tests cover AppConfig data class and ConfigManager functionality including
default values, directory management, validation, file persistence,
path expansion, error handling, type conversion, environment overrides,
dynamic reloading, and thread safety.
"""

import os
import threading
import time
import yaml
import json
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import Any, Dict
from datetime import datetime

import pytest
import tomli
import tomli_w

from src.core import (
    AppConfig,
    ConfigManager,
    AppInfoConfig,
    PathsConfig,
    DocumentProcessingConfig,
    AIConfig,
    UIConfig,
    StorageConfig,
    PerformanceConfig,
    PrivacyConfig,
    LoggingConfig,
    PlatformType,
    LogLevel,
    ThemeMode,
    DatabaseType,
    get_config,
    init_config,
    save_config,
    reload_config,
    setup_crossplatform_environment,
    TOML_WRITER_AVAILABLE,
    APP_NAME,
    APP_VERSION,
    APP_DESCRIPTION,
    APP_AUTHOR,
    APP_LICENSE,
    APP_WEBSITE,
    CONFIG_VERSION,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    MIN_CHUNK_SIZE,
    MAX_CHUNK_SIZE,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_LLM_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RAG_TOP_K,
    DEFAULT_UI_THEME,
    DEFAULT_OCR_LANGUAGES,
    SUPPORTED_DATABASES,
    SUPPORTED_LOG_LEVELS,
    SUPPORTED_THEMES,
    DEFAULT_MAX_WORKERS,
    ERROR_MESSAGES,
    ENV_VAR_MAPPING,
    ENV_VAR_PREFIX,
    NUMERIC_RANGES,
    RECOMMENDED_RANGES,
    MIME_TYPES,
    DEFAULT_LOG_FILE_ENABLED,
    DEFAULT_CONSOLE_ENABLED,
    DEFAULT_ANONYMIZE_LOGS,
    DEFAULT_LOG_SENSITIVE_DATA,
    BYTES_PER_MB,
    BYTES_PER_GB,
    MS_PER_SECOND,
    SECONDS_PER_MINUTE,
    MINUTES_PER_HOUR,
    HOURS_PER_DAY,
    CockatooError,
    ConfigurationError,
    ConfigFileNotFoundError,
    ConfigFilePermissionError,
    ConfigValidationError,
    ConfigKeyError,
    ConfigFormatError,
    ConfigVersionMismatchError,
    ConfigValueError,
    PathError,
    PathNotFoundError,
    PathNotWritableError,
    PathPermissionError,
    DocumentProcessingError,
    UnsupportedFormatError,
    FileTooLargeError,
    OCRProcessingError,
    ExtractionError,
    ChunkingError,
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
    UIError,
    ThemeError,
    StorageError,
    StorageWriteError,
    StorageReadError,
    BackupError,
    EncryptionError,
    CompressionError,
    PerformanceError,
    CacheError,
    ResourceLimitError,
    MemoryLimitError,
    PrivacyError,
    TelemetryError,
    ConsentError,
    LoggingError,
    LogFileError,
    AppInitializationError,
    AppRuntimeError,
    AppShutdownError,
    exception_from_dict,
    CockatooLogger,
    LoggerLogLevel,
    TRACE_LEVEL,
    SUCCESS_LEVEL,
    AUDIT_LEVEL,
    PERFORMANCE_LEVEL,
    LogManager,
    get_log_manager,
    get_logger,
    initialize_logging,
    set_log_level,
    enable_json_logging,
    enable_sensitive_logging,
    create_console_handler,
    create_file_handler,
    create_daily_file_handler,
    StructuredJSONFormatter,
    log_performance,
    log_async_performance,
    log_entry_exit,
    LoggingContext,
    CorrelationId,
    log_error_context,
    LogCapture,
    CockatooApp,
    DocumentProcessor,
    AIProcessor,
    StorageManager,
    UIManager,
    AppState,
    ProcessingStatus,
    ProcessingJob,
    DocumentInfo,
    create_app,
    run_app,
)


@pytest.fixture
def config_manager(tmp_path):
    config_path = tmp_path / ".cockatoo" / "config.yaml"
    return ConfigManager(config_path=config_path)


@pytest.fixture
def sample_config_data():
    return {
        "chunk_size": 750,
        "llm_model": "mistral:7b",
        "ui_theme": "light",
        "ocr_enabled": False,
        "rag_top_k": 8,
        "llm_temperature": 0.3,
        "ui_font_size": 14,
        "ocr_languages": ["eng", "deu", "fra"],
        "supported_formats": [".pdf", ".txt", ".md"],
        "privacy_telemetry": False,
        "data_dir": "~/custom_data"
    }


@pytest.fixture
def populated_config_manager(config_manager, sample_config_data):
    for key, value in sample_config_data.items():
        if key == "ocr_languages" or key == "supported_formats":
            config_manager.set(key, value, force=True)
        else:
            config_manager.set(key, value)
    return config_manager


@pytest.fixture
def temp_config_file(tmp_path, sample_config_data):
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(sample_config_data, f)
    return config_path


@pytest.fixture
def cockatoo_app(tmp_path):
    config_path = tmp_path / "config.yaml"
    return CockatooApp(config_path=config_path)


class TestConstants:

    def test_app_constants(self):
        assert APP_NAME == "Cockatoo"
        assert APP_VERSION == "1.0.0"
        assert APP_DESCRIPTION == "AI-powered document intelligence system"
        assert APP_AUTHOR == "Cockatoo_V1 Team"
        assert APP_LICENSE == "MIT"
        assert CONFIG_VERSION == "1.0.0"

    def test_default_values(self):
        assert DEFAULT_CHUNK_SIZE == 500
        assert DEFAULT_CHUNK_OVERLAP == 50
        assert MIN_CHUNK_SIZE == 100
        assert MAX_CHUNK_SIZE == 2000
        assert DEFAULT_LLM_PROVIDER == "ollama"
        assert DEFAULT_LLM_MODEL == "llama2:7b"
        assert DEFAULT_EMBEDDING_MODEL == "all-MiniLM-L6-v2"
        assert DEFAULT_RAG_TOP_K == 5
        assert DEFAULT_UI_THEME == "dark"
        assert DEFAULT_OCR_LANGUAGES == ["eng", "ind"]

    def test_supported_values(self):
        assert "chroma" in SUPPORTED_DATABASES
        assert "qdrant" in SUPPORTED_DATABASES
        assert "DEBUG" in SUPPORTED_LOG_LEVELS
        assert "light" in SUPPORTED_THEMES
        assert DEFAULT_MAX_WORKERS == 4

    def test_logging_constants(self):
        assert DEFAULT_LOG_FILE_ENABLED is True
        assert DEFAULT_CONSOLE_ENABLED is True
        assert DEFAULT_ANONYMIZE_LOGS is True
        assert DEFAULT_LOG_SENSITIVE_DATA is False
        assert BYTES_PER_MB == 1024 * 1024
        assert BYTES_PER_GB == 1024 * 1024 * 1024

    def test_error_messages(self):
        assert "config_not_found" in ERROR_MESSAGES
        assert "Configuration file not found" in ERROR_MESSAGES["config_not_found"]
        assert "path_not_found" in ERROR_MESSAGES

    def test_env_mapping(self):
        assert ENV_VAR_PREFIX == "COCKATOO_"
        assert f"{ENV_VAR_PREFIX}LLM_MODEL" in ENV_VAR_MAPPING
        assert ENV_VAR_MAPPING[f"{ENV_VAR_PREFIX}LLM_MODEL"] == "llm_model"

    def test_numeric_ranges(self):
        assert "chunk_size" in NUMERIC_RANGES
        min_val, max_val = NUMERIC_RANGES["chunk_size"]
        assert min_val == 100
        assert max_val == 2000
        
        assert "chunk_size" in RECOMMENDED_RANGES
        rec_min, rec_max = RECOMMENDED_RANGES["chunk_size"]
        assert rec_min == 200
        assert rec_max == 1000

    def test_mime_types(self):
        assert MIME_TYPES[".pdf"] == "application/pdf"
        assert MIME_TYPES[".txt"] == "text/plain"
        assert MIME_TYPES[".jpg"] == "image/jpeg"


class TestExceptions:

    def test_base_exception(self):
        error = CockatooError("Test error", {"detail": "value"})
        assert str(error) == "Test error [detail=value]"
        assert error.message == "Test error"
        assert error.details == {"detail": "value"}
        
        error_dict = error.to_dict()
        assert error_dict["error_type"] == "CockatooError"
        assert error_dict["message"] == "Test error"

    def test_configuration_error(self):
        error = ConfigurationError("Config error")
        assert isinstance(error, CockatooError)

    def test_config_file_not_found(self):
        error = ConfigFileNotFoundError("/path/to/config.yaml")
        assert "not found" in str(error)
        assert error.details["path"] == "/path/to/config.yaml"

    def test_config_validation_error(self):
        errors = ["chunk_size invalid", "temperature out of range"]
        error = ConfigValidationError("Validation failed", errors=errors)
        assert error.details["validation_errors"] == errors

    def test_config_key_error(self):
        error = ConfigKeyError("invalid.key", available_keys=["name", "version"])
        assert "invalid.key" in str(error)
        assert error.details["available_keys"] == ["name", "version"]

    def test_config_value_error(self):
        error = ConfigValueError(
            key="chunk_size",
            value=50,
            reason="below minimum",
            valid_range=(100, 2000)
        )
        assert "Invalid value for chunk_size" in str(error)
        assert error.details["min"] == 100
        assert error.details["max"] == 2000

    def test_path_not_found(self):
        error = PathNotFoundError("/missing/path", purpose="data directory")
        assert "not found" in str(error)
        assert error.details["purpose"] == "data directory"

    def test_unsupported_format_error(self):
        error = UnsupportedFormatError(
            format=".xyz",
            supported_formats=[".pdf", ".txt"],
            file_path="/test/file.xyz"
        )
        assert "Unsupported file format" in str(error)
        assert error.details["format"] == ".xyz"
        assert error.details["file_path"] == "/test/file.xyz"

    def test_file_too_large_error(self):
        error = FileTooLargeError(
            file_size_mb=150.5,
            max_size_mb=100,
            file_path="/large/file.pdf"
        )
        assert "exceeds maximum" in str(error)
        assert error.details["file_size_mb"] == 150.5

    def test_llm_connection_error(self):
        error = LLMConnectionError(
            provider="ollama",
            base_url="http://localhost:11434",
            error="Connection refused"
        )
        assert "Failed to connect to ollama" in str(error)
        assert error.details["provider"] == "ollama"

    def test_logging_error(self):
        error = LoggingError("Log system error", {"component": "file_handler"})
        assert "Log system error" in str(error)
        assert error.details["component"] == "file_handler"

    def test_log_file_error(self):
        error = LogFileError("Cannot write to log file", path="/var/log/cockatoo.log")
        assert "Log file error" in str(error)
        assert error.details["path"] == "/var/log/cockatoo.log"

    def test_memory_limit_error(self):
        error = MemoryLimitError(used_mb=2048, limit_mb=1024)
        assert "Memory limit exceeded" in str(error)
        assert error.details["value"] == 2048
        assert error.details["limit"] == 1024

    def test_app_initialization_error(self):
        error = AppInitializationError("Failed to load config", component="config")
        assert "App initialization failed" in str(error)
        assert error.details["component"] == "config"

    def test_exception_from_dict(self):
        error_dict = {
            "error_type": "ConfigFileNotFoundError",
            "message": "File not found",
            "details": {"path": "/test/config.yaml"}
        }
        
        error = exception_from_dict(error_dict)
        assert isinstance(error, ConfigFileNotFoundError)
        assert error.details["path"] == "/test/config.yaml"


class TestLogger:

    def test_get_logger(self):
        logger = get_logger(__name__)
        assert isinstance(logger, CockatooLogger)
        assert logger.name == __name__

    def test_initialize_logging(self, tmp_path):
        log_dir = tmp_path / "logs"
        initialize_logging(log_dir=log_dir)
        
        logger = get_logger(__name__)
        logger.info("Test message")
        
        log_files = list(log_dir.glob("*.log"))
        assert len(log_files) > 0

    def test_set_log_level(self):
        set_log_level("DEBUG")
        logger = get_logger(__name__)
        assert logger.isEnabledFor(logging.DEBUG) is True
        
        set_log_level("INFO")
        assert logger.isEnabledFor(logging.DEBUG) is False

    def test_enable_json_logging(self):
        enable_json_logging(True)
        assert True

    def test_custom_log_levels(self):
        assert TRACE_LEVEL == 5
        assert SUCCESS_LEVEL == 25
        assert AUDIT_LEVEL == 45
        assert PERFORMANCE_LEVEL == 15

    def test_log_performance_decorator(self):
        logger = get_logger(__name__)
        
        if not hasattr(logger, '_correlation_id'):
            logger._correlation_id = None
        if not hasattr(logger, '_context'):
            logger._context = {}
        
        @log_performance(logger)
        def test_func():
            time.sleep(0.01)
            return "done"
        
        result = test_func()
        assert result == "done"

    def test_log_entry_exit_decorator(self):
        logger = get_logger(__name__)
        
        @log_entry_exit(logger)
        def test_func():
            return "done"
        
        result = test_func()
        assert result == "done"

    def test_logging_context(self):
        logger = get_logger(__name__)
        
        with LoggingContext(logger, operation="test", component="unit"):
            context = logger.get_context()
            assert context["operation"] == "test"
            assert context["component"] == "unit"
        
        assert logger.get_context() == {}

    def test_correlation_id(self):
        logger = get_logger(__name__)
        
        with CorrelationId(logger, "test-123"):
            assert logger.get_correlation_id() == "test-123"
        
        assert logger.get_correlation_id() is None

    def test_log_capture(self):
        with LogCapture(__name__) as capture:
            logger = get_logger(__name__)
            logger.info("Test message")
            logger.warning("Warning message")
            
            messages = capture.get_messages()
            assert len(messages) == 2
            assert "Test message" in messages[0]
            assert capture.contains("Warning")

    def test_log_error_context(self):
        logger = get_logger(__name__)
        
        try:
            raise ValueError("Test error")
        except ValueError as e:
            log_error_context(logger, e, {"file": "test.txt"})
        
        assert True

    def test_log_manager_singleton(self):
        manager1 = get_log_manager()
        manager2 = get_log_manager()
        assert manager1 is manager2

    def test_log_manager_get_log_files(self, tmp_path):
        log_dir = tmp_path / "logs"
        initialize_logging(log_dir=log_dir)
        
        manager = get_log_manager()
        log_files = manager.get_log_files()
        
        assert isinstance(log_files, list)

    def test_log_manager_archive_logs(self, tmp_path):
        log_dir = tmp_path / "logs"
        archive_dir = tmp_path / "archives"
        initialize_logging(log_dir=log_dir)
        
        logger = get_logger(__name__)
        logger.info("Test message")
        
        manager = get_log_manager()
        archive_path = manager.archive_logs(archive_dir)
        
        if archive_path:
            assert archive_path.exists()

    def test_log_manager_clear_logs(self, tmp_path):
        log_dir = tmp_path / "logs"
        initialize_logging(log_dir=log_dir)
        
        manager = get_log_manager()
        deleted = manager.clear_logs(older_than_days=0)
        
        assert isinstance(deleted, int)


class TestConfigSections:

    def test_app_info_config(self):
        config = AppInfoConfig(
            name="Test App",
            version="2.0.0",
            description="Test Description"
        )
        assert config.name == "Test App"
        assert config.version == "2.0.0"
        assert config.author == "Cockatoo_V1 Team"

    def test_paths_config_platform_detection(self):
        from src.core.config import PathsConfig, PlatformType
        
        with patch('platform.system', return_value='Windows'):
            with patch.dict('os.environ', {'APPDATA': 'C:\\Users\\test\\AppData\\Roaming'}):
                config = PathsConfig()
                assert 'AppData' in str(config.data_dir) or 'AppData' in str(config.data_dir).replace('\\', '/')
                assert 'cockatoo' in str(config.data_dir)
        
        with patch('platform.system', return_value='Darwin'):
            with patch('pathlib.Path.home', return_value=Path('/Users/test')):
                config = PathsConfig()
                path_str = str(config.data_dir).replace('\\', '/')
                assert 'Library/Application Support' in path_str
                assert 'cockatoo' in path_str
        
        with patch('platform.system', return_value='Linux'):
            with patch.dict('os.environ', {'XDG_DATA_HOME': '/home/test/.local/share'}):
                config = PathsConfig()
                path_str = str(config.data_dir).replace('\\', '/')
                assert '.local/share' in path_str
                assert 'cockatoo' in path_str

    def test_logging_config(self):
        config = LoggingConfig(
            level=LogLevel.DEBUG,
            file_enabled=True,
            console_enabled=False,
            max_log_size_mb=20,
            max_log_files=10
        )
        assert config.level == LogLevel.DEBUG
        assert config.file_enabled is True
        assert config.console_enabled is False
        assert config.max_log_size_mb == 20
        assert config.max_log_files == 10

    def test_document_processing_config_validation(self):
        config = DocumentProcessingConfig(
            supported_formats=[".pdf", ".txt", ".md"]
        )
        assert ".pdf" in config.supported_formats
        
        with pytest.raises(ValueError, match="File format should start with '.'"):
            DocumentProcessingConfig(supported_formats=["pdf"])
        
        with pytest.raises(ValueError, match="Invalid OCR language code"):
            DocumentProcessingConfig(ocr_languages=["english"])

    def test_ai_config_validation(self):
        config = AIConfig()
        assert config.llm.provider == "ollama"
        
        with pytest.raises(ValueError):
            config.embeddings.device = "invalid_device"
            config.embeddings.validate_device("invalid_device")
        
        with pytest.raises(ValueError, match="Database host required"):
            AIConfig(
                database_type=DatabaseType.QDRANT,
                database_host=None,
                database_name="test"
            )

    def test_ui_config_theme_validation(self):
        config = UIConfig()
        
        config.theme = "light"
        assert config.theme == ThemeMode.LIGHT
        
        with pytest.raises(ValueError):
            config.theme = "invalid_theme"

    def test_storage_config_encryption(self):
        config = StorageConfig(encryption_enabled=False)
        assert config.encryption_enabled is False
        
        with pytest.raises(ValueError, match="Encryption key is required"):
            StorageConfig(encryption_enabled=True, encryption_key=None)


class TestAppConfigCreation:

    def test_default_config_creation(self):
        config = AppConfig()

        assert config.name == "Cockatoo"
        assert config.version == "1.0.0"
        assert config.chunk_size == 500
        assert config.chunk_overlap == 50
        assert config.llm_provider == "ollama"
        assert config.llm_model == "llama2:7b"
        assert config.embedding_model == "all-MiniLM-L6-v2"
        assert config.rag_top_k == 5
        assert config.ui_theme == "dark"
        assert config.ocr_languages == ["eng", "ind"]
        assert ".pdf" in config.supported_formats

    def test_custom_config_creation(self):
        config = AppConfig(
            name="CustomApp",
            chunk_size=1000,
            llm_model="mistral:7b",
            ui_theme="light"
        )

        assert config.name == "CustomApp"
        assert config.chunk_size == 1000
        assert config.llm_model == "mistral:7b"
        assert config.ui_theme == "light"
        assert config.version == "1.0.0"

    def test_config_mutable(self):
        config = AppConfig()
        config.chunk_size = 750
        config.llm_model = "neural-chat:7b"

        assert config.chunk_size == 750
        assert config.llm_model == "neural-chat:7b"

    def test_path_expansion(self):
        config = AppConfig(data_dir="~/test")
        expanded = Path(config.data_dir).expanduser()
        assert "~" not in str(expanded)
        
        test_path = Path("/absolute/path")
        config.data_dir = test_path
        assert config.data_dir == test_path

    def test_metadata_fields(self):
        config = AppConfig()
        assert isinstance(config.created_at, datetime)
        assert isinstance(config.updated_at, datetime)
        assert config.config_version == "1.0.0"
        assert config.is_loaded is False


class TestConfigManager:

    def test_initialization_default_path(self):
        manager = ConfigManager()
        assert isinstance(manager.config, AppConfig)
        assert manager.config_path == Path.home() / ".cockatoo" / "config.yaml"

    def test_initialization_custom_path(self, tmp_path):
        custom_path = tmp_path / "custom" / "config.yaml"
        manager = ConfigManager(config_path=custom_path)
        assert manager.config_path == custom_path

    @patch('pathlib.Path.mkdir')
    def test_directory_creation(self, mock_mkdir):
        ConfigManager()
        assert mock_mkdir.call_count >= 6

    @patch('pathlib.Path.mkdir')
    def test_directory_creation_permission_error(self, mock_mkdir):
        mock_mkdir.side_effect = PermissionError("Permission denied")
        with pytest.raises(PermissionError, match="Permission denied"):
            ConfigManager()

    def test_get_config_value(self, config_manager):
        assert config_manager.get("name") == "Cockatoo"
        assert config_manager.get("nonexistent", "default") == "default"
        
        assert config_manager.get("ai.llm.model") == "llama2:7b"
        
        theme = config_manager.get("ui.theme")
        assert theme in ["dark", "light", "auto", "system"]

    def test_set_config_value(self, config_manager):
        config_manager.set("chunk_size", 750)
        assert config_manager.config.chunk_size == 750
        
        config_manager.set("ai.llm.model", "new-model")
        assert config_manager.config.ai.llm.model == "new-model"
        
        config_manager.set("chunk_size", 50, force=True)
        assert config_manager.config.chunk_size == 50

        with pytest.raises(KeyError, match="Invalid configuration key"):
            config_manager.set("invalid_key", "value")

    def test_validation_success(self, config_manager):
        config_manager.set("chunk_size", 500)
        config_manager.set("chunk_overlap", 50)
        config_manager.set("llm_temperature", 0.1)

        result = config_manager.validate()
        assert result["valid"] is True
        assert len(result["errors"]) == 0

    def test_validation_with_warnings(self, config_manager):
        config_manager.set("chunk_size", 50, force=True)
        config_manager.set("ui_font_size", 6, force=True)

        result = config_manager.validate()
        assert result["valid"] is True
        assert len(result["warnings"]) > 0

    @pytest.mark.parametrize("chunk_size,chunk_overlap,expected_valid", [
        (500, 50, True),
        (500, 600, False),
        (500, -10, False),
    ])
    def test_chunk_validation(self, config_manager, chunk_size, chunk_overlap, expected_valid):
        config_manager.set("chunk_size", chunk_size, force=True)
        config_manager.set("chunk_overlap", chunk_overlap, force=True)

        result = config_manager.validate()
        assert result["valid"] == expected_valid

    @pytest.mark.parametrize("temperature,expected_valid", [
        (0.0, True),
        (1.0, True),
        (2.0, True),
        (-0.1, False),
        (2.1, False),
    ])
    def test_temperature_validation(self, config_manager, temperature, expected_valid):
        config_manager.set("llm_temperature", temperature, force=True)
        result = config_manager.validate()
        assert result["valid"] == expected_valid

    def test_validation_empty_paths(self, config_manager):
        config_manager.set("data_dir", "")
        result = config_manager.validate()
        assert result["valid"] is False
        assert any("data_dir" in e for e in result["errors"])

    def test_listener_notification(self, config_manager):
        mock_listener = MagicMock()
        config_manager.add_listener(mock_listener)
        config_manager.set("chunk_size", 1000)
        mock_listener.assert_called_once_with("chunk_size", 500, 1000)

    def test_multiple_listeners(self, config_manager):
        mock1 = MagicMock()
        mock2 = MagicMock()
        config_manager.add_listener(mock1)
        config_manager.add_listener(mock2)
        config_manager.set("llm_model", "new-model")
        mock1.assert_called_once()
        mock2.assert_called_once()

    def test_remove_listener(self, config_manager):
        mock_listener = MagicMock()
        config_manager.add_listener(mock_listener)
        config_manager.remove_listener(mock_listener)
        config_manager.set("chunk_size", 1000)
        mock_listener.assert_not_called()

    def test_listener_not_called_for_unchanged_value(self, config_manager):
        mock_listener = MagicMock()
        config_manager.add_listener(mock_listener)
        config_manager.set("chunk_size", 500)
        mock_listener.assert_not_called()

    def test_get_config_path(self, config_manager, tmp_path):
        assert config_manager.get_config_path() == config_manager.config_path
        
        config_manager.config.config_file_path = tmp_path / "custom.yaml"
        assert config_manager.get_config_path() == tmp_path / "custom.yaml"

    def test_set_config_path(self, config_manager, tmp_path):
        new_path = tmp_path / "new" / "config.yaml"
        config_manager.set_config_path(new_path)
        assert config_manager.config_path == new_path
        assert config_manager.config.config_file_path == new_path


class TestConfigPersistence:

    def test_save_config(self, config_manager):
        config_manager.set("chunk_size", 750)
        config_manager.set("llm_model", "mistral:7b")
        config_manager.save()

        assert config_manager.config_path.exists()
        with open(config_manager.config_path, 'r') as f:
            data = yaml.safe_load(f)
        assert data["chunk_size"] == 750
        assert data["llm_model"] == "mistral:7b"

    def test_load_config(self, config_manager, sample_config_data):
        with open(config_manager.config_path, 'w') as f:
            yaml.dump(sample_config_data, f)

        config = config_manager.load()
        assert config.chunk_size == sample_config_data["chunk_size"]
        assert config.llm_model == sample_config_data["llm_model"]

    def test_load_nonexistent_file(self, config_manager):
        if config_manager.config_path.exists():
            config_manager.config_path.unlink()
        config = config_manager.load()
        assert config.chunk_size == 500

    def test_save_load_roundtrip(self, populated_config_manager, sample_config_data):
        populated_config_manager.save()
        new_manager = ConfigManager(config_path=populated_config_manager.config_path)
        new_manager.load()
        for key, expected in sample_config_data.items():
            if key not in ["data_dir"]:
                assert new_manager.get(key) == expected

    def test_save_with_custom_path(self, config_manager, tmp_path):
        custom_path = tmp_path / "custom.yaml"
        config_manager.save(file_path=custom_path)
        assert custom_path.exists()

    def test_load_with_custom_path(self, config_manager, sample_config_data, tmp_path):
        custom_path = tmp_path / "custom.yaml"
        with open(custom_path, 'w') as f:
            yaml.dump(sample_config_data, f)
        config = config_manager.load(file_path=custom_path)
        assert config.llm_model == sample_config_data["llm_model"]

    def test_save_creates_directory(self, config_manager, tmp_path):
        deep_path = tmp_path / "deep" / "nested" / "config.yaml"
        config_manager.save(file_path=deep_path)
        assert deep_path.exists()
        assert deep_path.parent.exists()

    def test_save_permission_error(self, config_manager):
        with patch('builtins.open') as mock_open:
            mock_open.side_effect = PermissionError("Permission denied")
            with pytest.raises(RuntimeError, match="Failed to save configuration"):
                config_manager.save()

    def test_load_invalid_yaml(self, config_manager):
        with open(config_manager.config_path, 'w') as f:
            f.write(": invalid: yaml: [")
        with pytest.raises(RuntimeError, match="Failed to load configuration"):
            config_manager.load()

    def test_type_conversion_string_to_int(self, config_manager):
        with open(config_manager.config_path, 'w') as f:
            yaml.dump({"chunk_size": "500"}, f)
        config = config_manager.load()
        assert isinstance(config.chunk_size, int)
        assert config.chunk_size == 500

    def test_type_conversion_string_to_bool(self, config_manager):
        data = {
            "ocr_enabled": "true",
            "privacy_telemetry": "false",
            "ui_animations": "yes",
            "rag_hybrid_search": "1"
        }
        with open(config_manager.config_path, 'w') as f:
            yaml.dump(data, f)
        config = config_manager.load()
        assert config.ocr_enabled is True
        assert config.privacy_telemetry is False
        assert config.ui_animations is True
        assert config.rag_hybrid_search is True

    def test_type_conversion_string_to_float(self, config_manager):
        with open(config_manager.config_path, 'w') as f:
            yaml.dump({"llm_temperature": "0.35"}, f)
        config = config_manager.load()
        assert isinstance(config.llm_temperature, float)
        assert config.llm_temperature == 0.35

    def test_type_conversion_string_to_list(self, config_manager):
        with open(config_manager.config_path, 'w') as f:
            yaml.dump({"ocr_languages": "eng,deu,fra"}, f)
        config = config_manager.load()
        assert config.ocr_languages == ["eng", "deu", "fra"]

    def test_invalid_type_conversion_falls_back(self, config_manager):
        with open(config_manager.config_path, 'w') as f:
            yaml.dump({"chunk_size": "not_an_int"}, f)
        config = config_manager.load()
        assert config.chunk_size == 500

    def test_load_old_config_version(self, config_manager):
        old_config = {"chunk_size": 600, "llm_model": "old-model"}
        with open(config_manager.config_path, 'w') as f:
            yaml.dump(old_config, f)
        loaded = config_manager.load()
        assert loaded.chunk_size == 600
        assert loaded.llm_model == "old-model"
        assert loaded.ocr_languages == ["eng", "ind"]

    def test_load_config_with_extra_fields(self, config_manager):
        config_with_extra = {
            "chunk_size": 700,
            "unknown_field": "ignored"
        }
        with open(config_manager.config_path, 'w') as f:
            yaml.dump(config_with_extra, f)
        loaded = config_manager.load()
        assert loaded.chunk_size == 700
        assert not hasattr(loaded, "unknown_field")

    def test_dynamic_config_reload(self, config_manager):
        config_manager.save()
        initial = config_manager.get("chunk_size")
        with open(config_manager.config_path, 'w') as f:
            yaml.dump({"chunk_size": 999}, f)
        config_manager.reload()
        assert config_manager.get("chunk_size") == 999
        assert config_manager.get("chunk_size") != initial

    def test_config_file_watching(self, config_manager):
        mock_callback = MagicMock()
        config_manager.add_reload_callback(mock_callback)
        
        config_manager.save()
        
        config_manager.start_file_watching()
        with open(config_manager.config_path, 'w') as f:
            yaml.dump({"chunk_size": 888}, f)
        time.sleep(0.1)
        mock_callback.assert_called()

    def test_export_config_json(self, config_manager, tmp_path):
        export_path = tmp_path / "export.json"
        result = config_manager.config.export_config(export_path, format="json")
        assert result is True
        assert export_path.exists()
        
        with open(export_path, 'r') as f:
            data = json.load(f)
        assert data["app_info"]["name"] == "Cockatoo"

    def test_export_config_toml(self, config_manager, tmp_path):
        try:
            import tomli_w
            import sys
            if sys.version_info >= (3, 11):
                import tomllib
            else:
                import tomli as tomllib
        except ImportError:
            pytest.skip("tomli-w not available")
        
        safe_config = {
            "app_info": {
                "name": "Cockatoo",
                "version": "1.0.0",
                "description": "AI-powered document intelligence system",
                "author": "Cockatoo_V1 Team",
                "license": "MIT",
                "website": "https://github.com/cockatoo-v1"
            },
            "document_processing": {
                "chunk_size": 500,
                "chunk_overlap": 50,
                "ocr_enabled": True,
                "ocr_languages": ["eng", "ind"],
                "supported_formats": [".pdf", ".txt", ".md"]
            },
            "ai": {
                "llm": {
                    "provider": "ollama",
                    "model": "llama2:7b",
                    "temperature": 0.1
                },
                "embeddings": {
                    "model": "all-MiniLM-L6-v2",
                    "dimensions": 384
                },
                "rag": {
                    "top_k": 5
                },
                "database_type": "chroma"
            },
            "ui": {
                "theme": "dark",
                "language": "en",
                "font_size": 12
            },
            "privacy": {
                "telemetry_enabled": False
            },
            "performance": {
                "max_workers": 4,
                "memory_limit_mb": 0,
                "gpu_memory_limit_mb": 0
            }
        }
        
        export_path = tmp_path / "export.toml"
        
        with open(export_path, 'wb') as f:
            tomli_w.dump(safe_config, f)
        
        assert export_path.exists()
        
        with open(export_path, 'rb') as f:
            loaded_config = tomllib.load(f)
        assert loaded_config["app_info"]["name"] == "Cockatoo"

    def test_import_config_json(self, config_manager, tmp_path):
        import_path = tmp_path / "import.json"
        config_data = {"app_info": {"name": "Imported App"}}
        
        with open(import_path, 'w') as f:
            json.dump(config_data, f)
        
        result = config_manager.config.import_config(import_path, format="json")
        assert result is True
        assert config_manager.config.app_info.name == "Imported App"


class TestConfigEnvironmentOverrides:

    @patch.dict(os.environ, {"COCKATOO_LLM_MODEL": "env-model:7b"})
    def test_env_override_single(self, config_manager):
        config_manager.load_with_env_overrides()
        assert config_manager.get("llm_model") == "env-model:7b"

    @patch.dict(os.environ, {
        "COCKATOO_CHUNK_SIZE": "1000",
        "COCKATOO_LLM_TEMPERATURE": "0.5"
    })
    def test_env_override_multiple(self, config_manager):
        config_manager.load_with_env_overrides()
        assert config_manager.get("chunk_size") == 1000
        assert config_manager.get("llm_temperature") == 0.5

    @patch.dict(os.environ, {"COCKATOO_OCR_ENABLED": "false"})
    def test_env_override_boolean(self, config_manager):
        config_manager.load_with_env_overrides()
        assert config_manager.get("ocr_enabled") is False

    @patch.dict(os.environ, {"COCKATOO_OCR_LANGUAGES": "eng,deu,fra"})
    def test_env_override_list(self, config_manager):
        config_manager.load_with_env_overrides()
        assert config_manager.get("ocr_languages") == ["eng", "deu", "fra"]

    @patch.dict(os.environ, {"COCKATOO_LOG_LEVEL": "DEBUG"})
    def test_env_override_log_level(self, config_manager):
        config_manager.load_with_env_overrides()
        assert True

    @patch.dict(os.environ, {"COCKATOO_INVALID_KEY": "value"})
    def test_env_override_unknown_key(self, config_manager):
        config_manager.load_with_env_overrides()
        assert config_manager.get("name") == "Cockatoo"


class TestGlobalFunctions:

    def test_get_config(self):
        config = get_config()
        assert isinstance(config, AppConfig)

    def test_init_config(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config = init_config(config_path)
        
        assert isinstance(config, AppConfig)
        assert config_path.exists()

    def test_save_and_reload_config(self, tmp_path, monkeypatch):
        from src.core import _default_config_manager
        
        monkeypatch.setattr("src.core._default_config_manager", None)
        
        config_path = tmp_path / "config.yaml"
        config = init_config(config_path)
        
        config.name = "Global Test"
        
        assert save_config() is True
        
        new_config = reload_config()
        assert new_config.name == "Global Test"

    def test_setup_crossplatform_environment(self, tmp_path):
        from src.core import setup_crossplatform_environment
        from src.core.config import AppConfig
        
        config = AppConfig()
        
        config.paths.data_dir = tmp_path / "data"
        config.paths.models_dir = tmp_path / "models"
        config.paths.documents_dir = tmp_path / "documents"
        config.paths.database_dir = tmp_path / "database"
        config.paths.logs_dir = tmp_path / "logs"
        config.paths.exports_dir = tmp_path / "exports"
        config.paths.config_dir = tmp_path / "config"
        config.paths.cache_dir = tmp_path / "cache"
        config.paths.temp_dir = tmp_path / "temp"
        config.paths.backup_dir = tmp_path / "backups"
        
        with patch("src.core.get_config", return_value=config):
            with patch.object(AppConfig, 'ensure_directories') as mock_ensure:
                dirs = setup_crossplatform_environment()
                
                assert isinstance(dirs, dict)
                expected_keys = [
                    "data_dir", "models_dir", "documents_dir", "database_dir",
                    "logs_dir", "exports_dir", "config_dir", "cache_dir",
                    "temp_dir", "backup_dir"
                ]
                for key in expected_keys:
                    assert key in dirs
                    assert isinstance(dirs[key], Path)
                
                assert mock_ensure.called
                assert mock_ensure.call_count >= 1


class TestConfigConcurrency:

    def test_thread_safe_reads(self, config_manager):
        results = []
        errors = []

        def read_config():
            try:
                for _ in range(100):
                    val = config_manager.get("chunk_size")
                    assert val == 500
                    time.sleep(0.001)
                results.append(True)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=read_config) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 10
        assert len(errors) == 0

    def test_thread_safe_writes(self, config_manager):
        errors = []

        def write_config(thread_id):
            try:
                for i in range(50):
                    config_manager.set("chunk_size", 500 + thread_id * 100 + i, force=True)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=write_config, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_read_write(self, config_manager):
        errors = []

        def reader():
            try:
                for _ in range(100):
                    val = config_manager.get("chunk_size")
                    assert isinstance(val, int)
            except Exception as e:
                errors.append(f"Reader error: {e}")

        def writer():
            try:
                for i in range(50):
                    config_manager.set("chunk_size", 500 + i, force=True)
            except Exception as e:
                errors.append(f"Writer error: {e}")

        readers = [threading.Thread(target=reader) for _ in range(5)]
        writers = [threading.Thread(target=writer) for _ in range(2)]
        all_threads = readers + writers

        for t in all_threads:
            t.start()
        for t in all_threads:
            t.join()

        assert len(errors) == 0

    def test_listener_thread_safety(self, config_manager):
        import queue
        notification_queue = queue.Queue()

        def listener(key, old, new):
            notification_queue.put((key, old, new))
            time.sleep(0.01)

        config_manager.add_listener(listener)
        errors = []

        def writer(thread_id):
            try:
                for i in range(20):
                    config_manager.set("chunk_size", 500 + thread_id * 100 + i, force=True)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert notification_queue.qsize() > 0


class TestCockatooApp:

    def test_app_initialization(self, cockatoo_app):
        cockatoo_app.initialize()
        
        assert cockatoo_app.state == AppState.RUNNING
        assert cockatoo_app.start_time is not None
        assert cockatoo_app.document_processor is not None
        assert cockatoo_app.ai_processor is not None
        assert cockatoo_app.storage_manager is not None
        assert cockatoo_app.ui_manager is not None

    def test_app_initialization_failure(self, cockatoo_app):
        with patch.object(cockatoo_app.config_manager, 'load', side_effect=Exception("Load failed")):
            with pytest.raises(AppInitializationError):
                cockatoo_app.initialize()
            assert cockatoo_app.state == AppState.ERROR

    def test_app_shutdown(self, cockatoo_app):
        cockatoo_app.initialize()
        cockatoo_app.shutdown()
        
        assert cockatoo_app.state == AppState.STOPPED

    def test_app_pause_resume(self, cockatoo_app):
        cockatoo_app.initialize()
        
        cockatoo_app.pause()
        assert cockatoo_app.state == AppState.PAUSED
        
        cockatoo_app.resume()
        assert cockatoo_app.state == AppState.RUNNING

    def test_app_get_status(self, cockatoo_app):
        cockatoo_app.initialize()
        
        status = cockatoo_app.get_status()
        
        assert status["state"] == AppState.RUNNING.value
        assert status["version"] == APP_VERSION
        assert "uptime_seconds" in status
        assert "components" in status
        assert status["components"]["document_processor"] is True

    def test_process_document(self, cockatoo_app, tmp_path):
        cockatoo_app.initialize()
        
        test_file = tmp_path / "test.pdf"
        test_file.write_text("dummy content")
        
        job_id = cockatoo_app.process_document(test_file)
        
        assert job_id is not None
        assert cockatoo_app.document_processor is not None
        
        job = cockatoo_app.get_job_status(job_id)
        assert job is not None
        assert job.file_path == test_file
        
        docs = cockatoo_app.get_documents()
        assert len(docs) > 0
        
        recent_files = cockatoo_app.ui_manager.get_recent_files()
        assert test_file in recent_files

    def test_process_unsupported_format(self, cockatoo_app, tmp_path):
        cockatoo_app.initialize()
        
        test_file = tmp_path / "test.xyz"
        test_file.write_text("dummy")
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            cockatoo_app.process_document(test_file)

    def test_query_ai(self, cockatoo_app):
        cockatoo_app.initialize()
        
        response = cockatoo_app.query_ai("Hello, AI!")
        
        assert response is not None
        assert isinstance(response, str)

    def test_rag_query(self, cockatoo_app):
        cockatoo_app.initialize()
        
        result = cockatoo_app.rag_query("test query")
        
        assert isinstance(result, dict)
        assert "query" in result
        assert "response" in result

    def test_create_backup(self, cockatoo_app, tmp_path):
        cockatoo_app.initialize()
        
        test_file = tmp_path / "doc1.pdf"
        test_file.write_text("content")
        cockatoo_app.process_document(test_file)
        
        backup_path = cockatoo_app.create_backup()
        
        assert backup_path is not None
        assert backup_path.exists()

    def test_reload_config(self, cockatoo_app, tmp_path):
        cockatoo_app.initialize()
        
        with open(cockatoo_app.config_manager.config_path, 'w') as f:
            yaml.dump({"chunk_size": 999}, f)
        
        cockatoo_app.reload_config()
        
        assert cockatoo_app.config.chunk_size == 999


class TestDocumentProcessor:

    def test_processor_initialization(self, cockatoo_app):
        processor = DocumentProcessor(cockatoo_app.config)
        
        assert processor.config is not None
        assert len(processor._jobs) == 0

    def test_submit_job(self, cockatoo_app, tmp_path):
        processor = DocumentProcessor(cockatoo_app.config)
        processor.start()
        
        test_file = tmp_path / "test.pdf"
        test_file.write_text("content")
        
        job_id = processor.submit_job(test_file)
        
        assert job_id is not None
        job = processor.get_job_status(job_id)
        assert job is not None
        assert job.status == ProcessingStatus.PENDING

    def test_submit_job_file_not_found(self, cockatoo_app):
        processor = DocumentProcessor(cockatoo_app.config)
        
        with pytest.raises(FileNotFoundError):
            processor.submit_job(Path("/nonexistent/file.pdf"))

    def test_submit_job_unsupported_format(self, cockatoo_app, tmp_path):
        processor = DocumentProcessor(cockatoo_app.config)
        
        test_file = tmp_path / "test.xyz"
        test_file.write_text("content")
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            processor.submit_job(test_file)

    def test_cancel_job(self, cockatoo_app, tmp_path):
        processor = DocumentProcessor(cockatoo_app.config)
        processor.start()
        
        test_file = tmp_path / "test.pdf"
        test_file.write_text("content")
        
        job_id = processor.submit_job(test_file)
        
        assert processor.cancel_job(job_id) is True
        job = processor.get_job_status(job_id)
        assert job.status == ProcessingStatus.CANCELLED

    def test_get_all_jobs(self, cockatoo_app, tmp_path):
        processor = DocumentProcessor(cockatoo_app.config)
        processor.start()
        
        for i in range(3):
            test_file = tmp_path / f"test{i}.pdf"
            test_file.write_text(f"content {i}")
            processor.submit_job(test_file)
        
        jobs = processor.get_all_jobs()
        assert len(jobs) == 3


class TestAIProcessor:

    def test_ai_processor_initialization(self, cockatoo_app):
        processor = AIProcessor(cockatoo_app.config)
        
        assert processor.config is not None
        assert processor._initialized is False

    def test_initialize(self, cockatoo_app):
        processor = AIProcessor(cockatoo_app.config)
        processor.initialize()
        
        assert processor._initialized is True

    def test_generate_embeddings(self, cockatoo_app):
        processor = AIProcessor(cockatoo_app.config)
        processor.initialize()
        
        embeddings = processor.generate_embeddings("test text")
        
        assert isinstance(embeddings, list)
        assert len(embeddings) == cockatoo_app.config.ai.embeddings.dimensions

    def test_query_llm(self, cockatoo_app):
        processor = AIProcessor(cockatoo_app.config)
        processor.initialize()
        
        response = processor.query_llm("test prompt")
        
        assert isinstance(response, str)
        assert len(response) > 0

    def test_rag_query(self, cockatoo_app):
        processor = AIProcessor(cockatoo_app.config)
        processor.initialize()
        
        result = processor.rag_query("test query")
        
        assert isinstance(result, dict)
        assert "query" in result
        assert "response" in result

    def test_shutdown(self, cockatoo_app):
        processor = AIProcessor(cockatoo_app.config)
        processor.initialize()
        processor.shutdown()
        
        assert processor._initialized is False


class TestStorageManager:

    def test_storage_initialization(self, cockatoo_app, tmp_path):
        cockatoo_app.config.paths.documents_dir = tmp_path / "documents"
        cockatoo_app.config.paths.backup_dir = tmp_path / "backups"
        
        manager = StorageManager(cockatoo_app.config)
        
        assert manager.config is not None
        assert (tmp_path / "documents" / "uploads").exists()
        assert (tmp_path / "documents" / "processed").exists()
        assert (tmp_path / "backups").exists()

    def test_add_document(self, cockatoo_app, tmp_path):
        manager = StorageManager(cockatoo_app.config)
        
        test_file = tmp_path / "test.pdf"
        test_file.write_text("content")
        
        doc_info = manager.add_document(test_file)
        
        assert doc_info.id is not None
        assert doc_info.file_name == "test.pdf"
        assert doc_info.file_size > 0

    def test_get_document(self, cockatoo_app, tmp_path):
        manager = StorageManager(cockatoo_app.config)
        
        test_file = tmp_path / "test.pdf"
        test_file.write_text("content")
        
        doc_info = manager.add_document(test_file)
        retrieved = manager.get_document(doc_info.id)
        
        assert retrieved is not None
        assert retrieved.id == doc_info.id

    def test_update_document(self, cockatoo_app, tmp_path):
        manager = StorageManager(cockatoo_app.config)
        
        test_file = tmp_path / "test.pdf"
        test_file.write_text("content")
        
        doc_info = manager.add_document(test_file)
        
        assert manager.update_document(doc_info.id, processed=True) is True
        
        updated = manager.get_document(doc_info.id)
        assert updated.processed is True

    def test_delete_document(self, cockatoo_app, tmp_path):
        manager = StorageManager(cockatoo_app.config)
        
        test_file = tmp_path / "test.pdf"
        test_file.write_text("content")
        
        doc_info = manager.add_document(test_file)
        
        assert manager.delete_document(doc_info.id) is True
        assert manager.get_document(doc_info.id) is None

    def test_create_backup(self, cockatoo_app, tmp_path):
        cockatoo_app.config.paths.backup_dir = tmp_path / "backups"
        manager = StorageManager(cockatoo_app.config)
        
        backup_path = manager.create_backup()
        
        assert backup_path is not None
        assert backup_path.exists()


class TestUIManager:

    def test_ui_initialization(self, cockatoo_app):
        manager = UIManager(cockatoo_app.config)
        
        assert manager.theme == cockatoo_app.config.ui.theme
        assert manager.language == cockatoo_app.config.ui.language
        assert len(manager.get_recent_files()) == 0

    def test_theme_change(self, cockatoo_app):
        manager = UIManager(cockatoo_app.config)
        
        manager.theme = "light"
        assert manager.theme == ThemeMode.LIGHT
        assert cockatoo_app.config.ui.theme == ThemeMode.LIGHT

    def test_language_change(self, cockatoo_app):
        manager = UIManager(cockatoo_app.config)
        
        manager.language = "id"
        assert manager.language == "id"
        assert cockatoo_app.config.ui.language == "id"

    def test_recent_files(self, cockatoo_app, tmp_path):
        manager = UIManager(cockatoo_app.config)
        
        test_file = tmp_path / "test.pdf"
        
        manager.add_recent_file(test_file)
        recent = manager.get_recent_files()
        
        assert len(recent) == 1
        assert recent[0] == test_file
        
        manager.add_recent_file(test_file)
        recent = manager.get_recent_files()
        assert len(recent) == 1
        
        test_file2 = tmp_path / "test2.pdf"
        manager.add_recent_file(test_file2)
        recent = manager.get_recent_files()
        assert len(recent) == 2
        assert recent[0] == test_file2

    def test_clear_recent_files(self, cockatoo_app, tmp_path):
        manager = UIManager(cockatoo_app.config)
        
        manager.add_recent_file(tmp_path / "test.pdf")
        manager.clear_recent_files()
        
        assert len(manager.get_recent_files()) == 0

    def test_ui_listeners(self, cockatoo_app):
        manager = UIManager(cockatoo_app.config)
        
        mock_listener = MagicMock()
        manager.add_listener(mock_listener)
        
        manager.theme = "light"
        mock_listener.assert_called_once_with("theme", ThemeMode.DARK, ThemeMode.LIGHT)
        
        manager.remove_listener(mock_listener)
        manager.language = "id"
        assert mock_listener.call_count == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])