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
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from typing import Any, Dict
from dataclasses import asdict

import pytest

from src.core.config import AppConfig, ConfigManager


@pytest.fixture
def config_manager(tmp_path):
    """Create ConfigManager with temporary path."""
    config_path = tmp_path / ".cockatoo" / "config.yaml"
    return ConfigManager(config_path=config_path)


@pytest.fixture
def sample_config_data():
    """Provide sample configuration data."""
    return {
        "chunk_size": 750,
        "llm_model": "mistral:7b",
        "ui_theme": "light",
        "ocr_enabled": False,
        "rag_top_k": 8,
        "llm_temperature": 0.3,
        "ui_font_size": 14
    }


@pytest.fixture
def populated_config_manager(config_manager, sample_config_data):
    """Create ConfigManager with pre-populated values."""
    for key, value in sample_config_data.items():
        config_manager.set(key, value)
    return config_manager


class TestAppConfigCreation:
    """Test suite for AppConfig dataclass creation."""

    def test_default_config_creation(self):
        """Test AppConfig creation with default values."""
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
        """Test AppConfig creation with custom values."""
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
        """Test config attribute modification."""
        config = AppConfig()
        config.chunk_size = 750
        config.llm_model = "neural-chat:7b"

        assert config.chunk_size == 750
        assert config.llm_model == "neural-chat:7b"

    def test_path_expansion(self):
        """Test path tilde expansion."""
        config = AppConfig(data_dir="~/test")
        expanded = Path(config.data_dir).expanduser()
        assert "~" not in str(expanded)


class TestConfigManager:
    """Test suite for ConfigManager functionality."""

    def test_initialization_default_path(self):
        """Test initialization with default path."""
        manager = ConfigManager()
        assert isinstance(manager.config, AppConfig)
        assert manager.config_path == Path.home() / ".cockatoo" / "config.yaml"

    def test_initialization_custom_path(self, tmp_path):
        """Test initialization with custom path."""
        custom_path = tmp_path / "custom" / "config.yaml"
        manager = ConfigManager(config_path=custom_path)
        assert manager.config_path == custom_path

    @patch('pathlib.Path.mkdir')
    def test_directory_creation(self, mock_mkdir):
        """Test directory creation during initialization."""
        ConfigManager()
        assert mock_mkdir.call_count >= 6

    @patch('pathlib.Path.mkdir')
    def test_directory_creation_permission_error(self, mock_mkdir):
        """Test handling of permission errors."""
        mock_mkdir.side_effect = PermissionError("Permission denied")
        with pytest.raises(PermissionError, match="Permission denied"):
            ConfigManager()

    def test_get_config_value(self, config_manager):
        """Test getting config values by key."""
        assert config_manager.get("name") == "Cockatoo"
        assert config_manager.get("nonexistent", "default") == "default"

    def test_set_config_value(self, config_manager):
        """Test setting config values by key."""
        config_manager.set("chunk_size", 750)
        assert config_manager.config.chunk_size == 750

        with pytest.raises(KeyError, match="Invalid configuration key"):
            config_manager.set("invalid_key", "value")

    def test_validation_success(self, config_manager):
        """Test validation with valid values."""
        config_manager.set("chunk_size", 500)
        config_manager.set("chunk_overlap", 50)
        config_manager.set("llm_temperature", 0.1)

        result = config_manager.validate()
        assert result["valid"] is True
        assert len(result["errors"]) == 0

    def test_validation_with_warnings(self, config_manager):
        """Test validation with values outside recommended range."""
        config_manager.set("chunk_size", 50)
        config_manager.set("ui_font_size", 6)

        result = config_manager.validate()
        assert result["valid"] is True
        assert len(result["warnings"]) > 0

    @pytest.mark.parametrize("chunk_size,chunk_overlap,expected_valid", [
        (500, 50, True),
        (500, 600, False),
        (500, -10, False),
    ])
    def test_chunk_validation(self, config_manager, chunk_size, chunk_overlap, expected_valid):
        """Test chunk size and overlap validation."""
        config_manager.set("chunk_size", chunk_size)
        config_manager.set("chunk_overlap", chunk_overlap)

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
        """Test temperature validation."""
        config_manager.set("llm_temperature", temperature)
        result = config_manager.validate()
        assert result["valid"] == expected_valid

    def test_validation_empty_paths(self, config_manager):
        """Test validation catches empty paths."""
        config_manager.set("data_dir", "")
        result = config_manager.validate()
        assert result["valid"] is False
        assert any("data_dir" in e for e in result["errors"])

    def test_listener_notification(self, config_manager):
        """Test config change listener notification."""
        mock_listener = MagicMock()
        config_manager.add_listener(mock_listener)
        config_manager.set("chunk_size", 1000)
        mock_listener.assert_called_once_with("chunk_size", 500, 1000)

    def test_multiple_listeners(self, config_manager):
        """Test multiple listeners receive notifications."""
        mock1 = MagicMock()
        mock2 = MagicMock()
        config_manager.add_listener(mock1)
        config_manager.add_listener(mock2)
        config_manager.set("llm_model", "new-model")
        mock1.assert_called_once()
        mock2.assert_called_once()

    def test_remove_listener(self, config_manager):
        """Test removing a listener."""
        mock_listener = MagicMock()
        config_manager.add_listener(mock_listener)
        config_manager.remove_listener(mock_listener)
        config_manager.set("chunk_size", 1000)
        mock_listener.assert_not_called()

    def test_listener_not_called_for_unchanged_value(self, config_manager):
        """Test listeners not called when value unchanged."""
        mock_listener = MagicMock()
        config_manager.add_listener(mock_listener)
        config_manager.set("chunk_size", 500)
        mock_listener.assert_not_called()


class TestConfigPersistence:
    """Test suite for configuration persistence."""

    def test_save_config(self, config_manager):
        """Test saving configuration to file."""
        config_manager.set("chunk_size", 750)
        config_manager.set("llm_model", "mistral:7b")
        config_manager.save()

        assert config_manager.config_path.exists()
        with open(config_manager.config_path, 'r') as f:
            data = yaml.safe_load(f)
        assert data["chunk_size"] == 750
        assert data["llm_model"] == "mistral:7b"

    def test_load_config(self, config_manager, sample_config_data):
        """Test loading configuration from file."""
        with open(config_manager.config_path, 'w') as f:
            yaml.dump(sample_config_data, f)

        config = config_manager.load()
        assert config.chunk_size == sample_config_data["chunk_size"]
        assert config.llm_model == sample_config_data["llm_model"]

    def test_load_nonexistent_file(self, config_manager):
        """Test loading from nonexistent file."""
        if config_manager.config_path.exists():
            config_manager.config_path.unlink()
        config = config_manager.load()
        assert config.chunk_size == 500

    def test_save_load_roundtrip(self, populated_config_manager, sample_config_data):
        """Test full save-load cycle preserves values."""
        populated_config_manager.save()
        new_manager = ConfigManager(config_path=populated_config_manager.config_path)
        new_manager.load()
        for key, expected in sample_config_data.items():
            assert new_manager.get(key) == expected

    def test_save_with_custom_path(self, config_manager, tmp_path):
        """Test saving to custom path."""
        custom_path = tmp_path / "custom.yaml"
        config_manager.save(file_path=custom_path)
        assert custom_path.exists()

    def test_load_with_custom_path(self, config_manager, sample_config_data, tmp_path):
        """Test loading from custom path."""
        custom_path = tmp_path / "custom.yaml"
        with open(custom_path, 'w') as f:
            yaml.dump(sample_config_data, f)
        config = config_manager.load(file_path=custom_path)
        assert config.llm_model == sample_config_data["llm_model"]

    def test_save_creates_directory(self, config_manager, tmp_path):
        """Test save creates parent directories."""
        deep_path = tmp_path / "deep" / "nested" / "config.yaml"
        config_manager.save(file_path=deep_path)
        assert deep_path.exists()
        assert deep_path.parent.exists()

    def test_save_permission_error(self, config_manager):
        """Test handling of permission error during save."""
        with patch('builtins.open') as mock_open:
            mock_open.side_effect = PermissionError("Permission denied")
            with pytest.raises(RuntimeError, match="Failed to save configuration"):
                config_manager.save()

    def test_load_invalid_yaml(self, config_manager):
        """Test loading invalid YAML."""
        with open(config_manager.config_path, 'w') as f:
            f.write(": invalid: yaml: [")
        with pytest.raises(RuntimeError, match="Failed to load configuration"):
            config_manager.load()

    def test_type_conversion_string_to_int(self, config_manager):
        """Test string to int conversion during load."""
        with open(config_manager.config_path, 'w') as f:
            yaml.dump({"chunk_size": "500"}, f)
        config = config_manager.load()
        assert isinstance(config.chunk_size, int)
        assert config.chunk_size == 500

    def test_type_conversion_string_to_bool(self, config_manager):
        """Test string to bool conversion during load."""
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
        """Test string to float conversion during load."""
        with open(config_manager.config_path, 'w') as f:
            yaml.dump({"llm_temperature": "0.35"}, f)
        config = config_manager.load()
        assert isinstance(config.llm_temperature, float)
        assert config.llm_temperature == 0.35

    def test_type_conversion_string_to_list(self, config_manager):
        """Test string to list conversion during load."""
        with open(config_manager.config_path, 'w') as f:
            yaml.dump({"ocr_languages": "eng,deu,fra"}, f)
        config = config_manager.load()
        assert config.ocr_languages == ["eng", "deu", "fra"]

    def test_invalid_type_conversion_falls_back(self, config_manager):
        """Test invalid conversions fall back to defaults."""
        with open(config_manager.config_path, 'w') as f:
            yaml.dump({"chunk_size": "not_an_int"}, f)
        config = config_manager.load()
        assert config.chunk_size == 500

    def test_load_old_config_version(self, config_manager):
        """Test loading config with missing fields."""
        old_config = {"chunk_size": 600, "llm_model": "old-model"}
        with open(config_manager.config_path, 'w') as f:
            yaml.dump(old_config, f)
        loaded = config_manager.load()
        assert loaded.chunk_size == 600
        assert loaded.llm_model == "old-model"
        assert loaded.ocr_languages == ["eng", "ind"]

    def test_load_config_with_extra_fields(self, config_manager):
        """Test loading config with unknown fields."""
        config_with_extra = {
            "chunk_size": 700,
            "unknown_field": "ignored"
        }
        with open(config_manager.config_path, 'w') as f:
            yaml.dump(config_with_extra, f)
        loaded = config_manager.load()
        assert loaded.chunk_size == 700
        assert not hasattr(loaded, "unknown_field")

    def test_dynamic_config_reload(self, config_manager, tmp_path):
        """Test dynamic config reloading."""
        config_manager.save()
        initial = config_manager.get("chunk_size")
        with open(config_manager.config_path, 'w') as f:
            yaml.dump({"chunk_size": 999}, f)
        config_manager.reload()
        assert config_manager.get("chunk_size") == 999
        assert config_manager.get("chunk_size") != initial

    def test_config_file_watching(self, config_manager, tmp_path):
        """Test config file change detection."""
        mock_callback = MagicMock()
        config_manager.add_reload_callback(mock_callback)
        
        # Save config first to create the file
        config_manager.save()
        
        config_manager.start_file_watching()
        with open(config_manager.config_path, 'w') as f:
            yaml.dump({"chunk_size": 888}, f)
        time.sleep(0.1)
        mock_callback.assert_called()

class TestConfigEnvironmentOverrides:
    """Test environment variable overrides."""

    @patch.dict(os.environ, {"COCKATOO_LLM_MODEL": "env-model:7b"})
    def test_env_override_single(self, config_manager):
        """Test single environment variable override."""
        config_manager.load_with_env_overrides()
        assert config_manager.get("llm_model") == "env-model:7b"

    @patch.dict(os.environ, {
        "COCKATOO_CHUNK_SIZE": "1000",
        "COCKATOO_LLM_TEMPERATURE": "0.5"
    })
    def test_env_override_multiple(self, config_manager):
        """Test multiple environment variable overrides."""
        config_manager.load_with_env_overrides()
        assert config_manager.get("chunk_size") == 1000
        assert config_manager.get("llm_temperature") == 0.5

    @patch.dict(os.environ, {"COCKATOO_OCR_ENABLED": "false"})
    def test_env_override_boolean(self, config_manager):
        """Test boolean environment variable override."""
        config_manager.load_with_env_overrides()
        assert config_manager.get("ocr_enabled") is False

    @patch.dict(os.environ, {"COCKATOO_OCR_LANGUAGES": "eng,deu,fra"})
    def test_env_override_list(self, config_manager):
        """Test list environment variable override."""
        config_manager.load_with_env_overrides()
        assert config_manager.get("ocr_languages") == ["eng", "deu", "fra"]

    @patch.dict(os.environ, {"COCKATOO_INVALID_KEY": "value"})
    def test_env_override_unknown_key(self, config_manager):
        """Test unknown environment variable is ignored."""
        config_manager.load_with_env_overrides()
        assert config_manager.get("name") == "Cockatoo"


class TestConfigConcurrency:
    """Test concurrent access to configuration."""

    def test_thread_safe_reads(self, config_manager):
        """Test multiple threads reading simultaneously."""
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
        """Test multiple threads writing simultaneously."""
        errors = []

        def write_config(thread_id):
            try:
                for i in range(50):
                    config_manager.set("chunk_size", 500 + thread_id * 100 + i)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=write_config, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_read_write(self, config_manager):
        """Test concurrent reads and writes."""
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
                    config_manager.set("chunk_size", 500 + i)
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
        """Test listener notifications are thread-safe."""
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
                    config_manager.set("chunk_size", 500 + thread_id * 100 + i)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert notification_queue.qsize() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])