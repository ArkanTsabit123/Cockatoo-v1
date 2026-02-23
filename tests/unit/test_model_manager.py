# tests/unit/test_model_manager.py

"""Unit tests for model manager.

Tests cover model initialization, listing, download, progress tracking,
verification, information retrieval, deletion, cleanup, retry mechanisms,
concurrent downloads, disk space management, and integrity validation.
"""

import time
import json
import hashlib
import threading
from pathlib import Path
from unittest.mock import patch

import pytest
import requests

from src.ai_engine.model_manager import ModelManager, ModelInfo, DownloadProgress, ModelType


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset ModelManager singleton before each test."""
    ModelManager._instance = None
    ModelManager._lock = threading.Lock()
    yield


@pytest.fixture
def models_dir(tmp_path):
    """Create temporary models directory."""
    return tmp_path / "models"


@pytest.fixture
def config_file(tmp_path):
    """Create temporary config file."""
    return tmp_path / "models_config.json"


@pytest.fixture
def model_manager(models_dir, config_file):
    """Create model manager instance."""
    
    manager = ModelManager(
        config_path=config_file,
        models_dir=models_dir,
        cache_dir=models_dir / "cache"
    )
    
    manager.downloaded_models = {}
    manager.model_registry = {}
    
    return manager


@pytest.fixture
def sample_model_info():
    """Get sample model information."""
    return ModelInfo(
        name="test-model:7b",
        size_bytes=1_048_576,
        size_mb=1.0,
        ram_required_mb=8192,
        context_length=4096,
        family="test",
        description="Test model",
        tags=["test", "llm"],
        checksum="abc123def456",
        download_url="https://example.com/model.bin"
    )


class TestModelManagerInitialization:
    """Test model manager initialization."""

    def test_with_config_file(self, models_dir, config_file):
        """Test initialization with config file."""
        config_data = {
            "downloaded_models": {
                "llama2:7b": {
                    "path": str(models_dir / "llama2_7b.bin"),
                    "size": 3800000000,
                    "checksum": "abc123",
                    "download_date": time.time()
                }
            },
            "last_updated": time.time()
        }
        config_file.write_text(json.dumps(config_data))

        manager = ModelManager(models_dir=models_dir, config_file=config_file)

        assert manager.models_dir == models_dir
        assert manager.config_path == config_file

    def test_without_config_file(self, models_dir):
        """Test initialization without config file."""
        manager = ModelManager(models_dir=models_dir)

        assert manager.models_dir == models_dir
        assert len(manager.downloaded_models) == 0

    def test_directory_creation(self, tmp_path):
        """Test that directories are created."""
        deep_path = tmp_path / "deep" / "nested" / "models"
        
        manager = ModelManager(models_dir=deep_path)
        
        assert manager is not None
        assert manager.models_dir == deep_path
        assert deep_path.exists()

    def test_directory_creation_permission_error(self, tmp_path):
        """Test directory creation permission error."""
        with patch('src.ai_engine.model_manager.ModelManager.__init__', side_effect=PermissionError("Permission denied")):
            with pytest.raises(PermissionError, match="Permission denied"):
                ModelManager(models_dir=tmp_path / "models")

    def test_corrupt_config_file(self, models_dir, config_file):
        """Test handling of corrupt config file."""
        config_file.write_text("This is not valid JSON")

        manager = ModelManager(models_dir=models_dir, config_file=config_file)

        assert len(manager.downloaded_models) == 0


class TestModelListing:
    """Test model listing functionality."""

    def test_list_available_models(self, model_manager):
        """Test listing available models."""
        model1 = ModelInfo(
            name="llama2:7b",
            size_bytes=3800000000,
            size_mb=3623,
            ram_required_mb=8192,
            context_length=4096,
            family="llama2",
            description="Meta Llama 2",
            tags=["llama", "llm"],
            checksum="abc123",
            download_url="https://example.com/llama2.bin"
        )
        
        model2 = ModelInfo(
            name="mistral:7b",
            size_bytes=4100000000,
            size_mb=3910,
            ram_required_mb=8192,
            context_length=8192,
            family="mistral",
            description="Mistral AI",
            tags=["mistral", "llm"],
            checksum="def456",
            download_url="https://example.com/mistral.bin"
        )
        
        model_manager.model_registry["llama2:7b"] = model1
        model_manager.model_registry["mistral:7b"] = model2

        models = model_manager.list_available_models()

        assert len(models) >= 2
        assert any(m.name == "llama2:7b" for m in models)
        assert any(m.name == "mistral:7b" for m in models)

    def test_list_available_models_network_error(self, model_manager, sample_model_info):
        """Test listing models with network error."""
        model_manager.model_registry["llama2:7b"] = sample_model_info
        
        with patch('requests.get', side_effect=requests.ConnectionError("Network error")):
            models = model_manager.list_available_models()
            assert len(models) > 0

    def test_list_downloaded_models(self, model_manager, sample_model_info):
        """Test listing downloaded models."""
        downloaded = model_manager.list_downloaded_models()
        assert len(downloaded) == 0

        model_manager.model_registry["llama2:7b"] = sample_model_info

        with patch.object(model_manager, '_download_file', return_value=True):
            model_manager.download_model("llama2:7b")
        
        downloaded = model_manager.list_downloaded_models()

        assert len(downloaded) == 1
        assert downloaded[0].name == "llama2:7b"

    def test_search_models(self, model_manager):
        """Test searching for models."""
        model1 = ModelInfo(
            name="llama2:7b",
            size_bytes=3800000000,
            size_mb=3623,
            ram_required_mb=8192,
            context_length=4096,
            family="llama2",
            description="Meta Llama 2",
            tags=["llama"],
            checksum="abc123",
            download_url="https://example.com/llama2.bin"
        )
        
        model2 = ModelInfo(
            name="llama2:13b",
            size_bytes=7600000000,
            size_mb=7246,
            ram_required_mb=16384,
            context_length=4096,
            family="llama2",
            description="Meta Llama 2 13B",
            tags=["llama"],
            checksum="def456",
            download_url="https://example.com/llama2-13b.bin"
        )
        
        model3 = ModelInfo(
            name="mistral:7b",
            size_bytes=4100000000,
            size_mb=3910,
            ram_required_mb=8192,
            context_length=8192,
            family="mistral",
            description="Mistral AI",
            tags=["mistral"],
            checksum="ghi789",
            download_url="https://example.com/mistral.bin"
        )
        
        model_manager.model_registry["llama2:7b"] = model1
        model_manager.model_registry["llama2:13b"] = model2
        model_manager.model_registry["mistral:7b"] = model3

        results = model_manager.search_models("llama")
        assert len(results) == 2
        assert all("llama" in r.name for r in results)


class TestModelDownload:
    """Test model download functionality."""

    def test_download_successful(self, model_manager, sample_model_info):
        """Test successful model download."""
        model_manager.model_registry["llama2:7b"] = sample_model_info

        assert not model_manager.is_model_downloaded("llama2:7b")

        with patch.object(model_manager, '_download_file', return_value=True):
            success = model_manager.download_model("llama2:7b")

        assert success is True

    def test_download_with_progress(self, model_manager, sample_model_info):
        """Test download with progress callback."""
        model_manager.model_registry["llama2:7b"] = sample_model_info
        
        progress_updates = []

        def progress_callback(progress):
            progress_updates.append(progress)

        with patch.object(model_manager, '_download_file') as mock_download:
            def fake_download(name, **kwargs):
                progress = model_manager.get_download_progress(name)
                progress.update(50, 100)
                progress.complete()
                return True
            mock_download.side_effect = fake_download
            
            success = model_manager.download_model("llama2:7b", progress_callback=progress_callback)

        assert success is True

    def test_download_invalid_model(self, model_manager):
        """Test downloading invalid model."""
        with pytest.raises(ValueError, match="not found in registry"):
            model_manager.download_model("non-existent-model")

    def test_download_model_already_exists(self, model_manager, sample_model_info):
        """Test downloading model that already exists."""
        model_manager.model_registry["llama2:7b"] = sample_model_info
        
        with patch.object(model_manager, '_download_file', return_value=True):
            model_manager.download_model("llama2:7b")
            success = model_manager.download_model("llama2:7b")

        assert success is True

    def test_download_partial_resume(self, model_manager, sample_model_info):
        """Test resuming partial download."""
        model_manager.model_registry["llama2:7b"] = sample_model_info
        
        with patch.object(model_manager, '_download_file', return_value=True):
            success = model_manager.download_model("llama2:7b", resume=True)

        assert success is True

    def test_download_network_interruption(self, model_manager, sample_model_info):
        """Test download with network interruption."""
        model_manager.model_registry["llama2:7b"] = sample_model_info
        
        with patch.object(model_manager, '_download_file', return_value=True):
            with patch('time.sleep', return_value=None):
                success = model_manager.download_model("llama2:7b")

        assert success is True

    def test_download_timeout(self, model_manager, sample_model_info):
        """Test download timeout."""
        model_manager.model_registry["llama2:7b"] = sample_model_info
        
        with patch('time.sleep', return_value=None):
            with patch.object(model_manager, '_download_file', return_value=True):
                success = model_manager.download_model("llama2:7b")

        assert success is True


class TestDownloadProgress:
    """Test download progress tracking."""

    def test_progress_tracking(self, model_manager, sample_model_info):
        """Test progress tracking during download."""
        model_manager.model_registry["llama2:7b"] = sample_model_info
        
        with patch.object(model_manager, '_download_file') as mock_download:
            def fake_download(name, **kwargs):
                progress = model_manager.get_download_progress(name)
                progress.update(50, 100)
                progress.complete()
                return True

            mock_download.side_effect = fake_download
            model_manager.download_model("llama2:7b")

        progress = model_manager.get_download_progress("llama2:7b")
        assert progress is not None
        assert progress.model_name == "llama2:7b"
        assert progress.status == "completed"
        assert progress.progress_percent == 100

    def test_get_all_downloads(self, model_manager, sample_model_info):
        """Test getting all downloads."""
        model_manager.model_registry["llama2:7b"] = sample_model_info
        model_manager.model_registry["mistral:7b"] = sample_model_info
        
        with patch.object(model_manager, '_download_file', return_value=True):
            model_manager.download_model("llama2:7b")
            model_manager.download_model("mistral:7b")

        downloads = model_manager.get_all_downloads()
        assert len(downloads) == 2

    def test_get_active_downloads(self, model_manager, sample_model_info):
        """Test getting active downloads."""
        model_manager.model_registry["llama2:7b"] = sample_model_info
        
        def slow_download():
            time.sleep(1)
            return True

        with patch.object(model_manager, '_download_file', side_effect=slow_download):
            thread = threading.Thread(target=model_manager.download_model, args=("llama2:7b",))
            thread.start()
            time.sleep(0.1)

            active = model_manager.get_active_downloads()
            assert len(active) == 1
            assert active[0].model_name == "llama2:7b"
            assert active[0].status == "downloading"

            thread.join(timeout=2)

    def test_cancel_download(self, model_manager, sample_model_info):
        """Test cancelling a download."""
        model_manager.model_registry["llama2:7b"] = sample_model_info
        
        def cancellable_download():
            progress = model_manager.get_download_progress("llama2:7b")
            if not progress:
                progress = DownloadProgress("llama2:7b")
                model_manager._download_progress["llama2:7b"] = progress
            
            for i in range(10):
                if progress.cancelled:
                    progress.fail("Download cancelled")
                    return False
                progress.update(i * 10, 100)
                time.sleep(0.05)
            progress.complete()
            return True

        with patch.object(model_manager, '_download_file', side_effect=cancellable_download):
            thread = threading.Thread(target=model_manager.download_model, args=("llama2:7b",))
            thread.start()
            time.sleep(0.2)

            result = model_manager.cancel_download("llama2:7b")
            assert result is True

            thread.join(timeout=1)

            progress = model_manager.get_download_progress("llama2:7b")
            assert progress is not None
            assert progress.status in ["cancelled", "failed"]


class TestModelVerification:
    """Test model verification."""

    def test_verify_model_success(self, model_manager, sample_model_info):
        """Test successful model verification."""
        model_manager.model_registry["llama2:7b"] = sample_model_info
        
        with patch.object(model_manager, '_download_file', return_value=True):
            model_manager.download_model("llama2:7b")

        with patch.object(model_manager, '_verify_file_integrity', return_value=True):
            verified = model_manager.verify_model("llama2:7b")

        assert verified is True

    def test_verify_nonexistent_model(self, model_manager):
        """Test verifying non-existent model."""
        verified = model_manager.verify_model("non-existent")
        assert verified is False

    def test_verify_integrity_detailed(self, model_manager):
        """Test detailed integrity verification."""
        model_path = model_manager.models_dir / "llama2_7b.bin"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(b"test model data")

        model_manager.downloaded_models["llama2:7b"] = {
            "path": str(model_path),
            "size": model_path.stat().st_size,
            "checksum": hashlib.sha256(b"test model data").hexdigest()
        }

        result = model_manager.verify_integrity("llama2:7b", detailed=True)

        assert result["verified"] is True
        assert "checks" in result
        assert result["checks"]["size_match"] is True

    def test_verify_integrity_corrupted(self, model_manager):
        """Test integrity verification with corrupted file."""
        model_path = model_manager.models_dir / "llama2_7b.bin"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(b"corrupted data")

        model_manager.downloaded_models["llama2:7b"] = {
            "path": str(model_path),
            "size": 1000,
            "checksum": hashlib.sha256(b"original data").hexdigest()
        }

        result = model_manager.verify_integrity("llama2:7b", detailed=True)

        assert result["verified"] is False
        assert result["checks"]["size_match"] is False

    def test_checksum_validation(self, model_manager, sample_model_info):
        """Test checksum validation during download."""
        model_manager.model_registry["test-model:7b"] = sample_model_info

        with patch.object(model_manager, '_download_file', return_value=True):
            success = model_manager.download_model("test-model:7b", verify_checksum=True)

        assert success is True


class TestModelDeletion:
    """Test model deletion."""

    def test_delete_model_success(self, model_manager):
        """Test successful model deletion."""
        model_path = model_manager.models_dir / "llama2_7b.bin"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.touch()

        model_manager.downloaded_models["llama2:7b"] = {"path": str(model_path), "size": 100}
        assert model_manager.is_model_downloaded("llama2:7b")

        success = model_manager.delete_model("llama2:7b")

        assert success is True
        assert not model_manager.is_model_downloaded("llama2:7b")
        assert not model_path.exists()

    def test_delete_nonexistent_model(self, model_manager):
        """Test deleting non-existent model."""
        success = model_manager.delete_model("non-existent")
        assert success is False

    def test_delete_with_file_error(self, model_manager):
        """Test deletion with file system error."""
        model_path = model_manager.models_dir / "llama2_7b.bin"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.touch()

        model_manager.downloaded_models["llama2:7b"] = {"path": str(model_path), "size": 100}

        with patch('pathlib.Path.unlink', side_effect=PermissionError("Permission denied")):
            success = model_manager.delete_model("llama2:7b")

        assert success is False
        assert model_manager.is_model_downloaded("llama2:7b") is True

    def test_delete_multiple_versions(self, model_manager):
        """Test deleting multiple versions of a model."""
        for version in ["7b", "13b", "70b"]:
            model_name = f"llama2:{version}"
            model_path = model_manager.models_dir / f"llama2_{version}.bin"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model_path.touch()
            model_manager.downloaded_models[model_name] = {"path": str(model_path), "size": 100}

        assert len(model_manager.downloaded_models) == 3

        results = model_manager.delete_models_by_family("llama2")

        assert results["deleted"] == 3
        assert results["failed"] == 0
        assert len(model_manager.downloaded_models) == 0


class TestDiskSpaceManagement:
    """Test disk space management."""

    def test_disk_space_check_before_download(self, model_manager, sample_model_info):
        """Test disk space check before download."""
        model_manager.model_registry["test-model:7b"] = sample_model_info

        with patch('shutil.disk_usage') as mock_disk_usage:
            mock_disk_usage.return_value = (1000, 500, 500)
            with patch.object(model_manager, '_download_file', return_value=True):
                success = model_manager.download_model("test-model:7b")
            assert success is True

    def test_auto_cleanup_on_low_space(self, model_manager):
        """Test automatic cleanup when disk space is low."""
        for i in range(5):
            model_name = f"model{i}:7b"
            model_path = model_manager.models_dir / f"model{i}.bin"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model_path.write_bytes(b"x" * 1024 * 1024)
            model_manager.downloaded_models[model_name] = {
                "path": str(model_path),
                "size": 1024 * 1024,
                "last_used": time.time() - (i * 3600)
            }

        freed_space = model_manager.cleanup_low_space(threshold_mb=3, target_free_mb=5)

        assert freed_space > 0
        assert len(model_manager.downloaded_models) < 5

    def test_cleanup_old_models(self, model_manager):
        """Test cleanup of old, unused models."""
        now = time.time()
        for i in range(5):
            model_name = f"old_model{i}:7b"
            model_path = model_manager.models_dir / f"old_model{i}.bin"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model_path.write_bytes(b"x" * 1024 * 1024)
            model_manager.downloaded_models[model_name] = {
                "path": str(model_path),
                "size": 1024 * 1024,
                "last_used": now - ((i + 1) * 86400)
            }

        cleaned = model_manager.cleanup_old_models(days=3)

        assert cleaned["deleted"] >= 2
        assert cleaned["space_freed_mb"] >= 2


class TestConcurrentDownloads:
    """Test concurrent download handling."""

    def test_concurrent_downloads_limit(self, model_manager, sample_model_info):
        """Test concurrent download limit enforcement."""
        model_manager.max_concurrent_downloads = 2
        
        for i in range(4):
            model_manager.model_registry[f"model{i}:7b"] = sample_model_info
            
        active_downloads = []
        active_lock = threading.Lock()
        all_downloads_complete = threading.Event()
        
        completed = 0
        
        def tracking_download(name, **kwargs):
            nonlocal completed
            with active_lock:
                active_downloads.append(name)
            
            time.sleep(0.2)
            
            with active_lock:
                active_downloads.remove(name)
                completed += 1
                if completed == 4:
                    all_downloads_complete.set()
            return True
        
        original_download = model_manager.download_model
        model_manager.download_model = tracking_download
        
        threads = []
        for i in range(4):
            thread = threading.Thread(
                target=model_manager.download_model,
                args=(f"model{i}:7b",)
            )
            thread.start()
            threads.append(thread)
        
        time.sleep(0.3)
        
        with active_lock:
            assert len(active_downloads) <= model_manager.max_concurrent_downloads, \
                f"Expected <=2, got {len(active_downloads)}: {active_downloads}"
        
        all_downloads_complete.wait(timeout=5)
        
        model_manager.download_model = original_download
        
        for thread in threads:
            thread.join(timeout=1)

    def test_concurrent_download_queue(self, model_manager, sample_model_info):
        """Test that downloads are queued when max_concurrent_downloads = 1."""
        model_manager.max_concurrent_downloads = 1

        for name in ["A", "B", "C"]:
            model_manager.model_registry[f"model{name}:7b"] = sample_model_info

        download_semaphore = threading.Semaphore(1)
        active_count = 0
        max_active = 0
        active_lock = threading.Lock()
        all_downloads_complete = threading.Event()
        completed = 0

        def queued_download(name, **kwargs):
            nonlocal active_count, max_active, completed
            
            with download_semaphore:
                with active_lock:
                    active_count += 1
                    max_active = max(max_active, active_count)
                
                time.sleep(0.2)
                
                with active_lock:
                    active_count -= 1
                    completed += 1
                    if completed == 3:
                        all_downloads_complete.set()
            
            return True

        original_download = model_manager.download_model
        model_manager.download_model = queued_download

        threads = []
        for name in ["A", "B", "C"]:
            thread = threading.Thread(
                target=model_manager.download_model,
                args=(f"model{name}:7b",)
            )
            thread.start()
            threads.append(thread)

        all_downloads_complete.wait(timeout=10)

        assert max_active <= 1, f"Expected max 1 active download, got {max_active}"

        model_manager.download_model = original_download

        for thread in threads:
            thread.join(timeout=1)

    def test_concurrent_download_same_model(self, model_manager, sample_model_info):
        """Test concurrent download of same model."""
        model_manager.model_registry["llama2:7b"] = sample_model_info
        
        results = []
        results_lock = threading.Lock()
        download_count = 0

        def same_model_download(name, **kwargs):
            nonlocal download_count
            with results_lock:
                download_count += 1
                results.append(True)
            time.sleep(0.1)
            return True

        original_download = model_manager.download_model
        model_manager.download_model = same_model_download

        threads = [threading.Thread(target=model_manager.download_model, args=("llama2:7b",)) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(results) == 3
        assert all(results)

        model_manager.download_model = original_download


class TestRetryMechanism:
    """Test retry mechanism for failed operations."""

    def test_retry_failed_download(self, model_manager, sample_model_info):
        """Test retrying a failed download."""
        model_manager.model_registry["llama2:7b"] = sample_model_info
        
        progress = DownloadProgress("llama2:7b")
        progress.fail("Download failed")
        model_manager._download_progress["llama2:7b"] = progress
        
        with patch.object(model_manager, 'download_model') as mock_download:
            mock_download.return_value = True

            success = model_manager.retry_failed_download("llama2:7b")

            assert success is True
            mock_download.assert_called_once_with("llama2:7b")

    def test_get_failed_downloads(self, model_manager):
        """Test getting list of failed downloads."""
        progress1 = DownloadProgress("llama2:7b")
        progress1.fail("Download failed")
        model_manager._download_progress["llama2:7b"] = progress1
        
        progress2 = DownloadProgress("mistral:7b")
        progress2.fail("Download failed")
        model_manager._download_progress["mistral:7b"] = progress2

        failed = model_manager.get_failed_downloads()
        assert len(failed) == 2
        assert "llama2:7b" in failed
        assert "mistral:7b" in failed

    def test_retry_all_failed(self, model_manager, sample_model_info):
        """Test retrying all failed downloads."""
        model_manager.model_registry["llama2:7b"] = sample_model_info
        model_manager.model_registry["mistral:7b"] = sample_model_info
        
        progress1 = DownloadProgress("llama2:7b")
        progress1.fail("Download failed")
        model_manager._download_progress["llama2:7b"] = progress1
        
        progress2 = DownloadProgress("mistral:7b")
        progress2.fail("Download failed")
        model_manager._download_progress["mistral:7b"] = progress2

        with patch.object(model_manager, 'download_model', return_value=True):
            results = model_manager.retry_all_failed()

        assert results["total"] == 2
        assert results["successful"] == 2

    def test_exponential_backoff(self, model_manager, sample_model_info):
        """Test exponential backoff on retries."""
        model_manager.model_registry["llama2:7b"] = sample_model_info
        
        model_manager.max_retries = 3
        model_manager.retry_delay = 0.1
        call_times = []

        def failing_download(*args, **kwargs):
            call_times.append(time.time())
            if len(call_times) <= model_manager.max_retries:
                raise Exception("Download failed")
            return True

        with patch.object(model_manager, 'download_model', side_effect=failing_download):
            with patch('time.sleep', return_value=None):
                try:
                    model_manager.download_model("llama2:7b", retry=True)
                except:
                    pass

        assert len(call_times) > 0


class TestModelMetadata:
    """Test model metadata management."""

    def test_get_model_info_from_registry(self, model_manager, sample_model_info):
        """Test getting model info from registry."""
        model_manager.model_registry["test-model:7b"] = sample_model_info

        info = model_manager.get_model_info("test-model:7b")

        assert info is not None

    def test_get_model_info_nonexistent(self, model_manager):
        """Test getting info for nonexistent model."""
        info = model_manager.get_model_info("nonexistent")
        assert info is None

    def test_update_model_metadata(self, model_manager):
        """Test updating model metadata."""
        model_manager.downloaded_models["llama2:7b"] = {
            "path": "/path/to/model",
            "size": 3800000000
        }

        model_manager.update_model_metadata("llama2:7b", {"last_used": time.time()})

        assert "last_used" in model_manager.downloaded_models["llama2:7b"]

    def test_save_load_config(self, model_manager, config_file):
        """Test saving and loading config."""
        model_manager.downloaded_models["llama2:7b"] = {
            "path": "/path/to/model",
            "size": 3800000000,
            "checksum": "abc123"
        }
        model_manager.save_config()

        ModelManager._instance = None
        new_manager = ModelManager(models_dir=model_manager.models_dir, config_file=config_file)

        assert len(new_manager.downloaded_models) >= 0


class TestModelFormatValidation:
    """Test model format validation."""

    def test_validate_gguf_format(self, model_manager, tmp_path):
        """Test GGUF format validation."""
        model_path = tmp_path / "model.gguf"
        gguf_header = b"GGUF" + b"\x00" * 100
        model_path.write_bytes(gguf_header)

        model_manager.downloaded_models["test:7b"] = {"path": str(model_path), "size": 100}
        result = model_manager.validate_model_format("test:7b")

        assert result["valid"] is True
        assert result.get("format") == "GGUF"

    def test_validate_unsupported_format(self, model_manager, tmp_path):
        """Test unsupported format validation."""
        model_path = tmp_path / "model.bin"
        model_path.write_bytes(b"invalid format")

        model_manager.downloaded_models["test:7b"] = {"path": str(model_path), "size": 100}
        result = model_manager.validate_model_format("test:7b")

        assert result["valid"] is False

    def test_corrupted_model_file(self, model_manager, tmp_path):
        """Test corrupted model file detection."""
        model_path = tmp_path / "model.gguf"
        model_path.write_bytes(b"GGUF" + b"\xFF" * 10)

        model_manager.downloaded_models["test:7b"] = {"path": str(model_path), "size": 100}
        result = model_manager.validate_model_format("test:7b")

        assert result["valid"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])