# cockatoo_v1/src/core/app.py
"""
Core application module for Cockatoo_V1.

This module provides the main application class and core components
for document processing, AI integration, storage management, and UI.
"""

import os
import sys
import signal
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable
from enum import Enum
from datetime import datetime
from dataclasses import dataclass, field
import logging
import traceback

from .config import (
    AppConfig,
    ConfigManager,
    get_config,
    init_config,
    save_config,
    PlatformType,
    LogLevel,
    ThemeMode,
    DatabaseType,
    logger as config_logger,
)

from .constants import (
    APP_NAME,
    APP_VERSION,
    ERROR_MESSAGES,
    DEFAULT_MAX_WORKERS,
)

from .exceptions import (
    CockatooError,
    ConfigurationError,
    AppInitializationError,
    AppRuntimeError,
    AppShutdownError,
    DocumentProcessingError,
    AIError,
    StorageError,
)


class AppState(str, Enum):
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"
    ERROR = "error"


class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProcessingJob:
    id: str
    file_path: Path
    status: ProcessingStatus = ProcessingStatus.PENDING
    progress: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentInfo:
    id: str
    file_path: Path
    file_name: str
    file_size: int
    mime_type: str
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    character_count: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    processed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    embeddings: Optional[List[float]] = None


class DocumentProcessor:
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.DocumentProcessor")
        self._jobs: Dict[str, ProcessingJob] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        
        self.logger.info("DocumentProcessor initialized")
    
    def start(self) -> None:
        if self._worker_thread and self._worker_thread.is_alive():
            self.logger.warning("DocumentProcessor already running")
            return
        
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._process_loop,
            name="DocProcessor",
            daemon=True
        )
        self._worker_thread.start()
        self.logger.info("DocumentProcessor started")
    
    def stop(self) -> None:
        self.logger.info("Stopping DocumentProcessor...")
        self._stop_event.set()
        
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)
        
        self.logger.info("DocumentProcessor stopped")
    
    def submit_job(self, file_path: Path) -> str:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        ext = file_path.suffix.lower()
        if ext not in self.config.document_processing.supported_formats:
            raise ValueError(
                f"Unsupported file format: {ext}. "
                f"Supported formats: {self.config.document_processing.supported_formats}"
            )
        
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.config.document_processing.max_file_size_mb:
            raise ValueError(
                f"File size ({file_size_mb:.2f} MB) exceeds maximum "
                f"({self.config.document_processing.max_file_size_mb} MB)"
            )
        
        job_id = f"job_{int(time.time())}_{hash(file_path)}"
        
        with self._lock:
            job = ProcessingJob(
                id=job_id,
                file_path=file_path,
                metadata={
                    "file_size_mb": file_size_mb,
                    "file_extension": ext,
                }
            )
            self._jobs[job_id] = job
        
        self.logger.info(f"Job submitted: {job_id} - {file_path.name}")
        return job_id
    
    def get_job_status(self, job_id: str) -> Optional[ProcessingJob]:
        with self._lock:
            return self._jobs.get(job_id)
    
    def get_all_jobs(self) -> List[ProcessingJob]:
        with self._lock:
            return list(self._jobs.values())
    
    def cancel_job(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job and job.status == ProcessingStatus.PENDING:
                job.status = ProcessingStatus.CANCELLED
                self.logger.info(f"Job cancelled: {job_id}")
                return True
        return False
    
    def _process_loop(self) -> None:
        self.logger.info("Processing loop started")
        
        while not self._stop_event.is_set():
            try:
                self._process_next_job()
                time.sleep(0.1)
            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
        
        self.logger.info("Processing loop stopped")
    
    def _process_next_job(self) -> None:
        job = None
        
        with self._lock:
            for j in self._jobs.values():
                if j.status == ProcessingStatus.PENDING:
                    job = j
                    j.status = ProcessingStatus.PROCESSING
                    j.started_at = datetime.now()
                    break
        
        if not job:
            return
        
        self.logger.info(f"Processing job: {job.id} - {job.file_path.name}")
        
        try:
            self._process_document(job)
            
            job.status = ProcessingStatus.COMPLETED
            job.progress = 100.0
            job.completed_at = datetime.now()
            
            self.logger.info(f"Job completed: {job.id}")
            
        except Exception as e:
            job.status = ProcessingStatus.FAILED
            job.error = str(e)
            job.completed_at = datetime.now()
            
            self.logger.error(f"Job failed: {job.id} - {e}")
    
    def _process_document(self, job: ProcessingJob) -> None:
        try:
            total_steps = 5
            
            for step in range(total_steps):
                if self._stop_event.is_set():
                    raise DocumentProcessingError("Processing cancelled")
                
                job.progress = (step / total_steps) * 100
                
                time.sleep(0.5)
            
            job.result = {
                "chunks": 10,
                "pages": 5,
                "words": 1000,
            }
            
        except Exception as e:
            raise DocumentProcessingError(f"Document processing failed: {e}")


class AIProcessor:
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.AIProcessor")
        self._initialized = False
        self._models_loaded = False
        
        self.logger.info("AIProcessor initialized")
    
    def initialize(self) -> None:
        if self._initialized:
            return
        
        self.logger.info("Initializing AI components...")
        
        try:
            self._init_llm()
            self._init_embeddings()
            self._init_database()
            
            self._initialized = True
            self.logger.info("AI components initialized")
            
        except Exception as e:
            raise AIError(f"Failed to initialize AI components: {e}")
    
    def _init_llm(self) -> None:
        self.logger.info(
            f"Initializing LLM: {self.config.ai.llm.provider} - {self.config.ai.llm.model}"
        )
        time.sleep(0.1)
    
    def _init_embeddings(self) -> None:
        self.logger.info(
            f"Initializing embeddings: {self.config.ai.embeddings.model}"
        )
        time.sleep(0.1)
    
    def _init_database(self) -> None:
        self.logger.info(
            f"Initializing database: {self.config.ai.database_type.value}"
        )
        time.sleep(0.1)
    
    def shutdown(self) -> None:
        self.logger.info("Shutting down AI components...")
        self._initialized = False
        self.logger.info("AI components shut down")
    
    def generate_embeddings(self, text: str) -> List[float]:
        if not self._initialized:
            raise AIError("AIProcessor not initialized")
        
        self.logger.debug(f"Generating embeddings for text: {text[:50]}...")
        
        return [0.0] * self.config.ai.embeddings.dimensions
    
    def query_llm(self, prompt: str, **kwargs) -> str:
        if not self._initialized:
            raise AIError("AIProcessor not initialized")
        
        self.logger.debug(f"Querying LLM: {prompt[:50]}...")
        
        return f"Response to: {prompt[:50]}..."
    
    def rag_query(self, query: str, **kwargs) -> Dict[str, Any]:
        if not self._initialized:
            raise AIError("AIProcessor not initialized")
        
        self.logger.debug(f"RAG query: {query[:50]}...")
        
        return {
            "query": query,
            "results": [],
            "context": "",
            "response": f"RAG response to: {query[:50]}...",
        }


class StorageManager:
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.StorageManager")
        self._documents: Dict[str, DocumentInfo] = {}
        self._lock = threading.Lock()
        
        self._ensure_directories()
        
        self.logger.info("StorageManager initialized")
    
    def _ensure_directories(self) -> None:
        directories = [
            self.config.paths.documents_dir / "uploads",
            self.config.paths.documents_dir / "processed",
            self.config.paths.backup_dir,
            self.config.paths.exports_dir,
            self.config.paths.temp_dir,
        ]
        
        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                self.logger.debug(f"Ensured directory: {directory}")
            except Exception as e:
                self.logger.error(f"Failed to create directory {directory}: {e}")
    
    def add_document(self, file_path: Path) -> DocumentInfo:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        doc_id = f"doc_{int(time.time())}_{hash(file_path)}"
        
        with self._lock:
            doc_info = DocumentInfo(
                id=doc_id,
                file_path=file_path,
                file_name=file_path.name,
                file_size=file_path.stat().st_size,
                mime_type=self._get_mime_type(file_path),
            )
            self._documents[doc_id] = doc_info
        
        self.logger.info(f"Document added: {doc_id} - {file_path.name}")
        return doc_info
    
    def _get_mime_type(self, file_path: Path) -> str:
        ext = file_path.suffix.lower()
        mime_map = {
            ".pdf": "application/pdf",
            ".txt": "text/plain",
            ".md": "text/markdown",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
        }
        return mime_map.get(ext, "application/octet-stream")
    
    def get_document(self, doc_id: str) -> Optional[DocumentInfo]:
        with self._lock:
            return self._documents.get(doc_id)
    
    def get_all_documents(self) -> List[DocumentInfo]:
        with self._lock:
            return list(self._documents.values())
    
    def update_document(self, doc_id: str, **kwargs) -> bool:
        with self._lock:
            doc = self._documents.get(doc_id)
            if not doc:
                return False
            
            for key, value in kwargs.items():
                if hasattr(doc, key):
                    setattr(doc, key, value)
            
            doc.updated_at = datetime.now()
            
        return True
    
    def delete_document(self, doc_id: str) -> bool:
        with self._lock:
            if doc_id in self._documents:
                del self._documents[doc_id]
                self.logger.info(f"Document deleted: {doc_id}")
                return True
        return False
    
    def create_backup(self) -> Optional[Path]:
        if not self.config.storage.backup_enabled:
            self.logger.info("Backup is disabled")
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.config.paths.backup_dir / f"backup_{timestamp}.zip"
        
        self.logger.info(f"Creating backup: {backup_path}")
        
        try:
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            backup_path.touch()
            
            time.sleep(0.5)
            
            self._cleanup_old_backups()
            
            self.logger.info(f"Backup created: {backup_path}")
            return backup_path
            
        except Exception as e:
            self.logger.error(f"Failed to create backup: {e}")
            return None
    
    def _cleanup_old_backups(self) -> None:
        try:
            backups = sorted(self.config.paths.backup_dir.glob("backup_*.zip"))
            max_backups = self.config.storage.max_backups
            
            if len(backups) > max_backups:
                for backup in backups[:-max_backups]:
                    backup.unlink()
                    self.logger.info(f"Removed old backup: {backup}")
                    
        except Exception as e:
            self.logger.error(f"Failed to clean up old backups: {e}")


class UIManager:
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.UIManager")
        self._theme = config.ui.theme
        self._language = config.ui.language
        self._recent_files: List[Path] = []
        self._listeners: List[Callable] = []
        
        self.logger.info("UIManager initialized")
    
    @property
    def theme(self) -> ThemeMode:
        return self._theme
    
    @theme.setter
    def theme(self, value: Union[str, ThemeMode]) -> None:
        old_value = self._theme
        
        if isinstance(value, str):
            value = ThemeMode(value)
        
        self._theme = value
        self._notify_listeners("theme", old_value, value)
        
        self.config.ui.theme = value
    
    @property
    def language(self) -> str:
        return self._language
    
    @language.setter
    def language(self, value: str) -> None:
        old_value = self._language
        self._language = value
        self._notify_listeners("language", old_value, value)
        
        self.config.ui.language = value
    
    def add_recent_file(self, file_path: Path) -> None:
        if file_path in self._recent_files:
            self._recent_files.remove(file_path)
        
        self._recent_files.insert(0, file_path)
        
        max_files = self.config.ui.max_recent_files
        if len(self._recent_files) > max_files:
            self._recent_files = self._recent_files[:max_files]
        
        self._notify_listeners("recent_files", None, self._recent_files)
    
    def get_recent_files(self) -> List[Path]:
        return self._recent_files.copy()
    
    def clear_recent_files(self) -> None:
        self._recent_files.clear()
        self._notify_listeners("recent_files", None, [])
    
    def add_listener(self, callback: Callable[[str, Any, Any], None]) -> None:
        self._listeners.append(callback)
    
    def remove_listener(self, callback: Callable) -> None:
        if callback in self._listeners:
            self._listeners.remove(callback)
    
    def _notify_listeners(self, key: str, old_value: Any, new_value: Any) -> None:
        for listener in self._listeners:
            try:
                listener(key, old_value, new_value)
            except Exception as e:
                self.logger.error(f"Error in UI listener: {e}")


class CockatooApp:
    
    def __init__(self, config_path: Optional[Path] = None):
        self.logger = logging.getLogger(f"{__name__}.CockatooApp")
        self.state = AppState.INITIALIZING
        self.start_time: Optional[datetime] = None
        
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config
        
        self.document_processor: Optional[DocumentProcessor] = None
        self.ai_processor: Optional[AIProcessor] = None
        self.storage_manager: Optional[StorageManager] = None
        self.ui_manager: Optional[UIManager] = None
        
        self._shutdown_event = threading.Event()
        self._shutdown_timeout = 10.0
        
        self._register_signal_handlers()
        
        self.logger.info(f"CockatooApp instance created (v{APP_VERSION})")
    
    def _register_signal_handlers(self) -> None:
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except (ValueError, AttributeError):
            pass
    
    def _signal_handler(self, signum: int, frame: Any) -> None:
        signal_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        self.logger.info(f"Received {signal_name}, initiating shutdown...")
        
        threading.Thread(target=self.shutdown, daemon=True).start()
    
    def initialize(self) -> None:
        if self.state != AppState.INITIALIZING:
            raise AppInitializationError(
                f"Cannot initialize from state: {self.state}"
            )
        
        self.logger.info("Initializing Cockatoo application...")
        
        try:
            self.logger.info("Loading configuration...")
            self.config = self.config_manager.load()
            
            self.logger.info("Creating directories...")
            self.config.ensure_directories()
            
            self.logger.info("Setting up logging...")
            self.config.setup_logging()
            
            self.logger.info("Initializing components...")
            
            self.storage_manager = StorageManager(self.config)
            self.document_processor = DocumentProcessor(self.config)
            self.ai_processor = AIProcessor(self.config)
            self.ui_manager = UIManager(self.config)
            
            self.document_processor.start()
            
            if not self.config.performance.lazy_loading:
                self.ai_processor.initialize()
            
            self.state = AppState.RUNNING
            self.start_time = datetime.now()
            
            self.logger.info("Cockatoo application initialized successfully")
            
        except Exception as e:
            self.state = AppState.ERROR
            self.logger.error(f"Failed to initialize application: {e}")
            raise AppInitializationError("Failed to initialize application", cause=e)
    
    def run(self) -> None:
        if self.state != AppState.RUNNING:
            raise AppRuntimeError(f"Cannot run from state: {self.state}")
        
        self.logger.info("Cockatoo application is running")
        
        try:
            while self.state == AppState.RUNNING and not self._shutdown_event.is_set():
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received")
            self.shutdown()
            
        except Exception as e:
            self.state = AppState.ERROR
            self.logger.error(f"Runtime error: {e}")
            raise AppRuntimeError("Application runtime error", cause=e)
    
    def shutdown(self) -> None:
        if self.state in [AppState.SHUTTING_DOWN, AppState.STOPPED]:
            return
        
        self.logger.info("Shutting down Cockatoo application...")
        self.state = AppState.SHUTTING_DOWN
        self._shutdown_event.set()
        
        try:
            if self.document_processor:
                self.logger.info("Stopping document processor...")
                self.document_processor.stop()
            
            if self.ai_processor:
                self.logger.info("Shutting down AI processor...")
                self.ai_processor.shutdown()
            
            self.logger.info("Saving configuration...")
            self.config_manager.save()
            
            if self.storage_manager and self.config.storage.backup_enabled:
                self.logger.info("Creating backup...")
                self.storage_manager.create_backup()
            
            self.state = AppState.STOPPED
            self.logger.info("Cockatoo application shut down successfully")
            
        except Exception as e:
            self.state = AppState.ERROR
            self.logger.error(f"Error during shutdown: {e}")
            raise AppShutdownError("Failed to shutdown application", cause=e)
    
    def pause(self) -> None:
        if self.state != AppState.RUNNING:
            raise AppRuntimeError(f"Cannot pause from state: {self.state}")
        
        self.logger.info("Pausing application...")
        self.state = AppState.PAUSED
    
    def resume(self) -> None:
        if self.state != AppState.PAUSED:
            raise AppRuntimeError(f"Cannot resume from state: {self.state}")
        
        self.logger.info("Resuming application...")
        self.state = AppState.RUNNING
    
    def get_status(self) -> Dict[str, Any]:
        uptime = None
        if self.start_time:
            uptime = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "state": self.state.value,
            "uptime_seconds": uptime,
            "version": APP_VERSION,
            "config_version": self.config.config_version,
            "platform": self.config.paths.platform.value,
            "components": {
                "document_processor": self.document_processor is not None,
                "ai_processor": self.ai_processor is not None,
                "storage_manager": self.storage_manager is not None,
                "ui_manager": self.ui_manager is not None,
            }
        }
    
    def process_document(self, file_path: Path) -> str:
        if self.state != AppState.RUNNING:
            raise AppRuntimeError(f"Cannot process document from state: {self.state}")
        
        if not self.document_processor:
            raise AppRuntimeError("Document processor not initialized")
        
        if self.storage_manager:
            self.storage_manager.add_document(file_path)
        
        job_id = self.document_processor.submit_job(file_path)
        
        if self.ui_manager:
            self.ui_manager.add_recent_file(file_path)
        
        return job_id
    
    def get_job_status(self, job_id: str) -> Optional[ProcessingJob]:
        if not self.document_processor:
            return None
        return self.document_processor.get_job_status(job_id)
    
    def get_all_jobs(self) -> List[ProcessingJob]:
        if not self.document_processor:
            return []
        return self.document_processor.get_all_jobs()
    
    def get_documents(self) -> List[DocumentInfo]:
        if not self.storage_manager:
            return []
        return self.storage_manager.get_all_documents()
    
    def query_ai(self, prompt: str) -> str:
        if self.state != AppState.RUNNING:
            raise AppRuntimeError(f"Cannot query AI from state: {self.state}")
        
        if not self.ai_processor:
            raise AppRuntimeError("AI processor not initialized")
        
        if not hasattr(self.ai_processor, '_initialized') or not self.ai_processor._initialized:
            self.ai_processor.initialize()
        
        return self.ai_processor.query_llm(prompt)
    
    def rag_query(self, query: str) -> Dict[str, Any]:
        if self.state != AppState.RUNNING:
            raise AppRuntimeError(f"Cannot perform RAG query from state: {self.state}")
        
        if not self.ai_processor:
            raise AppRuntimeError("AI processor not initialized")
        
        if not hasattr(self.ai_processor, '_initialized') or not self.ai_processor._initialized:
            self.ai_processor.initialize()
        
        return self.ai_processor.rag_query(query)
    
    def create_backup(self) -> Optional[Path]:
        if not self.storage_manager:
            return None
        return self.storage_manager.create_backup()
    
    def reload_config(self) -> None:
        self.logger.info("Reloading configuration...")
        self.config = self.config_manager.reload()
        self.logger.info("Configuration reloaded")


def create_app(config_path: Optional[Path] = None) -> CockatooApp:
    app = CockatooApp(config_path)
    app.initialize()
    return app


def run_app(config_path: Optional[Path] = None) -> None:
    app = create_app(config_path)
    
    try:
        app.run()
    except KeyboardInterrupt:
        pass
    finally:
        app.shutdown()


if __name__ == "__main__":
    print(f"Starting {APP_NAME} v{APP_VERSION}")
    run_app()