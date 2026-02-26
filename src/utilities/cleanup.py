# src/utilities/cleanup.py

"""Resource cleanup utilities for temporary files, caches, and exit handlers."""

import os
import sys
import shutil
import tempfile
import atexit
import signal
import threading
import time
from pathlib import Path
from typing import List, Dict, Optional, Union, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta


class CleanupPolicy(Enum):
    """Defines when cleanup tasks should be executed."""
    ON_EXIT = "on_exit"
    ON_DEMAND = "on_demand"
    PERIODIC = "periodic"
    NEVER = "never"


@dataclass
class CleanupTask:
    """Represents a cleanup task with execution policy."""
    name: str
    cleanup_func: Callable[[], None]
    policy: CleanupPolicy
    created_at: datetime = field(default_factory=datetime.now)
    last_run: Optional[datetime] = None
    interval: Optional[timedelta] = None
    priority: int = 0


class CleanupManager:
    """Manages cleanup tasks, temporary files, and directories."""
    
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
        self.tasks: List[CleanupTask] = []
        self.temp_dirs: List[Path] = []
        self.temp_files: List[Path] = []
        self.cleanup_lock = threading.Lock()
        self.running = True
        
        atexit.register(self.cleanup_all)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        self._start_periodic_cleanup()
    
    def _signal_handler(self, signum, frame):
        print(f"Received signal {signum}, cleaning up...")
        self.cleanup_all()
        sys.exit(0)
    
    def _start_periodic_cleanup(self):
        def periodic_cleanup():
            while self.running:
                try:
                    time.sleep(60)
                    self.run_periodic_cleanup()
                except Exception as e:
                    print(f"Periodic cleanup error: {e}")
        
        thread = threading.Thread(target=periodic_cleanup, daemon=True)
        thread.start()
    
    def register_task(self, task: CleanupTask) -> None:
        """Register a cleanup task."""
        with self.cleanup_lock:
            self.tasks.append(task)
            self.tasks.sort(key=lambda t: t.priority, reverse=True)
    
    def register_temp_dir(self, path: Union[str, Path]) -> Path:
        """Register a temporary directory for cleanup."""
        path = Path(path)
        with self.cleanup_lock:
            if path not in self.temp_dirs:
                self.temp_dirs.append(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def register_temp_file(self, path: Union[str, Path]) -> Path:
        """Register a temporary file for cleanup."""
        path = Path(path)
        with self.cleanup_lock:
            if path not in self.temp_files:
                self.temp_files.append(path)
        return path
    
    def create_temp_dir(self, prefix: str = "temp_") -> Path:
        """Create and register a temporary directory."""
        temp_dir = Path(tempfile.mkdtemp(prefix=prefix))
        return self.register_temp_dir(temp_dir)
    
    def create_temp_file(self, suffix: str = "", prefix: str = "temp_") -> Path:
        """Create and register a temporary file."""
        fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
        os.close(fd)
        return self.register_temp_file(Path(path))
    
    def cleanup_temp_files(self, older_than: Optional[timedelta] = None) -> None:
        """Clean up registered temporary files."""
        with self.cleanup_lock:
            files_to_remove = []
            for file_path in self.temp_files:
                try:
                    if file_path.exists():
                        if older_than:
                            mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                            if datetime.now() - mtime < older_than:
                                continue
                        file_path.unlink()
                        files_to_remove.append(file_path)
                except Exception as e:
                    print(f"Error cleaning up file {file_path}: {e}")
            
            for file_path in files_to_remove:
                self.temp_files.remove(file_path)
    
    def cleanup_temp_dirs(self, older_than: Optional[timedelta] = None) -> None:
        """Clean up registered temporary directories."""
        with self.cleanup_lock:
            dirs_to_remove = []
            for dir_path in self.temp_dirs:
                try:
                    if dir_path.exists():
                        if older_than:
                            mtime = datetime.fromtimestamp(dir_path.stat().st_mtime)
                            if datetime.now() - mtime < older_than:
                                continue
                        shutil.rmtree(dir_path)
                        dirs_to_remove.append(dir_path)
                except Exception as e:
                    print(f"Error cleaning up directory {dir_path}: {e}")
            
            for dir_path in dirs_to_remove:
                self.temp_dirs.remove(dir_path)
    
    def run_cleanup_tasks(self, policy: Optional[CleanupPolicy] = None) -> None:
        """Run registered cleanup tasks."""
        with self.cleanup_lock:
            for task in self.tasks:
                if policy is None or task.policy == policy:
                    try:
                        task.cleanup_func()
                        task.last_run = datetime.now()
                    except Exception as e:
                        print(f"Error running cleanup task '{task.name}': {e}")
    
    def run_periodic_cleanup(self) -> None:
        """Run periodic cleanup tasks."""
        current_time = datetime.now()
        with self.cleanup_lock:
            for task in self.tasks:
                if task.policy == CleanupPolicy.PERIODIC and task.interval:
                    if (task.last_run is None or 
                        current_time - task.last_run >= task.interval):
                        try:
                            task.cleanup_func()
                            task.last_run = current_time
                        except Exception as e:
                            print(f"Error in periodic cleanup '{task.name}': {e}")
    
    def cleanup_all(self) -> None:
        """Run all cleanup tasks and remove temporary files/directories."""
        print("Running cleanup...")
        self.run_cleanup_tasks()
        self.cleanup_temp_files()
        self.cleanup_temp_dirs()
        with self.cleanup_lock:
            self.temp_files.clear()
            self.temp_dirs.clear()
        print("Cleanup completed")
    
    def shutdown(self) -> None:
        """Shutdown the cleanup manager."""
        self.running = False
        self.cleanup_all()


class TempFileCleaner:
    """Context manager for automatic temporary file cleanup."""
    
    def __init__(self, manager: CleanupManager, suffix: str = "", prefix: str = "temp_"):
        self.manager = manager
        self.suffix = suffix
        self.prefix = prefix
        self.temp_file: Optional[Path] = None
    
    def __enter__(self) -> Path:
        self.temp_file = self.manager.create_temp_file(
            suffix=self.suffix,
            prefix=self.prefix
        )
        return self.temp_file
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.temp_file and self.temp_file.exists():
            try:
                self.temp_file.unlink()
            except Exception as e:
                print(f"Error cleaning up temp file: {e}")


class CacheCleaner:
    """Context manager for cleaning old cache files."""
    
    def __init__(self, cache_dir: Union[str, Path], max_age: timedelta = timedelta(days=7)):
        self.cache_dir = Path(cache_dir)
        self.max_age = max_age
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
    
    def cleanup(self) -> None:
        """Remove cache files older than max_age."""
        if not self.cache_dir.exists():
            return
        
        current_time = datetime.now()
        for item in self.cache_dir.iterdir():
            try:
                if item.is_file():
                    mtime = datetime.fromtimestamp(item.stat().st_mtime)
                    if current_time - mtime > self.max_age:
                        item.unlink()
                elif item.is_dir():
                    cleaner = CacheCleaner(item, self.max_age)
                    cleaner.cleanup()
                    if not any(item.iterdir()):
                        item.rmdir()
            except Exception as e:
                print(f"Error cleaning cache item {item}: {e}")


_cleanup_manager = CleanupManager()


def cleanup_on_exit(func: Callable) -> Callable:
    """Decorator to register a function for cleanup on exit."""
    task = CleanupTask(
        name=func.__name__,
        cleanup_func=func,
        policy=CleanupPolicy.ON_EXIT
    )
    _cleanup_manager.register_task(task)
    return func


def get_cleanup_manager() -> CleanupManager:
    """Get the global cleanup manager instance."""
    return _cleanup_manager


def register_temp_dir(path: Union[str, Path]) -> Path:
    """Register a temporary directory for cleanup."""
    return _cleanup_manager.register_temp_dir(path)


def register_temp_file(path: Union[str, Path]) -> Path:
    """Register a temporary file for cleanup."""
    return _cleanup_manager.register_temp_file(path)


def create_temp_dir(prefix: str = "temp_") -> Path:
    """Create and register a temporary directory."""
    return _cleanup_manager.create_temp_dir(prefix)


def create_temp_file(suffix: str = "", prefix: str = "temp_") -> Path:
    """Create and register a temporary file."""
    return _cleanup_manager.create_temp_file(suffix, prefix)