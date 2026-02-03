"""
cockatoo_v1 Helper Utilities
Miscellaneous utility functions for common operations
"""

import os
import sys
import json
import hashlib
import uuid
import random
import string
import platform
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timedelta
import inspect
import threading
from contextlib import contextmanager
import time


def create_crossplatform_directories() -> Dict[str, Path]:
    """
    Create platform-appropriate directories for cockatoo_v1 application data.
    
    Returns:
        Dictionary mapping directory names to their Path objects
    """
    system = platform.system().lower()
    
    if system == "windows":
        base = Path(os.environ.get('APPDATA', Path.home() / 'AppData' / 'Roaming'))
        app_dir = base / "cockatoo_v1"
    elif system == "darwin":
        app_dir = Path.home() / "Library" / "Application Support" / "cockatoo_v1"
    elif system == "linux":
        xdg_data_home = os.environ.get('XDG_DATA_HOME', '')
        if xdg_data_home:
            app_dir = Path(xdg_data_home) / "cockatoo_v1"
        else:
            app_dir = Path.home() / ".local" / "share" / "cockatoo_v1"
    else:
        app_dir = Path.home() / ".cockatoo_v1"
    
    directories = {
        'data': app_dir,
        'models': app_dir / "models",
        'documents': app_dir / "documents",
        'database': app_dir / "database",
        'logs': app_dir / "logs",
        'exports': app_dir / "exports",
        'config': app_dir / "config",
        'cache': app_dir / "cache",
        'temp': app_dir / "temp",
        'backups': app_dir / "backups"
    }
    
    for name, path in directories.items():
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Directory creation failed for {name}: {e}")
    
    return directories


def get_crossplatform_paths() -> Dict[str, str]:
    """
    Retrieve all cross-platform directory paths as strings.
    
    Returns:
        Dictionary mapping directory names to their string paths
    """
    dirs = create_crossplatform_directories()
    return {key: str(value) for key, value in dirs.items()}


def generate_id(prefix: str = "", length: int = 16) -> str:
    """
    Generate a unique identifier with timestamp and random component.
    
    Args:
        prefix: Optional prefix for the ID
        length: Length of random component
        
    Returns:
        Generated unique ID string
    """
    random_part = ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    
    if prefix:
        return f"{prefix}_{timestamp}_{random_part}"
    return f"{timestamp}_{random_part}"


def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """
    Safely parse JSON string with error handling.
    
    Args:
        json_str: JSON string to parse
        default: Value to return if parsing fails
        
    Returns:
        Parsed JSON object or default value
    """
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default


def safe_json_dumps(data: Any, default: Optional[Callable] = None, indent: int = 2) -> str:
    """
    Safely convert data to JSON string with error handling.
    
    Args:
        data: Data to serialize
        default: Optional function to handle non-serializable objects
        indent: JSON indentation level
        
    Returns:
        JSON string representation of data
    """
    try:
        return json.dumps(data, default=default, indent=indent, ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps({"error": "Could not serialize data"}, ensure_ascii=False)


def calculate_file_hash(file_path: Union[str, Path], algorithm: str = "sha256") -> str:
    """
    Calculate cryptographic hash of a file.
    
    Args:
        file_path: Path to the file
        algorithm: Hash algorithm to use
        
    Returns:
        Hexadecimal hash string
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    hash_func = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()


def calculate_text_hash(text: str, algorithm: str = "sha256") -> str:
    """
    Calculate cryptographic hash of text.
    
    Args:
        text: Text to hash
        algorithm: Hash algorithm to use
        
    Returns:
        Hexadecimal hash string
    """
    hash_func = hashlib.new(algorithm)
    hash_func.update(text.encode('utf-8'))
    return hash_func.hexdigest()


def get_file_size(file_path: Union[str, Path], human_readable: bool = False) -> Union[int, str]:
    """
    Get file size in bytes or human-readable format.
    
    Args:
        file_path: Path to the file
        human_readable: Whether to return human-readable format
        
    Returns:
        File size as integer bytes or formatted string
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    size_bytes = file_path.stat().st_size
    
    if not human_readable:
        return size_bytes
    
    return format_bytes(size_bytes)


def format_bytes(bytes_size: int) -> str:
    """
    Format byte count to human-readable string.
    
    Args:
        bytes_size: Size in bytes
        
    Returns:
        Human-readable size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} PB"


def ensure_directory(directory: Union[str, Path]) -> Path:
    """
    Ensure directory exists, creating it if necessary.
    
    Args:
        directory: Directory path
        
    Returns:
        Path object of the directory
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def clean_filename(filename: str, max_length: int = 255) -> str:
    """
    Sanitize filename by removing invalid characters.
    
    Args:
        filename: Original filename
        max_length: Maximum allowed filename length
        
    Returns:
        Cleaned filename
    """
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    filename = filename.strip().strip('.')
    
    if len(filename) > max_length:
        name, ext = os.path.splitext(filename)
        name = name[:max_length - len(ext)]
        filename = name + ext
    
    return filename


def split_list(input_list: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split list into chunks of specified size.
    
    Args:
        input_list: List to split
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]


def flatten_list(nested_list: List[Any]) -> List[Any]:
    """
    Flatten a nested list structure.
    
    Args:
        nested_list: Nested list to flatten
        
    Returns:
        Flattened list
    """
    result = []
    for item in nested_list:
        if isinstance(item, list):
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result


def remove_duplicates(input_list: List[Any], key: Optional[Callable] = None) -> List[Any]:
    """
    Remove duplicates from list while preserving order.
    
    Args:
        input_list: List with potential duplicates
        key: Optional function to extract comparison key
        
    Returns:
        List without duplicates
    """
    seen = set()
    result = []
    
    for item in input_list:
        item_key = key(item) if key else item
        if item_key not in seen:
            seen.add(item_key)
            result.append(item)
    
    return result


@contextmanager
def timing_context(name: str = "operation"):
    """
    Context manager for measuring operation duration.
    
    Args:
        name: Name of the operation for logging
    """
    start_time = datetime.now()
    try:
        yield
    finally:
        elapsed = datetime.now() - start_time
        print(f"{name} took {elapsed.total_seconds():.2f} seconds")


class Singleton(type):
    """
    Singleton metaclass for ensuring single instance of classes.
    """
    _instances = {}
    _lock = threading.Lock()
    
    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class LRUCache:
    """
    Least Recently Used cache implementation.
    """
    
    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.cache = {}
        self.order = []
    
    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieve value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        if key not in self.cache:
            return None
        
        self.order.remove(key)
        self.order.append(key)
        return self.cache[key]
    
    def set(self, key: Any, value: Any):
        """
        Store value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        if key in self.cache:
            self.order.remove(key)
        elif len(self.cache) >= self.capacity:
            lru_key = self.order.pop(0)
            del self.cache[lru_key]
        
        self.cache[key] = value
        self.order.append(key)
    
    def clear(self):
        """Clear all cache entries."""
        self.cache.clear()
        self.order.clear()
    
    def __len__(self) -> int:
        return len(self.cache)


class RateLimiter:
    """
    Rate limiter for controlling operation frequency.
    """
    
    def __init__(self, max_requests: int, time_window: float):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
        self.lock = threading.Lock()
    
    def acquire(self) -> bool:
        """
        Attempt to acquire permission for operation.
        
        Returns:
            True if operation can proceed, False if rate limited
        """
        with self.lock:
            now = time.time()
            
            self.requests = [t for t in self.requests if now - t < self.time_window]
            
            if len(self.requests) >= self.max_requests:
                return False
            
            self.requests.append(now)
            return True
    
    def wait(self):
        """Wait until operation can be performed."""
        while not self.acquire():
            time.sleep(0.1)


def get_caller_info(depth: int = 2) -> Dict[str, Any]:
    """
    Get information about the calling function.
    
    Args:
        depth: Call stack depth to examine
        
    Returns:
        Dictionary with caller information
    """
    frame = inspect.currentframe()
    for _ in range(depth):
        if frame:
            frame = frame.f_back
    
    if frame:
        info = inspect.getframeinfo(frame)
        return {
            'filename': info.filename,
            'function': info.function,
            'line': info.lineno,
            'code_context': info.code_context
        }
    
    return {}


def validate_email(email: str) -> bool:
    """
    Validate email address format.
    
    Args:
        email: Email address to validate
        
    Returns:
        True if email format is valid
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_url(url: str) -> bool:
    """
    Validate URL format.
    
    Args:
        url: URL to validate
        
    Returns:
        True if URL format is valid
    """
    pattern = r'^https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b[-a-zA-Z0-9()@:%_\+.~#?&//=]*$'
    return bool(re.match(pattern, url))


def normalize_text(text: str) -> str:
    """
    Normalize text by removing extra whitespace and normalizing characters.
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text
    """
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = ' '.join(text.split())
    return text.strip()


def get_platform_data_dir() -> Path:
    """
    Get platform-specific data directory for cockatoo_v1.
    
    Returns:
        Path to platform-appropriate data directory
    """
    system = platform.system().lower()
    
    if system == "windows":
        appdata = os.environ.get('APPDATA', '')
        if appdata:
            return Path(appdata) / "cockatoo_v1"
        return Path.home() / "AppData" / "Roaming" / "cockatoo_v1"
    
    elif system == "darwin":
        return Path.home() / "Library" / "Application Support" / "cockatoo_v1"
    
    elif system == "linux":
        xdg_data_home = os.environ.get('XDG_DATA_HOME', '')
        if xdg_data_home:
            return Path(xdg_data_home) / "cockatoo_v1"
        return Path.home() / ".local" / "share" / "cockatoo_v1"
    
    else:
        return Path.home() / ".cockatoo_v1"


def setup_crossplatform_directories() -> Dict[str, Path]:
    """
    Setup all cross-platform directories and return paths.
    
    Returns:
        Dictionary mapping directory names to Path objects
    """
    data_dir = get_platform_data_dir()
    
    directories = {
        'data': data_dir,
        'models': data_dir / "models",
        'documents': data_dir / "documents",
        'database': data_dir / "database",
        'logs': data_dir / "logs",
        'exports': data_dir / "exports",
        'config': data_dir / "config",
        'cache': data_dir / "cache",
        'temp': data_dir / "temp"
    }
    
    for name, path in directories.items():
        try:
            path.mkdir(parents=True, exist_ok=True)
            readme = path / "README.txt"
            if not readme.exists():
                with open(readme, 'w') as f:
                    f.write(f"cockatoo_v1 {name} directory\nCreated: {datetime.now()}\n")
        except Exception as e:
            print(f"Directory creation failed for {path}: {e}")
    
    return directories


def ensure_data_directories() -> bool:
    """
    Ensure all data directories exist.
    
    Returns:
        True if all directories exist or were created
    """
    try:
        dirs = setup_crossplatform_directories()
        return all(path.exists() for path in dirs.values())
    except Exception:
        return False