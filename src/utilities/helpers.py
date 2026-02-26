# src/utilities/helpers.py

"""Common helper functions for list operations, string manipulation, and timing."""

import random
import string
import hashlib
import time
import uuid
import unicodedata
import re
from typing import Any, Dict, List, Optional, Union, TypeVar
from datetime import timedelta


T = TypeVar('T')


def chunk_list(lst: List[T], chunk_size: int) -> List[List[T]]:
    """Split a list into chunks of specified size."""
    if chunk_size <= 0:
        return [lst]
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def flatten_list(nested_list: List[Any], depth: int = -1) -> List[Any]:
    """Flatten a nested list up to specified depth."""
    result = []
    
    def _flatten(item: Any, current_depth: int):
        if isinstance(item, list) and (depth == -1 or current_depth < depth):
            for subitem in item:
                _flatten(subitem, current_depth + 1)
        else:
            result.append(item)
    
    _flatten(nested_list, 0)
    return result


def ensure_list(value: Any) -> List[Any]:
    """Ensure the value is a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def merge_dicts(dict1: Dict, dict2: Dict, deep: bool = True) -> Dict:
    """Merge two dictionaries, optionally performing deep merge."""
    result = dict1.copy()
    
    for key, value in dict2.items():
        if deep and key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value, deep)
        else:
            result[key] = value
    
    return result


def safe_get(obj: Any, *keys: Any, default: Any = None) -> Any:
    """Safely get a value from nested structures."""
    current = obj
    
    for key in keys:
        if current is None:
            return default
        
        if isinstance(current, dict):
            current = current.get(key)
        elif isinstance(current, (list, tuple)) and isinstance(key, (int, slice)):
            try:
                current = current[key]
            except (IndexError, TypeError):
                return default
        elif isinstance(key, str):
            if hasattr(current, key):
                current = getattr(current, key)
            else:
                return default
        else:
            return default
        
        if current is None:
            return default
    
    return current


def safe_divide(a: Union[int, float], b: Union[int, float], 
                default: float = 0.0) -> float:
    """Safely divide two numbers, returning default on division by zero."""
    try:
        return a / b
    except ZeroDivisionError:
        return default


def parse_bool(value: Any) -> bool:
    """Parse various value types to boolean."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.lower() in ('true', 'yes', 'y', '1', 'on')
    return bool(value)


def format_timedelta(delta: timedelta, format: str = "auto") -> str:
    """Format a timedelta to human-readable string."""
    total_seconds = int(delta.total_seconds())
    days = total_seconds // 86400
    hours = (total_seconds % 86400) // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    
    if format == "iso":
        return str(delta)
    
    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if seconds > 0 or not parts:
        parts.append(f"{seconds}s")
    
    if format == "short":
        return " ".join(parts)
    
    long_parts = []
    if days > 0:
        long_parts.append(f"{days} day{'s' if days != 1 else ''}")
    if hours > 0:
        long_parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes > 0:
        long_parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if seconds > 0 or not long_parts:
        long_parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")
    
    if len(long_parts) == 1:
        return long_parts[0]
    elif len(long_parts) == 2:
        return f"{long_parts[0]} and {long_parts[1]}"
    else:
        return f"{', '.join(long_parts[:-1])}, and {long_parts[-1]}"


def slugify(text: str, separator: str = "-") -> str:
    """Convert text to URL-friendly slug."""
    text = text.lower()
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ASCII', 'ignore').decode('ASCII')
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', separator, text)
    return text.strip(separator)


def truncate_string(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate string to specified length with suffix."""
    if not text or len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def generate_id(prefix: str = "", length: int = 8) -> str:
    """Generate a random ID with optional prefix."""
    random_part = ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))
    return f"{prefix}_{random_part}" if prefix else random_part


def generate_uuid() -> str:
    """Generate a UUID string."""
    return str(uuid.uuid4())


def generate_hash(content: str, algorithm: str = "sha256") -> str:
    """Generate hash of content using specified algorithm."""
    if algorithm == "md5":
        return hashlib.md5(content.encode()).hexdigest()
    elif algorithm == "sha1":
        return hashlib.sha1(content.encode()).hexdigest()
    else:
        return hashlib.sha256(content.encode()).hexdigest()


class Timer:
    """Context manager for timing code execution."""
    
    def __init__(self, auto_start: bool = True):
        self.start_time: Optional[float] = None
        self.elapsed: float = 0.0
        self.running = False
        
        if auto_start:
            self.start()
    
    def start(self) -> None:
        """Start the timer."""
        self.start_time = time.time()
        self.running = True
    
    def stop(self) -> float:
        """Stop the timer and return elapsed time."""
        if self.running and self.start_time is not None:
            self.elapsed += time.time() - self.start_time
            self.running = False
        return self.elapsed
    
    def reset(self) -> None:
        """Reset the timer."""
        self.start_time = None
        self.elapsed = 0.0
        self.running = False
    
    def get_elapsed(self) -> float:
        """Get current elapsed time without stopping."""
        if self.running and self.start_time is not None:
            return self.elapsed + (time.time() - self.start_time)
        return self.elapsed
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()
    
    def __str__(self) -> str:
        return format_timedelta(timedelta(seconds=self.get_elapsed()))