# src/utilities/retry.py

"""Retry mechanism with configurable strategies and async support."""

import time
import asyncio
import random
from functools import wraps
from typing import Type, Union, Callable, Optional, List, Any
from enum import Enum
from dataclasses import dataclass


class RetryStrategy(Enum):
    """Available retry delay strategies."""
    FIXED = "fixed"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    RANDOM = "random"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    retry_on_exceptions: Optional[List[Type[Exception]]] = None
    retry_on_result: Optional[Callable[[Any], bool]] = None
    jitter: bool = True
    jitter_factor: float = 0.1


class RetryError(Exception):
    """Exception raised when max retries are exceeded."""
    
    def __init__(self, message: str, last_exception: Optional[Exception] = None):
        super().__init__(message)
        self.last_exception = last_exception


def exponential_backoff(attempt: int, base_delay: float, 
                       max_delay: float, multiplier: float = 2.0) -> float:
    """Calculate delay using exponential backoff."""
    delay = base_delay * (multiplier ** (attempt - 1))
    return min(delay, max_delay)


def fixed_delay(attempt: int, base_delay: float, **kwargs) -> float:
    """Calculate fixed delay."""
    return base_delay


def linear_delay(attempt: int, base_delay: float, **kwargs) -> float:
    """Calculate linear delay."""
    return base_delay * attempt


def random_delay(attempt: int, base_delay: float, max_delay: float, **kwargs) -> float:
    """Calculate random delay within range."""
    return random.uniform(base_delay, max_delay)


def add_jitter(delay: float, jitter_factor: float = 0.1) -> float:
    """Add random jitter to delay."""
    jitter = delay * jitter_factor * random.uniform(-1, 1)
    return max(0, delay + jitter)


def _calculate_delay(attempt: int, config: RetryConfig) -> float:
    """Calculate delay based on retry strategy."""
    if config.strategy == RetryStrategy.FIXED:
        delay = fixed_delay(attempt, config.delay)
    elif config.strategy == RetryStrategy.EXPONENTIAL:
        delay = exponential_backoff(attempt, config.delay, config.max_delay, config.backoff_multiplier)
    elif config.strategy == RetryStrategy.LINEAR:
        delay = linear_delay(attempt, config.delay)
    elif config.strategy == RetryStrategy.RANDOM:
        delay = random_delay(attempt, config.delay, config.max_delay)
    else:
        delay = exponential_backoff(attempt, config.delay, config.max_delay)
    
    if config.jitter:
        delay = add_jitter(delay, config.jitter_factor)
    
    return delay


def retry(config: Optional[RetryConfig] = None):
    """Decorator for retrying synchronous functions."""
    if config is None:
        config = RetryConfig()
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(1, config.max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    
                    if config.retry_on_result and config.retry_on_result(result):
                        if attempt < config.max_retries:
                            delay = _calculate_delay(attempt, config)
                            time.sleep(delay)
                            continue
                    
                    return result
                    
                except Exception as e:
                    last_exception = e
                    
                    if config.retry_on_exceptions:
                        should_retry = any(
                            isinstance(e, exc_type) 
                            for exc_type in config.retry_on_exceptions
                        )
                        if not should_retry:
                            raise
                    
                    if attempt == config.max_retries:
                        raise RetryError(
                            f"Max retries ({config.max_retries}) exceeded",
                            last_exception
                        ) from e
                    
                    delay = _calculate_delay(attempt, config)
                    time.sleep(delay)
            
            raise RetryError("Unexpected retry loop exit", last_exception)
        
        return wrapper
    return decorator


def async_retry(config: Optional[RetryConfig] = None):
    """Decorator for retrying asynchronous functions."""
    if config is None:
        config = RetryConfig()
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(1, config.max_retries + 1):
                try:
                    result = await func(*args, **kwargs)
                    
                    if config.retry_on_result and config.retry_on_result(result):
                        if attempt < config.max_retries:
                            delay = _calculate_delay(attempt, config)
                            await asyncio.sleep(delay)
                            continue
                        else:
                            return result
                    
                    return result
                    
                except Exception as e:
                    last_exception = e
                    
                    if config.retry_on_exceptions:
                        should_retry = any(
                            isinstance(e, exc_type) 
                            for exc_type in config.retry_on_exceptions
                        )
                        if not should_retry:
                            raise
                    
                    if attempt == config.max_retries:
                        raise RetryError(
                            f"Max retries ({config.max_retries}) exceeded",
                            last_exception
                        ) from e
                    
                    delay = _calculate_delay(attempt, config)
                    await asyncio.sleep(delay)
            
            raise RetryError("Unexpected retry loop exit", last_exception)
        
        return wrapper
    return decorator