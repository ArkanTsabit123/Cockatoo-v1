# src/ai_engine/llm_client.py

"""LLM client for interacting with various language model providers.

Provides unified interface for OpenAI, Anthropic, Ollama, and local models.
"""

import json
import time
import asyncio
import logging
import queue
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional, Union, Generator, AsyncGenerator
from datetime import datetime, timedelta
import requests
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try importing optional dependencies
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    openai = None
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    anthropic = None
    ANTHROPIC_AVAILABLE = False

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    TRANSFORMERS_AVAILABLE = False


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation, requests allowed
    OPEN = "open"      # Failure threshold exceeded, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before trying again
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self._lock = threading.RLock()
    
    def record_success(self) -> None:
        """Record a successful request."""
        with self._lock:
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                logger.info("Circuit breaker closed after successful test")
            elif self.state == CircuitBreakerState.CLOSED:
                self.failure_count = 0
    
    def record_failure(self) -> None:
        """Record a failed request."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state == CircuitBreakerState.CLOSED:
                if self.failure_count >= self.failure_threshold:
                    self.state = CircuitBreakerState.OPEN
                    logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
            
            elif self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.OPEN
                logger.warning("Circuit breaker re-opened after test failure")
    
    def allow_request(self) -> bool:
        """Check if request should be allowed."""
        with self._lock:
            if self.state == CircuitBreakerState.CLOSED:
                return True
            
            elif self.state == CircuitBreakerState.OPEN:
                # Check if recovery timeout has elapsed
                if self.last_failure_time and (time.time() - self.last_failure_time) > self.recovery_timeout:
                    self.state = CircuitBreakerState.HALF_OPEN
                    logger.info("Circuit breaker half-open, allowing test request")
                    return True
                return False
            
            elif self.state == CircuitBreakerState.HALF_OPEN:
                return True
            
            return False
    
    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state."""
        with self._lock:
            return {
                "state": self.state.value,
                "failure_count": self.failure_count,
                "failure_threshold": self.failure_threshold,
                "recovery_timeout": self.recovery_timeout,
                "last_failure_time": self.last_failure_time
            }
    
    def force_open(self) -> None:
        """Force circuit breaker to open state (for testing)."""
        with self._lock:
            self.state = CircuitBreakerState.OPEN
            self.last_failure_time = time.time()
    
    def force_closed(self) -> None:
        """Force circuit breaker to closed state (for testing)."""
        with self._lock:
            self.state = CircuitBreakerState.CLOSED
            self.failure_count = 0
            self.last_failure_time = None


class RateLimiter:
    """Rate limiter using token bucket algorithm."""
    
    def __init__(self, requests_per_second: float = 0):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_second: Maximum requests per second (0 = no limit)
        """
        self.requests_per_second = requests_per_second
        self.tokens = 0.0
        self.last_refill = time.time()
        self._lock = threading.RLock()
    
    def wait_if_needed(self) -> float:
        """
        Wait if rate limit exceeded.
        
        Returns:
            Wait time in seconds (0 if no wait)
        """
        if self.requests_per_second <= 0:
            return 0.0
        
        with self._lock:
            now = time.time()
            elapsed = now - self.last_refill
            
            # Refill tokens
            self.tokens += elapsed * self.requests_per_second
            if self.tokens > self.requests_per_second:
                self.tokens = self.requests_per_second
            self.last_refill = now
            
            if self.tokens >= 1.0:
                # Have enough tokens, consume one
                self.tokens -= 1.0
                return 0.0
            else:
                # Need to wait
                wait_time = (1.0 - self.tokens) / self.requests_per_second
                self.tokens = 0.0
                return wait_time


class RequestPriority(Enum):
    """Request priority levels."""
    HIGH = 0
    MEDIUM = 1
    LOW = 2


@dataclass
class QueuedRequest:
    """Request in the queue."""
    priority: RequestPriority
    timestamp: float
    func: callable
    args: tuple
    kwargs: dict
    future: Any


class RequestQueue:
    """Priority request queue with timeout."""
    
    def __init__(self, maxsize: int = 100, timeout: float = 30.0):
        """
        Initialize request queue.
        
        Args:
            maxsize: Maximum queue size
            timeout: Request timeout in seconds
        """
        self.maxsize = maxsize
        self.timeout = timeout
        self._queue = queue.PriorityQueue()
        self._lock = threading.RLock()
        self._worker_thread = None
        self._running = False
    
    def start(self):
        """Start queue worker."""
        with self._lock:
            if not self._running:
                self._running = True
                self._worker_thread = threading.Thread(target=self._worker, daemon=True)
                self._worker_thread.start()
    
    def stop(self):
        """Stop queue worker."""
        with self._lock:
            self._running = False
            # Clear queue
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
    
    def put(self, request: QueuedRequest) -> bool:
        """
        Add request to queue.
        
        Returns:
            True if added, False if queue full
        """
        with self._lock:
            if self._queue.qsize() >= self.maxsize:
                return False
            # Priority queue uses (priority, timestamp, id) as key
            # id() ensures uniqueness and prevents comparison of QueuedRequest objects
            self._queue.put(((request.priority.value, request.timestamp, id(request)), request))
            return True
    
    def _worker(self):
        """Queue worker thread."""
        while self._running:
            try:
                # Get request with timeout
                item = self._queue.get(timeout=1.0)
                if item is None:
                    continue
                
                # Unpack the tuple correctly
                if isinstance(item, tuple) and len(item) == 2:
                    _, request = item
                else:
                    logger.error(f"Invalid queue item: {item}")
                    continue
                
                # Check timeout
                if time.time() - request.timestamp > self.timeout:
                    if hasattr(request, 'future') and request.future and not request.future.done():
                        request.future.set_exception(
                            TimeoutError(f"Request timeout after {self.timeout}s")
                        )
                    continue
                
                # Execute request
                try:
                    result = request.func(*request.args, **request.kwargs)
                    if hasattr(request, 'future') and request.future and not request.future.done():
                        request.future.set_result(result)
                except Exception as e:
                    if hasattr(request, 'future') and request.future and not request.future.done():
                        request.future.set_exception(e)
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Queue worker error: {e}")
    
    def qsize(self) -> int:
        """Get queue size."""
        return self._queue.qsize()


@dataclass
class PerformanceMetrics:
    """Performance metrics collector."""
    request_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    token_usage: Dict[str, int] = field(default_factory=dict)
    request_timestamps: List[float] = field(default_factory=list)
    
    def record_request(self, success: bool, tokens: int = 0, latency_ms: float = 0):
        """Record a request."""
        self.request_count += 1
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
        
        self.total_tokens += tokens
        self.total_latency_ms += latency_ms
        self.request_timestamps.append(time.time())
        
        # Keep only last 100 timestamps
        if len(self.request_timestamps) > 100:
            self.request_timestamps = self.request_timestamps[-100:]
    
    def get_success_rate(self) -> float:
        """Get success rate."""
        if self.request_count == 0:
            return 1.0
        return self.success_count / self.request_count
    
    def get_average_latency_ms(self) -> float:
        """Get average latency."""
        if self.request_count == 0:
            return 0.0
        return self.total_latency_ms / self.request_count
    
    def get_requests_per_second(self) -> float:
        """Get requests per second (last 100 requests)."""
        if len(self.request_timestamps) < 2:
            return 0.0
        
        duration = self.request_timestamps[-1] - self.request_timestamps[0]
        if duration <= 0:
            return 0.0
        
        return (len(self.request_timestamps) - 1) / duration
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_count": self.request_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": self.get_success_rate(),
            "total_tokens": self.total_tokens,
            "average_latency_ms": self.get_average_latency_ms(),
            "requests_per_second": self.get_requests_per_second()
        }


@dataclass
class LLMResponse:
    """Response from LLM generation."""
    text: str
    model: str
    usage: Dict[str, int]
    finish_reason: str
    latency: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class LLMConfig:
    """Configuration for LLM client."""
    model_name: str
    provider: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 60
    max_retries: int = 3
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.95
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    rate_limit_rps: float = 0.0  # Requests per second (0 = no limit)
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    queue_maxsize: int = 100
    queue_timeout: float = 30.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class LLMModel:
    """Model information and metadata."""
    name: str
    provider: str
    context_length: int = 4096
    size: Optional[int] = None
    description: str = ""
    capabilities: List[str] = field(default_factory=lambda: ["chat", "completion"])
    quantization: Optional[str] = None
    ram_required: Optional[int] = None
    available: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LLMModel':
        """Create from dictionary."""
        return cls(**data)
    
    def get_context_length(self) -> int:
        """Get context length."""
        return self.context_length
    
    def is_available(self) -> bool:
        """Check if model is available."""
        return self.available
    
    def supports_capability(self, capability: str) -> bool:
        """Check if model supports a specific capability."""
        return capability in self.capabilities
    
    def get_estimated_memory_mb(self) -> Optional[float]:
        """Get estimated memory usage in MB."""
        if self.ram_required:
            return self.ram_required / (1024 * 1024)
        return None
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.name} ({self.provider})"


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    def __init__(self, config: LLMConfig):
        """Initialize LLM client.
        
        Args:
            config: LLM configuration
        """
        self.config = config
        self._client = None
        self._initialized = False
        
        # Rate limiting
        self.rate_limiter = RateLimiter(config.rate_limit_rps)
        
        # Circuit breaker
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=config.circuit_breaker_threshold,
            recovery_timeout=config.circuit_breaker_timeout
        )
        
        # Request queue
        self.request_queue = RequestQueue(
            maxsize=config.queue_maxsize,
            timeout=config.queue_timeout
        )
        self.request_queue.start()
        
        # Performance metrics
        self.metrics = PerformanceMetrics()
        
        logger.info(f"Initialized {self.__class__.__name__} with model: {config.model_name}")
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the client connection."""
        pass
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                temperature: Optional[float] = None, max_tokens: Optional[int] = None,
                stop_sequences: Optional[List[str]] = None,
                priority: RequestPriority = RequestPriority.MEDIUM) -> LLMResponse:
        """Generate text from prompt.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Override temperature
            max_tokens: Override max tokens
            stop_sequences: Stop sequences
            priority: Request priority
            
        Returns:
            LLMResponse object
        """
        pass
    
    @abstractmethod
    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None,
                       temperature: Optional[float] = None,
                       max_tokens: Optional[int] = None) -> Generator[str, None, None]:
        """Stream generated text from prompt.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Override temperature
            max_tokens: Override max tokens
            
        Yields:
            Text chunks
        """
        pass
    
    def chat(self, messages: List[Dict[str, str]], 
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None) -> LLMResponse:
        """Chat completion.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override temperature
            max_tokens: Override max tokens
            
        Returns:
            LLMResponse object
        """
        if not messages:
            raise ValueError("Conversation history cannot be empty")
        
        # Extract system prompt if present
        system_prompt = None
        chat_messages = []
        
        for msg in messages:
            if msg.get("role") == "system":
                system_prompt = msg.get("content")
            else:
                chat_messages.append(msg)
        
        # Use last user message as prompt
        user_messages = [m for m in chat_messages if m.get("role") == "user"]
        if not user_messages:
            raise ValueError("No user message found in conversation")
        
        prompt = user_messages[-1]["content"]
        
        return self.generate(prompt, system_prompt, temperature, max_tokens)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text.
        
        Args:
            text: Input text
            
        Returns:
            Token count
        """
        # Simple approximation: 1 token ~ 4 characters
        return len(text) // 4
    
    def validate_connection(self) -> bool:
        """Validate connection to LLM provider.
        
        Returns:
            True if connection is valid
        """
        try:
            response = self.generate("test", max_tokens=5)
            return response is not None
        except Exception as e:
            logger.error(f"Connection validation failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information.
        
        Returns:
            Model information dictionary
        """
        return {
            "model_name": self.config.model_name,
            "provider": self.config.provider,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return self.metrics.to_dict()
    
    def reset_stats(self) -> None:
        """Reset performance statistics."""
        self.metrics = PerformanceMetrics()
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        health = {
            "success": True,
            "status": "healthy",
            "timestamp": time.time(),
            "connected": self._initialized,
            "circuit_breaker": self.circuit_breaker.get_state(),
            "metrics": self.metrics.to_dict(),
            "queue_size": self.request_queue.qsize()
        }
        
        # Check circuit breaker
        if self.circuit_breaker.state == CircuitBreakerState.OPEN:
            health["status"] = "degraded"
            health["health_score"] = 0.3
        elif self.circuit_breaker.state == CircuitBreakerState.HALF_OPEN:
            health["status"] = "degraded"
            health["health_score"] = 0.6
        else:
            health["health_score"] = 1.0
        
        # Check queue
        if self.request_queue.qsize() > self.config.queue_maxsize * 0.8:
            health["status"] = "degraded"
            health["health_score"] = min(health["health_score"], 0.5)
        
        return health
    
    def _wait_for_rate_limit(self) -> None:
        """Wait for rate limit if needed."""
        wait_time = self.rate_limiter.wait_if_needed()
        if wait_time > 0:
            time.sleep(wait_time)
    
    def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker allows request."""
        return self.circuit_breaker.allow_request()
    
    def _enqueue_request(self, func, *args, **kwargs) -> Any:
        """Enqueue request with priority."""
        priority = kwargs.pop('priority', RequestPriority.MEDIUM)
        
        # Create future
        future = asyncio.Future() if asyncio.iscoroutinefunction(func) else None
        
        request = QueuedRequest(
            priority=priority,
            timestamp=time.time(),
            func=func,
            args=args,
            kwargs=kwargs,
            future=future
        )
        
        if not self.request_queue.put(request):
            raise queue.Full("Request queue is full")
        
        if future:
            return future
        return None


class OpenAIClient(LLMClient):
    """OpenAI API client."""
    
    def __init__(self, config: LLMConfig):
        """Initialize OpenAI client."""
        super().__init__(config)
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")
    
    def initialize(self) -> None:
        """Initialize OpenAI client."""
        try:
            self._client = openai.OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout
            )
            self._initialized = True
            logger.info("OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
    
    def _prepare_messages(self, prompt: str, system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
        """Prepare messages for OpenAI API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                temperature: Optional[float] = None, max_tokens: Optional[int] = None,
                stop_sequences: Optional[List[str]] = None,
                priority: RequestPriority = RequestPriority.MEDIUM) -> LLMResponse:
        """Generate text using OpenAI."""
        # Check circuit breaker
        if not self._check_circuit_breaker():
            self.metrics.record_request(False)
            return LLMResponse(
                text="Error: Circuit breaker is open",
                model=self.config.model_name,
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                finish_reason="error",
                latency=0
            )
        
        # Wait for rate limit
        self._wait_for_rate_limit()
        
        if not self._initialized:
            self.initialize()
        
        start_time = time.time()
        
        try:
            messages = self._prepare_messages(prompt, system_prompt)
            
            response = self._client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=temperature or self.config.temperature,
                max_tokens=max_tokens or self.config.max_tokens,
                stop=stop_sequences,
                top_p=self.config.top_p,
                frequency_penalty=self.config.frequency_penalty,
                presence_penalty=self.config.presence_penalty
            )
            
            latency = (time.time() - start_time) * 1000  # Convert to ms
            token_usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            # Record success
            self.circuit_breaker.record_success()
            self.metrics.record_request(
                success=True,
                tokens=response.usage.total_tokens,
                latency_ms=latency
            )
            
            return LLMResponse(
                text=response.choices[0].message.content,
                model=self.config.model_name,
                usage=token_usage,
                finish_reason=response.choices[0].finish_reason,
                latency=latency
            )
            
        except Exception as e:
            # Record failure
            self.circuit_breaker.record_failure()
            self.metrics.record_request(False)
            logger.error(f"OpenAI generation failed: {e}")
            raise
    
    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None,
                       temperature: Optional[float] = None,
                       max_tokens: Optional[int] = None) -> Generator[str, None, None]:
        """Stream text from OpenAI."""
        if not self._check_circuit_breaker():
            yield "Error: Circuit breaker is open"
            return
        
        self._wait_for_rate_limit()
        
        if not self._initialized:
            self.initialize()
        
        try:
            messages = self._prepare_messages(prompt, system_prompt)
            
            stream = self._client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=temperature or self.config.temperature,
                max_tokens=max_tokens or self.config.max_tokens,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"OpenAI streaming failed: {e}")
            raise


class AnthropicClient(LLMClient):
    """Anthropic Claude API client."""
    
    def __init__(self, config: LLMConfig):
        """Initialize Anthropic client."""
        super().__init__(config)
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic package not installed. Install with: pip install anthropic")
    
    def initialize(self) -> None:
        """Initialize Anthropic client."""
        try:
            self._client = anthropic.Anthropic(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout
            )
            self._initialized = True
            logger.info("Anthropic client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {e}")
            raise
    
    def _prepare_messages(self, prompt: str, system_prompt: Optional[str] = None) -> Dict:
        """Prepare messages for Anthropic API."""
        messages = [{"role": "user", "content": prompt}]
        return {
            "messages": messages,
            "system": system_prompt
        }
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                temperature: Optional[float] = None, max_tokens: Optional[int] = None,
                stop_sequences: Optional[List[str]] = None,
                priority: RequestPriority = RequestPriority.MEDIUM) -> LLMResponse:
        """Generate text using Anthropic."""
        if not self._check_circuit_breaker():
            self.metrics.record_request(False)
            return LLMResponse(
                text="Error: Circuit breaker is open",
                model=self.config.model_name,
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                finish_reason="error",
                latency=0
            )
        
        self._wait_for_rate_limit()
        
        if not self._initialized:
            self.initialize()
        
        start_time = time.time()
        
        try:
            prepared = self._prepare_messages(prompt, system_prompt)
            
            response = self._client.messages.create(
                model=self.config.model_name,
                messages=prepared["messages"],
                system=prepared["system"],
                temperature=temperature or self.config.temperature,
                max_tokens=max_tokens or self.config.max_tokens,
                stop_sequences=stop_sequences,
                top_p=self.config.top_p
            )
            
            latency = (time.time() - start_time) * 1000
            total_tokens = response.usage.input_tokens + response.usage.output_tokens
            
            self.circuit_breaker.record_success()
            self.metrics.record_request(
                success=True,
                tokens=total_tokens,
                latency_ms=latency
            )
            
            return LLMResponse(
                text=response.content[0].text,
                model=self.config.model_name,
                usage={
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": total_tokens
                },
                finish_reason=response.stop_reason,
                latency=latency
            )
            
        except Exception as e:
            self.circuit_breaker.record_failure()
            self.metrics.record_request(False)
            logger.error(f"Anthropic generation failed: {e}")
            raise
    
    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None,
                       temperature: Optional[float] = None,
                       max_tokens: Optional[int] = None) -> Generator[str, None, None]:
        """Stream text from Anthropic."""
        if not self._check_circuit_breaker():
            yield "Error: Circuit breaker is open"
            return
        
        self._wait_for_rate_limit()
        
        if not self._initialized:
            self.initialize()
        
        try:
            prepared = self._prepare_messages(prompt, system_prompt)
            
            with self._client.messages.stream(
                model=self.config.model_name,
                messages=prepared["messages"],
                system=prepared["system"],
                temperature=temperature or self.config.temperature,
                max_tokens=max_tokens or self.config.max_tokens
            ) as stream:
                for text in stream.text_stream:
                    yield text
                    
        except Exception as e:
            logger.error(f"Anthropic streaming failed: {e}")
            raise


class OllamaClient(LLMClient):
    """Ollama local LLM client."""
    
    def __init__(self, config: LLMConfig):
        """Initialize Ollama client."""
        super().__init__(config)
        self.base_url = config.base_url or "http://localhost:11434"
    
    def initialize(self) -> None:
        """Initialize Ollama client."""
        try:
            # Test connection
            response = requests.get(f"{self.base_url}/api/tags", timeout=self.config.timeout)
            if response.status_code == 200:
                self._initialized = True
                logger.info(f"Ollama client initialized at {self.base_url}")
            else:
                raise ConnectionError(f"Failed to connect to Ollama: {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama client: {e}")
            raise
    
    def list_models(self) -> List[str]:
        """List available models in Ollama."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=self.config.timeout)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [model["name"] for model in models]
            return []
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def pull_model(self, model_name: str) -> bool:
        """Pull a model in Ollama.
        
        Args:
            model_name: Name of model to pull
            
        Returns:
            True if successful
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                timeout=None  # No timeout for long pulls
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            return False
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                temperature: Optional[float] = None, max_tokens: Optional[int] = None,
                stop_sequences: Optional[List[str]] = None,
                priority: RequestPriority = RequestPriority.MEDIUM) -> LLMResponse:
        """Generate text using Ollama."""
        if not self._check_circuit_breaker():
            self.metrics.record_request(False)
            return LLMResponse(
                text="Error: Circuit breaker is open",
                model=self.config.model_name,
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                finish_reason="error",
                latency=0
            )
        
        self._wait_for_rate_limit()
        
        if not self._initialized:
            self.initialize()
        
        start_time = time.time()
        
        try:
            payload = {
                "model": self.config.model_name,
                "prompt": prompt,
                "system": system_prompt,
                "temperature": temperature or self.config.temperature,
                "max_tokens": max_tokens or self.config.max_tokens,
                "stop": stop_sequences,
                "top_p": self.config.top_p,
                "stream": False
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.config.timeout
            )
            
            if response.status_code != 200:
                raise Exception(f"Ollama API error: {response.status_code}")
            
            result = response.json()
            latency = (time.time() - start_time) * 1000
            total_tokens = result.get("prompt_eval_count", 0) + result.get("eval_count", 0)
            
            self.circuit_breaker.record_success()
            self.metrics.record_request(
                success=True,
                tokens=total_tokens,
                latency_ms=latency
            )
            
            return LLMResponse(
                text=result.get("response", ""),
                model=self.config.model_name,
                usage={
                    "prompt_tokens": result.get("prompt_eval_count", 0),
                    "completion_tokens": result.get("eval_count", 0),
                    "total_tokens": total_tokens
                },
                finish_reason=result.get("done_reason", "stop"),
                latency=latency
            )
            
        except Exception as e:
            self.circuit_breaker.record_failure()
            self.metrics.record_request(False)
            logger.error(f"Ollama generation failed: {e}")
            raise
    
    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None,
                       temperature: Optional[float] = None,
                       max_tokens: Optional[int] = None) -> Generator[str, None, None]:
        """Stream text from Ollama."""
        if not self._check_circuit_breaker():
            yield "Error: Circuit breaker is open"
            return
        
        self._wait_for_rate_limit()
        
        if not self._initialized:
            self.initialize()
        
        try:
            payload = {
                "model": self.config.model_name,
                "prompt": prompt,
                "system": system_prompt,
                "temperature": temperature or self.config.temperature,
                "max_tokens": max_tokens or self.config.max_tokens,
                "top_p": self.config.top_p,
                "stream": True
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=self.config.timeout
            )
            
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if chunk.get("response"):
                        yield chunk["response"]
                    if chunk.get("done"):
                        break
                    
        except Exception as e:
            logger.error(f"Ollama streaming failed: {e}")
            raise


class LocalLLMClient(LLMClient):
    """Local transformer-based LLM client."""
    
    def __init__(self, config: LLMConfig):
        """Initialize local LLM client."""
        super().__init__(config)
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers package not installed. Install with: pip install transformers torch")
        
        self.model_path = config.base_url or config.model_name
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch and torch.cuda.is_available() else "cpu"
    
    def load_model(self) -> bool:
        """Load model into memory.
        
        Returns:
            True if successful
        """
        try:
            logger.info(f"Loading model {self.model_path} on {self.device}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device
            )
            self._initialized = True
            logger.info("Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def unload_model(self) -> bool:
        """Unload model from memory.
        
        Returns:
            True if successful
        """
        try:
            if self.model:
                del self.model
                self.model = None
            if self.tokenizer:
                del self.tokenizer
                self.tokenizer = None
            if torch and self.device == "cuda":
                torch.cuda.empty_cache()
            self._initialized = False
            logger.info("Model unloaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to unload model: {e}")
            return False
    
    def initialize(self) -> None:
        """Initialize local model."""
        self.load_model()
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                temperature: Optional[float] = None, max_tokens: Optional[int] = None,
                stop_sequences: Optional[List[str]] = None,
                priority: RequestPriority = RequestPriority.MEDIUM) -> LLMResponse:
        """Generate text using local model."""
        if not self._check_circuit_breaker():
            self.metrics.record_request(False)
            return LLMResponse(
                text="Error: Circuit breaker is open",
                model=self.config.model_name,
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                finish_reason="error",
                latency=0
            )
        
        self._wait_for_rate_limit()
        
        if not self._initialized:
            self.initialize()
        
        start_time = time.time()
        
        try:
            # Prepare full prompt with system message
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens or self.config.max_tokens,
                    temperature=temperature or self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove input prompt from generated text
            response_text = generated_text[len(self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)):].strip()
            
            latency = (time.time() - start_time) * 1000
            prompt_tokens = inputs['input_ids'].shape[1]
            completion_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
            total_tokens = prompt_tokens + completion_tokens
            
            self.circuit_breaker.record_success()
            self.metrics.record_request(
                success=True,
                tokens=total_tokens,
                latency_ms=latency
            )
            
            return LLMResponse(
                text=response_text,
                model=self.config.model_name,
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                },
                finish_reason="stop",
                latency=latency
            )
            
        except Exception as e:
            self.circuit_breaker.record_failure()
            self.metrics.record_request(False)
            logger.error(f"Local generation failed: {e}")
            raise
    
    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None,
                       temperature: Optional[float] = None,
                       max_tokens: Optional[int] = None) -> Generator[str, None, None]:
        """Stream text from local model (simulated)."""
        response = self.generate(prompt, system_prompt, temperature, max_tokens)
        # Simulate streaming by yielding words
        for word in response.text.split():
            yield word + " "
            time.sleep(0.01)
    
    def get_device(self) -> str:
        """Get current device.
        
        Returns:
            Device string
        """
        return self.device
    
    def chat(self, messages: List[Dict[str, str]], 
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None) -> LLMResponse:
        """Chat completion for local model."""
        if not messages:
            raise ValueError("Conversation history cannot be empty")
        
        # Format conversation for local model
        formatted = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted += f"{role}: {content}\n"
        
        formatted += "assistant: "
        
        return self.generate(formatted, temperature=temperature, max_tokens=max_tokens)


# Factory function
def get_llm_client(config: Union[LLMConfig, Dict]) -> LLMClient:
    """Get LLM client instance based on configuration.
    
    Args:
        config: LLM configuration
        
    Returns:
        LLMClient instance
    """
    if isinstance(config, dict):
        config = LLMConfig(**config)
    
    providers = {
        "openai": OpenAIClient,
        "anthropic": AnthropicClient,
        "ollama": OllamaClient,
        "local": LocalLLMClient
    }
    
    client_class = providers.get(config.provider)
    if not client_class:
        raise ValueError(f"Unsupported provider: {config.provider}")
    
    client = client_class(config)
    return client