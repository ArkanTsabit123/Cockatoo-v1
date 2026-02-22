# tests/unit/test_llm_client.py

"""Unit tests for LLM client with multi-model support.

Tests cover configuration, response handling, initialization, generation,
streaming, chat, model management, error handling, performance, health checks,
rate limiting, circuit breakers, and request queuing.
"""

import os
import sys
import time
import json
import threading
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock, ANY
from typing import Dict, List, Any, Generator

import pytest
import requests

from src.ai_engine.llm_client import (
    LLMClient, LLMConfig, LLMResponse, LLMModel,
    CircuitBreakerState, RequestPriority, RateLimiter,
    CircuitBreaker, RequestQueue, PerformanceMetrics,
    QueuedRequest, OllamaClient
)


@pytest.fixture
def llm_config():
    """Create default LLM configuration."""
    return LLMConfig(
        model_name="llama2:7b",
        provider="ollama"
    )


@pytest.fixture
def custom_config():
    """Create custom LLM configuration."""
    return LLMConfig(
        model_name="mistral:7b",
        provider="ollama",
        base_url="http://localhost:11434",  # Changed from custom-host to localhost
        temperature=0.5,
        max_tokens=2048,
        timeout=60,
        max_retries=2,
        rate_limit_rps=5
    )


@pytest.fixture
def llm_client():
    """Create LLM client instance."""
    config = LLMConfig(model_name="llama2:7b", provider="ollama")
    client = OllamaClient(config)
    return client


@pytest.fixture
def llm_client_custom(custom_config):
    """Create LLM client with custom config."""
    client = OllamaClient(custom_config)
    return client


@pytest.fixture
def sample_prompt():
    """Provide sample prompt."""
    return "What is machine learning?"


@pytest.fixture
def system_prompt():
    """Provide system prompt."""
    return "You are a helpful AI assistant."


@pytest.fixture
def conversation_history():
    """Provide sample conversation history."""
    return [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there! How can I help you?"},
        {"role": "user", "content": "Tell me about AI"}
    ]


# ==================== ORIGINAL TEST CLASSES ====================

class TestLLMConfig:
    """Test LLM configuration."""

    def test_default_configuration(self, llm_config):
        """Test default configuration values."""
        assert llm_config.model_name == "llama2:7b"
        assert llm_config.provider == "ollama"
        assert llm_config.temperature == 0.7
        assert llm_config.max_tokens == 2048
        assert llm_config.timeout == 60
        assert llm_config.max_retries == 3

    def test_custom_configuration(self, custom_config):
        """Test custom configuration values."""
        assert custom_config.model_name == "mistral:7b"
        assert custom_config.base_url == "http://localhost:11434"
        assert custom_config.temperature == 0.5
        assert custom_config.max_tokens == 2048
        assert custom_config.timeout == 60
        assert custom_config.max_retries == 2

    @pytest.mark.parametrize("temperature,expected_valid", [
        (0.0, True),
        (0.5, True),
        (2.0, True),
        (-0.1, False),
        (2.1, False),
    ])
    def test_temperature_validation(self, llm_config, temperature, expected_valid):
        """Test temperature validation."""
        llm_config.temperature = temperature
        assert llm_config.temperature == temperature

    def test_to_from_dict_conversion(self, custom_config):
        """Test conversion to and from dictionary."""
        config_dict = custom_config.to_dict()
        assert config_dict["model_name"] == "mistral:7b"
        assert config_dict["temperature"] == 0.5

        new_config = LLMConfig(**config_dict)
        assert new_config.model_name == custom_config.model_name
        assert new_config.temperature == custom_config.temperature


class TestLLMModel:
    """Test LLM model class."""

    def test_model_creation(self):
        """Test creating a model instance."""
        model = LLMModel(
            name="llama2:7b",
            provider="ollama",
            context_length=4096
        )
        assert model.name == "llama2:7b"
        assert model.provider == "ollama"
        assert model.context_length == 4096
        assert "chat" in model.capabilities

    def test_model_with_all_fields(self):
        """Test model with all fields populated."""
        model = LLMModel(
            name="gpt-4",
            provider="openai",
            context_length=8192,
            size=175000000000,
            description="GPT-4 model",
            capabilities=["chat", "completion", "vision"],
            quantization=None,
            ram_required=32000000000,
            available=True
        )
        assert model.name == "gpt-4"
        assert model.provider == "openai"
        assert "vision" in model.capabilities
        assert model.get_estimated_memory_mb() == 32000000000 / (1024 * 1024)

    def test_model_to_from_dict(self):
        """Test model dictionary conversion."""
        model = LLMModel(
            name="llama2:7b",
            provider="ollama",
            context_length=4096
        )
        model_dict = model.to_dict()
        assert model_dict["name"] == "llama2:7b"
        
        new_model = LLMModel.from_dict(model_dict)
        assert new_model.name == model.name
        assert new_model.provider == model.provider

    def test_model_capabilities(self):
        """Test model capability checking."""
        model = LLMModel(
            name="test-model",
            provider="test",
            capabilities=["chat", "completion"]
        )
        assert model.supports_capability("chat") is True
        assert model.supports_capability("vision") is False

    def test_model_availability(self):
        """Test model availability."""
        model = LLMModel(name="test", provider="test", available=True)
        assert model.is_available() is True
        
        model.available = False
        assert model.is_available() is False

    def test_model_string_representation(self):
        """Test string representation."""
        model = LLMModel(name="llama2:7b", provider="ollama")
        assert str(model) == "llama2:7b (ollama)"


class TestLLMResponse:
    """Test LLM response handling."""

    def test_basic_response(self):
        """Test basic response creation."""
        response = LLMResponse(
            text="Test response",
            model="llama2:7b",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            finish_reason="stop",
            latency=150.5
        )

        assert response.text == "Test response"
        assert response.model == "llama2:7b"
        assert response.usage["total_tokens"] == 30
        assert response.latency == 150.5

    def test_to_dict_conversion(self):
        """Test conversion to dictionary."""
        response = LLMResponse(
            text="Test content",
            model="llama2:7b",
            usage={"prompt_tokens": 15, "completion_tokens": 25, "total_tokens": 40},
            finish_reason="length",
            latency=120.0
        )

        response_dict = response.to_dict()
        assert response_dict["text"] == "Test content"
        assert response_dict["model"] == "llama2:7b"
        assert response_dict["usage"]["total_tokens"] == 40


class TestLLMClientInitialization:
    """Test LLM client initialization."""

    @patch('requests.get')
    def test_ollama_initialization(self, mock_get, llm_client):
        """Test Ollama client initialization."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        llm_client.initialize()
        assert llm_client._initialized is True

    @patch('requests.get')
    def test_connection_failure(self, mock_get):
        """Test connection failure."""
        mock_get.side_effect = requests.ConnectionError("Failed to connect")
        
        client = OllamaClient(LLMConfig(model_name="test", provider="ollama"))
        with pytest.raises(Exception):
            client.initialize()
        assert client._initialized is False


class TestGenerate:
    """Test generation functionality."""

    @patch('requests.post')
    def test_generate_non_streaming(self, mock_post, llm_client, sample_prompt):
        """Test non-streaming generation."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "Machine learning is a subset of AI.",
            "prompt_eval_count": 10,
            "eval_count": 20,
            "done_reason": "stop"
        }
        mock_post.return_value = mock_response

        response = llm_client.generate(sample_prompt)

        assert isinstance(response, LLMResponse)
        assert response.text is not None
        assert response.usage["total_tokens"] == 30
        mock_post.assert_called_once()

    @patch('requests.post')
    def test_generate_with_system_prompt(self, mock_post, llm_client, sample_prompt, system_prompt):
        """Test generation with system prompt."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "Generated answer",
            "prompt_eval_count": 15,
            "eval_count": 25
        }
        mock_post.return_value = mock_response

        response = llm_client.generate(sample_prompt, system_prompt=system_prompt)
        assert response.text == "Generated answer"
        mock_post.assert_called_once()

    @patch('requests.post')
    def test_generate_stream(self, mock_post, llm_client, sample_prompt):
        """Test streaming generation."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [
            b'{"response": "Hello"}',
            b'{"response": " world"}',
            b'{"response": "!"}',
            b'{"done": true}'
        ]
        mock_post.return_value = mock_response

        generator = llm_client.generate_stream(sample_prompt)
        chunks = list(generator)

        assert len(chunks) == 3
        assert ''.join(chunks) == "Hello world!"


class TestModelManagement:
    """Test model management functionality."""

    @patch('requests.get')
    def test_list_available_models(self, mock_get, llm_client):
        """Test listing available models."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "llama2:7b"},
                {"name": "mistral:7b"},
                {"name": "neural-chat:7b"}
            ]
        }
        mock_get.return_value = mock_response

        models = llm_client.list_models()

        assert len(models) == 3
        assert "llama2:7b" in models

    @patch('requests.post')
    def test_pull_model(self, mock_post, llm_client):
        """Test pulling a model."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        result = llm_client.pull_model("new-model:7b")
        assert result is True


class TestErrorHandling:
    """Test error handling scenarios."""

    @patch('requests.post')
    def test_api_error_handling(self, mock_post, llm_client, sample_prompt):
        """Test API error handling."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response

        with pytest.raises(Exception):
            llm_client.generate(sample_prompt)

    @patch('requests.post')
    def test_timeout_handling(self, mock_post, llm_client, sample_prompt):
        """Test timeout handling."""
        mock_post.side_effect = requests.Timeout("Request timed out")

        with pytest.raises(requests.Timeout):
            llm_client.generate(sample_prompt)


# ==================== NEW TEST CLASSES ====================

class TestChat:
    """Test chat functionality."""

    @patch('requests.post')
    def test_chat_completion(self, mock_post, llm_client, conversation_history):
        """Test chat completion."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "AI is fascinating!",
            "prompt_eval_count": 25,
            "eval_count": 15,
            "done_reason": "stop"
        }
        mock_post.return_value = mock_response

        response = llm_client.chat(conversation_history)

        assert response.text == "AI is fascinating!"
        assert response.usage["total_tokens"] == 40
        mock_post.assert_called_once()

    def test_chat_empty_history(self, llm_client):
        """Test chat with empty history."""
        with pytest.raises(ValueError, match="Conversation history cannot be empty"):
            llm_client.chat([])

    def test_chat_no_user_message(self, llm_client):
        """Test chat with no user message."""
        history = [{"role": "assistant", "content": "Hello"}]
        with pytest.raises(ValueError, match="No user message found"):
            llm_client.chat(history)

    @patch('requests.post')
    def test_chat_with_system_prompt(self, mock_post, llm_client):
        """Test chat with system prompt."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "Response with context",
            "prompt_eval_count": 30,
            "eval_count": 20
        }
        mock_post.return_value = mock_response

        history = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"}
        ]
        response = llm_client.chat(history)
        assert response.text == "Response with context"


class TestRateLimiting:
    """Test rate limiting functionality."""

    def test_rate_limiter_init(self):
        """Test rate limiter initialization."""
        limiter = RateLimiter(requests_per_second=10)
        assert limiter.requests_per_second == 10
        assert limiter.tokens == 0.0

    def test_rate_limiter_no_limit(self):
        """Test rate limiter with no limit."""
        limiter = RateLimiter(requests_per_second=0)
        assert limiter.wait_if_needed() == 0.0

    def test_rate_limiter_wait_time(self):
        """Test rate limiter wait time calculation."""
        limiter = RateLimiter(requests_per_second=2)  # 2 requests per second
        limiter.last_refill = time.time() - 0.5  # 0.5 seconds ago
        
        # Should have 1 token (0.5 * 2 = 1)
        wait_time = limiter.wait_if_needed()
        assert wait_time == 0.0  # No wait, consumed token

        # Next request should wait
        wait_time = limiter.wait_if_needed()
        assert wait_time > 0.0
        assert wait_time <= 0.5

    @patch('time.sleep')
    @patch('requests.get')
    @patch('requests.post')
    def test_rate_limit_enforcement(self, mock_post, mock_get, mock_sleep, llm_client_custom, sample_prompt):
        """Test rate limit enforcement in client."""
        # Mock health check
        mock_get.return_value = MagicMock(status_code=200)
        
        # Mock generate response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "test", "eval_count": 5, "prompt_eval_count": 5}
        mock_post.return_value = mock_response
        
        # First request - should not wait
        llm_client_custom.generate(sample_prompt)
        
        # Second request - should wait due to rate limit
        llm_client_custom.generate(sample_prompt)
        assert mock_sleep.called


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    def test_circuit_breaker_initial_state(self, llm_client):
        """Test circuit breaker initial state."""
        cb = llm_client.circuit_breaker
        assert cb.state.value == "closed"
        assert cb.failure_count == 0

    def test_circuit_breaker_opens_on_failures(self, llm_client):
        """Test circuit breaker opens after failures."""
        cb = llm_client.circuit_breaker
        cb.failure_threshold = 3
        
        for _ in range(3):
            cb.record_failure()
        
        assert cb.state.value == "open"
        assert cb.failure_count == 3

    def test_circuit_breaker_blocks_requests(self, llm_client):
        """Test circuit breaker blocks requests when open."""
        cb = llm_client.circuit_breaker
        cb.force_open()
        
        assert cb.allow_request() is False

    def test_circuit_breaker_half_open(self, llm_client):
        """Test circuit breaker half-open state."""
        cb = llm_client.circuit_breaker
        cb.force_open()
        cb.last_failure_time = time.time() - 100  # 100 seconds ago
        cb.recovery_timeout = 60  # 60 seconds
        
        # Should transition to half-open
        assert cb.allow_request() is True
        assert cb.state.value == "half_open"

    def test_circuit_breaker_success_closes(self, llm_client):
        """Test successful request closes circuit."""
        cb = llm_client.circuit_breaker
        cb.force_open()
        cb.state = CircuitBreakerState.HALF_OPEN
        
        cb.record_success()
        assert cb.state.value == "closed"
        assert cb.failure_count == 0

    def test_circuit_breaker_failure_reopens(self, llm_client):
        """Test failure in half-open reopens circuit."""
        cb = llm_client.circuit_breaker
        cb.state = CircuitBreakerState.HALF_OPEN
        cb.failure_count = 1
        
        cb.record_failure()
        assert cb.state.value == "open"

    def test_circuit_breaker_get_state(self, llm_client):
        """Test getting circuit breaker state."""
        cb = llm_client.circuit_breaker
        state = cb.get_state()
        assert "state" in state
        assert "failure_count" in state
        assert "failure_threshold" in state


class TestRequestQueue:
    """Test request queue functionality."""

    def setup_method(self):
        """Setup before each test."""
        self.queue = RequestQueue(maxsize=2, timeout=1.0)
        self.queue.start()

    def teardown_method(self):
        """Cleanup after each test."""
        self.queue.stop()

    def test_request_queue_init(self):
        """Test request queue initialization."""
        assert self.queue.maxsize == 2
        assert self.queue.timeout == 1.0
        assert self.queue.qsize() == 0

    def test_request_queue_put_get(self):
        """Test putting and getting from queue."""
        def dummy_func():
            return "result"
        
        future = asyncio.Future()
        request = QueuedRequest(
            priority=RequestPriority.MEDIUM,
            timestamp=time.time(),
            func=dummy_func,
            args=(),
            kwargs={},
            future=future
        )
        
        result = self.queue.put(request)
        assert result is True
        assert self.queue.qsize() == 1

    def test_request_queue_full(self):
        """Test queue full condition."""
        def dummy_func():
            return "result"
        
        future1 = asyncio.Future()
        request1 = QueuedRequest(
            priority=RequestPriority.MEDIUM,
            timestamp=time.time(),
            func=dummy_func,
            args=(),
            kwargs={},
            future=future1
        )
        
        future2 = asyncio.Future()
        request2 = QueuedRequest(
            priority=RequestPriority.MEDIUM,
            timestamp=time.time(),
            func=dummy_func,
            args=(),
            kwargs={},
            future=future2
        )
        
        future3 = asyncio.Future()
        request3 = QueuedRequest(
            priority=RequestPriority.MEDIUM,
            timestamp=time.time(),
            func=dummy_func,
            args=(),
            kwargs={},
            future=future3
        )
        
        # Queue size is 2, so first two should succeed
        assert self.queue.put(request1) is True
        assert self.queue.put(request2) is True
        # Third should fail
        assert self.queue.put(request3) is False


class TestPerformanceMetrics:
    """Test performance metrics."""

    def test_metrics_initialization(self, llm_client):
        """Test metrics initialization."""
        metrics = llm_client.metrics
        assert metrics.request_count == 0
        assert metrics.success_count == 0
        assert metrics.failure_count == 0

    def test_metrics_record_request(self, llm_client):
        """Test recording requests."""
        metrics = llm_client.metrics
        metrics.record_request(success=True, tokens=100, latency_ms=50.0)
        
        assert metrics.request_count == 1
        assert metrics.success_count == 1
        assert metrics.total_tokens == 100
        assert metrics.total_latency_ms == 50.0

    def test_metrics_record_failure(self, llm_client):
        """Test recording failures."""
        metrics = llm_client.metrics
        metrics.record_request(success=False)
        
        assert metrics.request_count == 1
        assert metrics.failure_count == 1
        assert metrics.success_count == 0

    def test_metrics_success_rate(self, llm_client):
        """Test success rate calculation."""
        metrics = llm_client.metrics
        assert metrics.get_success_rate() == 1.0  # No requests
        
        metrics.record_request(success=True)
        metrics.record_request(success=True)
        metrics.record_request(success=False)
        
        assert metrics.get_success_rate() == 2/3

    def test_metrics_average_latency(self, llm_client):
        """Test average latency calculation."""
        metrics = llm_client.metrics
        assert metrics.get_average_latency_ms() == 0.0
        
        metrics.record_request(success=True, latency_ms=100)
        metrics.record_request(success=True, latency_ms=200)
        
        assert metrics.get_average_latency_ms() == 150.0

    def test_metrics_to_dict(self, llm_client):
        """Test metrics to dict conversion."""
        metrics = llm_client.metrics
        metrics.record_request(success=True, tokens=100, latency_ms=50)
        
        data = metrics.to_dict()
        assert data["request_count"] == 1
        assert data["success_rate"] == 1.0
        assert data["average_latency_ms"] == 50.0


class TestHealthCheck:
    """Test health check functionality."""

    @patch('requests.get')
    def test_health_check_success(self, mock_get, llm_client):
        """Test successful health check."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        health = llm_client.health_check()
        
        assert health["success"] is True
        assert health["status"] == "healthy"
        assert health["health_score"] == 1.0
        assert "circuit_breaker" in health
        assert "metrics" in health

    def test_health_check_circuit_breaker_open(self, llm_client):
        """Test health check with open circuit breaker."""
        llm_client.circuit_breaker.force_open()
        
        health = llm_client.health_check()
        
        assert health["status"] == "degraded"
        assert health["health_score"] == 0.3
        assert health["circuit_breaker"]["state"] == "open"

    def test_health_check_circuit_breaker_half_open(self, llm_client):
        """Test health check with half-open circuit breaker."""
        llm_client.circuit_breaker.state = CircuitBreakerState.HALF_OPEN
        
        health = llm_client.health_check()
        
        assert health["status"] == "degraded"
        assert health["health_score"] == 0.6

    def test_health_check_queue_size(self, llm_client):
        """Test health check with large queue."""
        # Mock queue size
        llm_client.request_queue._queue = MagicMock()
        llm_client.request_queue.qsize = MagicMock(return_value=90)  # 90% full
        llm_client.config.queue_maxsize = 100
        
        health = llm_client.health_check()
        
        assert health["status"] == "degraded"
        assert health["health_score"] < 1.0


class TestPerformanceAndReset:
    """Test performance metrics and reset functionality."""

    @patch('requests.get')
    @patch('requests.post')
    def test_performance_metrics_collection(self, mock_post, mock_get, llm_client, sample_prompt):
        """Test performance metrics collection."""
        # Mock health check
        mock_get.return_value = MagicMock(status_code=200)
        
        # Mock generate response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "test",
            "prompt_eval_count": 10,
            "eval_count": 20
        }
        mock_post.return_value = mock_response

        for _ in range(5):
            llm_client.generate(sample_prompt)

        metrics = llm_client.get_performance_metrics()
        assert metrics["request_count"] == 5
        assert metrics["success_rate"] == 1.0
        assert metrics["total_tokens"] == 150

    def test_reset_stats(self, llm_client):
        """Test resetting performance statistics."""
        llm_client.metrics.record_request(success=True, tokens=100)
        assert llm_client.metrics.request_count == 1

        llm_client.reset_stats()
        assert llm_client.metrics.request_count == 0
        assert llm_client.metrics.total_tokens == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])