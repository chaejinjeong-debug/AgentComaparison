"""Tests for Observability module (OB-001~OB-003, TS-004)."""

import asyncio
import time

import pytest

from agent_engine.config import ObservabilityConfig
from agent_engine.observability import (
    LoggingManager,
    MetricsManager,
    TracingManager,
    logged_tool,
    metered,
    traced,
)


class TestTracingManager:
    """Tests for TracingManager (OB-001)."""

    def test_tracing_manager_initialization(self, tracing_manager: TracingManager):
        """Test TracingManager initialization."""
        assert tracing_manager is not None
        assert tracing_manager.config is not None

    def test_tracing_disabled(self):
        """Test TracingManager with tracing disabled."""
        config = ObservabilityConfig(tracing_enabled=False)
        manager = TracingManager(config=config)
        manager.setup()

        # Should not be enabled
        assert not manager.is_enabled

    def test_tracing_span_context(self, tracing_manager: TracingManager):
        """Test span context manager."""
        tracing_manager.setup()

        with tracing_manager.span("test-span", {"key": "value"}) as span:
            # Span may be None if tracing not fully initialized
            pass  # Just verify no errors

    def test_get_instance_singleton(self):
        """Test TracingManager singleton pattern."""
        # Reset singleton
        TracingManager._instance = None

        manager1 = TracingManager.get_instance()
        manager2 = TracingManager.get_instance()

        assert manager1 is manager2


class TestLoggingManager:
    """Tests for LoggingManager (OB-002)."""

    def test_logging_manager_initialization(self, logging_manager: LoggingManager):
        """Test LoggingManager initialization."""
        assert logging_manager is not None

    def test_logging_setup(self, logging_manager: LoggingManager):
        """Test logging setup."""
        logging_manager.setup()
        assert logging_manager.is_enabled

    def test_logging_disabled(self):
        """Test LoggingManager with logging disabled."""
        config = ObservabilityConfig(logging_enabled=False)
        manager = LoggingManager(config=config)
        manager.setup()

        # Should not be enabled
        assert not manager.is_enabled

    def test_get_logger(self, logging_manager: LoggingManager):
        """Test getting a logger instance."""
        logging_manager.setup()
        logger = logging_manager.get_logger("test")
        assert logger is not None

    def test_bind_unbind_context(self, logging_manager: LoggingManager):
        """Test context binding and unbinding."""
        logging_manager.setup()

        # Should not raise any errors
        logging_manager.bind_context(user_id="test-user", session_id="test-session")
        logging_manager.unbind_context("user_id")
        logging_manager.clear_context()


class TestMetricsManager:
    """Tests for MetricsManager (OB-003)."""

    def test_metrics_manager_initialization(self, metrics_manager: MetricsManager):
        """Test MetricsManager initialization."""
        assert metrics_manager is not None
        assert metrics_manager.is_enabled

    def test_record_request(self, metrics_manager: MetricsManager):
        """Test request metric recording."""
        initial_stats = metrics_manager.get_stats()
        initial_requests = initial_stats["requests"]

        metrics_manager.record_request(method="query", status="success")

        stats = metrics_manager.get_stats()
        assert stats["requests"] == initial_requests + 1

    def test_record_error(self, metrics_manager: MetricsManager):
        """Test error metric recording."""
        initial_stats = metrics_manager.get_stats()
        initial_errors = initial_stats["errors"]

        metrics_manager.record_error(error_type="TestError", method="query")

        stats = metrics_manager.get_stats()
        assert stats["errors"] == initial_errors + 1

    def test_record_tokens(self, metrics_manager: MetricsManager):
        """Test token usage recording."""
        initial_stats = metrics_manager.get_stats()
        initial_input = initial_stats["tokens"]["input"]
        initial_output = initial_stats["tokens"]["output"]

        metrics_manager.record_tokens(input_tokens=100, output_tokens=50)

        stats = metrics_manager.get_stats()
        assert stats["tokens"]["input"] == initial_input + 100
        assert stats["tokens"]["output"] == initial_output + 50

    def test_record_latency(self, metrics_manager: MetricsManager):
        """Test latency recording."""
        metrics_manager.record_latency(150.5, method="query")

        stats = metrics_manager.get_stats()
        assert "latency" in stats
        assert len(stats["latency"]) > 0

    def test_update_active_sessions(self, metrics_manager: MetricsManager):
        """Test active sessions gauge."""
        initial_stats = metrics_manager.get_stats()
        initial_sessions = initial_stats["active_sessions"]

        metrics_manager.update_active_sessions(1)
        metrics_manager.update_active_sessions(1)

        stats = metrics_manager.get_stats()
        assert stats["active_sessions"] == initial_sessions + 2

        metrics_manager.update_active_sessions(-1)
        stats = metrics_manager.get_stats()
        assert stats["active_sessions"] == initial_sessions + 1

    def test_error_rate_calculation(self, metrics_manager: MetricsManager):
        """Test error rate calculation."""
        # Reset by creating new manager
        config = ObservabilityConfig(metrics_enabled=True)
        manager = MetricsManager(config=config)
        manager.setup()

        # Record 10 requests, 2 errors
        for _ in range(10):
            manager.record_request()
        for _ in range(2):
            manager.record_error("TestError")

        stats = manager.get_stats()
        assert stats["error_rate"] == pytest.approx(0.2, rel=0.01)

    def test_latency_percentiles(self, metrics_manager: MetricsManager):
        """Test latency percentile calculation."""
        # Reset
        config = ObservabilityConfig(metrics_enabled=True)
        manager = MetricsManager(config=config)
        manager.setup()

        # Record various latencies
        latencies = [100, 150, 200, 250, 300, 350, 400, 450, 500, 1000]
        for lat in latencies:
            manager.record_latency(lat)

        stats = manager.get_stats()
        assert "p50" in stats["latency"]
        assert "p99" in stats["latency"]
        assert stats["latency"]["min"] == 100
        assert stats["latency"]["max"] == 1000


class TestDecorators:
    """Tests for observability decorators (TS-004)."""

    def test_logged_tool_sync(self):
        """TS-004: Test @logged_tool decorator with sync function."""

        @logged_tool
        def sample_tool(x: int, y: int) -> int:
            return x + y

        result = sample_tool(2, 3)
        assert result == 5

    @pytest.mark.asyncio
    async def test_logged_tool_async(self):
        """TS-004: Test @logged_tool decorator with async function."""

        @logged_tool
        async def async_sample_tool(x: int) -> int:
            await asyncio.sleep(0.01)
            return x * 2

        result = await async_sample_tool(5)
        assert result == 10

    def test_logged_tool_with_options(self):
        """TS-004: Test @logged_tool with options."""

        @logged_tool(log_args=False, log_result=False)
        def private_tool(secret: str) -> str:
            return f"processed: {secret}"

        result = private_tool("my-secret")
        assert result == "processed: my-secret"

    def test_logged_tool_error_handling(self):
        """TS-004: Test @logged_tool error logging."""

        @logged_tool
        def failing_tool() -> None:
            raise ValueError("Intentional error")

        with pytest.raises(ValueError, match="Intentional error"):
            failing_tool()

    def test_metered_decorator_sync(self):
        """Test @metered decorator with sync function."""
        # Create metrics manager
        config = ObservabilityConfig(metrics_enabled=True)
        manager = MetricsManager(config=config)
        manager.setup()
        MetricsManager._instance = manager

        @metered(name="test_operation")
        def metered_function(x: int) -> int:
            time.sleep(0.01)  # Small delay
            return x * 2

        result = metered_function(5)
        assert result == 10

    @pytest.mark.asyncio
    async def test_metered_decorator_async(self):
        """Test @metered decorator with async function."""

        @metered(name="async_test_operation")
        async def async_metered_function(x: int) -> int:
            await asyncio.sleep(0.01)
            return x * 3

        result = await async_metered_function(4)
        assert result == 12

    def test_traced_decorator_sync(self):
        """Test @traced decorator with sync function."""
        # Reset tracing manager
        TracingManager._instance = None
        config = ObservabilityConfig(tracing_enabled=False)
        manager = TracingManager.get_instance(config=config)

        @traced(name="test_span")
        def traced_function(x: int) -> int:
            return x + 1

        result = traced_function(10)
        assert result == 11

    @pytest.mark.asyncio
    async def test_traced_decorator_async(self):
        """Test @traced decorator with async function."""

        @traced(name="async_test_span", attributes={"key": "value"})
        async def async_traced_function(x: int) -> int:
            await asyncio.sleep(0.01)
            return x - 1

        result = await async_traced_function(10)
        assert result == 9

    def test_traced_decorator_error_handling(self):
        """Test @traced decorator error handling."""

        @traced(name="error_span")
        def failing_traced_function() -> None:
            raise RuntimeError("Test error")

        with pytest.raises(RuntimeError, match="Test error"):
            failing_traced_function()


class TestTimerContext:
    """Tests for Timer context manager."""

    def test_timer_basic(self):
        """Test Timer context manager."""
        from agent_engine.observability.metrics import Timer

        with Timer() as timer:
            time.sleep(0.1)

        assert timer.elapsed_ms >= 100
        assert timer.elapsed_ms < 200  # Should be reasonably close

    def test_timer_with_metrics(self, metrics_manager: MetricsManager):
        """Test Timer with metrics recording."""
        from agent_engine.observability.metrics import Timer

        with Timer(metrics_manager=metrics_manager, method="test") as timer:
            time.sleep(0.05)

        assert timer.elapsed_ms >= 50
