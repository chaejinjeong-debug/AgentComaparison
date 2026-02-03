"""Observability decorators for the Agent Engine platform.

Implements TS-004: Tool execution logging.
"""

import functools
import time
from typing import Any, Callable, ParamSpec, TypeVar

import structlog

from agent_engine.observability.logging import log_tool_execution
from agent_engine.observability.metrics import MetricsManager
from agent_engine.observability.tracing import TracingManager

logger = structlog.get_logger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


def traced(
    name: str | None = None,
    attributes: dict[str, Any] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to trace function execution.

    OB-001: Cloud Trace integration for function tracing.

    Args:
        name: Span name (defaults to function name)
        attributes: Additional span attributes

    Returns:
        Decorated function
    """
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        span_name = name or func.__name__

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            tracing = TracingManager.get_instance()

            if not tracing.is_enabled:
                return func(*args, **kwargs)

            with tracing.span(span_name, attributes) as span:
                try:
                    result = func(*args, **kwargs)
                    if span is not None:
                        span.set_attribute("success", True)
                    return result
                except Exception as e:
                    if span is not None:
                        TracingManager.record_exception(span, e)
                    raise

        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            tracing = TracingManager.get_instance()

            if not tracing.is_enabled:
                return await func(*args, **kwargs)  # type: ignore

            with tracing.span(span_name, attributes) as span:
                try:
                    result = await func(*args, **kwargs)  # type: ignore
                    if span is not None:
                        span.set_attribute("success", True)
                    return result
                except Exception as e:
                    if span is not None:
                        TracingManager.record_exception(span, e)
                    raise

        # Return appropriate wrapper based on function type
        if asyncio_iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper

    return decorator


def metered(
    name: str | None = None,
    record_latency: bool = True,
    record_errors: bool = True,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to record metrics for function execution.

    OB-003: Metrics collection for function execution.

    Args:
        name: Metric name (defaults to function name)
        record_latency: Whether to record latency
        record_errors: Whether to record errors

    Returns:
        Decorated function
    """
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        metric_name = name or func.__name__

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            metrics = MetricsManager.get_instance()

            if not metrics.is_enabled:
                return func(*args, **kwargs)

            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                metrics.record_request(method=metric_name, status="success")
                return result
            except Exception as e:
                if record_errors:
                    metrics.record_error(
                        error_type=type(e).__name__,
                        method=metric_name,
                    )
                raise
            finally:
                if record_latency:
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    metrics.record_latency(elapsed_ms, method=metric_name)

        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            metrics = MetricsManager.get_instance()

            if not metrics.is_enabled:
                return await func(*args, **kwargs)  # type: ignore

            start_time = time.perf_counter()
            try:
                result = await func(*args, **kwargs)  # type: ignore
                metrics.record_request(method=metric_name, status="success")
                return result
            except Exception as e:
                if record_errors:
                    metrics.record_error(
                        error_type=type(e).__name__,
                        method=metric_name,
                    )
                raise
            finally:
                if record_latency:
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    metrics.record_latency(elapsed_ms, method=metric_name)

        if asyncio_iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper

    return decorator


def logged_tool(
    func: Callable[P, R] | None = None,
    *,
    log_args: bool = True,
    log_result: bool = True,
    truncate_at: int = 200,
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to log tool execution.

    TS-004: Tool execution logging.

    Can be used with or without arguments:
        @logged_tool
        def my_tool(...): ...

        @logged_tool(log_result=False)
        def my_tool(...): ...

    Args:
        func: Function to decorate (when used without parentheses)
        log_args: Whether to log function arguments
        log_result: Whether to log function result
        truncate_at: Maximum length for logged values

    Returns:
        Decorated function
    """
    def truncate(value: Any) -> str:
        """Truncate value for logging."""
        s = str(value)
        if len(s) > truncate_at:
            return s[:truncate_at] + "..."
        return s

    def decorator(f: Callable[P, R]) -> Callable[P, R]:
        tool_name = f.__name__

        @functools.wraps(f)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start_time = time.perf_counter()
            log = structlog.get_logger("tool")

            # Log start
            log_data: dict[str, Any] = {"tool": tool_name}
            if log_args:
                log_data["args"] = truncate(args) if args else None
                log_data["kwargs"] = truncate(kwargs) if kwargs else None

            log.debug("Tool execution started", **log_data)

            try:
                result = f(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - start_time) * 1000

                # Log success
                log_data["success"] = True
                log_data["execution_time_ms"] = round(elapsed_ms, 2)
                if log_result:
                    log_data["result"] = truncate(result)

                log.info("Tool executed", **log_data)
                log_tool_execution(
                    tool_name=tool_name,
                    success=True,
                    execution_time_ms=elapsed_ms,
                )

                return result

            except Exception as e:
                elapsed_ms = (time.perf_counter() - start_time) * 1000

                # Log failure
                log.error(
                    "Tool execution failed",
                    tool=tool_name,
                    error=str(e),
                    error_type=type(e).__name__,
                    execution_time_ms=round(elapsed_ms, 2),
                )
                log_tool_execution(
                    tool_name=tool_name,
                    success=False,
                    execution_time_ms=elapsed_ms,
                    error=str(e),
                )

                raise

        @functools.wraps(f)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start_time = time.perf_counter()
            log = structlog.get_logger("tool")

            # Log start
            log_data: dict[str, Any] = {"tool": tool_name}
            if log_args:
                log_data["args"] = truncate(args) if args else None
                log_data["kwargs"] = truncate(kwargs) if kwargs else None

            log.debug("Tool execution started", **log_data)

            try:
                result = await f(*args, **kwargs)  # type: ignore
                elapsed_ms = (time.perf_counter() - start_time) * 1000

                # Log success
                log_data["success"] = True
                log_data["execution_time_ms"] = round(elapsed_ms, 2)
                if log_result:
                    log_data["result"] = truncate(result)

                log.info("Tool executed", **log_data)
                log_tool_execution(
                    tool_name=tool_name,
                    success=True,
                    execution_time_ms=elapsed_ms,
                )

                return result

            except Exception as e:
                elapsed_ms = (time.perf_counter() - start_time) * 1000

                # Log failure
                log.error(
                    "Tool execution failed",
                    tool=tool_name,
                    error=str(e),
                    error_type=type(e).__name__,
                    execution_time_ms=round(elapsed_ms, 2),
                )
                log_tool_execution(
                    tool_name=tool_name,
                    success=False,
                    execution_time_ms=elapsed_ms,
                    error=str(e),
                )

                raise

        if asyncio_iscoroutinefunction(f):
            return async_wrapper  # type: ignore
        return sync_wrapper

    # Handle being called with or without parentheses
    if func is not None:
        return decorator(func)
    return decorator


def asyncio_iscoroutinefunction(func: Callable[..., Any]) -> bool:
    """Check if a function is a coroutine function.

    Args:
        func: Function to check

    Returns:
        True if coroutine function
    """
    import asyncio
    return asyncio.iscoroutinefunction(func)
