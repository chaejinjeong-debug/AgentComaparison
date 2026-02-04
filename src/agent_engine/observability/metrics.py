"""Metrics collection for the Agent Engine platform.

Implements OB-003: Metrics collection (request count, error rate, token usage).
"""

import time
from typing import Any

import structlog

try:
    from opentelemetry import metrics
    from opentelemetry.exporter.cloud_monitoring import CloudMonitoringMetricsExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

    OTEL_METRICS_AVAILABLE = True
except ImportError:
    OTEL_METRICS_AVAILABLE = False

from agent_engine.config import ObservabilityConfig

logger = structlog.get_logger(__name__)


class MetricsManager:
    """Manages metrics collection with Cloud Monitoring integration.

    OB-003: Metrics collection for request count, error rate, token usage.

    Attributes:
        config: Observability configuration
        _initialized: Whether metrics has been set up
    """

    _instance: "MetricsManager | None" = None

    def __init__(
        self,
        config: ObservabilityConfig | None = None,
        project_id: str | None = None,
        service_name: str = "agent-engine",
    ) -> None:
        """Initialize the metrics manager.

        Args:
            config: Observability configuration
            project_id: GCP project ID for Cloud Monitoring
            service_name: Service name for metrics
        """
        self.config = config or ObservabilityConfig()
        self.project_id = project_id
        self.service_name = service_name
        self._initialized = False

        # Metrics instruments
        self._request_counter: Any = None
        self._error_counter: Any = None
        self._token_counter: Any = None
        self._latency_histogram: Any = None
        self._active_sessions_gauge: Any = None
        self._memory_count_gauge: Any = None

        # In-memory metrics for local use
        self._local_metrics: dict[str, Any] = {
            "requests": 0,
            "errors": 0,
            "tokens": {"input": 0, "output": 0},
            "latency_samples": [],
            "active_sessions": 0,
            "memory_count": 0,
        }

    @classmethod
    def get_instance(
        cls,
        config: ObservabilityConfig | None = None,
        project_id: str | None = None,
    ) -> "MetricsManager":
        """Get or create the singleton metrics manager.

        Args:
            config: Observability configuration
            project_id: GCP project ID

        Returns:
            MetricsManager instance
        """
        if cls._instance is None:
            cls._instance = cls(config=config, project_id=project_id)
        return cls._instance

    def setup(self) -> None:
        """Set up OpenTelemetry metrics with Cloud Monitoring exporter."""
        if not self.config.metrics_enabled:
            logger.info("Metrics disabled by configuration")
            return

        if self._initialized:
            logger.debug("Metrics already initialized")
            return

        if not OTEL_METRICS_AVAILABLE:
            logger.warning(
                "OpenTelemetry metrics not available. "
                "Install with: pip install opentelemetry-sdk "
                "opentelemetry-exporter-gcp-monitoring"
            )
            self._setup_local_metrics()
            return

        try:
            # Create metric reader
            if self.project_id:
                exporter = CloudMonitoringMetricsExporter(project_id=self.project_id)
                reader = PeriodicExportingMetricReader(
                    exporter,
                    export_interval_millis=60000,  # 1 minute
                )
                logger.info(
                    "Cloud Monitoring exporter configured",
                    project_id=self.project_id,
                )
            else:
                # Use console exporter for local development
                from opentelemetry.sdk.metrics.export import ConsoleMetricExporter

                reader = PeriodicExportingMetricReader(
                    ConsoleMetricExporter(),
                    export_interval_millis=60000,
                )
                logger.info("Console metric exporter configured (no project_id)")

            # Create meter provider
            provider = MeterProvider(metric_readers=[reader])
            metrics.set_meter_provider(provider)

            # Get meter
            meter = metrics.get_meter(self.service_name)

            # Create metrics instruments
            self._request_counter = meter.create_counter(
                name="agent_engine.requests",
                description="Total number of agent requests",
                unit="1",
            )

            self._error_counter = meter.create_counter(
                name="agent_engine.errors",
                description="Total number of errors",
                unit="1",
            )

            self._token_counter = meter.create_counter(
                name="agent_engine.tokens",
                description="Total tokens used",
                unit="1",
            )

            self._latency_histogram = meter.create_histogram(
                name="agent_engine.latency",
                description="Request latency",
                unit="ms",
            )

            self._active_sessions_gauge = meter.create_up_down_counter(
                name="agent_engine.active_sessions",
                description="Number of active sessions",
                unit="1",
            )

            self._memory_count_gauge = meter.create_up_down_counter(
                name="agent_engine.memory_count",
                description="Number of stored memories",
                unit="1",
            )

            self._initialized = True
            logger.info("Metrics initialized", service_name=self.service_name)

        except Exception as e:
            logger.error("Failed to initialize metrics", error=str(e))
            self._setup_local_metrics()

    def _setup_local_metrics(self) -> None:
        """Set up local in-memory metrics for testing."""
        self._initialized = True
        logger.info("Local metrics initialized (no Cloud Monitoring)")

    def record_request(
        self,
        method: str = "query",
        status: str = "success",
        user_id: str | None = None,
    ) -> None:
        """Record a request metric.

        Args:
            method: Request method (query, aquery)
            status: Request status (success, error)
            user_id: User identifier
        """
        self._local_metrics["requests"] += 1

        if self._request_counter is not None:
            self._request_counter.add(
                1,
                {"method": method, "status": status},
            )

        logger.debug(
            "Request recorded",
            method=method,
            status=status,
            user_id=user_id,
        )

    def record_error(
        self,
        error_type: str,
        method: str = "query",
    ) -> None:
        """Record an error metric.

        Args:
            error_type: Type of error
            method: Request method where error occurred
        """
        self._local_metrics["errors"] += 1

        if self._error_counter is not None:
            self._error_counter.add(
                1,
                {"error_type": error_type, "method": method},
            )

        logger.debug("Error recorded", error_type=error_type, method=method)

    def record_tokens(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str = "gemini-2.5-pro",
    ) -> None:
        """Record token usage metrics.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Model name
        """
        self._local_metrics["tokens"]["input"] += input_tokens
        self._local_metrics["tokens"]["output"] += output_tokens

        if self._token_counter is not None:
            self._token_counter.add(
                input_tokens,
                {"type": "input", "model": model},
            )
            self._token_counter.add(
                output_tokens,
                {"type": "output", "model": model},
            )

        logger.debug(
            "Tokens recorded",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
        )

    def record_latency(
        self,
        latency_ms: float,
        method: str = "query",
    ) -> None:
        """Record request latency.

        Args:
            latency_ms: Latency in milliseconds
            method: Request method
        """
        self._local_metrics["latency_samples"].append(latency_ms)

        # Keep only last 1000 samples
        if len(self._local_metrics["latency_samples"]) > 1000:
            self._local_metrics["latency_samples"] = self._local_metrics["latency_samples"][-1000:]

        if self._latency_histogram is not None:
            self._latency_histogram.record(
                latency_ms,
                {"method": method},
            )

        logger.debug("Latency recorded", latency_ms=latency_ms, method=method)

    def update_active_sessions(self, delta: int) -> None:
        """Update active sessions count.

        Args:
            delta: Change in session count (+1 or -1)
        """
        self._local_metrics["active_sessions"] += delta

        if self._active_sessions_gauge is not None:
            self._active_sessions_gauge.add(delta)

    def update_memory_count(self, delta: int) -> None:
        """Update memory count.

        Args:
            delta: Change in memory count
        """
        self._local_metrics["memory_count"] += delta

        if self._memory_count_gauge is not None:
            self._memory_count_gauge.add(delta)

    def get_stats(self) -> dict[str, Any]:
        """Get current metrics statistics.

        Returns:
            Dictionary of metrics
        """
        latency_samples = self._local_metrics["latency_samples"]
        latency_stats = {}

        if latency_samples:
            sorted_samples = sorted(latency_samples)
            latency_stats = {
                "min": min(latency_samples),
                "max": max(latency_samples),
                "avg": sum(latency_samples) / len(latency_samples),
                "p50": sorted_samples[len(sorted_samples) // 2],
                "p99": sorted_samples[int(len(sorted_samples) * 0.99)],
            }

        return {
            "requests": self._local_metrics["requests"],
            "errors": self._local_metrics["errors"],
            "error_rate": (
                self._local_metrics["errors"] / self._local_metrics["requests"]
                if self._local_metrics["requests"] > 0
                else 0.0
            ),
            "tokens": self._local_metrics["tokens"],
            "latency": latency_stats,
            "active_sessions": self._local_metrics["active_sessions"],
            "memory_count": self._local_metrics["memory_count"],
        }

    @property
    def is_enabled(self) -> bool:
        """Check if metrics is enabled and initialized."""
        return self._initialized


class Timer:
    """Context manager for timing operations."""

    def __init__(self, metrics_manager: MetricsManager | None = None, method: str = "query"):
        """Initialize timer.

        Args:
            metrics_manager: Optional metrics manager to record to
            method: Method name for metrics
        """
        self.metrics_manager = metrics_manager
        self.method = method
        self.start_time: float = 0
        self.elapsed_ms: float = 0

    def __enter__(self) -> "Timer":
        """Start timing."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        """Stop timing and record."""
        self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000

        if self.metrics_manager is not None:
            self.metrics_manager.record_latency(self.elapsed_ms, self.method)
