"""Cloud Trace integration for the Agent Engine platform.

Implements OB-001: Cloud Trace integration.
"""

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import structlog

try:
    from opentelemetry import trace
    from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
    from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

from agent_engine.config import ObservabilityConfig

logger = structlog.get_logger(__name__)


class TracingManager:
    """Manages distributed tracing with Cloud Trace.

    OB-001: Cloud Trace integration for request tracing.

    Attributes:
        config: Observability configuration
        tracer: OpenTelemetry tracer instance
        _initialized: Whether tracing has been set up
    """

    _instance: "TracingManager | None" = None
    _tracer: Any = None

    def __init__(
        self,
        config: ObservabilityConfig | None = None,
        project_id: str | None = None,
        service_name: str = "agent-engine",
    ) -> None:
        """Initialize the tracing manager.

        Args:
            config: Observability configuration
            project_id: GCP project ID for Cloud Trace
            service_name: Service name for traces
        """
        self.config = config or ObservabilityConfig()
        self.project_id = project_id
        self.service_name = service_name
        self._initialized = False

    @classmethod
    def get_instance(
        cls,
        config: ObservabilityConfig | None = None,
        project_id: str | None = None,
    ) -> "TracingManager":
        """Get or create the singleton tracing manager.

        Args:
            config: Observability configuration
            project_id: GCP project ID

        Returns:
            TracingManager instance
        """
        if cls._instance is None:
            cls._instance = cls(config=config, project_id=project_id)
        return cls._instance

    def setup(self) -> None:
        """Set up OpenTelemetry with Cloud Trace exporter.

        This configures the tracer provider with appropriate sampling
        and export settings.
        """
        if not self.config.tracing_enabled:
            logger.info("Tracing disabled by configuration")
            return

        if not OTEL_AVAILABLE:
            logger.warning(
                "OpenTelemetry not available, tracing disabled. "
                "Install with: pip install opentelemetry-api opentelemetry-sdk "
                "opentelemetry-exporter-gcp-trace"
            )
            return

        if self._initialized:
            logger.debug("Tracing already initialized")
            return

        try:
            # Create resource with service information
            resource = Resource.create(
                {
                    "service.name": self.service_name,
                    "service.version": "0.1.0",
                }
            )

            # Create sampler based on config
            sampler = TraceIdRatioBased(self.config.sample_rate)

            # Create tracer provider
            provider = TracerProvider(
                resource=resource,
                sampler=sampler,
            )

            # Add Cloud Trace exporter if project_id is set
            if self.project_id:
                cloud_trace_exporter = CloudTraceSpanExporter(project_id=self.project_id)
                provider.add_span_processor(BatchSpanProcessor(cloud_trace_exporter))
                logger.info(
                    "Cloud Trace exporter configured",
                    project_id=self.project_id,
                )
            else:
                # Use simple processor for local development
                from opentelemetry.sdk.trace.export import ConsoleSpanExporter

                provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
                logger.info("Console span exporter configured (no project_id)")

            # Set as global tracer provider
            trace.set_tracer_provider(provider)
            self._tracer = trace.get_tracer(self.service_name)
            TracingManager._tracer = self._tracer
            self._initialized = True

            logger.info(
                "Tracing initialized",
                service_name=self.service_name,
                sample_rate=self.config.sample_rate,
            )

        except Exception as e:
            logger.error("Failed to initialize tracing", error=str(e))
            self._initialized = False

    @contextmanager
    def span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> Generator[Any, None, None]:
        """Create a trace span context.

        Args:
            name: Span name
            attributes: Span attributes

        Yields:
            OpenTelemetry span or None if tracing disabled
        """
        if not self._initialized or self._tracer is None:
            yield None
            return

        with self._tracer.start_as_current_span(name) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, str(value))
            yield span

    def start_span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> Any:
        """Start a new span (manual management).

        Args:
            name: Span name
            attributes: Span attributes

        Returns:
            OpenTelemetry span or None
        """
        if not self._initialized or self._tracer is None:
            return None

        span = self._tracer.start_span(name)
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, str(value))
        return span

    @staticmethod
    def end_span(span: Any) -> None:
        """End a span.

        Args:
            span: Span to end
        """
        if span is not None:
            span.end()

    @staticmethod
    def add_event(
        span: Any,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """Add an event to a span.

        Args:
            span: Target span
            name: Event name
            attributes: Event attributes
        """
        if span is not None:
            span.add_event(name, attributes=attributes or {})

    @staticmethod
    def record_exception(span: Any, exception: Exception) -> None:
        """Record an exception on a span.

        Args:
            span: Target span
            exception: Exception to record
        """
        if span is not None:
            span.record_exception(exception)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(exception)))

    def get_tracer(self) -> Any:
        """Get the underlying tracer.

        Returns:
            OpenTelemetry tracer or None
        """
        return self._tracer

    @property
    def is_enabled(self) -> bool:
        """Check if tracing is enabled and initialized."""
        return self._initialized and self._tracer is not None
