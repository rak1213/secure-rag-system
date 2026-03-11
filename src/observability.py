"""Phoenix (Arize) observability with OpenTelemetry tracing.

Phoenix runs locally for free — no API key needed.
Provides a trace UI at localhost:6006 for debugging LLM calls.
"""

from .config import Settings
from .logging_config import get_logger

log = get_logger(__name__)


def setup_tracing(settings: Settings) -> None:
    """Launch Phoenix and set up OpenTelemetry tracing.

    This instruments LangChain so every LLM call, embedding call, and
    retriever operation is automatically traced.

    Args:
        settings: Application settings (uses enable_tracing flag).
    """
    if not settings.enable_tracing:
        log.info("observability.tracing.disabled")
        return

    try:
        import phoenix as px
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from openinference.instrumentation.langchain import LangChainInstrumentor

        # Launch Phoenix
        px.launch_app()
        log.info("observability.phoenix.launched", url="http://localhost:6006")

        # Set up OpenTelemetry
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(
            SimpleSpanProcessor(
                OTLPSpanExporter(endpoint="http://localhost:6006/v1/traces")
            )
        )
        trace.set_tracer_provider(tracer_provider)

        # Instrument LangChain
        LangChainInstrumentor().instrument()
        log.info("observability.langchain.instrumented")

    except ImportError as e:
        log.warning("observability.tracing.import_error",
                    detail="Install tracing deps: pip install 'rag-agent-week2[tracing]'",
                    error=str(e))
    except Exception as e:
        log.error("observability.tracing.setup_error", error=str(e))
