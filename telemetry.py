"""
Módulo de Observabilidade e OpenTelemetry para o BioDash AI Service (FastAPI).
Isola a inicialização do Azure Monitor OpenTelemetry e o logging estruturado JSON
com correlação automática de traceId e spanId (W3C Trace Context).
"""

import importlib
import json
import logging
import os
from typing import Any, Dict

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [biodash-ai-service]: %(message)s",
)
# Silencia logs verbosos internos do SDK da Azure (evita loop infinito de telemetria e poluição no stderr)
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
logging.getLogger("azure.monitor.opentelemetry").setLevel(logging.WARNING)

logger = logging.getLogger("biodash-ai")

_is_initialized = False


def _get_otel_context_meta() -> Dict[str, str]:
    """Extrai traceId e spanId do span ativo do OpenTelemetry, se presente."""
    try:
        otel_trace = importlib.import_module("opentelemetry.trace")
        span = otel_trace.get_current_span()
        if not span:
            return {}
        ctx = span.get_span_context()
        if not ctx or not ctx.is_valid:
            return {}
        return {
            "traceId": format(ctx.trace_id, "032x"),
            "spanId": format(ctx.span_id, "016x"),
        }
    except Exception:
        return {}


def log_event(level: str, message: str, **meta: Any) -> None:
    """Emite um log estruturado enriquecido com metadados do serviço e contexto OTel."""
    payload: Dict[str, Any] = {
        "service": "biodash-ai-service",
        **_get_otel_context_meta(),
        **meta,
    }
    formatted = f"{message} {json.dumps(payload, ensure_ascii=False)}"
    getattr(logger, level, logger.info)(formatted)


def init_telemetry() -> None:
    """Inicializa o exportador Azure Monitor OpenTelemetry caso a Connection String exista."""
    global _is_initialized
    if _is_initialized:
        return

    conn_str = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING") or os.getenv(
        "APPINSIGHTS_CONNECTIONSTRING"
    )
    if conn_str:
        try:
            az_otel = importlib.import_module("azure.monitor.opentelemetry")
            az_otel.configure_azure_monitor(
                connection_string=conn_str,
                logger_name="biodash-ai",
            )
            _is_initialized = True
            log_event(
                "info",
                "Starting BioDash AI Service - OpenTelemetry instrumentado com sucesso",
                telemetryProvider="azure-monitor-otel",
            )
        except Exception as err:
            log_event(
                "warning",
                "Falha ao inicializar OpenTelemetry no AI Service",
                telemetryProvider="azure-monitor-otel",
                error=str(err),
            )
    else:
        _is_initialized = True
        log_event(
            "info",
            "Starting BioDash AI Service - APPLICATIONINSIGHTS_CONNECTION_STRING ausente (modo local)",
            telemetryProvider="local-noop",
        )

