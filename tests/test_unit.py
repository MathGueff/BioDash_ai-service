import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import main
import telemetry


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("  PRESSÃO e Temperatura  ", "pressao e temperatura"),
        ("Março", "marco"),
        ("ÁÉÍÓÚ àèìòù ãõ âêîôû ç ñ", "aeiou aeiou ao aeiou c n"),
        ("", ""),
    ],
)
def test_normalize_text(text, expected):
    assert main.normalize_text(text) == expected


def test_extract_entities_interprets_date_priority_and_values():
    entities = main.extract_entities("Urgente: 125,5 kg e 42 kWh e R$ 310,75 em 15/03/2026")
    assert entities["priority"] == "high"
    assert entities["waste_processed"] == 125.5
    assert entities["energy_generated"] == 42.0
    assert entities["tax_savings"] == 310.75
    assert (entities["day"], entities["month"], entities["year"]) == (15, 2, 2026)


@pytest.mark.parametrize(
    ("text", "expected_day", "expected_month", "expected_year"),
    [
        ("relatório de 05-11-25", 5, 10, 2025),
        ("manutenção dia 18 de outubro de 2026", 18, 9, 2026),
        ("dados de 7 de janeiro", 7, 0, None),
    ],
)
def test_extract_entities_date_formats_and_isolated_days(text, expected_day, expected_month, expected_year):
    entities = main.extract_entities(text)
    assert entities.get("day") == expected_day
    assert entities.get("month") == expected_month
    assert entities.get("year") == expected_year


def test_extract_entities_ignores_invalid_isolated_day():
    entities = main.extract_entities("revisão no dia 45")
    assert "day" not in entities


@pytest.mark.parametrize(
    ("text", "expected_priority"),
    [
        ("manutenção crítica no tanque", "high"),
        ("revisão moderada mensal", "medium"),
        ("ajuste leve na tampa", "low"),
    ],
)
def test_extract_entities_priorities(text, expected_priority):
    assert main.extract_entities(text)["priority"] == expected_priority


@pytest.mark.parametrize(
    ("text", "expected_format"),
    [
        ("exportar em pdf agora", "pdf"),
        ("gerar planilha xlsx", "excel"),
        ("baixar dados em csv", "csv"),
    ],
)
def test_extract_entities_export_formats(text, expected_format):
    assert main.extract_entities(text)["format"] == expected_format


@pytest.mark.parametrize(
    ("text", "topic"),
    [
        ("pressão alta na válvula de alívio", "pressao"),
        ("cheiro de gás sulfídrico h2s", "h2s"),
        ("acidez do biodigestor e ph", "ph"),
        ("temperatura caiu e houve choque térmico", "temperatura"),
        ("alimentação de biomassa e carga orgânica", "alimentacao"),
        ("vazamento com odor forte e emergência de segurança", "seguranca"),
        ("procedimento padrão de operação da planta", "operacao_geral"),
    ],
)
def test_extract_operational_topics(text, topic):
    assert main.extract_entities(text)["operational_topic"] == topic


def test_build_marker_text_handles_dict_and_string_addresses():
    marker_dict = {
        "title": "Planta Norte",
        "address": {
            "street": "Rua das Flores",
            "cep": "13800-000",
            "city": "Mogi Mirim",
            "complement": "Galpão 2",
        },
        "description": "Unidade IoT principal",
    }
    text_dict = main.build_marker_text(marker_dict)
    assert "Planta Norte" in text_dict
    assert "Rua das Flores" in text_dict
    assert "13800-000" in text_dict
    assert "Mogi Mirim" in text_dict
    assert "Unidade IoT principal" in text_dict

    marker_str = {
        "title": "Planta Sul",
        "address": "Rodovia SP-340 Km 120",
    }
    text_str = main.build_marker_text(marker_str)
    assert text_str == "Planta Sul Rodovia SP-340 Km 120"


def test_format_marker_address_handles_complete_partial_and_missing_fields():
    full_marker = {
        "title": "Biodigestor Central",
        "address": {
            "street": "Av. Brasil",
            "number": "500",
            "complement": "Bloco B",
            "cep": "13000-000",
            "city": "Campinas",
        },
    }
    formatted_full = main.format_marker_address(full_marker)
    assert "📍 *Biodigestor Central*" in formatted_full
    assert "Av. Brasil, 500" in formatted_full
    assert "Bloco B" in formatted_full
    assert "CEP: 13000-000" in formatted_full
    assert "Campinas" in formatted_full

    empty_dict_marker = {"title": "Planta Vazia", "address": {}, "description": "Setor Rural"}
    assert "Setor Rural" in main.format_marker_address(empty_dict_marker)

    str_addr_marker = {"title": "Planta String", "address": "Fazenda Boa Vista"}
    assert "Endereço não informado" in main.format_marker_address(str_addr_marker)


def test_format_indicators_handles_empty_and_iot_metrics_list():
    empty_msg = main.format_indicators([])
    assert "Nenhuma métrica disponível" in empty_msg

    iot_indicators = [
        {"waste_processed": 450.5, "energy_generated": 89.25, "tax_savings": 150.0},
        {"waste_processed": 100.0, "energy_generated": 20.0, "tax_savings": 30.0},
    ]
    formatted = main.format_indicators(iot_indicators)
    assert "450.50 kg" in formatted
    assert "89.25 kWh" in formatted
    assert "R$ 150.00" in formatted


def test_build_default_fallback_response_updates_context_and_suggestions():
    ctx = main.ContextState(last_intent="pedido_energia", last_topic="energia")
    res = main.build_default_fallback_response(0.123456, {"month": 2}, ctx)
    assert res.intent == "nao_compreendido"
    assert res.question_type == main.QuestionType.FORA_DE_ESCOPO
    assert res.confidence == 0.1235
    assert res.context.last_question_type == main.QuestionType.FORA_DE_ESCOPO
    assert len(res.suggestions) == 6


def test_transcribe_file_joins_segments_without_loading_whisper(monkeypatch):
    model = Mock()
    model.transcribe.return_value = (
        [SimpleNamespace(text=" Olá "), SimpleNamespace(text=" biodigestor ")],
        SimpleNamespace(language="pt", duration=1.234),
    )
    monkeypatch.setattr(main, "get_whisper_model", lambda: model)

    result = main.transcribe_file("audio.wav")

    assert result == {
        "text": "Olá biodigestor",
        "language": "pt",
        "duration": 1.23,
        "model": main.WHISPER_MODEL,
    }
    model.transcribe.assert_called_once()
    assert model.transcribe.call_args.args == ("audio.wav",)
    assert model.transcribe.call_args.kwargs["language"] == "pt"


def test_transcribe_file_raises_error_when_audio_duration_exceeds_limit(monkeypatch):
    model = Mock()
    max_allowed = main.MAX_AUDIO_DURATION_SECONDS + main.AUDIO_DURATION_TOLERANCE_SECONDS
    model.transcribe.return_value = (
        [SimpleNamespace(text="fala longa")],
        SimpleNamespace(language="pt", duration=max_allowed + 2.5),
    )
    monkeypatch.setattr(main, "get_whisper_model", lambda: model)

    with pytest.raises(main.AudioDurationExceededError) as exc_info:
        main.transcribe_file("long_audio.wav")

    assert exc_info.value.duration == pytest.approx(max_allowed + 2.5)


def test_telemetry_get_otel_context_meta_handles_valid_invalid_and_error(monkeypatch):
    # Cenário 1: Span válido com trace_id e span_id
    fake_ctx = SimpleNamespace(is_valid=True, trace_id=0x1234, span_id=0x5678)
    fake_span = SimpleNamespace(get_span_context=lambda: fake_ctx)
    fake_otel = SimpleNamespace(get_current_span=lambda: fake_span)
    monkeypatch.setattr(telemetry.importlib, "import_module", lambda name: fake_otel)

    meta = telemetry._get_otel_context_meta()
    assert meta == {
        "traceId": f"{0x1234:032x}",
        "spanId": f"{0x5678:016x}",
    }

    # Cenário 2: Span inválido
    invalid_ctx = SimpleNamespace(is_valid=False)
    invalid_span = SimpleNamespace(get_span_context=lambda: invalid_ctx)
    monkeypatch.setattr(
        telemetry.importlib,
        "import_module",
        lambda name: SimpleNamespace(get_current_span=lambda: invalid_span),
    )
    assert telemetry._get_otel_context_meta() == {}

    # Cenário 3: Exceção ao importar opentelemetry.trace
    def raise_import_error(name):
        raise ImportError("opentelemetry ausente")

    monkeypatch.setattr(telemetry.importlib, "import_module", raise_import_error)
    assert telemetry._get_otel_context_meta() == {}


def test_telemetry_log_event_formats_structured_json(monkeypatch):
    captured = []
    monkeypatch.setattr(telemetry, "_get_otel_context_meta", lambda: {"traceId": "abc", "spanId": "def"})
    monkeypatch.setattr(telemetry.logger, "info", lambda msg: captured.append(msg))

    telemetry.log_event("info", "Evento de teste", customKey="valor123")

    assert len(captured) == 1
    assert captured[0].startswith("Evento de teste ")
    json_part = json.loads(captured[0].split("Evento de teste ", 1)[1])
    assert json_part["service"] == "biodash-ai-service"
    assert json_part["traceId"] == "abc"
    assert json_part["customKey"] == "valor123"


def test_telemetry_init_handles_local_noop_and_azure_failure(monkeypatch):
    events = []
    monkeypatch.setattr(telemetry, "log_event", lambda level, msg, **kw: events.append((level, msg, kw)))

    # Modo local sem connection string
    monkeypatch.setattr(telemetry, "_is_initialized", False)
    monkeypatch.delenv("APPLICATIONINSIGHTS_CONNECTION_STRING", raising=False)
    monkeypatch.delenv("APPINSIGHTS_CONNECTIONSTRING", raising=False)
    telemetry.init_telemetry()
    assert telemetry._is_initialized is True
    assert events[-1][2]["telemetryProvider"] == "local-noop"

    # Modo com connection string mas falha no SDK da Azure (cenário de erro contemplado)
    monkeypatch.setattr(telemetry, "_is_initialized", False)
    monkeypatch.setenv("APPLICATIONINSIGHTS_CONNECTION_STRING", "InstrumentationKey=00000000-0000-0000-0000-000000000000")

    def broken_azure_import(name):
        raise RuntimeError("Falha simulada no exportador Azure Monitor")

    monkeypatch.setattr(telemetry.importlib, "import_module", broken_azure_import)
    telemetry.init_telemetry()
    assert events[-1][0] == "warning"
    assert "Falha simulada" in events[-1][2]["error"]
