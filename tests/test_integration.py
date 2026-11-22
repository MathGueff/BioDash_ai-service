from types import SimpleNamespace
from unittest.mock import Mock

from fastapi.testclient import TestClient

import main

client = TestClient(main.app)


# ──────────────────────────────────────────────────────────────────────────────
# 1. TESTES DE INTEGRAÇÃO DAS PRINCIPAIS APIs (HTTP / FastAPI)
# ──────────────────────────────────────────────────────────────────────────────


def test_root_and_health_endpoints():
    res_root = client.get("/")
    assert res_root.status_code == 200
    body_root = res_root.json()
    assert body_root["service"] == "BioDash AI Service"
    assert body_root["status"] == "running"
    assert "version" in body_root

    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json() == {"status": "ok"}


def test_legacy_biodigestores_endpoint():
    res = client.get("/biodigestores", params={"user_id": "123e4567-e89b-12d3-a456-426614174000"})
    assert res.status_code == 200
    body = res.json()
    assert body["success"] is True
    assert body["data"] == []
    assert "PostgreSQL" in body["message"]


# ──────────────────────────────────────────────────────────────────────────────
# 2. VALIDAÇÃO DA INTEGRAÇÃO COM DADOS IoT (Sensores, Telemetria e Indicadores)
# ──────────────────────────────────────────────────────────────────────────────


def test_iot_metrics_and_markers_integration_via_chatbot_api():
    iot_markers = [
        {
            "id": "bio-iot-01",
            "title": "Biodigestor IoT Fazenda Santa Clara",
            "address": {
                "street": "Rodovia SP-340",
                "number": "Km 145",
                "city": "Mogi Mirim",
                "cep": "13800-000",
            },
        }
    ]
    iot_indicators = [
        {
            "waste_processed": 1540.75,
            "energy_generated": 312.40,
            "tax_savings": 490.00,
            "measured_at": "2026-11-20T14:30:00Z",
        }
    ]

    # 1. Consulta endereço da planta IoT -> atualiza selected_biodigestor no contexto
    res_addr = client.post(
        "/chatbot",
        json={"message": "Onde fica o biodigestor?", "markers": iot_markers, "indicators": iot_indicators},
    )
    assert res_addr.status_code == 200
    data_addr = res_addr.json()
    assert data_addr["intent"] == "pedido_endereco"
    assert data_addr["question_type"] == "localizacao"
    assert "Biodigestor IoT Fazenda Santa Clara" in data_addr["response"]
    assert data_addr["context"]["selected_biodigestor"] == "Biodigestor IoT Fazenda Santa Clara"

    # 2. Continuidade multi-turn sobre energia gerada pela unidade IoT selecionada
    res_energy = client.post(
        "/chatbot",
        json={
            "message": "Quanta energia foi gerada?",
            "markers": iot_markers,
            "indicators": iot_indicators,
            "context": data_addr["context"],
        },
    )
    assert res_energy.status_code == 200
    data_energy = res_energy.json()
    assert data_energy["intent"] == "pedido_energia"
    assert "Biodigestor IoT Fazenda Santa Clara" in data_energy["response"]
    assert "312.40 kWh" in data_energy["response"]

    # 3. Follow-up elíptico ("E os resíduos?") preservando contexto da planta IoT
    res_waste = client.post(
        "/chatbot",
        json={
            "message": "E os resíduos?",
            "markers": iot_markers,
            "indicators": iot_indicators,
            "context": data_energy["context"],
        },
    )
    assert res_waste.status_code == 200
    data_waste = res_waste.json()
    assert data_waste["intent"] == "pedido_residuos"
    assert data_waste["question_type"] == "contextual"
    assert "1540.75 kg" in data_waste["response"]
    assert "Biodigestor IoT Fazenda Santa Clara" in data_waste["response"]


def test_iot_sensor_anomaly_alerts_and_multi_turn_operational_diagnostics():
    # Alerta IoT de H2S elevado
    res_h2s = client.post("/chatbot", json={"message": "Alerta de H2S elevado no sensor do biodigestor"})
    assert res_h2s.status_code == 200
    body_h2s = res_h2s.json()
    assert body_h2s["intent"] == "duvida_operacional"
    assert body_h2s["action"] == "view_alerts"
    assert body_h2s["entities"]["operational_topic"] == "h2s"
    assert body_h2s["context"]["last_operational_topic"] == "h2s"

    # Alerta IoT de sobrepressão crítica no gasômetro
    res_press = client.post("/chatbot", json={"message": "Pressão alta no biodigestor, válvula em sobrepressão"})
    assert res_press.status_code == 200
    body_press = res_press.json()
    assert body_press["intent"] == "duvida_operacional"
    assert body_press["action"] == "view_alerts"
    assert "Sobrepressão" in body_press["response"]

    # Leitura IoT de temperatura ideal seguida de diagnóstico de queda térmica e superaquecimento
    res_temp = client.post("/chatbot", json={"message": "Qual a temperatura ideal de operação do biodigestor?"})
    assert res_temp.status_code == 200
    ctx_temp = res_temp.json()["context"]
    assert ctx_temp["last_operational_topic"] == "temperatura"

    res_cool = client.post(
        "/chatbot",
        json={"message": "E se esfriar demais?", "context": ctx_temp},
    )
    assert res_cool.status_code == 200
    body_cool = res_cool.json()
    assert body_cool["question_type"] == "contextual"
    assert "Queda de Temperatura" in body_cool["response"]

    res_heat = client.post(
        "/chatbot",
        json={"message": "E se esquentar muito e passar de 40 graus?", "context": body_cool["context"]},
    )
    assert res_heat.status_code == 200
    body_heat = res_heat.json()
    assert "Superaquecimento" in body_heat["response"]

    # Leitura IoT de pH: acidificação (< 6.5) e alcalinização (> 8.0)
    res_acid = client.post("/chatbot", json={"message": "E se o pH cair e acidificar o biodigestor?"})
    assert res_acid.status_code == 200
    assert "Acidificação" in res_acid.json()["response"]

    res_alk = client.post("/chatbot", json={"message": "E se o pH subir com alcalinidade e amônia alta?"})
    assert res_alk.status_code == 200
    assert "Alcalinização" in res_alk.json()["response"]


# ──────────────────────────────────────────────────────────────────────────────
# 3. VALIDAÇÃO DA COMUNICAÇÃO ENTRE COMPONENTES (Whisper -> NLP / TF-IDF Search / Telemetry)
# ──────────────────────────────────────────────────────────────────────────────


def test_component_communication_audio_transcription_to_chatbot_pipeline(monkeypatch):
    """
    Valida a comunicação ponta a ponta entre os componentes:
    1. Upload de áudio na API /transcribe -> processamento pelo adaptador Whisper -> telemetry log_event
    2. Texto transcrito enviado à API /chatbot -> classificador TF-IDF + SVM -> extração de entidades e resposta IoT
    """
    fake_whisper = Mock()
    fake_whisper.transcribe.return_value = (
        [SimpleNamespace(text="Quanta energia foi gerada em março de 2026?")],
        SimpleNamespace(language="pt", duration=3.45),
    )
    monkeypatch.setattr(main, "get_whisper_model", lambda: fake_whisper)

    telemetry_logs = []
    orig_log_event = main.log_event

    def spy_log_event(level, message, **meta):
        telemetry_logs.append((level, message, meta))
        return orig_log_event(level, message, **meta)

    monkeypatch.setattr(main, "log_event", spy_log_event)

    # Passo 1: Transcrição do áudio
    audio_bytes = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 128
    res_transcribe = client.post(
        "/transcribe",
        files={"audio": ("comando_voz.wav", audio_bytes, "audio/wav")},
    )
    assert res_transcribe.status_code == 200
    transcribed_payload = res_transcribe.json()
    assert transcribed_payload["text"] == "Quanta energia foi gerada em março de 2026?"
    assert transcribed_payload["language"] == "pt"
    assert transcribed_payload["duration"] == 3.45

    # Passo 2: Alimenta o texto transcrito diretamente no componente de Chatbot com indicadores IoT
    res_chat = client.post(
        "/chatbot",
        json={
            "message": transcribed_payload["text"],
            "indicators": [{"waste_processed": 800.0, "energy_generated": 215.75, "tax_savings": 180.0}],
        },
    )
    assert res_chat.status_code == 200
    chat_payload = res_chat.json()
    assert chat_payload["intent"] == "pedido_energia"
    assert "215.75 kWh" in chat_payload["response"]
    assert chat_payload["entities"]["month"] == 2
    assert chat_payload["entities"]["year"] == 2026

    # Verifica que ambos os componentes registraram eventos estruturados no componente de telemetria
    logged_messages = [msg for _, msg, _ in telemetry_logs]
    assert "Transcrição Whisper concluída" in logged_messages
    assert "Chatbot intent classified" in logged_messages


def test_semantic_search_api_ranks_iot_biodigestor_markers():
    markers = [
        {
            "id": "1",
            "title": "Biodigestor Norte - Suinocultura",
            "address": {"street": "Estrada Municipal", "city": "Campinas", "cep": "13000-000"},
            "description": "Planta de biogás para dejetos suínos",
        },
        {
            "id": "2",
            "title": "Usina Solar Sul",
            "address": {"street": "Av. Paulista", "city": "São Paulo", "cep": "01310-000"},
            "description": "Painéis fotovoltaicos",
        },
        {
            "id": "3",
            "title": "Biodigestor Leste - Laticínios",
            "address": {"street": "Rodovia SP-340", "city": "Mogi Mirim", "cep": "13800-000"},
            "description": "Tratamento anaeróbico de efluentes de laticínios",
        },
    ]

    res = client.post(
        "/semantic-search",
        json={"query": "Mogi Mirim laticínios SP-340", "markers": markers},
    )
    assert res.status_code == 200
    results = res.json()["results"]
    assert len(results) >= 1
    assert results[0]["id"] == "3"
    assert results[0]["similarity_score"] > 0.1

    # Lista vazia de marcadores retorna lista vazia sem erro
    res_empty = client.post("/semantic-search", json={"query": "Campinas", "markers": []})
    assert res_empty.status_code == 200
    assert res_empty.json() == {"results": []}


def test_chatbot_transactional_and_export_flows():
    flows = [
        ("agendar manutenção", "agendar_manutencao", "start_flow_manutencao"),
        ("adicionar métricas", "incluir_metrica", "start_flow_metrica"),
        ("editar métricas", "editar_metrica", "start_flow_editar_metrica"),
        ("adicionar biodigestor", "adicionar_endereco", "start_flow_endereco"),
        ("relatório por período", "relatorio_periodo", "start_flow_relatorio"),
        ("gerar relatório pdf", "pedido_exportar_pdf", "export_pdf"),
        ("exportar csv", "pedido_exportar_csv", "export_csv"),
        ("exportar excel", "pedido_exportar_excel", "export_excel"),
        ("sim", "confirmar", "flow_confirm"),
        ("cancelar", "cancelar", "cancel_flow"),
    ]
    for msg, expected_intent, expected_action in flows:
        res = client.post("/chatbot", json={"message": msg})
        assert res.status_code == 200
        body = res.json()
        assert body["intent"] == expected_intent
        assert body["action"] == expected_action


# ──────────────────────────────────────────────────────────────────────────────
# 4. CENÁRIOS DE ERRO E EXCEÇÕES NAS APIs
# ──────────────────────────────────────────────────────────────────────────────


def test_chatbot_error_scenarios_empty_whitespace_and_out_of_scope():
    # Mensagem vazia -> HTTP 400
    res_empty = client.post("/chatbot", json={"message": "   "})
    assert res_empty.status_code == 400
    assert "vazia" in res_empty.json()["detail"].lower()

    # Payload inválido sem campo obrigatório `message` -> HTTP 422
    res_invalid = client.post("/chatbot", json={})
    assert res_invalid.status_code == 422

    # Solicitação fora de escopo -> fallback estruturado 200 com intent `nao_compreendido`
    res_fallback = client.post(
        "/chatbot",
        json={"message": "quero pedir uma pizza calabresa com borda recheada"},
    )
    assert res_fallback.status_code == 200
    body_fallback = res_fallback.json()
    assert body_fallback["intent"] == "nao_compreendido"
    assert body_fallback["question_type"] == "fora_de_escopo"
    assert body_fallback["action"] is None

    # Solicitação de métricas/endereços quando não há dados cadastrados
    res_no_markers = client.post("/chatbot", json={"message": "onde fica o biodigestor?", "markers": []})
    assert res_no_markers.status_code == 200
    assert "Não encontrei nenhum biodigestor" in res_no_markers.json()["response"]

    res_no_indicators = client.post("/chatbot", json={"message": "quanta energia foi gerada?", "indicators": []})
    assert res_no_indicators.status_code == 200
    assert "Nenhuma métrica de energia encontrada" in res_no_indicators.json()["response"]


def test_semantic_search_error_scenarios():
    markers = [{"title": "Planta 01", "address": "Rua A"}]

    # Query em branco -> HTTP 400
    res_blank = client.post("/semantic-search", json={"query": "   ", "markers": markers})
    assert res_blank.status_code == 400
    assert "vazia" in res_blank.json()["detail"].lower()

    # Payload sem campo `markers` -> HTTP 422
    res_missing = client.post("/semantic-search", json={"query": "Planta"})
    assert res_missing.status_code == 422


def test_transcribe_error_scenarios_unsupported_type_empty_oversize_duration_no_speech_and_crash(monkeypatch):
    # 1. Formato de arquivo não suportado (415)
    res_415 = client.post(
        "/transcribe",
        files={"audio": ("documento.pdf", b"%PDF-1.4", "application/pdf")},
    )
    assert res_415.status_code == 415
    assert "não suportado" in res_415.json()["detail"].lower()

    # 2. Arquivo de áudio vazio (400)
    res_400 = client.post(
        "/transcribe",
        files={"audio": ("vazio.wav", b"", "audio/wav")},
    )
    assert res_400.status_code == 400
    assert "vazio" in res_400.json()["detail"].lower()

    # 3. Arquivo excedendo limite de bytes MAX_AUDIO_BYTES (413)
    oversized_bytes = b"a" * (main.MAX_AUDIO_BYTES + 10)
    res_413_size = client.post(
        "/transcribe",
        files={"audio": ("grande.webm", oversized_bytes, "audio/webm")},
    )
    assert res_413_size.status_code == 413
    assert "limite permitido" in res_413_size.json()["detail"].lower()

    # 4. Áudio excedendo duração máxima permitida (413 via AudioDurationExceededError)
    def raise_duration_exceeded(_path):
        raise main.AudioDurationExceededError(18.4)

    monkeypatch.setattr(main, "transcribe_file", raise_duration_exceeded)
    res_413_dur = client.post(
        "/transcribe",
        files={"audio": ("longo.ogg", b"OggS-fake-audio", "audio/ogg")},
    )
    assert res_413_dur.status_code == 413
    assert "18.4s" in res_413_dur.json()["detail"]

    # 5. Transcrição sem fala reconhecida / texto vazio (422)
    monkeypatch.setattr(
        main,
        "transcribe_file",
        lambda _path: {"text": "", "language": "pt", "duration": 1.0, "model": "base"},
    )
    res_422 = client.post(
        "/transcribe",
        files={"audio": ("silencio.wav", b"RIFF-silence", "audio/wav")},
    )
    assert res_422.status_code == 422
    assert "nenhuma fala" in res_422.json()["detail"].lower()

    # 6. Falha interna inesperada no motor Whisper (503)
    def raise_internal_error(_path):
        raise RuntimeError("Corrupted audio stream")

    monkeypatch.setattr(main, "transcribe_file", raise_internal_error)
    res_503 = client.post(
        "/transcribe",
        files={"audio": ("corrompido.mp3", b"ID3-corrupt", "audio/mpeg")},
    )
    assert res_503.status_code == 503
    assert "falha ao transcrever" in res_503.json()["detail"].lower()
