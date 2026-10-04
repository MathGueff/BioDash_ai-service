"""
Validação Automatizada da Checklist de Interpretação de Contexto do Chatbot BioDash:
Item 1: Definir tipos de perguntas.
Item 2: Definir contexto das solicitações.
Item 3: Implementar interpretação.
Item 4: Testar diferentes perguntas.
Item 5: Validar respostas.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from main import (
    chatbot_endpoint,
    ChatRequest,
    ContextState,
    ChatMessage,
    QuestionType,
    extract_entities,
    CONFIDENCE_THRESHOLD,
)


def test_item1_definir_tipos_de_perguntas():
    print("\n=======================================================")
    print("CHECKLIST ITEM 1: Definir tipos de perguntas")
    print("=======================================================")
    expected_types = {
        "operacional",
        "metrica",
        "localizacao",
        "transacional",
        "contextual",
        "fora_de_escopo",
        "social",
    }
    actual_types = {t.value for t in QuestionType}
    print(f"Tipos definidos no sistema: {actual_types}")
    assert expected_types.issubset(actual_types), f"Tipos esperados não encontrados: {expected_types - actual_types}"
    print("[OK] Item 1 validado com sucesso!")


def test_item2_definir_contexto_das_solicitacoes():
    print("\n=======================================================")
    print("CHECKLIST ITEM 2: Definir contexto das solicitações")
    print("=======================================================")
    ctx = ContextState(
        last_intent="pedido_energia",
        last_topic="energia",
        last_question_type=QuestionType.METRICA,
        selected_biodigestor="Biodigestor Norte",
        period="fevereiro 2026",
    )
    print(f"Estrutura do contexto (ContextState): {ctx.model_dump()}")
    assert ctx.last_intent == "pedido_energia"
    assert ctx.last_topic == "energia"
    assert ctx.last_question_type == QuestionType.METRICA
    assert ctx.selected_biodigestor == "Biodigestor Norte"
    print("[OK] Item 2 validado com sucesso!")


def test_item3_implementar_interpretacao():
    print("\n=======================================================")
    print("CHECKLIST ITEM 3: Implementar interpretação")
    print("=======================================================")
    
    # 3.1 Extração de entidades operacionais
    ent_h2s = extract_entities("o filtro de gas sulfidrico e h2s está saturado")
    assert ent_h2s.get("operational_topic") == "h2s"
    
    ent_press = extract_entities("alerta de sobrepressao e valvula travada")
    assert ent_press.get("operational_topic") == "pressao"

    ent_ph = extract_entities("qual o ph e acidez da biomassa?")
    assert ent_ph.get("operational_topic") == "ph"

    ent_temp = extract_entities("temperatura de 37 graus no biodigestor")
    assert ent_temp.get("operational_topic") == "temperatura"

    # 3.2 Interpretação e threshold de confiança
    print(f"Limiar de corte para confiança de interpretação: {CONFIDENCE_THRESHOLD}")
    assert CONFIDENCE_THRESHOLD >= 0.20 and CONFIDENCE_THRESHOLD <= 0.40

    print("[OK] Item 3 validado com sucesso!")


def test_item4_testar_diferentes_perguntas():
    print("\n=======================================================")
    print("CHECKLIST ITEM 4: Testar diferentes perguntas")
    print("=======================================================")
    sample_indicators = [{"waste_processed": 1200.0, "energy_generated": 180.5, "tax_savings": 320.0, "measured_at": "2026-02-15"}]
    sample_markers = [{"title": "Planta 01", "address": {"street": "Rodovia SP 340", "city": "Mogi Mirim"}}]

    test_cases = [
        # Tipo 1: OPERACIONAL
        ("o que fazer com alerta de h2s?", QuestionType.OPERACIONAL, "duvida_operacional"),
        ("pressão alta no biodigestor", QuestionType.OPERACIONAL, "duvida_operacional"),
        ("qual a temperatura ideal de operação?", QuestionType.OPERACIONAL, "duvida_operacional"),
        ("qual o ph ideal do biodigestor?", QuestionType.OPERACIONAL, "duvida_operacional"),
        ("como alimentar a biomassa no biodigestor?", QuestionType.OPERACIONAL, "duvida_operacional"),
        
        # Tipo 2: METRICA
        ("quanta energia foi gerada?", QuestionType.METRICA, "pedido_energia"),
        ("quantos resíduos foram processados?", QuestionType.METRICA, "pedido_residuos"),
        ("quais são as métricas do biodigestor?", QuestionType.METRICA, "pedido_metricas"),
        
        # Tipo 3: LOCALIZACAO
        ("onde fica o biodigestor?", QuestionType.LOCALIZACAO, "pedido_endereco"),
        
        # Tipo 4: TRANSACIONAL
        ("agendar manutenção", QuestionType.TRANSACIONAL, "agendar_manutencao"),
        ("gera um pdf", QuestionType.TRANSACIONAL, "pedido_exportar_pdf"),
        ("exportar excel", QuestionType.TRANSACIONAL, "pedido_exportar_excel"),
        
        # Tipo 5: FORA_DE_ESCOPO / ININTELIGÍVEL
        ("asdfghjkl qwerty 123", QuestionType.FORA_DE_ESCOPO, "nao_compreendido"),
        ("quero comprar um pastel de queijo com refrigerante", QuestionType.FORA_DE_ESCOPO, "nao_compreendido"),
        ("qual a cor do cavalo branco de napoleao?", QuestionType.FORA_DE_ESCOPO, "nao_compreendido"),
        
        # Tipo 6: SOCIAL
        ("olá bom dia", QuestionType.SOCIAL, "saudacao"),
        ("muito obrigado até logo", QuestionType.SOCIAL, "despedida"),
    ]

    for q, exp_type, exp_intent in test_cases:
        req = ChatRequest(message=q, indicators=sample_indicators, markers=sample_markers)
        res = chatbot_endpoint(req)
        print(f"Pergunta: '{q[:35]:35}' -> Tipo: {res.question_type.value:12} | Intent: {res.intent:18} | Conf: {res.confidence:.4f}")
        assert res.question_type == exp_type, f"Para '{q}', esperado tipo {exp_type}, recebido {res.question_type}"
        assert res.intent == exp_intent, f"Para '{q}', esperado intent {exp_intent}, recebido {res.intent}"

    # Tipo 7: CONTEXTUAL_FOLLOWUP
    req_contextual = ChatRequest(
        message="e os resíduos?",
        indicators=sample_indicators,
        history=[ChatMessage(role="user", text="quanta energia foi gerada?"), ChatMessage(role="bot", text="Energia: 180.50 kWh")],
        context=ContextState(last_intent="pedido_energia", last_topic="energia", last_question_type=QuestionType.METRICA)
    )
    res_ctx = chatbot_endpoint(req_contextual)
    print(f"Pergunta: '{'e os resíduos?':35}' -> Tipo: {res_ctx.question_type.value:12} | Intent: {res_ctx.intent:18} | Conf: {res_ctx.confidence:.4f}")
    assert res_ctx.question_type == QuestionType.CONTEXTUAL_FOLLOWUP
    assert res_ctx.intent == "pedido_residuos"

    # Tipo 7.2: CONTEXTUAL_FOLLOWUP Operacional (continuidade de temperatura/parâmetros)
    req_temp_contextual = ChatRequest(
        message="E se esfriar demais?",
        context=ContextState(
            last_intent="duvida_operacional",
            last_topic="operacao",
            last_operational_topic="temperatura",
            last_question_type=QuestionType.OPERACIONAL,
        )
    )
    res_temp_ctx = chatbot_endpoint(req_temp_contextual)
    print(f"Pergunta: '{'E se esfriar demais?':35}' -> Tipo: {res_temp_ctx.question_type.value:12} | Intent: {res_temp_ctx.intent:18} | Conf: {res_temp_ctx.confidence:.4f}")
    assert res_temp_ctx.question_type == QuestionType.CONTEXTUAL_FOLLOWUP
    assert res_temp_ctx.intent == "duvida_operacional"
    assert "esfriar" in res_temp_ctx.response.lower() or "resfriamento" in res_temp_ctx.response.lower()

    print("[OK] Item 4 validado com sucesso!")


def test_item5_validar_respostas():
    print("\n=======================================================")
    print("CHECKLIST ITEM 5: Validar respostas")
    print("=======================================================")
    sample_indicators = [{"waste_processed": 985.4, "energy_generated": 142.2, "tax_savings": 210.0, "measured_at": "2026-02-15"}]
    
    # 5.1 Validação de resposta operacional com ação direcionada
    res_h2s = chatbot_endpoint(ChatRequest(message="o que fazer com alerta de h2s?"))
    assert "H2S" in res_h2s.response
    assert "Filtro" in res_h2s.response or "filtro" in res_h2s.response
    assert res_h2s.action == "view_alerts"
    assert any(s.value == "view_alerts" for s in res_h2s.suggestions)
    print("✓ Resposta operacional para H2S validada com direcionamento e ação correta.")

    # 5.2 Validação de resposta com dados reais contextualizados
    res_energia = chatbot_endpoint(ChatRequest(message="quanta energia foi gerada?", indicators=sample_indicators))
    assert "142.20 kWh" in res_energia.response
    assert res_energia.context.last_topic == "energia"
    print("✓ Resposta métrica validada com dados reais e atualização de contexto.")

    # 5.3 Validação de resposta para solicitação não compreendida (fallback estruturado)
    res_fallback = chatbot_endpoint(ChatRequest(message="abracadabra batata xyz 999"))
    assert "Não consegui compreender" in res_fallback.response
    assert res_fallback.suggestions is not None and len(res_fallback.suggestions) >= 4
    print("✓ Resposta de fallback validada com alternativas estruturadas.")

    print("[OK] Item 5 validado com sucesso!")


if __name__ == "__main__":
    test_item1_definir_tipos_de_perguntas()
    test_item2_definir_contexto_das_solicitacoes()
    test_item3_implementar_interpretacao()
    test_item4_testar_diferentes_perguntas()
    test_item5_validar_respostas()
    print("\n=======================================================")
    print("🎉 TODA A CHECKLIST FOI ATENDIDA E VALIDADA COM SUCESSO!")
    print("=======================================================")
