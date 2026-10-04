"""
BioDash AI Service
==================
Microserviço Python para PLN (Processamento de Linguagem Natural).
- Chatbot com TF-IDF + SVM (scikit-learn)
- Interpretação de Contexto e Histórico Multi-turn
- Direcionamento de Dúvidas Operacionais (H2S, pressão, pH, temperatura, alimentação, procedimentos)
- Tratamento de Fallback com Threshold de Confiança para solicitações não compreendidas
- Busca Semântica por biodigestores (similaridade de cosseno)
- Extração de Entidades (datas, números, prioridade)
- Extração de Entidades (datas, números, prioridade, tópicos operacionais)
- Fluxos conversacionais: agendamento, métricas, endereços, relatórios por período

Run: uvicorn main:app --host 0.0.0.0 --port 5000 --reload
"""

import os
import re
from enum import Enum
import tempfile
import threading
from pathlib import Path
from typing import Optional, List, Any, Dict

import httpx
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity

# Carrega variáveis de ambiente do .env
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

# Limiar mínimo de confiança do classificador SVM
CONFIDENCE_THRESHOLD = 0.25
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
MAX_AUDIO_BYTES = int(os.getenv("MAX_AUDIO_BYTES", str(20 * 1024 * 1024)))
WHISPER_INITIAL_PROMPT = os.getenv(
    "WHISPER_INITIAL_PROMPT",
    (
        "Conversa em português brasileiro sobre o sistema BioDash. "
        "Vocabulário: biodigestor, biodigestores, resíduos, energia, kWh, "
        "quilogramas, benefícios fiscais, métricas, manutenção, relatório, "
        "endereço, localização e CEP."
    ),
)

_whisper_model = None
_whisper_model_lock = threading.Lock()


def get_whisper_model():
    """Carrega o modelo apenas na primeira transcrição."""
    global _whisper_model
    if _whisper_model is None:
        with _whisper_model_lock:
            if _whisper_model is None:
                from faster_whisper import WhisperModel
                _whisper_model = WhisperModel(
                    WHISPER_MODEL,
                    device=WHISPER_DEVICE,
                    compute_type=WHISPER_COMPUTE_TYPE,
                )
    return _whisper_model


def transcribe_file(path: str) -> Dict[str, Any]:
    model = get_whisper_model()
    segments, info = model.transcribe(
        path,
        language="pt",
        task="transcribe",
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": 350,
            "speech_pad_ms": 300,
        },
        beam_size=5,
        best_of=5,
        patience=1.0,
        temperature=0.0,
        condition_on_previous_text=False,
        initial_prompt=WHISPER_INITIAL_PROMPT,
    )
    text = " ".join(segment.text.strip() for segment in segments).strip()
    return {
        "text": text,
        "language": info.language,
        "duration": round(float(info.duration), 2),
        "model": WHISPER_MODEL,
    }

# ──────────────────────────────────────────────────────────────────────────────
# APP SETUP
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="BioDash AI Service",
    description="Chatbot (TF-IDF + SVM), Interpretação de Contexto, Operação e Busca Semântica",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────────────────────
# 1. TREINAMENTO DO CHATBOT (TF-IDF + SVM)
# ──────────────────────────────────────────────────────────────────────────────

TRAINING_DATA = [
    # ─── SAUDAÇÃO ───────────────────────────────────────────────────────────
    ("saudacao", "olá"),
    ("saudacao", "oi"),
    ("saudacao", "bom dia"),
    ("saudacao", "boa tarde"),
    ("saudacao", "boa noite"),
    ("saudacao", "olá tudo bem"),
    ("saudacao", "ei"),
    ("saudacao", "oi tudo bem"),
    ("saudacao", "olá como vai"),
    ("saudacao", "oi biodash"),
    ("saudacao", "hey"),
    ("saudacao", "salve"),
    ("saudacao", "e aí"),
    ("saudacao", "oi oi"),
    ("saudacao", "olá tenho uma pergunta"),
    ("saudacao", "bom dia preciso de ajuda"),

    # ─── PEDIDO: Endereço ────────────────────────────────────────────────────
    ("pedido_endereco", "qual é o endereço do biodigestor"),
    ("pedido_endereco", "onde fica o biodigestor"),
    ("pedido_endereco", "me diz o endereço"),
    ("pedido_endereco", "localização do biodigestor"),
    ("pedido_endereco", "onde está o biodigestor"),
    ("pedido_endereco", "endereço"),
    ("pedido_endereco", "qual a localização"),
    ("pedido_endereco", "me mostra o endereço"),
    ("pedido_endereco", "qual o local do biodigestor"),
    ("pedido_endereco", "me informa o endereço do biodigestor"),
    ("pedido_endereco", "onde fica"),
    ("pedido_endereco", "localização"),
    ("pedido_endereco", "onde fica ele"),
    ("pedido_endereco", "qual o endereço dele"),

    # ─── PEDIDO: Métricas (Resíduos) ────────────────────────────────────────
    ("pedido_residuos", "quantos resíduos foram processados"),
    ("pedido_residuos", "me mostra os resíduos"),
    ("pedido_residuos", "qual a quantidade de resíduos"),
    ("pedido_residuos", "resíduos processados"),
    ("pedido_residuos", "quanto de resíduo"),
    ("pedido_residuos", "me informa os resíduos"),
    ("pedido_residuos", "resíduos do biodigestor"),
    ("pedido_residuos", "quantos kg de resíduos"),
    ("pedido_residuos", "processamento de resíduos"),
    ("pedido_residuos", "dados de resíduos"),
    ("pedido_residuos", "e os resíduos"),
    ("pedido_residuos", "e de resíduos"),
    ("pedido_residuos", "e quanto a resíduos"),
    ("pedido_residuos", "quanto de lixo orgânico"),

    # ─── PEDIDO: Métricas (Energia) ─────────────────────────────────────────
    ("pedido_energia", "quanta energia foi gerada"),
    ("pedido_energia", "me mostra a energia gerada"),
    ("pedido_energia", "energia gerada"),
    ("pedido_energia", "geração de energia"),
    ("pedido_energia", "quanto de energia"),
    ("pedido_energia", "energia do biodigestor"),
    ("pedido_energia", "me informa a energia"),
    ("pedido_energia", "quantos kwh"),
    ("pedido_energia", "energia em kwh"),
    ("pedido_energia", "dados de energia"),
    ("pedido_energia", "e a energia"),
    ("pedido_energia", "e quanto de energia"),
    ("pedido_energia", "e a geração"),

    # ─── PEDIDO: Métricas (Ambos) ────────────────────────────────────────────
    ("pedido_metricas", "quais são as métricas do biodigestor"),
    ("pedido_metricas", "me mostra as métricas"),
    ("pedido_metricas", "como está o biodigestor"),
    ("pedido_metricas", "status do biodigestor"),
    ("pedido_metricas", "dados do biodigestor"),
    ("pedido_metricas", "indicadores do biodigestor"),
    ("pedido_metricas", "me informa as métricas"),
    ("pedido_metricas", "relatório do biodigestor"),
    ("pedido_metricas", "energia e resíduos"),
    ("pedido_metricas", "como está funcionando o biodigestor"),
    ("pedido_metricas", "informações sobre o biodigestor"),
    ("pedido_metricas", "me mostra tudo sobre o biodigestor"),

    # ─── DÚVIDAS E PROCEDIMENTOS OPERACIONAIS ─────────────────────────────
    ("duvida_operacional", "como funciona a operação do biodigestor"),
    ("duvida_operacional", "como operar o biodigestor"),
    ("duvida_operacional", "procedimento operacional do biodigestor"),
    ("duvida_operacional", "operação do biodigestor"),
    ("duvida_operacional", "como funciona o biodigestor"),
    ("duvida_operacional", "como funciona a digestão anaeróbica"),
    ("duvida_operacional", "o que fazer com alerta de h2s"),
    ("duvida_operacional", "alerta de h2s"),
    ("duvida_operacional", "alerta de h2s no sistema"),
    ("duvida_operacional", "como tratar gás sulfídrico"),
    ("duvida_operacional", "filtro de h2s saturado"),
    ("duvida_operacional", "filtro de gas sulfidrico saturado"),
    ("duvida_operacional", "concentração de h2s muito alta"),
    ("duvida_operacional", "o que fazer se a pressão estiver alta"),
    ("duvida_operacional", "pressão alta no biodigestor"),
    ("duvida_operacional", "sobrepressão no tanque"),
    ("duvida_operacional", "válvula de alívio de pressão"),
    ("duvida_operacional", "pressão de biogás elevada"),
    ("duvida_operacional", "qual o ph ideal do biodigestor"),
    ("duvida_operacional", "qual a faixa de ph ideal"),
    ("duvida_operacional", "faixa de ph da digestão anaeróbica"),
    ("duvida_operacional", "ph da biomassa"),
    ("duvida_operacional", "temperatura ideal do biodigestor"),
    ("duvida_operacional", "temperatura ideal de operação"),
    ("duvida_operacional", "qual a temperatura ideal de operação do biodigestor"),
    ("duvida_operacional", "temperatura de operação da biomassa"),
    ("duvida_operacional", "temperatura recomendada"),
    ("duvida_operacional", "e se esfriar demais"),
    ("duvida_operacional", "e se a temperatura cair"),
    ("duvida_operacional", "o que acontece se esfriar"),
    ("duvida_operacional", "e se esfriar o biodigestor"),
    ("duvida_operacional", "biodigestor esfriou"),
    ("duvida_operacional", "queda de temperatura na biomassa"),
    ("duvida_operacional", "e se esquentar muito"),
    ("duvida_operacional", "e se a temperatura subir"),
    ("duvida_operacional", "e se passar de 40 graus"),
    ("duvida_operacional", "superaquecimento no biodigestor"),
    ("duvida_operacional", "choque térmico no biodigestor"),
    ("duvida_operacional", "e se o ph cair"),
    ("duvida_operacional", "e se o ph subir"),
    ("duvida_operacional", "e se acidificar o biodigestor"),
    ("duvida_operacional", "o que fazer se o ph baixar"),
    ("duvida_operacional", "e se a pressão subir"),
    ("duvida_operacional", "e se a pressão cair"),
    ("duvida_operacional", "e se a pressão passar do limite"),
    ("duvida_operacional", "o biodigestor está com cheiro forte"),
    ("duvida_operacional", "vazamento de biogás"),
    ("duvida_operacional", "e se o h2s subir"),
    ("duvida_operacional", "e se o filtro de gas saturar"),
    ("duvida_operacional", "e se passar do limite"),
    ("duvida_operacional", "e se sair da faixa ideal"),
    ("duvida_operacional", "como alimentar a biomassa"),
    ("duvida_operacional", "como alimentar a biomassa no biodigestor"),
    ("duvida_operacional", "alimentação de resíduos no biodigestor"),
    ("duvida_operacional", "taxa de carga orgânica"),
    ("duvida_operacional", "o biodigestor parou de produzir biogás"),
    ("duvida_operacional", "queda na produção de gás"),
    ("duvida_operacional", "acidificação do biodigestor"),
    ("duvida_operacional", "espumamento no biodigestor"),
    ("duvida_operacional", "como resolver incidente operacional"),
    ("duvida_operacional", "problema operacional na planta"),
    ("duvida_operacional", "procedimento de emergência do biodigestor"),
    ("duvida_operacional", "dúvida sobre a operação"),
    ("duvida_operacional", "como proceder com problema na operação"),
    ("duvida_operacional", "preciso de suporte operacional"),
    ("duvida_operacional", "contatar equipe de operações"),
    ("duvida_operacional", "suporte operacional"),
    ("duvida_operacional", "falar com suporte sobre o biodigestor"),
    ("duvida_operacional", "procedimento de segurança do biogás"),
    ("duvida_operacional", "como resolvo isso"),
    ("duvida_operacional", "o que devo fazer"),

    # ─── PEDIDO: Exportar PDF ────────────────────────────────────────────────
    ("pedido_exportar_pdf", "gera um pdf"),
    ("pedido_exportar_pdf", "exportar pdf"),
    ("pedido_exportar_pdf", "quero um relatório em pdf"),
    ("pedido_exportar_pdf", "me manda um pdf"),
    ("pedido_exportar_pdf", "gerar relatório pdf"),
    ("pedido_exportar_pdf", "baixar pdf"),
    ("pedido_exportar_pdf", "criar pdf"),

    # ─── PEDIDO: Exportar CSV ────────────────────────────────────────────────
    ("pedido_exportar_csv", "gera um csv"),
    ("pedido_exportar_csv", "exportar csv"),
    ("pedido_exportar_csv", "quero um arquivo csv"),
    ("pedido_exportar_csv", "me manda um csv"),
    ("pedido_exportar_csv", "gerar csv"),
    ("pedido_exportar_csv", "baixar csv"),
    ("pedido_exportar_csv", "criar csv"),

    # ─── PEDIDO: Exportar Excel ──────────────────────────────────────────────
    ("pedido_exportar_excel", "gera um excel"),
    ("pedido_exportar_excel", "exportar excel"),
    ("pedido_exportar_excel", "quero um arquivo excel"),
    ("pedido_exportar_excel", "me manda um excel"),
    ("pedido_exportar_excel", "gerar excel"),
    ("pedido_exportar_excel", "baixar excel"),
    ("pedido_exportar_excel", "criar excel"),
    ("pedido_exportar_excel", "planilha excel"),

    # ─── AGENDAR MANUTENÇÃO ──────────────────────────────────────────────────
    ("agendar_manutencao", "agendar manutenção"),
    ("agendar_manutencao", "quero agendar uma manutenção"),
    ("agendar_manutencao", "criar manutenção"),
    ("agendar_manutencao", "programar manutenção"),
    ("agendar_manutencao", "nova manutenção"),
    ("agendar_manutencao", "agendar revisão"),
    ("agendar_manutencao", "marcar manutenção"),
    ("agendar_manutencao", "preciso agendar manutenção"),
    ("agendar_manutencao", "quero criar um agendamento"),
    ("agendar_manutencao", "agendar serviço de manutenção"),
    ("agendar_manutencao", "programar revisão do biodigestor"),
    ("agendar_manutencao", "adicionar manutenção"),
    ("agendar_manutencao", "agendar troca de filtro"),
    ("agendar_manutencao", "quero marcar uma revisão"),

    # ─── INCLUIR MÉTRICA ────────────────────────────────────────────────────
    ("incluir_metrica", "adicionar métricas"),
    ("incluir_metrica", "registrar resíduos"),
    ("incluir_metrica", "incluir dados"),
    ("incluir_metrica", "inserir indicadores"),
    ("incluir_metrica", "adicionar dados do biodigestor"),
    ("incluir_metrica", "registrar métricas"),
    ("incluir_metrica", "incluir métricas"),
    ("incluir_metrica", "inserir métricas"),
    ("incluir_metrica", "adicionar indicadores"),
    ("incluir_metrica", "novo registro de métricas"),
    ("incluir_metrica", "inserir dados de energia e resíduos"),
    ("incluir_metrica", "quero adicionar as métricas do mês"),
    ("incluir_metrica", "registrar produção do biodigestor"),
    ("incluir_metrica", "lançar métricas"),

    # ─── EDITAR MÉTRICA ─────────────────────────────────────────────────────
    ("editar_metrica", "editar métricas"),
    ("editar_metrica", "atualizar resíduos"),
    ("editar_metrica", "corrigir dados"),
    ("editar_metrica", "alterar indicadores"),
    ("editar_metrica", "modificar métricas"),
    ("editar_metrica", "quero editar as métricas"),
    ("editar_metrica", "preciso corrigir os dados"),
    ("editar_metrica", "alterar os dados do mês"),
    ("editar_metrica", "atualizar métricas"),
    ("editar_metrica", "editar indicadores"),
    ("editar_metrica", "corrigir métricas"),
    ("editar_metrica", "modificar dados do biodigestor"),
    ("editar_metrica", "atualizar dados do mês passado"),

    # ─── ADICIONAR ENDEREÇO ─────────────────────────────────────────────────
    ("adicionar_endereco", "adicionar biodigestor"),
    ("adicionar_endereco", "cadastrar endereço"),
    ("adicionar_endereco", "novo biodigestor"),
    ("adicionar_endereco", "registrar localização"),
    ("adicionar_endereco", "adicionar localização"),
    ("adicionar_endereco", "cadastrar biodigestor"),
    ("adicionar_endereco", "quero adicionar um biodigestor"),
    ("adicionar_endereco", "incluir novo biodigestor"),
    ("adicionar_endereco", "registrar biodigestor"),
    ("adicionar_endereco", "novo ponto no mapa"),
    ("adicionar_endereco", "adicionar ponto no mapa"),
    ("adicionar_endereco", "quero cadastrar um novo biodigestor"),
    ("adicionar_endereco", "incluir endereço no mapa"),

    # ─── RELATÓRIO POR PERÍODO ───────────────────────────────────────────────
    ("relatorio_periodo", "relatório de outubro"),
    ("relatorio_periodo", "relatório do mês"),
    ("relatorio_periodo", "exportar período"),
    ("relatorio_periodo", "relatório entre datas"),
    ("relatorio_periodo", "relatório por período"),
    ("relatorio_periodo", "relatório de um período específico"),
    ("relatorio_periodo", "quero relatório de um período"),
    ("relatorio_periodo", "gerar relatório do período"),
    ("relatorio_periodo", "relatório de janeiro a junho"),
    ("relatorio_periodo", "dados do semestre"),
    ("relatorio_periodo", "relatório mensal"),
    ("relatorio_periodo", "dados do período"),
    ("relatorio_periodo", "exportar dados de um período"),
    ("relatorio_periodo", "relatório do trimestre"),
    ("relatorio_periodo", "relatório específico de um período"),
    ("relatorio_periodo", "e no mês passado"),
    ("relatorio_periodo", "e no mês anterior"),

    # ─── CONFIRMAR ──────────────────────────────────────────────────────────
    ("confirmar", "sim"),
    ("confirmar", "s"),
    ("confirmar", "yes"),
    ("confirmar", "confirmar"),
    ("confirmar", "confirmo"),
    ("confirmar", "ok"),
    ("confirmar", "pode ser"),
    ("confirmar", "certo"),
    ("confirmar", "correto"),
    ("confirmar", "isso mesmo"),
    ("confirmar", "exato"),
    ("confirmar", "com certeza"),
    ("confirmar", "pode"),
    ("confirmar", "claro"),
    ("confirmar", "tá bom"),

    # ─── CANCELAR ───────────────────────────────────────────────────────────
    ("cancelar", "não"),
    ("cancelar", "nao"),
    ("cancelar", "n"),
    ("cancelar", "no"),
    ("cancelar", "cancelar"),
    ("cancelar", "cancela"),
    ("cancelar", "desistir"),
    ("cancelar", "para"),
    ("cancelar", "chega"),
    ("cancelar", "voltar"),
    ("cancelar", "esqueça"),
    ("cancelar", "esqueça isso"),
    ("cancelar", "não quero mais"),
    ("cancelar", "para tudo"),

    # ─── DESPEDIDA ───────────────────────────────────────────────────────────
    ("despedida", "tchau"),
    ("despedida", "até logo"),
    ("despedida", "até mais"),
    ("despedida", "obrigado"),
    ("despedida", "valeu"),
    ("despedida", "encerrar"),
    ("despedida", "finalizar"),
    ("despedida", "até"),
    ("despedida", "muito obrigado"),
    ("despedida", "obrigada"),
    ("despedida", "agradeço"),
    ("despedida", "encerrando"),
    ("despedida", "foi ótimo obrigado"),
    ("despedida", "tudo certo obrigado"),
]

# Separa labels e frases
train_labels, train_texts = zip(*TRAINING_DATA)

# Treina o TF-IDF (char_wb com uni e bi-gramas — robusto para português)
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=1,
    analyzer='char_wb',
    sublinear_tf=True
)
X_train = vectorizer.fit_transform(train_texts)

# Treina o SVM
svm_model = SVC(kernel='linear', C=1.0, probability=True)
svm_model.fit(X_train, train_labels)

print("[OK] Modelo TF-IDF + SVM v2.0 treinado com sucesso!")
print(f"   Classes: {list(svm_model.classes_)}")
print("[OK] Modelo TF-IDF + SVM v2.1 treinado com sucesso!")
print(f"   Classes ({len(svm_model.classes_)}): {list(svm_model.classes_)}")
print(f"   Total de exemplos: {len(TRAINING_DATA)}")

def normalize_text(text: str) -> str:
    """Normaliza texto: minúsculas, remove acentos básicos."""
    text = text.lower().strip()
    replacements = {
        'á': 'a', 'à': 'a', 'ã': 'a', 'â': 'a',
        'é': 'e', 'ê': 'e', 'è': 'e',
        'í': 'i', 'î': 'i', 'ì': 'i',
        'ó': 'o', 'ô': 'o', 'õ': 'o', 'ò': 'o',
        'ú': 'u', 'û': 'u', 'ù': 'u',
        'ç': 'c', 'ñ': 'n',
    }
    for accented, plain in replacements.items():
        text = text.replace(accented, plain)
    return text

# ─── VOCABULÁRIO DE DOMÍNIO PARA VALIDAÇÃO DE ESCOPO ─────────────────────────
STOPWORDS = {
    'a', 'o', 'as', 'os', 'um', 'uma', 'uns', 'umas', 'de', 'do', 'da', 'dos', 'das',
    'em', 'no', 'na', 'nos', 'nas', 'por', 'para', 'com', 'e', 'ou', 'que', 'se', 'me',
    'te', 'eu', 'voce', 'qual', 'quais', 'quero', 'gostaria', 'preciso', 'favor'
}

DOMAIN_VOCABULARY = set()
for _text in train_texts:
    for _tok in re.findall(r'\b\w+\b', normalize_text(_text)):
        if _tok not in STOPWORDS and len(_tok) > 1:
            DOMAIN_VOCABULARY.add(_tok)

# ──────────────────────────────────────────────────────────────────────────────
# 2. DEFINIÇÃO DE TIPOS DE PERGUNTAS E CONTEXTO (Schemas)
# ──────────────────────────────────────────────────────────────────────────────

class QuestionType(str, Enum):
    OPERACIONAL = "operacional"          # Parâmetros técnicos, segurança, H2S, pressão, pH, temperatura
    METRICA = "metrica"                  # Resíduos, energia, indicadores consolidados
    LOCALIZACAO = "localizacao"          # Endereços e plantas no mapa
    TRANSACIONAL = "transacional"        # Ações, agendamentos, exportação de arquivos
    CONTEXTUAL_FOLLOWUP = "contextual"   # Continuações de assunto ("e os resíduos?", "como resolvo?")
    FORA_DE_ESCOPO = "fora_de_escopo"    # Perguntas ininteligíveis ou fora de domínio
    SOCIAL = "social"                    # Saudações, despedidas, confirmações, cancelamentos


class ChatMessage(BaseModel):
    role: str  # "user" | "bot"
    text: str


class ContextState(BaseModel):
    last_intent: Optional[str] = None
    last_topic: Optional[str] = None  # "energia" | "residuos" | "operacao" | "manutencao" | "endereco"
    last_operational_topic: Optional[str] = None  # "temperatura" | "ph" | "h2s" | "pressao" | "alimentacao" | "seguranca"
    last_question_type: Optional[QuestionType] = None
    selected_biodigestor: Optional[str] = None
    period: Optional[str] = None


class QuickSuggestion(BaseModel):
    label: str
    action_type: str  # "message" | "action"
    value: str        # texto da mensagem ou identificador da ação


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = []
    context: Optional[ContextState] = None
    markers: Optional[List[Dict[str, Any]]] = []
    indicators: Optional[List[Dict[str, Any]]] = []


class ChatResponse(BaseModel):
    intent: str
    question_type: QuestionType = QuestionType.OPERACIONAL
    response: str
    confidence: float
    action: Optional[str] = None
    entities: Optional[Dict[str, Any]] = {}
    suggestions: Optional[List[QuickSuggestion]] = []
    context: Optional[ContextState] = None


class SemanticSearchRequest(BaseModel):
    query: str
    markers: List[Dict[str, Any]]


class SemanticSearchResponse(BaseModel):
    results: List[Dict[str, Any]]


# ──────────────────────────────────────────────────────────────────────────────
# 3. HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Normaliza texto: minúsculas, remove acentos básicos."""
    text = text.lower().strip()
    replacements = {
        'á': 'a', 'à': 'a', 'ã': 'a', 'â': 'a',
        'é': 'e', 'ê': 'e', 'è': 'e',
        'í': 'i', 'î': 'i', 'ì': 'i',
        'ó': 'o', 'ô': 'o', 'õ': 'o', 'ò': 'o',
        'ú': 'u', 'û': 'u', 'ù': 'u',
        'ç': 'c', 'ñ': 'n',
    }
    for accented, plain in replacements.items():
        text = text.replace(accented, plain)
    return text


def extract_entities(text: str) -> Dict[str, Any]:
    """
    Extrai entidades do texto em linguagem natural:
    - Meses (por nome ou número)
    - Ano (4 dígitos)
    - Dia
    - Prioridade (alta/média/baixa)
    - Métricas numéricas (kg, kWh, R$)
    - Formato de exportação (pdf/csv/excel)
    - Tópicos operacionais (h2s, pressão, temperatura, ph, alimentação, segurança)
    """
    entities: Dict[str, Any] = {}
    normalized = normalize_text(text)

    # ─── Tópicos Operacionais ────────────────────────────────────────────────
    if any(w in normalized for w in ['h2s', 'sulfidrico', 'gas sulfidrico', 'dessulfurizacao', 'cheiro de ovo']):
        entities['operational_topic'] = 'h2s'
    elif any(w in normalized for w in ['pressao', 'sobrepressao', 'valvula', 'despressurizacao', 'gasometro', 'estufar', 'alivio']):
        entities['operational_topic'] = 'pressao'
    elif any(w in normalized for w in ['temperatura', 'calor', 'graus', 'termica', 'termico', 'esfriar', 'esfria', 'esfriou', 'esfriamento', 'esquentar', 'esquenta', 'aquecer', 'aquecimento', 'congelar', 'resfriar', 'resfriamento', 'choque termico']):
        entities['operational_topic'] = 'temperatura'
    elif any(w in normalized for w in ['ph', 'acidez', 'acidificacao', 'alcalinidade', 'acido', 'acidificar', 'acidose', 'alcalino', 'alcalinizacao', 'amonia']):
        entities['operational_topic'] = 'ph'
    elif any(w in normalized for w in ['alimentacao', 'biomassa', 'carga organica', 'alimentar', 'sobrecarga', 'substrato', 'esterco', 'dejeto']):
        entities['operational_topic'] = 'alimentacao'
    elif any(w in normalized for w in ['seguranca', 'vazamento', 'cheiro', 'odor', 'emergencia']):
        entities['operational_topic'] = 'seguranca'
    elif any(w in normalized for w in ['operacao', 'operar', 'funcionamento', 'procedimento', 'incidente']):
        entities['operational_topic'] = 'operacao_geral'

    # ─── Meses ───────────────────────────────────────────────────────────────
    month_map = {
        'janeiro': 0, 'fevereiro': 1, 'marco': 2, 'abril': 3,
        'maio': 4, 'junho': 5, 'julho': 6, 'agosto': 7,
        'setembro': 8, 'outubro': 9, 'novembro': 10, 'dezembro': 11,
    }
    for name, idx in month_map.items():
        if name in normalized:
            entities['month'] = idx
            break

    # ─── Ano (4 dígitos) ─────────────────────────────────────────────────────
    year_match = re.search(r'\b(20\d{2})\b', text)
    if year_match:
        entities['year'] = int(year_match.group(1))

    # ─── Data no formato dd/mm ou dd/mm/aaaa ─────────────────────────────────
    date_match = re.search(r'(\d{1,2})[/\-](\d{1,2})(?:[/\-](\d{2,4}))?', text)
    if date_match:
        entities['day'] = int(date_match.group(1))
        entities['month'] = int(date_match.group(2)) - 1  # 0-indexed
        if date_match.group(3):
            yr = int(date_match.group(3))
            entities['year'] = yr if yr > 100 else 2000 + yr

    # ─── Dia isolado ─────────────────────────────────────────────────────────
    if 'day' not in entities:
        day_match = re.search(r'\bdia\s+(\d{1,2})\b|\b(\d{1,2})\s+de\b', normalized)
        if day_match:
            d = int(day_match.group(1) or day_match.group(2))
            if 1 <= d <= 31:
                entities['day'] = d

    # ─── Prioridade ──────────────────────────────────────────────────────────
    if any(w in normalized for w in ['alta', 'urgente', 'critica', 'critico', 'importante']):
        entities['priority'] = 'high'
    elif any(w in normalized for w in ['media', 'moderada', 'normal']):
        entities['priority'] = 'medium'
    elif any(w in normalized for w in ['baixa', 'leve', 'pequena']):
        entities['priority'] = 'low'

    # ─── Resíduos (kg) ───────────────────────────────────────────────────────
    waste_match = re.search(r'(\d+(?:[.,]\d+)?)\s*(?:kg|quilos?)', normalized)
    if waste_match:
        entities['waste_processed'] = float(waste_match.group(1).replace(',', '.'))

    # ─── Energia (kWh) ───────────────────────────────────────────────────────
    energy_match = re.search(r'(\d+(?:[.,]\d+)?)\s*(?:kwh|kw/h|kilowatt)', normalized)
    if energy_match:
        entities['energy_generated'] = float(energy_match.group(1).replace(',', '.'))

    # ─── Valor monetário (R$) ────────────────────────────────────────────────
    money_match = re.search(
        r'(?:r\$|reais?|brl)\s*(\d+(?:[.,]\d+)?)|(\d+(?:[.,]\d+)?)\s*(?:reais?)',
        normalized
    )
    if money_match:
        val = money_match.group(1) or money_match.group(2)
        entities['tax_savings'] = float(val.replace(',', '.'))

    # ─── Formato de exportação ───────────────────────────────────────────────
    if 'pdf' in normalized:
        entities['format'] = 'pdf'
    elif 'excel' in normalized or 'xlsx' in normalized or 'planilha' in normalized:
        entities['format'] = 'excel'
    elif 'csv' in normalized:
        entities['format'] = 'csv'

    return entities


def build_marker_text(marker: Dict[str, Any]) -> str:
    """Cria uma string de texto searchable para um marcador."""
    parts = [marker.get("title", "")]
    addr = marker.get("address", {})
    if isinstance(addr, dict):
        parts.extend([
            addr.get("street", ""), addr.get("cep", ""),
            addr.get("city", ""), addr.get("complement", ""),
        ])
    elif isinstance(addr, str):
        parts.append(addr)
    desc = marker.get("description", "")
    if desc:
        parts.append(desc)
    return " ".join(filter(None, parts))


def format_marker_address(marker: Dict[str, Any]) -> str:
    """Formata o endereço de um marcador para exibição."""
    addr = marker.get("address", {})
    title = marker.get("title", "Biodigestor")
    if isinstance(addr, dict):
        street = addr.get("street", "")
        number = addr.get("number", "")
        cep = addr.get("cep", "")
        complement = addr.get("complement", "")
        city = addr.get("city", "")
        parts = []
        if street:
            parts.append(f"{street}{', ' + number if number else ''}")
        if complement:
            parts.append(complement)
        if cep:
            parts.append(f"CEP: {cep}")
        if city:
            parts.append(city)
        address_str = "\n".join(parts) if parts else marker.get("description", "Endereço não informado")
        return f"📍 *{title}*\n{address_str}"
    desc = marker.get("description", "Endereço não informado")
    return f"📍 *{title}*\n{desc}"


def format_indicators(indicators: List[Dict[str, Any]]) -> str:
    """Formata as métricas mais recentes para exibição."""
    if not indicators:
        return "Nenhuma métrica disponível. Diga *'adicionar métricas'* para registrar dados pelo chat."
    latest = indicators[0]
    waste = latest.get("waste_processed", 0)
    energy = latest.get("energy_generated", 0)
    tax = latest.get("tax_savings", 0)
    return (
        f"📊 *Métricas do Biodigestor*\n"
        f"• Resíduos Processados: {waste:.2f} kg\n"
        f"• Energia Gerada: {energy:.2f} kWh\n"
        f"• Benefícios Fiscais: R$ {tax:.2f}"
    )


def build_default_fallback_response(confidence: float, entities: Dict[str, Any], context: ContextState) -> ChatResponse:
    """Gera uma resposta amigável e estruturada quando a solicitação não puder ser interpretada."""
    response = (
        "Não consegui compreender a sua solicitação com clareza. 🤔\n\n"
        "Posso te orientar nos seguintes temas:\n"
        "• ⚙️ **Operação**: dúvidas sobre biodigestores, H2S, pressão, pH e procedimentos\n"
        "• 📊 **Métricas**: resíduos processados e energia gerada\n"
        "• 📍 **Biodigestores**: endereços e localização das plantas\n"
        "• 🛠️ **Manutenções**: agendamento e acompanhamento\n"
        "• 📄 **Relatórios**: emissão em PDF, CSV ou Excel\n"
        "• 📞 **Suporte**: contato com a equipe técnica operacional"
    )
    suggestions = [
        QuickSuggestion(label="⚙️ Dúvidas de Operação", action_type="message", value="Como funciona a operação do biodigestor?"),
        QuickSuggestion(label="⚡ Energia Gerada", action_type="message", value="Quanta energia foi gerada?"),
        QuickSuggestion(label="♻️ Resíduos Processados", action_type="message", value="Quantos resíduos foram processados?"),
        QuickSuggestion(label="📍 Onde fica?", action_type="message", value="Onde fica o biodigestor?"),
        QuickSuggestion(label="🛠️ Agendar Manutenção", action_type="message", value="Agendar manutenção"),
        QuickSuggestion(label="📞 Suporte Operacional", action_type="action", value="contact_support"),
    ]
    if context:
        context.last_question_type = QuestionType.FORA_DE_ESCOPO
    return ChatResponse(
        intent="nao_compreendido",
        question_type=QuestionType.FORA_DE_ESCOPO,
        response=response,
        confidence=round(confidence, 4),
        action=None,
        entities=entities,
        suggestions=suggestions,
        context=context,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 4. ENDPOINTS
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"service": "BioDash AI Service", "status": "running", "version": "2.1.0"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """Transcreve áudio gravado pelo browser usando Whisper local."""
    allowed_prefixes = ("audio/", "video/webm")
    content_type = (audio.content_type or "").lower()
    if content_type and not content_type.startswith(allowed_prefixes):
        raise HTTPException(status_code=415, detail="Formato de áudio não suportado.")

    contents = await audio.read(MAX_AUDIO_BYTES + 1)
    await audio.close()
    if not contents:
        raise HTTPException(status_code=400, detail="O arquivo de áudio está vazio.")
    if len(contents) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="O áudio excede o limite permitido.")

    suffix = Path(audio.filename or "voice-message.webm").suffix.lower()
    if suffix not in {".webm", ".ogg", ".mp4", ".m4a", ".wav", ".mpeg", ".mp3"}:
        suffix = ".webm"

    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_file.write(contents)
            temp_path = temp_file.name
        result = await run_in_threadpool(transcribe_file, temp_path)
        if not result["text"]:
            raise HTTPException(status_code=422, detail="Nenhuma fala foi reconhecida.")
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Falha ao transcrever áudio: {exc}") from exc
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)


@app.get("/biodigestores")
def get_biodigestores(user_id: str = Query(..., description="UUID do usuário autenticado")):
    """
    Busca os biodigestores cadastrados para um usuário diretamente do Supabase.
    Consulta a tabela `biodigestor_maps` (id, user_id, address json, created_at).
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise HTTPException(status_code=500, detail="SUPABASE_URL ou SUPABASE_SERVICE_ROLE_KEY não configurados.")

    url = f"{SUPABASE_URL}/rest/v1/biodigestor_maps"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }
    params = {
        "user_id": f"eq.{user_id}",
        "order": "created_at.desc",
        "select": "id,user_id,address,created_at",
    }

    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(url, headers=headers, params=params)

        if resp.status_code != 200:
            raise HTTPException(
                status_code=resp.status_code,
                detail=f"Erro ao consultar Supabase: {resp.text}"
            )

        rows = resp.json()
        biodigestores = [
            {
                "id": str(row["id"]),
                "user_id": row["user_id"],
                "title": (row.get("address") or {}).get("title", "Biodigestor"),
                "latitude": (row.get("address") or {}).get("latitude", -14.235),
                "longitude": (row.get("address") or {}).get("longitude", -51.925),
                "description": (row.get("address") or {}).get("description", ""),
                "address": row.get("address") or {},
                "created_at": row["created_at"],
            }
            for row in rows
        ]

        return {"success": True, "data": biodigestores, "total": len(biodigestores)}

    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=f"Erro de conexão com Supabase: {exc}")


@app.post("/chatbot", response_model=ChatResponse)
def chatbot_endpoint(req: ChatRequest):
    """
    Classifica a intenção com TF-IDF + SVM, extrai entidades e retorna
    resposta contextualizada. Suporta fluxos conversacionais multi-etapa.
    Classifica a solicitação com TF-IDF + SVM, analisa contexto histórico/multi-turn,
    direciona procedimentos operacionais e retorna respostas contextualizadas ou fallback estruturado.
    """
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="Mensagem não pode ser vazia.")

    normalized = normalize_text(req.message)
    context = req.context or ContextState()
    history = req.history or []
    entities = extract_entities(req.message)
    markers = req.markers or []
    indicators = req.indicators or []

    # ─── Classificação com TF-IDF + SVM ──────────────────────────────────────
    X_input = vectorizer.transform([normalized])

    probs = svm_model.predict_proba(X_input)[0]
    classes = svm_model.classes_
    best_idx = int(np.argmax(probs))
    intent = classes[best_idx]
    confidence = float(probs[best_idx])

    markers = req.markers or []
    indicators = req.indicators or []
    # ─── Análise de Contexto Multi-Turn (perguntas elípticas ou de continuidade) ───
    # Ex: "e os resíduos?", "e a energia?", "e no mês passado?", "como resolvo isso?"
    is_elliptical = bool(re.search(r'\b(e\s+os?|e\s+as?|e\s+quanto|e\s+no|e\s+na|como\s+resolvo|como\s+proceder|o\s+que\s+faco|disso|dele)\b', normalized))
    # Ex: "e os resíduos?", "e a energia?", "e no mês passado?", "como resolvo isso?", "e se esfriar demais?"
    is_elliptical = bool(re.search(
        r'\b(e\s+se|e\s+os?|e\s+as?|e\s+quanto|e\s+no|e\s+na|como\s+resolvo|como\s+proceder|o\s+que\s+faco|o\s+que\s+acontece|o\s+que\s+ocorre|se\s+esfriar|se\s+esquentar|se\s+subir|se\s+cair|disso|dele|dela|nisso)\b',
        normalized
    ))

    if is_elliptical or confidence < CONFIDENCE_THRESHOLD:
        is_op_context = (
            context.last_intent == "duvida_operacional"
            or context.last_topic == "operacao"
            or bool(context.last_operational_topic)
        )
        op_followup_terms = [
            'esfriar', 'esfria', 'esfriou', 'esfriamento', 'esquentar', 'esquenta', 'esquentou',
            'aquecer', 'aquecimento', 'subir', 'sobe', 'cair', 'cai', 'descer', 'aumentar', 'diminuir',
            'passar', 'limite', 'faixa', 'choque', 'travar', 'vazar', 'cheiro', 'acido',
            'acidificar', 'alcalino', 'amonia', 'pressao', 'h2s', 'temperatura', 'ph',
            'resolver', 'proceder', 'ajuda', 'fazer'
        ]

        # Se for dúvida/continuação operacional vinculada ao contexto anterior
        if is_op_context and (is_elliptical or any(w in normalized for w in op_followup_terms)):
            intent = "duvida_operacional"
            confidence = max(confidence, 0.88)
            if not entities.get('operational_topic'):
                entities['operational_topic'] = (
                    context.last_operational_topic
                    or (context.last_topic if context.last_topic != "operacao" else "operacao_geral")
                )

        # Se for pergunta sobre resíduos em continuidade
        elif any(w in normalized for w in ['residuo', 'residuos', 'lixo', 'biomassa']):
            if 'alimentacao' not in entities.get('operational_topic', ''):
                intent = "pedido_residuos"
                confidence = max(confidence, 0.85)

        # Se for pergunta sobre energia em continuidade
        elif any(w in normalized for w in ['energia', 'kwh', 'geracao']):
            intent = "pedido_energia"
            confidence = max(confidence, 0.85)

        # Se for pergunta de período ("e no mês passado?")
        elif any(w in normalized for w in ['mes passado', 'anterior', 'ultimo mes']):
            intent = "relatorio_periodo"
            confidence = max(confidence, 0.80)

        # Se for pergunta de resolução/procedimento operacional
        elif any(w in normalized for w in ['como resolvo', 'o que faco', 'como proceder', 'ajuda com isso']):
            intent = "duvida_operacional"
            confidence = max(confidence, 0.85)
            if not entities.get('operational_topic') and context.last_topic:
                entities['operational_topic'] = context.last_topic
            if not entities.get('operational_topic'):
                entities['operational_topic'] = context.last_operational_topic or context.last_topic or 'operacao_geral'

    # ─── Verificação de Escopo e Threshold para Solicitações Não Compreendidas ────────
    tokens = [tok for tok in re.findall(r'\b\w+\b', normalized) if tok not in STOPWORDS]
    matches = [tok for tok in tokens if tok in DOMAIN_VOCABULARY]
    is_short_intent = normalized in ['sim', 's', 'nao', 'n', 'ok', 'oi', 'ola', 'ei', 'tchau', 'obrigado', 'valeu']
    has_domain_keywords = len(matches) > 0 or is_short_intent
    has_active_context = bool(context and (context.last_intent or context.last_topic or context.last_operational_topic))
    has_domain_keywords = len(matches) > 0 or is_short_intent or (is_elliptical and has_active_context)

    # Se a mensagem não contiver palavras do domínio ou a confiança for insuficiente
    if not has_domain_keywords or confidence < CONFIDENCE_THRESHOLD:
        return build_default_fallback_response(confidence, entities, context)

    # ─── Construção de Resposta Contextualizada ──────────────────────────────
    action = None
    preserved_op_topic = entities.get('operational_topic')
    entities = extract_entities(req.message)
    if preserved_op_topic and not entities.get('operational_topic'):
        entities['operational_topic'] = preserved_op_topic
    suggestions: List[QuickSuggestion] = []

    # ─── Respostas por intenção ──────────────────────────────────────────────

    if intent == "saudacao":
        response = (
            "Olá! Em que posso ajudar você hoje? 😊\n\n"
            "Posso:\n"
            "• Responder sobre endereços e métricas\n"
            "• Agendar manutenções\n"
            "• Registrar ou editar métricas\n"
            "• Cadastrar novos biodigestores\n"
            "• Gerar relatórios por período"
            "• Esclarecer dúvidas operacionais (pH, H2S, pressão e temperatura)\n"
            "• Responder sobre endereços e métricas (energia e resíduos)\n"
            "• Agendar manutenções preventivas ou corretivas\n"
            "• Registrar ou editar dados de produção\n"
            "• Gerar relatórios em PDF, CSV ou Excel"
        )
        suggestions = [
            QuickSuggestion(label="⚙️ Dúvidas de Operação", action_type="message", value="Como funciona a operação do biodigestor?"),
            QuickSuggestion(label="⚡ Energia Gerada", action_type="message", value="Quanta energia foi gerada?"),
            QuickSuggestion(label="📍 Onde fica?", action_type="message", value="Onde fica o biodigestor?"),
            QuickSuggestion(label="📄 Gerar Relatório", action_type="message", value="Gerar relatório PDF"),
        ]

    elif intent == "pedido_endereco":
        context.last_topic = "endereco"
        context.last_intent = "pedido_endereco"
        if not markers:
            response = "Não encontrei nenhum biodigestor cadastrado.\n\nDiga *'adicionar biodigestor'* para cadastrar um agora pelo chat."
            suggestions = [
                QuickSuggestion(label="➕ Adicionar Biodigestor", action_type="message", value="Adicionar biodigestor"),
            ]
        elif len(markers) == 1:
            response = format_marker_address(markers[0])
            context.selected_biodigestor = markers[0].get("title")
            suggestions = [
                QuickSuggestion(label="📊 Ver Métricas", action_type="message", value="Quais são as métricas do biodigestor?"),
                QuickSuggestion(label="⚙️ Operação da Planta", action_type="message", value="Como funciona a operação do biodigestor?"),
            ]
        else:
            addresses = "\n\n".join([format_marker_address(m) for m in markers[:5]])
            response = f"Encontrei {len(markers)} biodigestores cadastrados:\n\n{addresses}"
            suggestions = [
                QuickSuggestion(label="📊 Ver Métricas", action_type="message", value="Quais são as métricas do biodigestor?"),
                QuickSuggestion(label="🛠️ Manutenções", action_type="message", value="Agendar manutenção"),
            ]

    elif intent == "pedido_residuos":
        context.last_topic = "residuos"
        context.last_intent = "pedido_residuos"
        if not indicators:
            response = "Nenhuma métrica de resíduos encontrada.\n\nDiga *'adicionar métricas'* para registrar agora pelo chat."
            suggestions = [
                QuickSuggestion(label="➕ Registrar Métricas", action_type="message", value="Adicionar métricas"),
            ]
        else:
            latest = indicators[0]
            waste = latest.get("waste_processed", 0)
            response = f"♻️ *Resíduos Processados*\nÚltimo registro: *{waste:.2f} kg*"
            context_prefix = ""
            if context.selected_biodigestor:
                context_prefix = f"Para a unidade *{context.selected_biodigestor}*:\n"
            response = f"{context_prefix}♻️ *Resíduos Processados*\nÚltimo registro: *{waste:.2f} kg*"
            suggestions = [
                QuickSuggestion(label="⚡ Ver Energia", action_type="message", value="E a energia gerada?"),
                QuickSuggestion(label="📄 Gerar PDF", action_type="message", value="Gerar relatório PDF"),
                QuickSuggestion(label="📊 Resumo Completo", action_type="message", value="Métricas do biodigestor"),
            ]

    elif intent == "pedido_energia":
        context.last_topic = "energia"
        context.last_intent = "pedido_energia"
        if not indicators:
            response = "Nenhuma métrica de energia encontrada.\n\nDiga *'adicionar métricas'* para registrar agora pelo chat."
            suggestions = [
                QuickSuggestion(label="➕ Registrar Métricas", action_type="message", value="Adicionar métricas"),
            ]
        else:
            latest = indicators[0]
            energy = latest.get("energy_generated", 0)
            response = f"⚡ *Energia Gerada*\nÚltimo registro: *{energy:.2f} kWh*"
            context_prefix = ""
            if context.selected_biodigestor:
                context_prefix = f"Para a unidade *{context.selected_biodigestor}*:\n"
            response = f"{context_prefix}⚡ *Energia Gerada*\nÚltimo registro: *{energy:.2f} kWh*"
            suggestions = [
                QuickSuggestion(label="♻️ Ver Resíduos", action_type="message", value="E os resíduos?"),
                QuickSuggestion(label="📄 Gerar PDF", action_type="message", value="Gerar relatório PDF"),
                QuickSuggestion(label="📊 Resumo Completo", action_type="message", value="Métricas do biodigestor"),
            ]

    elif intent == "pedido_metricas":
        context.last_topic = "metricas"
        context.last_intent = "pedido_metricas"
        response = format_indicators(indicators)
        suggestions = [
            QuickSuggestion(label="📄 Gerar Relatório PDF", action_type="message", value="Gerar relatório PDF"),
            QuickSuggestion(label="📊 Exportar Planilha", action_type="message", value="Exportar Excel"),
            QuickSuggestion(label="⚙️ Dúvida de Operação", action_type="message", value="Como funciona a operação?"),
        ]

    # ─── DIRECIONAMENTO OPERACIONAL ──────────────────────────────────────────
    elif intent == "duvida_operacional":
        op_topic = entities.get('operational_topic') or context.last_operational_topic or 'operacao_geral'
        context.last_topic = "operacao"
        context.last_operational_topic = op_topic
        context.last_intent = "duvida_operacional"
        op_topic = entities.get('operational_topic', 'operacao_geral')

        if op_topic == 'h2s':
            response = (
                "⚠️ **Procedimento Operacional — Alerta de H2S (Gás Sulfídrico)**\n\n"
                "O H2S é um gás tóxico e corrosivo presente no biogás. Ações operacionais recomendadas:\n"
                "1. **Segurança**: Mantenha a área ventilada e utilize EPIs adequados para proteção respiratória.\n"
                "2. **Filtro de Dessulfurização**: Verifique o filtro de carvão ativado ou esponja de ferro. Níveis altos indicam saturação do filtro.\n"
                "3. **Monitoramento**: Acompanhe o sensor de alerta na aba Painel.\n"
                "4. **Manutenção**: Caso persista elevado, agende a troca de filtro imediatamente."
            )
            action = "view_alerts"
            suggestions = [
                QuickSuggestion(label="🚨 Ver Alertas no Painel", action_type="action", value="view_alerts"),
                QuickSuggestion(label="🛠️ Agendar Troca de Filtro", action_type="message", value="Agendar troca de filtro"),
                QuickSuggestion(label="📞 Falar com Suporte", action_type="action", value="contact_support"),
            ]

        elif op_topic == 'pressao':
            response = (
                "⚡ **Procedimento Operacional — Pressão do Biogás**\n\n"
                "Instruções para controle de pressão na tubulação e gasômetro:\n"
                "1. **Válvula de Alívio**: Verifique se a válvula de segurança/selo hidráulico está desobstruída.\n"
                "2. **Consumo de Gás**: Certifique-se de que o motogerador ou queimador (flare) está funcionando para queimar o excedente.\n"
                "3. **Purgadores de Condensado**: Cheque e drene a água condensada acumulada nas linhas.\n"
                "4. **Alerta**: Pressão acima do limite operacional exige intervenção preventiva imediata."
            )
            is_high_press = any(w in normalized for w in ['alta', 'alto', 'sobrepressao', 'subir', 'sobe', 'estufar', 'perigoso', 'perigo', 'limite', 'passar'])
            if is_high_press:
                response = (
                    "🚨 **Alerta Crítico — Sobrepressão no Gasômetro / Linhas**\n\n"
                    "Pressão acima do limite operacional seguro ameaça romper a manta ou conexões:\n"
                    "1. **Válvula de Alívio**: Verifique e acione a válvula de alívio manual ou cheque o selo hidráulico imediatamente.\n"
                    "2. **Flare / Queima de Emergência**: Ative o queimador (flare) para queimar o biogás excedente e despressurizar o sistema.\n"
                    "3. **Purgador de Condensado**: Drene a água condensada nos pontos baixos da linha que possam estar obstruindo a vazão do gás."
                )
            else:
                response = (
                    "⚡ **Procedimento Operacional — Pressão do Biogás**\n\n"
                    "Instruções para controle de pressão na tubulação e gasômetro:\n"
                    "1. **Válvula de Alívio**: Verifique se a válvula de segurança/selo hidráulico está desobstruída.\n"
                    "2. **Consumo de Gás**: Certifique-se de que o motogerador ou queimador (flare) está funcionando para queimar o excedente.\n"
                    "3. **Purgadores de Condensado**: Cheque e drene a água condensada acumulada nas linhas.\n"
                    "4. **Alerta**: Pressão acima do limite operacional exige intervenção preventiva imediata."
                )
            action = "view_alerts"
            suggestions = [
                QuickSuggestion(label="🚨 Ver Alertas no Painel", action_type="action", value="view_alerts"),
                QuickSuggestion(label="🛠️ Agendar Manutenção", action_type="message", value="Agendar manutenção"),
                QuickSuggestion(label="📞 Suporte Operacional", action_type="action", value="contact_support"),
            ]

        elif op_topic == 'temperatura':
            response = (
                "🌡️ **Parâmetro Operacional — Temperatura da Biomassa**\n\n"
                "A temperatura é crucial para manter os microrganismos anaeróbicos ativos:\n"
                "• **Faixa Mesofílica Ideal**: 35°C a 40°C (ótimo: 37°C).\n"
                "• **Estabilidade**: Oscilações térmicas maiores que 2°C/dia prejudicam a produção metanogênica.\n"
                "• **Procedimento**: Verifique o isolamento térmico do digestor e o sistema de aquecimento/recirculação."
            )
            suggestions = [
                QuickSuggestion(label="📊 Ver Métricas", action_type="message", value="Quais são as métricas do biodigestor?"),
                QuickSuggestion(label="⚙️ Outras Dúvidas", action_type="message", value="Como funciona a operação do biodigestor?"),
            ]
            is_cooling = any(w in normalized for w in ['esfriar', 'esfria', 'frio', 'queda', 'cair', 'cai', 'baixa', 'descer', 'abaixo', 'congelar', 'resfriar', 'resfriamento', 'esfriou'])
            is_overheating = any(w in normalized for w in ['esquentar', 'esquenta', 'calor', 'subir', 'sobe', 'alta', 'acima', 'passar', 'superaquecer', 'aumentar', 'esquentou', '40 graus', '42'])

            if is_cooling:
                response = (
                    "❄️ **Queda de Temperatura / Resfriamento da Biomassa**\n\n"
                    "Se a temperatura cair abaixo da faixa mesofílica (< 35°C, e especialmente < 30°C):\n"
                    "• **Impacto Biológico**: A atividade metabólica das bactérias metanogênicas diminui drasticamente, reduzindo ou paralisando a produção de biogás (queda de 50% a 90%).\n"
                    "• **Risco de Acidose**: As bactérias fermentativas continuam ativas produzindo ácidos no frio, acumulando AGVs e derrubando o pH.\n\n"
                    "**Procedimentos de Ação Imediata**:\n"
                    "1. **Aquecimento**: Acione ou aumente a recirculação do trocador de calor / serpentina de aquecimento.\n"
                    "2. **Isolamento**: Verifique a integridade da cúpula térmica e proteções contra correntes de ar frio.\n"
                    "3. **Alimentação**: Reduza temporariamente a carga de novos resíduos frios para não derrubar ainda mais a temperatura interna."
                )
                suggestions = [
                    QuickSuggestion(label="📊 Ver Métricas", action_type="message", value="Quais são as métricas do biodigestor?"),
                    QuickSuggestion(label="🛠️ Agendar Manutenção", action_type="message", value="Agendar manutenção"),
                    QuickSuggestion(label="📞 Suporte Técnico", action_type="action", value="contact_support"),
                ]
            elif is_overheating:
                response = (
                    "🔥 **Superaquecimento da Biomassa (> 40°C - 42°C)**\n\n"
                    "O aquecimento excessivo em digestores mesofílicos causa choque térmico severo:\n"
                    "• **Impacto**: Mortalidade em massa dos microrganismos mesofílicos, cessando a geração de biogás.\n"
                    "• **Ação Imediata**: Desligue o sistema de aquecimento imediatamente, aumente a recirculação sem calor para dispersar temperatura e monitore a massa de hora em hora."
                )
                suggestions = [
                    QuickSuggestion(label="🚨 Ver Alertas no Painel", action_type="action", value="view_alerts"),
                    QuickSuggestion(label="📞 Suporte Técnico", action_type="action", value="contact_support"),
                ]
            else:
                response = (
                    "🌡️ **Parâmetro Operacional — Temperatura da Biomassa**\n\n"
                    "A temperatura é crucial para manter os microrganismos anaeróbicos ativos:\n"
                    "• **Faixa Mesofílica Ideal**: 35°C a 40°C (ótimo: 37°C).\n"
                    "• **Estabilidade**: Oscilações térmicas maiores que 2°C/dia prejudicam a produção metanogênica.\n"
                    "• **Procedimento**: Verifique o isolamento térmico do digestor e o sistema de aquecimento/recirculação."
                )
                suggestions = [
                    QuickSuggestion(label="📊 Ver Métricas", action_type="message", value="Quais são as métricas do biodigestor?"),
                    QuickSuggestion(label="⚙️ Outras Dúvidas", action_type="message", value="Como funciona a operação do biodigestor?"),
                ]

        elif op_topic == 'ph':
            response = (
                "🧪 **Parâmetro Operacional — Faixa de pH**\n\n"
                "O equilíbrio de pH é indicador fundamental de saúde da digestão anaeróbica:\n"
                "• **Faixa Ideal**: pH entre **6.8 e 7.4**.\n"
                "• **Acidificação (pH < 6.5)**: Indica sobrecarga orgânica por excesso de resíduos frescos. Reduza a alimentação imediatamente e adicione corretivo de alcalinidade (ex: bicarbonato ou cal).\n"
                "• **Inibição por Amônia (pH > 8.0)**: Verifique se houve excesso de dejetos com alta concentração de nitrogênio."
            )
            is_acidic = any(w in normalized for w in ['acido', 'acida', 'acidez', 'acidificar', 'acidificacao', 'acidose', 'cair', 'cai', 'queda', 'baixo', 'descer'])
            is_alkaline = any(w in normalized for w in ['alcalino', 'alcalina', 'alcalinidade', 'amonia', 'subir', 'sobe', 'alto'])

            if is_acidic:
                response = (
                    "⚠️ **Acidificação do Biodigestor (Acidose / pH < 6.5)**\n\n"
                    "A queda brusca de pH é uma emergência microbiológica:\n"
                    "• **Causa**: Sobrecarga de resíduos de fácil fermentação gerando excesso de AGVs.\n"
                    "• **Ações Imediatas**:\n"
                    "  1. **Suspender Alimentação**: Pare de alimentar o digestor com resíduos frescos.\n"
                    "  2. **Adição de Alcalinizante**: Dose cal hidratada ou bicarbonato de sódio lentamente até o pH estabilizar.\n"
                    "  3. **Acompanhamento**: Monitore o pH diariamente antes de reintroduzir nova biomassa."
                )
            elif is_alkaline:
                response = (
                    "⚠️ **Alcalinização Excessiva (pH > 8.0)**\n\n"
                    "• **Causa**: Excesso de dejetos com alta concentração de nitrogênio gerando amônia livre inibitória.\n"
                    "• **Ações**: Dilua com água limpa ou efluente estabilizado e ajuste o balanço C:N da alimentação."
                )
            else:
                response = (
                    "🧪 **Parâmetro Operacional — Faixa de pH**\n\n"
                    "O equilíbrio de pH é indicador fundamental de saúde da digestão anaeróbica:\n"
                    "• **Faixa Ideal**: pH entre **6.8 e 7.4**.\n"
                    "• **Acidificação (pH < 6.5)**: Indica sobrecarga orgânica por excesso de resíduos frescos. Reduza a alimentação imediatamente e adicione corretivo de alcalinidade (ex: bicarbonato ou cal).\n"
                    "• **Inibição por Amônia (pH > 8.0)**: Verifique se houve excesso de dejetos com alta concentração de nitrogênio."
                )
            suggestions = [
                QuickSuggestion(label="🛠️ Agendar Manutenção", action_type="message", value="Agendar manutenção"),
                QuickSuggestion(label="📞 Suporte Técnico", action_type="action", value="contact_support"),
            ]

        elif op_topic == 'alimentacao':
            response = (
                "🌱 **Procedimento Operacional — Alimentação de Biomassa**\n\n"
                "Boas práticas de carga orgânica do biodigestor:\n"
                "1. **Uniformidade**: Alimente em bateladas regulares para evitar choques de carga orgânica.\n"
                "2. **Teor de Sólidos**: Mantenha o teor de sólidos totais (TS) diluído adequadamente (tipicamente 8% a 10%).\n"
                "3. **Inibidores**: Evite introduzir água sanitária, desinfetantes ou substâncias químicas bactericidas."
            )
            suggestions = [
                QuickSuggestion(label="♻️ Resíduos Processados", action_type="message", value="Quantos resíduos foram processados?"),
                QuickSuggestion(label="📊 Métricas Gerais", action_type="message", value="Métricas do biodigestor"),
            ]

        else:
            response = (
                "⚙️ **Diretrizes e Suporte Operacional — BioDash**\n\n"
                "Acompanhamento da operação dos biodigestores:\n"
                "• **Monitoramento Diário**: Verifique pressão, H2S e temperatura no Painel.\n"
                "• **Procedimentos de Segurança**: Mantenha filtros de biogás e válvulas de alívio sempre revisados.\n"
                "• **Suporte Dedicado**: Caso ocorra qualquer anomalia técnica ou incidente em campo, contate diretamente nossa equipe de operações."
            )
            action = "contact_support"
            suggestions = [
                QuickSuggestion(label="🚨 Ver Alertas no Painel", action_type="action", value="view_alerts"),
                QuickSuggestion(label="🛠️ Agendar Manutenção", action_type="message", value="Agendar manutenção"),
                QuickSuggestion(label="📞 Falar com Suporte", action_type="action", value="contact_support"),
            ]

    elif intent == "pedido_exportar_pdf":
        response = "📄 Vou gerar o relatório em *PDF* para você agora! Aguarde..."
        action = "export_pdf"

    elif intent == "pedido_exportar_csv":
        response = "📊 Vou exportar os dados em *CSV* para você agora! Aguarde..."
        action = "export_csv"

    elif intent == "pedido_exportar_excel":
        response = "📋 Vou exportar os dados em *Excel* para você agora! Aguarde..."
        action = "export_excel"

    # ─── Fluxos conversacionais ──────────────────────────────────────────────

    elif intent == "agendar_manutencao":
        response = (
            "📅 Certo! Vou agendar uma manutenção.\n\n"
            "Qual o **nome** da manutenção?\n"
            "(ex: Troca de filtros, Limpeza do tanque, Inspeção geral)"
        )
        action = "start_flow_manutencao"

    elif intent == "incluir_metrica":
        response = (
            "📊 Vou registrar novas métricas do biodigestor.\n\n"
            "Qual a quantidade de **resíduos processados** em kg?"
        )
        action = "start_flow_metrica"

    elif intent == "editar_metrica":
        response = (
            "✏️ Vou editar métricas existentes.\n\n"
            "Qual **mês e ano** deseja atualizar?\n"
            "(ex: outubro 2025 ou 10/2025)"
        )
        action = "start_flow_editar_metrica"

    elif intent == "adicionar_endereco":
        response = (
            "📍 Vou cadastrar um novo biodigestor no mapa.\n\n"
            "Qual o **nome** do biodigestor?\n"
            "(ex: Biodigestor Norte, Planta 01)"
        )
        action = "start_flow_endereco"

    elif intent == "relatorio_periodo":
        response = (
            "📅 Vou gerar um relatório por período específico.\n\n"
            "Qual o **mês e ano inicial**?\n"
            "(ex: janeiro 2025 ou 01/2025)"
        )
        action = "start_flow_relatorio"

    elif intent == "confirmar":
        response = "✅ Entendido!"
        action = "flow_confirm"

    elif intent == "cancelar":
        response = "❌ Operação cancelada. Como mais posso ajudar?"
        action = "cancel_flow"

    elif intent == "despedida":
        response = "Foi um prazer te ajudar! Até logo e continuo à disposição! 🌿"

    else:
        response = (
            "Desculpe, não entendi muito bem. Posso te ajudar com:\n"
            "• Endereços dos biodigestores\n"
            "• Métricas de energia e resíduos\n"
            "• Agendar manutenções\n"
            "• Registrar ou editar métricas\n"
            "• Cadastrar novos biodigestores\n"
            "• Gerar relatórios por período"
        )
        return build_default_fallback_response(confidence, entities, context)

    if is_elliptical:
        question_type = QuestionType.CONTEXTUAL_FOLLOWUP
    elif intent == "duvida_operacional":
        question_type = QuestionType.OPERACIONAL
    elif intent in ["pedido_energia", "pedido_residuos", "pedido_metricas"]:
        question_type = QuestionType.METRICA
    elif intent == "pedido_endereco":
        question_type = QuestionType.LOCALIZACAO
    elif intent in ["agendar_manutencao", "incluir_metrica", "editar_metrica", "adicionar_endereco", "relatorio_periodo", "pedido_exportar_pdf", "pedido_exportar_csv", "pedido_exportar_excel"]:
        question_type = QuestionType.TRANSACIONAL
    elif intent in ["saudacao", "despedida", "confirmar", "cancelar"]:
        question_type = QuestionType.SOCIAL
    else:
        question_type = QuestionType.FORA_DE_ESCOPO

    if context:
        context.last_question_type = question_type

    return ChatResponse(
        intent=intent,
        question_type=question_type,
        response=response,
        confidence=round(confidence, 4),
        action=action,
        entities=entities,
        suggestions=suggestions,
        context=context,
    )


@app.post("/semantic-search", response_model=SemanticSearchResponse)
def semantic_search(req: SemanticSearchRequest):
    """
    Busca semântica entre a query e os marcadores usando TF-IDF + cosseno.
    """
    if not req.markers:
        return SemanticSearchResponse(results=[])
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query não pode ser vazia.")

    marker_texts = [build_marker_text(m) for m in req.markers]
    all_texts = [req.query] + marker_texts

    search_vectorizer = TfidfVectorizer(ngram_range=(1, 2), analyzer='char_wb')
    try:
        tfidf_matrix = search_vectorizer.fit_transform(all_texts)
    except ValueError:
        return SemanticSearchResponse(results=[])

    query_vec = tfidf_matrix[0]
    marker_vecs = tfidf_matrix[1:]
    similarities = cosine_similarity(query_vec, marker_vecs)[0]

    ranked = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)

    results = []
    for idx, score in ranked:
        if score > 0.01:
            marker = req.markers[idx].copy()
            marker["similarity_score"] = round(float(score), 4)
            results.append(marker)

    return SemanticSearchResponse(results=results[:10])


# ──────────────────────────────────────────────────────────────────────────────
# 5. ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("AI_SERVICE_PORT", 5000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
