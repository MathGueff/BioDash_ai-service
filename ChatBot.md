# 🤖 BioDash — ChatBot com PLN (TF-IDF + SVM)

> Documentação técnica do módulo de Inteligência Artificial integrado ao BioDash Mobile.  
> Disciplina: Processamento de Linguagem Natural — 6º Semestre

---

## 📌 Índice

1. [Visão Geral](#visão-geral)
2. [Arquitetura](#arquitetura)
3. [Microserviço Python (ai-service)](#microserviço-python)
4. [Chatbot — TF-IDF + SVM](#chatbot--tfidf--svm)
5. [Busca Semântica](#busca-semântica)
6. [Reconhecimento de Voz](#reconhecimento-de-voz)
7. [Integração com o App (Frontend)](#integração-com-o-app)
8. [Endpoints da API](#endpoints-da-api)
9. [Como Executar](#como-executar)
10. [Estrutura de Arquivos](#estrutura-de-arquivos)

---

## Visão Geral

O módulo de ChatBot foi desenvolvido como entrega do **Projeto Integrador de PLN** e adiciona ao aplicativo BioDash Mobile três funcionalidades principais:

| Funcionalidade | Tecnologia |
|---|---|
| **Chatbot inteligente** | TF-IDF + SVM (`scikit-learn`) |
| **Busca Semântica** | TF-IDF + Similaridade de Cosseno |
| **Reconhecimento de Voz** | Web Speech API (nativa do browser) |

O chatbot compreende linguagem natural em **português brasileiro** e responde com informações reais dos biodigestores cadastrados pelo usuário — como endereços, métricas de energia e resíduos — além de acionar exportações de relatórios (PDF, CSV ou Excel) diretamente pela conversa.

---

## Arquitetura

```
BioDash Mobile (Expo / React Native)
│
├── Frontend (React Native / Expo Web)
│   ├── src/screens/ChatbotScreen.tsx  ← Interface do chat + microfone + busca
│   └── src/lib/api.ts                ← chatbotApi + semanticSearchApi
│
├── Backend Node.js (porta 3003)
│   └── (autenticação, indicadores, marcadores — PostgreSQL + MongoDB)
│
└── AI Service Python (FastAPI, porta 5000)  ← NOVO
    └── ai-service/main.py
        ├── POST /chatbot              ← TF-IDF + SVM (classificação de intenção)
        └── POST /semantic-search      ← TF-IDF + Cosseno (busca por biodigestores)
```

O microserviço Python é independente e se comunica com o frontend diretamente via HTTP/REST. O frontend detecta automaticamente se está rodando em `localhost` (desenvolvimento) ou no IP de rede (dispositivo móvel) para ajustar a URL de conexão.

---

## Microserviço Python

### Localização
```
BioDash_mobile/
└── ai-service/
    ├── main.py           ← Servidor FastAPI + lógica de PLN
    ├── requirements.txt  ← Dependências Python
    └── Dockerfile        ← Containerização (opcional)
```

### Dependências (`requirements.txt`)
```
fastapi==0.115.0
uvicorn==0.30.6
scikit-learn==1.5.2
numpy==1.26.4
pydantic==2.9.2
python-dotenv==1.0.1
```

### Inicialização
```powershell
cd ai-service
$env:PYTHONIOENCODING="utf-8"
python -m uvicorn main:app --host 0.0.0.0 --port 5000 --reload
```

---

## Chatbot — TF-IDF + SVM

### Fluxo de Classificação

```
Texto do usuário
      ↓
  Normalização (lowercase, remoção de acentos)
      ↓
  TfidfVectorizer (char_wb, ngram (1,2))
      ↓
  SVC — Support Vector Classifier (kernel linear)
      ↓
  Intenção classificada + confiança (probabilidade)
      ↓
  Resposta contextualizada (dados reais do usuário)
```

### Intenções Reconhecidas

| Intenção | Exemplos de Frase | Resposta e Comportamento |
|---|---|---|
| `saudacao` | "Oi", "Bom dia", "Olá" | "Olá! Em que posso ajudar você hoje? 😊" + sugestões rápidas |
| `duvida_operacional` | "Alerta de H2S", "Pressão alta", "pH ideal", "Como operar" | Instruções operacionais detalhadas + ações (`view_alerts`, `contact_support`) |
| `pedido_endereco` | "Onde fica o biodigestor?", "Qual o endereço?" | Endereço real dos marcadores do mapa |
| `pedido_residuos` | "Quantos resíduos foram processados?", "E os resíduos?" | Valor em kg do último registro (considera contexto anterior) |
| `pedido_energia` | "Quanta energia foi gerada?", "E a energia?" | Valor em kWh do último registro (considera contexto anterior) |
| `pedido_metricas` | "Como está o biodigestor?", "Status" | Resumo de resíduos + energia + benefícios |
| `agendar_manutencao` | "Agendar manutenção", "Troca de filtro" | Fluxo conversacional guiado de agendamento |
| `pedido_exportar_pdf` | "Gera um PDF", "Relatório PDF" | Abre modal de impressão / download PDF |
| `pedido_exportar_csv` | "Exportar CSV", "Arquivo CSV" | Download do CSV com os indicadores |
| `pedido_exportar_excel` | "Exportar Excel", "Planilha" | Download do Excel (.csv formatado) |
| `nao_compreendido` | Solicitações fora de escopo ou ininteligíveis | Resposta orientativa amigável + botões interativos de ajuda |
| `despedida` | "Tchau", "Obrigado", "Encerrar" | "Foi um prazer te ajudar! Até logo e continuo à disposição! 🌿" |

### Interpretação de Contexto e Histórico Multi-turn
- O frontend envia o histórico das últimas 5 mensagens (`history`) e o estado contextual (`context`).
- Perguntas elípticas ou de continuidade (ex: *"e os resíduos?"*, *"e no mês passado?"*, *"como resolvo isso?"*) preservam o tópico ou biodigestor previamente consultado.

### Direcionamento Operacional e Segurança
- Perguntas sobre parâmetros críticos (**H2S**, **pressão de biogás**, **temperatura** e **faixa de pH**) fornecem diretrizes técnicas imediatas.
- Ações automáticas (`action: "view_alerts"`, `action: "contact_support"`) acionam alertas visuais ou direcionamento para suporte operacional especializado (`suporte@biodash.com`).

### Tratamento de Fallback
- Limiar de confiança (`CONFIDENCE_THRESHOLD = 0.40`) e validação de vocabulário de domínio.
- Caso a solicitação não possa ser interpretada, o chatbot não chuta respostas aleatórias; ele responde com um menu contextual de opções e sugestões clicáveis.

### Parâmetros do Modelo
```python
TfidfVectorizer(
    ngram_range=(1, 2),   # uni e bi-gramas
    analyzer='char_wb',   # análise por caractere (melhor para português)
    sublinear_tf=True,    # suavização logarítmica
    min_df=1
)

SVC(
    kernel='linear',
    C=1.0,
    probability=True      # habilita predict_proba para retornar confiança
)
```

### Dataset de Treinamento
O modelo foi treinado com **250+ frases** em português distribuídas entre as intenções do sistema, incluindo procedimentos operacionais e de emergência. O dataset está embutido no `ai-service/main.py`.

---

## Busca Semântica

A busca semântica permite encontrar biodigestores cadastrados pelo usuário usando **similaridade de cosseno** entre vetores TF-IDF — diferente de uma busca por palavra-chave exata.

### Fluxo
```
Query do usuário  +  Lista de marcadores do mapa
         ↓
TfidfVectorizer.fit_transform([query] + [textos dos marcadores])
         ↓
cosine_similarity(query_vec, marker_vecs)
         ↓
Resultados ordenados por relevância (score 0.0 → 1.0)
```

### Dados Usados na Indexação
Para cada marcador/biodigestor, o seguinte texto é gerado para indexação:
- `title` (nome do biodigestor)
- `address.street`, `address.cep`, `address.city`, `address.complement`
- `description` (endereço reverso do mapa)

### Interface
O painel de busca semântica fica acessível pelo ícone 🔍 no canto superior direito da tela do chatbot.

---

## Reconhecimento de Voz

### Plataforma Web (Chrome / Edge)
Utiliza a **Web Speech API** nativa do browser, sem necessidade de bibliotecas externas:

```typescript
const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)()
recognition.lang = 'pt-BR'
recognition.continuous = false
recognition.start()
```

**Comportamento:**
1. Usuário clica no botão 🎙️
2. Estado muda imediatamente para "Ouvindo" (campo fica vermelho)
3. Placeholder muda para `🔴 Ouvindo... fale agora`
4. Após detectar fala, a mensagem é transcrita e **enviada automaticamente**

### Plataforma Mobile (Expo Go)
O Expo Go não suporta módulos nativos compilados (como `@react-native-voice/voice`). A solução atual orienta o usuário a usar o **ditado por voz nativo do teclado do dispositivo** (disponível em Android e iOS por padrão, basta tocar no ícone de microfone no teclado).

> Para habilitar reconhecimento de voz nativo no mobile em produção (APK/IPA compilado), seria necessário adicionar o módulo `@react-native-voice/voice` e rebuild da aplicação.

---

## Integração com o App

### Nova Aba — "Assistente"
A aba **Assistente** 🤖 foi adicionada na barra de navegação inferior do `App.tsx`, ao lado de "Painel" e "Ajustes".

### ChatbotScreen.tsx
Componente completo com:
- Bolhas de mensagem estilizadas (estilo WhatsApp)
- Atalhos rápidos horizontais (Endereço, Energia, Resíduos, Métricas, PDF, Excel)
- Painel de Busca Semântica (colapsável pelo ícone 🔍)
- Indicador "Assistente digitando..." com animação
- Botão de microfone com animação de pulso
- Tema automático (dark/light)

### Novas funções em `api.ts`
```typescript
// Envia mensagem para o chatbot (TF-IDF + SVM)
chatbotApi.send({ message, markers?, indicators? })

// Busca semântica por biodigestores
semanticSearchApi.search({ query, markers })
```

A URL do serviço é detectada automaticamente:
```typescript
const AI_SERVICE_URL = window?.location?.hostname === 'localhost'
  ? 'http://localhost:5000'           // desenvolvimento web
  : process.env.EXPO_PUBLIC_AI_SERVICE_URL  // IP de rede (mobile)
```

---

## Endpoints da API

### `POST /chatbot`
**Body:**
```json
{
  "message": "Quanta energia foi gerada?",
  "markers": [...],      // opcional — biodigestores do usuário
  "indicators": [...]    // opcional — métricas do usuário
}
```
**Response:**
```json
{
  "intent": "pedido_energia",
  "response": "⚡ Energia Gerada\nÚltimo registro: 92.50 kWh",
  "confidence": 0.7621,
  "action": null
}
```

### `POST /semantic-search`
**Body:**
```json
{
  "query": "biodigestor perto do rio",
  "markers": [...]
}
```
**Response:**
```json
{
  "results": [
    {
      "title": "Biodigestor Norte",
      "similarity_score": 0.8231,
      ...
    }
  ]
}
```

### `GET /health`
```json
{ "status": "ok" }
```

---

## Como Executar

### 1. Instalar dependências Python (uma vez)
```powershell
cd ai-service
pip install -r requirements.txt
```

### 2. Iniciar o serviço de IA
```powershell
cd ai-service
$env:PYTHONIOENCODING="utf-8"
python -m uvicorn main:app --host 0.0.0.0 --port 5000 --reload
```

### 3. Iniciar o backend Node.js
```powershell
cd backend
npm run dev
```

### 4. Iniciar o frontend Expo
```powershell
npm run start
# Pressione W para abrir no browser
```

### Ordem de inicialização recomendada
```
1. AI Service Python  (porta 5000)
2. Backend Node.js    (porta 3003)
3. Frontend Expo      (porta 8081)
```

---

## Estrutura de Arquivos

```
BioDash_mobile/
│
├── ai-service/                        ← NOVO — Microserviço de IA
│   ├── main.py                        ← FastAPI + TF-IDF + SVM
│   ├── requirements.txt               ← Dependências Python
│   └── Dockerfile                     ← Containerização
│
├── src/
│   ├── screens/
│   │   └── ChatbotScreen.tsx          ← NOVO — Tela do Assistente Virtual
│   └── lib/
│       └── api.ts                     ← MODIFICADO — chatbotApi + semanticSearchApi
│
├── App.tsx                            ← MODIFICADO — Nova aba "Assistente"
├── .env                               ← MODIFICADO — EXPO_PUBLIC_AI_SERVICE_URL
│
└── backend/
    ├── src/routes/auth.js             ← MODIFICADO — Fix login (password_hash null)
    └── .env                           ← MODIFICADO — MongoDB Atlas URI
```

---

## Problemas Conhecidos e Soluções

| Problema | Causa | Solução |
|---|---|---|
| Microfone não abre | Permissão negada no browser | Clicar em 🔒 → Microfone → Permitir → Recarregar |
| Chatbot retorna erro 500 | Serviço Python não rodando | Iniciar `uvicorn` na porta 5000 |
| Timeout de conexão no login | IP da máquina mudou | Atualizar `.env` com novo IP (`ipconfig`) |
| Erro `data and hash required` | Usuário sem senha no banco | Criar nova conta pelo app |
| MongoDB timeout | URI local incorreta | Usar URI do MongoDB Atlas no `backend/.env` |

---

*BioDash Intelligence Systems — Projeto Integrador 6º Semestre*
