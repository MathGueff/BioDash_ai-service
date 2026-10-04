# BioDash AI Service 🤖

Microserviço de PLN (Processamento de Linguagem Natural), Transcrição de Áudio (Whisper) e Chatbot do ecossistema BioDash.

## 🚀 Como Rodar

### Com Docker (Recomendado)
```bash
# Iniciar o container
docker compose up -d --build

# Ver logs
docker logs -f biodash_chatbot

# Parar o container
docker compose down
```

### Com Python Localmente
```bash
python -m venv .venv
source .venv/bin/activate # ou .venv\Scripts\activate no Windows
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 5000 --reload
```

## 🌐 Endpoints
- `GET /health` - Healthcheck
- `POST /chatbot` - Envio de mensagens para o chatbot
- `POST /chatbot/transcribe` - Transcrição de áudio via Whisper
- `GET /semantic-search` - Busca semântica de biodigestores
