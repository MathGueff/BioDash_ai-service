# BioDash AI Service 🤖

Microserviço de PLN (Processamento de Linguagem Natural), Transcrição de Áudio (Whisper) e Chatbot do ecossistema BioDash.

## 🚀 Como Rodar

### Opção 1: Via NPM com UV (Modo Desenvolvimento)
```bash
# Inicia a aplicação usando uv + python 3.11 automaticamente
npm run dev

# Instalar dependências de execução e desenvolvimento (Python 3.11)
python -m pip install -r requirements.txt -r requirements-dev.txt

# Verificar lint e executar os testes unitários e de contexto
npm run lint
npm test
```

### Opção 2: Com Docker
```bash
# Iniciar o container
npm run docker:build
# ou: docker compose up -d --build

# Ver logs
npm run docker:logs
# ou: docker logs -f biodash_ai-service

# Parar o container
npm run docker:down
# ou: docker compose down
```

## 🌐 Endpoints
- `GET /health` - Healthcheck da API
- `GET /docs` - Documentação interativa (Swagger UI)
- `POST /chatbot` - Envio de mensagens para o chatbot
- `POST /chatbot/transcribe` - Transcrição de áudio via Whisper
- `GET /semantic-search` - Busca semântica de biodigestores


## Integração contínua

O GitHub Actions executa o Ruff e os testes com pytest antes do build da imagem
e do deploy existente no Azure. Os testes unitários de transcrição simulam o
modelo Whisper e não baixam modelos nem acessam serviços externos.
