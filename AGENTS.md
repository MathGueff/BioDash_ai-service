# Regras e Diretrizes para IA (Agent Guidelines)

## 🚨 REGRA CRÍTICA: Conventional Commits e Versionamento SemVer

Este repositório utiliza automação de CI/CD baseada em **Conventional Commits** para cálculo automático de versão SemVer e publicação de tags/releases Docker.

### 1. NUNCA faça commits fora do padrão
Qualquer commit criado DEVE seguir rigorosamente a convenção:
`<tipo>(<escopo opcional>): <descrição no imperativo>`

### 2. Mapeamento de Tipos e Impacto no SemVer
- `feat: ...` -> **MINOR** (ex: `1.0.0` -> `1.1.0`). Use para novas rotas FastAPI, novos modelos de NLP, novas funcionalidades de chatbot/whisper.
- `fix: ...` -> **PATCH** (ex: `1.0.0` -> `1.0.1`). Use para correções de bugs, prompts, erros de transcrição ou inferência.
- `feat!: ...` ou corpo com `BREAKING CHANGE:` -> **MAJOR** (ex: `1.0.0` -> `2.0.0`). Use para alterações incompatíveis em endpoints ou contratos de payload.
- `chore: ...` -> Atualização de requirements.txt, uv, docker, configs (NÃO gera bump de versão).
- `docs: ...` -> Documentação apenas (NÃO gera bump de versão).
- `refactor: ...` -> Refatoração interna sem alterar contratos.
- `test: ...` -> Adição ou atualização de testes em `test_chatbot_context.py`.
- `ci: ...` -> Mudanças no GitHub Actions (`ci-cd.yml`).
- `style: ...` -> Formatação PEP8 / Black.

> ⚠️ **Atenção:** Se um commit de nova feature ou bugfix for feito com outro prefixo (ou sem prefixo), o pipeline de CI **NÃO gerará a tag SemVer**, causando falha na release.

### 3. Exemplos Práticos
- ✅ `feat(whisper): add streaming audio transcription endpoint`
- ✅ `fix(nlp): handle out-of-context fallback response in chatbot`
- ✅ `chore(deps): update fastrtc and whisper dependencies`
- ❌ `ajuste no prompt do bot` (Inválido - commit será bloqueado pelo git hook)
- ❌ `update fastAPI` (Inválido)
- ❌ `corrigindo whisper` (Inválido)

### 4. Git Hooks
Existe um hook local em `.githooks/commit-msg` e um workflow no GitHub (`.github/workflows/commitlint.yml`). Qualquer mensagem fora desse padrão será rejeitada no terminal e na CI.

