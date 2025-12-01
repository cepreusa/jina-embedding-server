# Jina Embeddings v4 Server

OpenAI/Jina-compatible API server for jina-embeddings-v4.

## Quick Start

```bash
# OpenAI-compatible (for EMBEDDING_BINDING=openai)
./start.sh

# Jina-compatible with base64 (for EMBEDDING_BINDING=jina)
./start.sh jina

# With API key
API_KEY=your-secret-key ./start.sh

# Custom port
PORT=8000 ./start.sh
```

## Test

```bash
# Health
curl http://localhost:8080/health

# Embeddings (OpenAI mode)
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-key" \
  -d '{"input": "Hello world"}'

# Embeddings (Jina mode with base64)
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-key" \
  -d '{"input": "Hello world", "embedding_type": "base64"}'
```

## LightRAG Config

### OpenAI binding (recommended)
```bash
EMBEDDING_BINDING=openai
EMBEDDING_BINDING_HOST=http://your-server:8080/v1
EMBEDDING_MODEL=jina-embeddings-v4
EMBEDDING_DIM=2048
EMBEDDING_BINDING_API_KEY=your-key
```

### Jina binding (with base64)
```bash
EMBEDDING_BINDING=jina
EMBEDDING_BINDING_HOST=http://your-server:8080/v1/embeddings
EMBEDDING_MODEL=jina-embeddings-v4
EMBEDDING_DIM=2048
EMBEDDING_BINDING_API_KEY=your-key
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 8080 | Server port |
| `API_KEY` | *(empty)* | API key (empty = no auth) |
| `EMBEDDING_TASK` | text-matching | Default task |
| `EMBEDDING_DIM` | 2048 | Default dimensions |
| `LOG_LEVEL` | INFO | Log level |
