# Jina Embeddings v4 Server

OpenAI-compatible API server for jina-embeddings-v4 with Prometheus metrics.

## Quick Start (Local)

```bash
# Start server
./start.sh

# With API key
API_KEY=your-secret-key ./start.sh

# Custom port
PORT=8000 ./start.sh
```

## Vast.ai Deployment

```bash
vastai create instance <OFFER_ID> \
  --image vastai/base-image:@vastai-automatic-tag \
  --env '-p 8080:8080 \
    -e JINA_PORT=8080 \
    -e API_KEY=your-secret-key \
    -e SCRIPT_URL=https://raw.githubusercontent.com/YOUR_USER/YOUR_REPO/main/server.py \
    -e PROVISIONING_SCRIPT=https://raw.githubusercontent.com/YOUR_USER/YOUR_REPO/main/jina_vast.sh' \
  --onstart-cmd 'entrypoint.sh' \
  --disk 50 --ssh --direct
```

## Endpoints

| Endpoint | Description |
|----------|-------------|
| `/health` | Health check + stats |
| `/metrics` | Prometheus metrics |
| `/v1/embeddings` | Create embeddings |
| `/v1/models` | List models |
| `/docs` | OpenAPI docs |

## Prometheus Metrics

```bash
curl http://localhost:8080/metrics
```

Метрики:
- `jina_requests_total` — всего запросов
- `jina_requests_failed_total` — ошибки
- `jina_texts_processed_total` — обработано текстов
- `jina_avg_latency_seconds` — средняя задержка
- `jina_texts_per_second` — throughput
- `jina_gpu_memory_used_bytes` — память GPU
- `jina_gpu_utilization_percent` — загрузка GPU
- `jina_gpu_temperature_celsius` — температура GPU

## LightRAG Config

```bash
EMBEDDING_BINDING=openai
EMBEDDING_BINDING_HOST=http://your-server:8080/v1
EMBEDDING_MODEL=jina-embeddings-v4
EMBEDDING_DIM=2048
EMBEDDING_BINDING_API_KEY=your-key

# Parallelism for RTX 4090
EMBEDDING_BATCH_NUM=16
EMBEDDING_FUNC_MAX_ASYNC=4
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 8080 | Server port |
| `API_KEY` | *(empty)* | API key (empty = no auth) |
| `EMBEDDING_TASK` | text-matching | Default task |
| `EMBEDDING_DIM` | 2048 | Default dimensions |
| `LOG_LEVEL` | INFO | Log level |
| `SCRIPT_URL` | *(required for vast.ai)* | URL to server.py |

## Grafana Integration

Add to Prometheus `prometheus.yml`:
```yaml
scrape_configs:
  - job_name: 'jina-embeddings'
    static_configs:
      - targets: ['gpu-server:8080']
```
