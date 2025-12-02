"""
Jina Embeddings v4 - OpenAI-compatible API Server with Prometheus metrics
"""

import os
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Header, Depends, Response
from pydantic import BaseModel, Field

# Конфигурация
MODEL_NAME = os.getenv("MODEL_NAME", "jinaai/jina-embeddings-v4")
DEFAULT_TASK = os.getenv("EMBEDDING_TASK", "text-matching")
DEFAULT_DIM = int(os.getenv("EMBEDDING_DIM", "2048"))
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "32"))
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8080"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
API_KEY = os.getenv("API_KEY", "")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

VALID_DIMENSIONS = {128, 256, 512, 1024, 2048}

# Промпты для разных задач (из документации Jina)
TASK_PROMPTS = {
    "text-matching": "text-matching",
    "retrieval.query": "retrieval.query",
    "retrieval.passage": "retrieval.passage",
    "code": "code",
}

model = None
inner_model = None

# =============================================================================
# Метрики
# =============================================================================

class Metrics:
    def __init__(self):
        self.requests_total = 0
        self.requests_failed = 0
        self.texts_processed = 0
        self.processing_seconds = 0.0
        self.last_request_time = 0
        self.start_time = time.time()
    
    def record_request(self, texts_count: int, duration: float, success: bool):
        self.requests_total += 1
        if success:
            self.texts_processed += texts_count
            self.processing_seconds += duration
        else:
            self.requests_failed += 1
        self.last_request_time = time.time()
    
    def get_gpu_metrics(self):
        if not torch.cuda.is_available():
            return {}
        
        try:
            return {
                "gpu_memory_used_bytes": torch.cuda.memory_allocated(),
                "gpu_memory_cached_bytes": torch.cuda.memory_reserved(),
                "gpu_memory_total_bytes": torch.cuda.get_device_properties(0).total_memory,
                "gpu_utilization": self._get_gpu_utilization(),
                "gpu_temperature": self._get_gpu_temperature(),
            }
        except:
            return {}
    
    def _get_gpu_utilization(self):
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            return float(result.stdout.strip())
        except:
            return 0.0
    
    def _get_gpu_temperature(self):
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            return float(result.stdout.strip())
        except:
            return 0.0
    
    def to_prometheus(self):
        gpu = self.get_gpu_metrics()
        uptime = time.time() - self.start_time
        successful = max(self.requests_total - self.requests_failed, 1)
        avg_latency = self.processing_seconds / successful
        texts_per_sec = self.texts_processed / max(self.processing_seconds, 0.001)
        
        lines = [
            "# HELP jina_requests_total Total embedding requests",
            "# TYPE jina_requests_total counter",
            f"jina_requests_total {self.requests_total}",
            "",
            "# HELP jina_requests_failed_total Failed requests",
            "# TYPE jina_requests_failed_total counter",
            f"jina_requests_failed_total {self.requests_failed}",
            "",
            "# HELP jina_texts_processed_total Total texts embedded",
            "# TYPE jina_texts_processed_total counter",
            f"jina_texts_processed_total {self.texts_processed}",
            "",
            "# HELP jina_processing_seconds_total Total processing time",
            "# TYPE jina_processing_seconds_total counter",
            f"jina_processing_seconds_total {self.processing_seconds:.3f}",
            "",
            "# HELP jina_avg_latency_seconds Average request latency",
            "# TYPE jina_avg_latency_seconds gauge",
            f"jina_avg_latency_seconds {avg_latency:.4f}",
            "",
            "# HELP jina_texts_per_second Throughput",
            "# TYPE jina_texts_per_second gauge",
            f"jina_texts_per_second {texts_per_sec:.2f}",
            "",
            "# HELP jina_uptime_seconds Server uptime",
            "# TYPE jina_uptime_seconds gauge",
            f"jina_uptime_seconds {uptime:.0f}",
            "",
            "# HELP jina_model_loaded Model loaded status",
            "# TYPE jina_model_loaded gauge",
            f"jina_model_loaded {1 if inner_model else 0}",
        ]
        
        if gpu:
            lines.extend([
                "",
                "# HELP jina_gpu_memory_used_bytes GPU memory used",
                "# TYPE jina_gpu_memory_used_bytes gauge",
                f"jina_gpu_memory_used_bytes {gpu.get('gpu_memory_used_bytes', 0)}",
                "",
                "# HELP jina_gpu_memory_total_bytes GPU memory total",
                "# TYPE jina_gpu_memory_total_bytes gauge",
                f"jina_gpu_memory_total_bytes {gpu.get('gpu_memory_total_bytes', 0)}",
                "",
                "# HELP jina_gpu_utilization_percent GPU utilization",
                "# TYPE jina_gpu_utilization_percent gauge",
                f"jina_gpu_utilization_percent {gpu.get('gpu_utilization', 0)}",
                "",
                "# HELP jina_gpu_temperature_celsius GPU temperature",
                "# TYPE jina_gpu_temperature_celsius gauge",
                f"jina_gpu_temperature_celsius {gpu.get('gpu_temperature', 0)}",
            ])
        
        return "\n".join(lines) + "\n"

metrics = Metrics()


# =============================================================================
# Auth
# =============================================================================

async def verify_api_key(authorization: Optional[str] = Header(None)):
    if not API_KEY:
        return True
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = authorization[7:] if authorization.lower().startswith("bearer ") else authorization
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


# =============================================================================
# Models
# =============================================================================

class EmbeddingRequest(BaseModel):
    input: str | list[str]
    model: str = "jina-embeddings-v4"
    dimensions: Optional[int] = None
    task: Optional[str] = None
    prompt_name: Optional[str] = None


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: list[float]
    index: int


class UsageInfo(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingData]
    model: str
    usage: UsageInfo


# =============================================================================
# Model loading
# =============================================================================

def load_model():
    global model, inner_model
    from transformers import AutoModel
    
    logger.info(f"Загрузка модели {MODEL_NAME}...")
    start_time = time.time()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    
    if device == "cuda":
        model = model.cuda()
    
    model.eval()
    
    # Получаем внутреннюю модель с методами encode_text
    if hasattr(model, 'get_base_model'):
        inner_model = model.get_base_model()
    else:
        inner_model = model.base_model.model
    
    logger.info(f"Модель загружена за {time.time() - start_time:.1f}s")
    logger.info(f"Inner model: {type(inner_model).__name__}")
    
    # Прогрев
    logger.info("Прогрев модели...")
    with torch.no_grad():
        _ = inner_model.encode_text(["warmup"], task=DEFAULT_TASK)
    logger.info("Модель готова")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="Jina Embeddings v4 API", lifespan=lifespan)


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health")
async def health_check():
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_memory_used_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
            "gpu_memory_total_mb": round(torch.cuda.get_device_properties(0).total_memory / 1024**2, 1),
        }
    
    return {
        "status": "healthy" if inner_model else "loading",
        "model": MODEL_NAME,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "default_task": DEFAULT_TASK,
        "default_dim": DEFAULT_DIM,
        "auth_enabled": bool(API_KEY),
        "requests_total": metrics.requests_total,
        "texts_processed": metrics.texts_processed,
        **gpu_info
    }


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus metrics endpoint"""
    return Response(content=metrics.to_prometheus(), media_type="text/plain")


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{"id": "jina-embeddings-v4", "object": "model"}]}


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest, authorized: bool = Depends(verify_api_key)):
    global inner_model
    
    if inner_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    texts = [request.input] if isinstance(request.input, str) else request.input
    if not texts:
        raise HTTPException(status_code=400, detail="Empty input")
    if len(texts) > MAX_BATCH_SIZE:
        raise HTTPException(status_code=400, detail=f"Batch size exceeds {MAX_BATCH_SIZE}")
    
    # Определяем task
    task = request.task or DEFAULT_TASK
    if task == "retrieval" and request.prompt_name:
        task = f"retrieval.{request.prompt_name}"
    
    dimensions = request.dimensions or DEFAULT_DIM
    if dimensions not in VALID_DIMENSIONS:
        raise HTTPException(status_code=400, detail=f"Invalid dimensions. Valid: {sorted(VALID_DIMENSIONS)}")
    
    start = time.time()
    
    try:
        with torch.no_grad():
            embeddings = inner_model.encode_text(
                texts, 
                task=task,
                truncate_dim=dimensions if dimensions != 2048 else None
            )
        
        # encode_text возвращает list of tensors
        if isinstance(embeddings, list):
            embeddings = torch.stack(embeddings).cpu().float().numpy()
        elif isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().float().numpy()
        
        # L2 нормализация
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-12)
        
        elapsed = time.time() - start
        metrics.record_request(len(texts), elapsed, True)
        
        logger.info(f"Encoded {len(texts)} texts | task={task} | dim={dimensions} | {elapsed:.3f}s")
        
        data = [EmbeddingData(embedding=emb.tolist(), index=i) for i, emb in enumerate(embeddings)]
        tokens = sum(len(t) for t in texts) // 4
        
        return EmbeddingResponse(
            data=data,
            model=request.model,
            usage=UsageInfo(prompt_tokens=tokens, total_tokens=tokens)
        )
    except Exception as e:
        elapsed = time.time() - start
        metrics.record_request(len(texts), elapsed, False)
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"service": "Jina Embeddings v4 API", "docs": "/docs", "metrics": "/metrics"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
