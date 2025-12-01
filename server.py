"""
Jina Embeddings v4 - OpenAI-compatible API Server
"""

import os
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel, Field

# Configuration
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

# Prompts for different tasks (from Jina documentation)
TASK_PROMPTS = {
    "text-matching": "text-matching",
    "retrieval.query": "retrieval.query",
    "retrieval.passage": "retrieval.passage",
    "code": "code",
}

model = None
inner_model = None


async def verify_api_key(authorization: Optional[str] = Header(None)):
    if not API_KEY:
        return True
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = authorization[7:] if authorization.lower().startswith("bearer ") else authorization
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


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
    
    # We obtain an internal model with encode_text methods
    if hasattr(model, 'get_base_model'):
        inner_model = model.get_base_model()
    else:
        inner_model = model.base_model.model
    
    logger.info(f"Модель загружена за {time.time() - start_time:.1f}s")
    logger.info(f"Inner model: {type(inner_model).__name__}")
    
    # Warm-up
    logger.info("Прогрев модели...")
    with torch.no_grad():
        _ = inner_model.encode_text(["warmup"], task=DEFAULT_TASK)
    logger.info("Модель готова")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="Jina Embeddings v4 API", lifespan=lifespan)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "default_task": DEFAULT_TASK,
        "default_dim": DEFAULT_DIM,
        "auth_enabled": bool(API_KEY)
    }


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
    
    # Defining the task
    task = request.task or DEFAULT_TASK
    if task == "retrieval" and request.prompt_name:
        task = f"retrieval.{request.prompt_name}"
    
    dimensions = request.dimensions or DEFAULT_DIM
    if dimensions not in VALID_DIMENSIONS:
        raise HTTPException(status_code=400, detail=f"Invalid dimensions. Valid: {sorted(VALID_DIMENSIONS)}")
    
    try:
        start = time.time()
        
        with torch.no_grad():
            embeddings = inner_model.encode_text(
                texts, 
                task=task,
                truncate_dim=dimensions if dimensions != 2048 else None
            )
        
        # encode_text returns a list of tensors
        if isinstance(embeddings, list):
            embeddings = torch.stack(embeddings).cpu().float().numpy()
        elif isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().float().numpy()
        
        # L2 normalization
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-12)
        
        elapsed = time.time() - start
        logger.info(f"Encoded {len(texts)} texts | task={task} | dim={dimensions} | {elapsed:.3f}s")
        
        data = [EmbeddingData(embedding=emb.tolist(), index=i) for i, emb in enumerate(embeddings)]
        tokens = sum(len(t) for t in texts) // 4
        
        return EmbeddingResponse(
            data=data,
            model=request.model,
            usage=UsageInfo(prompt_tokens=tokens, total_tokens=tokens)
        )
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"service": "Jina Embeddings v4 API", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
