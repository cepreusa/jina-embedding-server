#!/bin/bash
# =============================================================================
# Jina Embeddings v4 Server - Start Script
# =============================================================================
#
# Usage:
#   ./start.sh                    # OpenAI-compatible (default)
#   ./start.sh jina               # Jina-compatible (base64 support)
#   API_KEY=xxx ./start.sh        # With API key
#   PORT=8000 ./start.sh          # Custom port
#
# =============================================================================

set -e

MODE="${1:-openai}"

echo "=== Jina Embeddings v4 Server ==="

# Install missing dependencies
echo "Checking dependencies..."
pip install -q fastapi uvicorn pydantic torch torchvision transformers einops peft pillow numpy

# Default config
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8080}"
export API_KEY="${API_KEY:-}"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"
export EMBEDDING_TASK="${EMBEDDING_TASK:-text-matching}"
export EMBEDDING_DIM="${EMBEDDING_DIM:-2048}"

echo "Config:"
echo "  Mode: $MODE"
echo "  Host: $HOST:$PORT"
echo "  Auth: ${API_KEY:+enabled}${API_KEY:-disabled}"
echo "  Task: $EMBEDDING_TASK"
echo "  Dim:  $EMBEDDING_DIM"
echo ""

# Start server
if [ "$MODE" = "jina" ]; then
    echo "Starting Jina-compatible server (base64 support)..."
    python3 server_jina.py
else
    echo "Starting OpenAI-compatible server..."
    python3 server.py
fi
