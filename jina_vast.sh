#!/bin/bash
# =========================================
# Jina Embeddings v4 Provisioning Script for Vast.ai
# =========================================
#
# Usage in vast.ai:
#   vastai create instance <OFFER_ID> \
#     --image vastai/base-image:@vastai-automatic-tag \
#     --env '-p 8080:8080 -e JINA_PORT=8080 -e API_KEY=your-key -e SCRIPT_URL=https://raw.githubusercontent.com/user/repo/main/server.py -e PROVISIONING_SCRIPT=https://raw.githubusercontent.com/.../jina_vast.sh' \
#     --onstart-cmd 'entrypoint.sh' \
#     --disk 50 --ssh --direct
#
# =========================================

set -e

# =========================================
# Переменные
# =========================================
UBUNTU_HOME="/home/ubuntu"
JINA_DIR="${UBUNTU_HOME}/jina-v4-server"
VENV_DIR="${UBUNTU_HOME}/venv"
LOG_DIR="/var/log/jina-embeddings"
PROVISION_LOG="${LOG_DIR}/provisioning.log"
SUPERVISOR_CONF="/etc/supervisor/conf.d/jina-embeddings.conf"

# URL скрипта сервера (обязательный параметр)
SCRIPT_URL="${SCRIPT_URL:-}"

# Порт сервера
JINA_PORT="${JINA_PORT:-8080}"

# API ключ
API_KEY="${API_KEY:-}"

# Настройки модели
EMBEDDING_TASK="${EMBEDDING_TASK:-text-matching}"
EMBEDDING_DIM="${EMBEDDING_DIM:-2048}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-32}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

APT_PACKAGES=(
    curl
    supervisor
)

# =========================================
# Подготовка логирования
# =========================================
mkdir -p "$LOG_DIR"

# =========================================
# Логирование
# =========================================
exec > >(tee -a "$PROVISION_LOG") 2>&1

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" >&2
}

# =========================================
# Обработка ошибок
# =========================================
cleanup_on_error() {
    log_error "Provisioning failed! Check logs: $PROVISION_LOG"
    exit 1
}

trap cleanup_on_error ERR

# =========================================
# Основные функции
# =========================================

provisioning_print_header() {
    log "##############################################"
    log "# Starting Jina Embeddings v4 provisioning..."
    log "# Port: ${JINA_PORT}"
    log "# API Key: ${API_KEY:+enabled}${API_KEY:-disabled}"
    log "# Script URL: ${SCRIPT_URL:-NOT SET}"
    log "##############################################"
}

provisioning_print_end() {
    log "##############################################"
    log "# Provisioning complete!"
    log "# Server: http://localhost:${JINA_PORT}"
    log "# Health: http://localhost:${JINA_PORT}/health"
    log "# Metrics: http://localhost:${JINA_PORT}/metrics"
    log "# Docs: http://localhost:${JINA_PORT}/docs"
    log "# Logs: ${LOG_DIR}/jina-embeddings.log"
    log "##############################################"
}

provisioning_check_requirements() {
    log "Проверяем обязательные параметры..."
    
    if [[ -z "$SCRIPT_URL" ]]; then
        log_error "SCRIPT_URL не задан! Укажите URL к server.py"
        log_error "Пример: -e SCRIPT_URL=https://raw.githubusercontent.com/user/repo/main/server.py"
        exit 1
    fi
    
    log "✓ Параметры проверены"
}

provisioning_create_ubuntu_user() {
    log "Проверяем пользователя ubuntu..."

    if id ubuntu &>/dev/null; then
        log "Пользователь ubuntu найден"
        
        if ! id -nG ubuntu | grep -qw "sudo"; then
            log "Добавляем ubuntu в группу sudo..."
            usermod -aG sudo ubuntu
        fi

        if [[ ! -f /etc/sudoers.d/90-ubuntu ]]; then
            echo "ubuntu ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-ubuntu
            chmod 440 /etc/sudoers.d/90-ubuntu
        fi
    else
        log "Создаём пользователя ubuntu..."
        adduser --disabled-password --gecos "" ubuntu
        usermod -aG sudo ubuntu
        echo "ubuntu ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-ubuntu
        chmod 440 /etc/sudoers.d/90-ubuntu
    fi
    
    log "✓ Пользователь готов"
}

provisioning_prepare_dirs() {
    log "Создаём директории..."
    
    chown ubuntu:ubuntu "$LOG_DIR"
    
    sudo -u ubuntu mkdir -p "${JINA_DIR}"
    
    # HuggingFace cache - vast.ai использует /workspace/.hf_home
    if [[ -d "/workspace" ]]; then
        mkdir -p /workspace/.hf_home
        chown -R ubuntu:ubuntu /workspace/.hf_home
    fi
    
    # Fallback - домашняя директория
    sudo -u ubuntu mkdir -p "${UBUNTU_HOME}/.cache/huggingface"
    
    log "✓ Директории созданы"
}

provisioning_get_apt_packages() {
    log "Устанавливаем системные пакеты..."
    apt-get update -qq
    apt-get install -y -qq "${APT_PACKAGES[@]}"
    log "✓ Пакеты установлены"
}

provisioning_setup_venv() {
    log "Настраиваем Python окружение..."
    
    if [[ ! -d "${VENV_DIR}" ]]; then
        sudo -u ubuntu python3 -m venv "${VENV_DIR}"
    fi
    
    sudo -u ubuntu "${VENV_DIR}/bin/pip" install --upgrade pip -q
    
    log "✓ Виртуальное окружение готово"
}

provisioning_install_dependencies() {
    log "Устанавливаем Python зависимости..."
    
    sudo -u ubuntu "${VENV_DIR}/bin/pip" install -q \
        fastapi \
        uvicorn \
        pydantic \
        torch \
        torchvision \
        transformers \
        einops \
        peft \
        pillow \
        numpy
    
    log "✓ Зависимости установлены"
}

provisioning_download_server() {
    log "Скачиваем server.py из ${SCRIPT_URL}..."
    
    if ! curl -fsSL -o "${JINA_DIR}/server.py" "$SCRIPT_URL"; then
        log_error "Не удалось скачать server.py"
        exit 1
    fi
    
    chown ubuntu:ubuntu "${JINA_DIR}/server.py"
    chmod +x "${JINA_DIR}/server.py"
    
    log "✓ server.py скачан"
}

provisioning_preload_model() {
    log "Предзагружаем модель jinaai/jina-embeddings-v4..."
    log "Это займёт 3-5 минут (~8GB)..."
    
    if ! sudo -u ubuntu "${VENV_DIR}/bin/python" -c "
from transformers import AutoModel
import torch

print('Downloading model...')
model = AutoModel.from_pretrained(
    'jinaai/jina-embeddings-v4',
    trust_remote_code=True,
    torch_dtype=torch.float16
)
print('Model downloaded successfully!')
"; then
        log_error "Не удалось загрузить модель"
        exit 1
    fi
    
    log "✓ Модель загружена в кэш"
}

provisioning_setup_supervisor() {
    log "Настраиваем Supervisor..."
    
    cat > "$SUPERVISOR_CONF" <<EOL
[program:jina-embeddings]
directory=${JINA_DIR}
command=${VENV_DIR}/bin/python server.py
autostart=true
autorestart=true
startsecs=60
startretries=3
stdout_logfile=${LOG_DIR}/jina-embeddings.log
stderr_logfile=${LOG_DIR}/jina-embeddings.err
stopsignal=TERM
user=ubuntu
environment=HOME="${UBUNTU_HOME}",PYTHONUNBUFFERED="1",PORT="${JINA_PORT}",API_KEY="${API_KEY}",EMBEDDING_TASK="${EMBEDDING_TASK}",EMBEDDING_DIM="${EMBEDDING_DIM}",MAX_BATCH_SIZE="${MAX_BATCH_SIZE}",LOG_LEVEL="${LOG_LEVEL}"
EOL

    supervisorctl reread
    supervisorctl update
    supervisorctl start jina-embeddings || true

    log "✓ Supervisor настроен"
}

provisioning_wait_ready() {
    log "Ожидаем готовности сервера (до 5 минут)..."
    
    for i in {1..60}; do
        if curl -s "http://localhost:${JINA_PORT}/health" | grep -q "healthy"; then
            log "✓ Сервер готов!"
            
            # Показываем статус
            curl -s "http://localhost:${JINA_PORT}/health" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f\"  Model: {d.get('model')}\" )
print(f\"  Device: {d.get('device')}\")
print(f\"  GPU Memory: {d.get('gpu_memory_used_mb', 'N/A')} / {d.get('gpu_memory_total_mb', 'N/A')} MB\")
" 2>/dev/null || true
            
            return 0
        fi
        log "  Ожидание... ($i/60)"
        sleep 5
    done
    
    log_error "Сервер не ответил за 5 минут"
    log_error "Проверьте логи: tail -f ${LOG_DIR}/jina-embeddings.err"
    return 1
}

# =========================================
# Основной процесс
# =========================================
provisioning_start() {
    provisioning_print_header
    provisioning_check_requirements
    provisioning_create_ubuntu_user
    provisioning_prepare_dirs
    provisioning_get_apt_packages
    provisioning_setup_venv
    provisioning_install_dependencies
    provisioning_download_server
    provisioning_preload_model
    provisioning_setup_supervisor
    provisioning_wait_ready
    provisioning_print_end
}

# =========================================
# Запуск
# =========================================
if [[ ! -f /.noprovisioning ]]; then
    provisioning_start
fi
