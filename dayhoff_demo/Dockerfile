FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/tmp/huggingface \
    TRANSFORMERS_CACHE=/tmp/huggingface \
    BIOEMU_OUTPUT_BASE_DIR=/tmp/bioemu_runs

RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY dayhoff_demo/requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -U pip \
    && python -m pip install --no-cache-dir -r /app/requirements.txt \
    && (python -m pip uninstall -y torchvision || true)

COPY dayhoff_demo /app/dayhoff_demo
COPY images /app/images

WORKDIR /app/dayhoff_demo

EXPOSE 7860

CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "7860"]
