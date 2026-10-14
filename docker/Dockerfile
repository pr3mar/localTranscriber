FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models/hf \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv python3-dev build-essential ffmpeg libsndfile1 \
 && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv

COPY docker/requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir -r /tmp/requirements.lock

WORKDIR /app
COPY scripts/spike_gpu.py scripts/spike_gpu.py
