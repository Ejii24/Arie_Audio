FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS builder

ENV PYTHONUNBUFFERED=1 DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    git build-essential libsndfile1-dev ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip

# Install PyTorch with CUDA 12.4
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.5.1 torchaudio==2.5.1

# Install audio processing libs and runpod (latest stable)
RUN pip install --no-cache-dir \
    transformers==4.48.0 \
    librosa==0.10.2.post1 \
    soundfile==0.12.1 \
    numpy==1.26.4 \
    pydantic==2.9.0 \
    accelerate==0.34.2 \
    huggingface-hub==0.27.0 \
    runpod==1.9.1

# Runtime stage
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y \
    python3.11 ffmpeg libsndfile1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY handler.py .

EXPOSE 8000

CMD ["python3.11", "-u", "handler.py"]