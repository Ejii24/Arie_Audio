FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS builder
ENV PYTHONUNBUFFERED=1 DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3.11 python3.11-dev python3.11-venv python3-pip git build-essential libsndfile1-dev ffmpeg && rm -rf /var/lib/apt/lists/*
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 torch==2.8.0 torchaudio==2.8.0
RUN pip install --no-cache-dir transformers==4.48.0 librosa soundfile numpy fastapi uvicorn pydantic accelerate huggingface-hub

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04
ENV PYTHONUNBUFFERED=1 PATH="/opt/venv/bin:$PATH"
RUN apt-get update && apt-get install -y python3.11 ffmpeg libsndfile1 ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY handler.py .
EXPOSE 8000
CMD ["python", "-u", "handler.py"]
