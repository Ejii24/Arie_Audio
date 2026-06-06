#!/usr/bin/env python3
import os
import torch
import logging
import io
import base64
import librosa
from transformers import WhisperForConditionalGeneration, WhisperProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("arie-audio")

model = None
processor = None

def load_model():
    global model, processor
    if model is not None:
        return
    
    model_id = os.getenv("MODEL_ID", "SoundSource/arie-audio")
    hf_token = os.getenv("HF_TOKEN")
    
    logger.info(f"Loading {model_id}...")
    processor = WhisperProcessor.from_pretrained(model_id, token=hf_token, trust_remote_code=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        token=hf_token,
        trust_remote_code=True,
    )
    model.eval()
    logger.info("✅ Model loaded")

def handler(job):
    try:
        load_model()
        job_input = job["input"]
        audio_base64 = job_input.get("audio_base64")
        
        if not audio_base64:
            return {"error": "audio_base64 required"}
        
        audio_bytes = base64.b64decode(audio_base64)
        audio_buffer = io.BytesIO(audio_bytes)
        arr, sr = librosa.load(audio_buffer, sr=None)
        arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
        
        with torch.no_grad():
            features = processor(arr, sampling_rate=16000, return_tensors="pt").input_features
            features = features.to(model.device).to(model.dtype)
            outputs = model.generate(input_features=features, task="transcribe", max_new_tokens=225)
            text = processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        
        return {"text": text, "model": "arie-audio-v1"}
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return {"error": str(e)}
