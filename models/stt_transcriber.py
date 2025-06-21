from transformers import pipeline
import torch
asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=0 if torch.cuda.is_available() else -1)

def speech_to_text(audio_path: str) -> str:
    result = asr_pipeline(audio_path)
    return result["text"]