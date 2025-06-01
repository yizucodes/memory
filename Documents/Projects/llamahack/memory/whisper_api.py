from fastapi import FastAPI, File, UploadFile
import ffmpeg
import torchaudio
import torch
import os
import uvicorn
from transformers import WhisperProcessor, WhisperForConditionalGeneration

app = FastAPI()

# Load Whisper model once at startup
processor = WhisperProcessor.from_pretrained("openai/whisper-base")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base").to("cpu")

def should_trigger_llama(transcript, keywords=None):
    """Determine if LLaMA should be triggered based on transcript content."""
    if keywords is None:
        keywords = ["remember", "important", "note", "don't forget", 
                   "remind me", "save this", "record this"]
    
    return any(word in transcript.lower() for word in keywords)

@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...), trigger_keywords: str = None):
    """
    Transcribe video and conditionally return result based on trigger words.
    
    Args:
        file: The uploaded video file
        trigger_keywords: Comma-separated list of trigger words (optional)
    """
    # Create temp directory
    os.makedirs("temp", exist_ok=True)
    video_path = f"temp/{file.filename}"
    audio_path = video_path.replace(".mp4", "_audio.wav")

    # Save uploaded file
    with open(video_path, "wb") as f:
        f.write(await file.read())

    # Extract audio
    ffmpeg.input(video_path).output(audio_path, ac=1, ar='16k').run(overwrite_output=True)

    # Load and process audio
    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = waveform.squeeze().numpy()

    # Transcribe with Whisper
    input_features = processor(waveform, sampling_rate=sample_rate, return_tensors="pt").input_features.to("cpu")
    predicted_ids = model.generate(input_features)
    transcript = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    # Parse keywords if provided
    keywords = None
    if trigger_keywords:
        keywords = [k.strip() for k in trigger_keywords.split(",")]

    # Check if we should trigger LLaMA
    should_trigger = should_trigger_llama(transcript, keywords)
    
    # Return different responses based on trigger condition
    if should_trigger:
        return {
            "transcript": transcript,
            "trigger_detected": True,
            "message": "Trigger words detected, proceed with LLaMA API call"
        }
    else:
        return {
            "transcript": transcript,
            "trigger_detected": False,
            "message": "No trigger words detected, skipping LLaMA API call"
        }

if __name__ == "__main__":
    uvicorn.run("whisper_api:app", host="0.0.0.0", port=8000, reload=True)
