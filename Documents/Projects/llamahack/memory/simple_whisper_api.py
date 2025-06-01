from fastapi import FastAPI, File, UploadFile, Form
import subprocess
import os
import uvicorn
import json

app = FastAPI()

@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...), trigger_keywords: str = Form(None)):
    """
    Transcribe video and conditionally return result based on trigger words.
    
    Args:
        file: The uploaded video file
        trigger_keywords: Comma-separated list of trigger words (optional)
    """
    # Create temp directory
    os.makedirs("temp", exist_ok=True)
    video_path = f"temp/{file.filename}"
    
    # Save uploaded file
    with open(video_path, "wb") as f:
        f.write(await file.read())
    
    # Use whisper directly via command line (avoiding torchaudio dependency)
    # This requires whisper to be installed: pip install -U openai-whisper
    try:
        result = subprocess.run(
            ["whisper", video_path, "--model", "base", "--output_format", "json"],
            capture_output=True,
            text=True,
            check=True
        )
        
        # First try to get the transcript from the JSON file
        json_file = video_path.replace(os.path.splitext(video_path)[1], ".json")
        if os.path.exists(json_file):
            with open(json_file, "r") as f:
                whisper_json = json.load(f)
                transcript = whisper_json.get("text", "").strip()
        else:
            # Fallback to stdout or txt file
            transcript = result.stdout.strip()
            if not transcript:
                txt_file = video_path.replace(os.path.splitext(video_path)[1], ".txt")
                if os.path.exists(txt_file):
                    with open(txt_file, "r") as f:
                        transcript = f.read().strip()
    except subprocess.CalledProcessError as e:
        return {"error": f"Transcription failed: {str(e)}", "stderr": e.stderr}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}
    
    # Parse keywords if provided
    keywords = None
    if trigger_keywords:
        keywords = [k.strip() for k in trigger_keywords.split(",")]
    else:
        keywords = ["remember", "important", "note", "don't forget", 
                   "remind me", "save this", "record this"]

    # Check if we should trigger LLaMA
    should_trigger = any(word.lower() in transcript.lower() for word in keywords)
    
    # Return different responses based on trigger condition
    if should_trigger:
        return {
            "transcript": transcript,
            "trigger_detected": True,
            "message": "Trigger words detected, proceed with LLaMA API call",
            "matched_keywords": [word for word in keywords if word.lower() in transcript.lower()]
        }
    else:
        return {
            "transcript": transcript,
            "trigger_detected": False,
            "message": "No trigger words detected, skipping LLaMA API call"
        }

@app.get("/")
async def root():
    return {"status": "Whisper API is running", "endpoints": ["/transcribe/"]}

if __name__ == "__main__":
    uvicorn.run("simple_whisper_api:app", host="0.0.0.0", port=8000, reload=True)
