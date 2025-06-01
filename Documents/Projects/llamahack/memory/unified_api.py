from fastapi import FastAPI, File, UploadFile, Form
import subprocess
import os
import uvicorn
import json
import requests

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
    
    # Use whisper directly via command line
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
    
    # If trigger detected, extract frames (optional)
    frames = []
    if should_trigger and os.path.exists(video_path):
        # Extract a few key frames (optional)
        try:
            frames_dir = f"temp/frames_{os.path.basename(video_path)}"
            os.makedirs(frames_dir, exist_ok=True)
            
            # Extract frames at 1-second intervals
            subprocess.run([
                "ffmpeg", "-i", video_path, "-vf", "fps=1", 
                f"{frames_dir}/frame_%03d.jpg"
            ], check=True, capture_output=True)
            
            # Get paths to extracted frames
            if os.path.exists(frames_dir):
                frames = [f"{frames_dir}/{f}" for f in os.listdir(frames_dir) 
                          if f.endswith(".jpg")]
                frames.sort()
        except Exception as e:
            print(f"Frame extraction error: {str(e)}")
    
    # Return different responses based on trigger condition
    if should_trigger:
        return {
            "transcript": transcript,
            "trigger_detected": True,
            "message": "Trigger words detected, proceed with LLaMA API call",
            "matched_keywords": [word for word in keywords if word.lower() in transcript.lower()],
            "frames": frames[:5] if frames else []  # Limit to 5 frames
        }
    else:
        return {
            "transcript": transcript,
            "trigger_detected": False,
            "message": "No trigger words detected, skipping LLaMA API call"
        }

@app.post("/process_memory/")
async def process_memory(transcript: str = Form(...), frames: list = Form([])):
    """
    Process a memory with LLaMA (placeholder for your LLaMA integration)
    
    Args:
        transcript: The transcript text
        frames: Optional list of frame paths
    """
    # This is where you would call LLaMA API
    # For now, we'll just return a placeholder response
    
    return {
        "memory_id": "mem_12345",
        "summary": f"Memory created from transcript: {transcript[:50]}...",
        "processed_frames": len(frames),
        "status": "success"
    }

@app.get("/")
async def root():
    return {
        "status": "Unified Memory Assistant API is running", 
        "endpoints": ["/transcribe/", "/process_memory/"]
    }

if __name__ == "__main__":
    uvicorn.run("unified_api:app", host="0.0.0.0", port=8000, reload=True)
