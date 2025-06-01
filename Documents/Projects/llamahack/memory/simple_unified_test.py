import subprocess
import os
import json
import sys

def transcribe_video(video_path, trigger_keywords=None):
    """
    Transcribe a video file using Whisper and check for trigger words
    
    Args:
        video_path: Path to the video file
        trigger_keywords: Optional comma-separated list of trigger words
    """
    print(f"Processing video: {video_path}")
    
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
                print(f"\nTranscript from JSON: {transcript}")
        else:
            # Fallback to stdout or txt file
            transcript = result.stdout.strip()
            if not transcript:
                txt_file = video_path.replace(os.path.splitext(video_path)[1], ".txt")
                if os.path.exists(txt_file):
                    with open(txt_file, "r") as f:
                        transcript = f.read().strip()
                    print(f"\nTranscript from TXT: {transcript}")
    except subprocess.CalledProcessError as e:
        print(f"Transcription failed: {str(e)}")
        print(f"Error output: {e.stderr}")
        return None
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return None
    
    # Parse keywords if provided
    keywords = None
    if trigger_keywords:
        keywords = [k.strip() for k in trigger_keywords.split(",")]
        print(f"Using custom keywords: {keywords}")
    else:
        keywords = ["remember", "important", "note", "don't forget", 
                   "remind me", "save this", "record this"]
        print(f"Using default keywords: {keywords}")

    # Check if we should trigger LLaMA
    matched_keywords = [word for word in keywords if word.lower() in transcript.lower()]
    should_trigger = len(matched_keywords) > 0
    
    # If trigger detected, extract frames (optional)
    frames = []
    if should_trigger and os.path.exists(video_path):
        # Extract a few key frames (optional)
        try:
            frames_dir = f"temp/frames_{os.path.basename(video_path)}"
            os.makedirs(frames_dir, exist_ok=True)
            
            print(f"Extracting frames to {frames_dir}...")
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
                print(f"Extracted {len(frames)} frames")
        except Exception as e:
            print(f"Frame extraction error: {str(e)}")
    
    # Return different responses based on trigger condition
    result = {
        "transcript": transcript,
        "trigger_detected": should_trigger,
        "message": "Trigger words detected, proceed with LLaMA API call" if should_trigger else "No trigger words detected, skipping LLaMA API call",
    }
    
    if should_trigger:
        result["matched_keywords"] = matched_keywords
        result["frames"] = frames[:5] if frames else []  # Limit to 5 frames
    
    return result

def process_memory(transcript, frames=[]):
    """
    Process a memory with LLaMA (placeholder for your LLaMA integration)
    
    Args:
        transcript: The transcript text
        frames: Optional list of frame paths
    """
    print("\n--- Processing Memory with LLaMA ---")
    print(f"Transcript: {transcript[:100]}..." if len(transcript) > 100 else f"Transcript: {transcript}")
    print(f"Number of frames: {len(frames)}")
    
    # This is where you would call LLaMA API
    # For now, we'll just return a placeholder response
    
    memory_result = {
        "memory_id": "mem_12345",
        "summary": f"Memory created from transcript: {transcript[:50]}...",
        "processed_frames": len(frames),
        "status": "success"
    }
    
    print(f"Memory created with ID: {memory_result['memory_id']}")
    print(f"Summary: {memory_result['summary']}")
    
    return memory_result

def main():
    # Create temp directory
    os.makedirs("temp", exist_ok=True)
    
    # Check if a video path was provided as an argument
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        # Default to a sample video if none provided
        video_path = "../IMG_4032.MOV"
    
    print(f"\n=== Testing Unified Memory Assistant Pipeline ===")
    print(f"Video path: {video_path}")
    
    # Step 1: Transcribe video and check for triggers
    print("\n--- Step 1: Transcribing Video ---")
    transcription_result = transcribe_video(video_path, "remember,important,note")
    
    if not transcription_result:
        print("Transcription failed, exiting.")
        return
    
    print(f"\nTranscription complete!")
    print(f"Transcript: {transcription_result['transcript']}")
    print(f"Trigger detected: {transcription_result['trigger_detected']}")
    print(f"Message: {transcription_result['message']}")
    
    # Step 2: If trigger detected, process memory
    if transcription_result['trigger_detected']:
        print(f"\nMatched keywords: {transcription_result.get('matched_keywords', [])}")
        if 'frames' in transcription_result and transcription_result['frames']:
            print(f"Extracted {len(transcription_result['frames'])} frames for processing")
        
        # Process memory with LLaMA
        print("\n--- Step 2: Processing Memory ---")
        memory_result = process_memory(
            transcription_result['transcript'], 
            transcription_result.get('frames', [])
        )
    else:
        print("\nNo trigger words detected, skipping memory processing.")
        print("To test memory processing, add trigger words to your video or modify the keywords.")
        
        # Force a test of memory processing anyway
        print("\n--- Forcing memory processing for demonstration ---")
        memory_result = process_memory(transcription_result['transcript'])
    
    print("\n=== Test Complete ===")
    print("This demonstrates the full pipeline that would run on your Intel server.")
    print("In WebAI Navigator, you would configure API elements to call these endpoints.")

if __name__ == "__main__":
    main()
