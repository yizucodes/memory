#!/usr/bin/env python3
"""
Video transcription using OpenAI Whisper (no HuggingFace auth needed)
"""

import whisper
import sys
import os

def transcribe_video_simple(video_path, model_size="base"):
    """Simple transcription using OpenAI Whisper"""
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return None
    
    print(f"🎵 Loading Whisper model '{model_size}'...")
    model = whisper.load_model(model_size)  # "tiny", "base", "small", "medium", "large"
    
    print(f"🎥 Transcribing: {video_path}")
    print("This may take a few minutes...")
    
    # Whisper handles video files directly!
    result = model.transcribe(video_path)
    
    # Save transcript
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_file = f"{base_name}_transcript.txt"
    
    with open(output_file, 'w') as f:
        f.write(result["text"])
    
    print(f"✅ Transcript saved to: {output_file}")
    print(f"📝 Preview: {result['text'][:200]}...")
    
    return result["text"]

def main():
    if len(sys.argv) < 2:
        print("Usage: python simple_transcribe.py <video_file> [model_size]")
        print("Model sizes: tiny, base, small, medium, large")
        print("Example: python simple_transcribe.py data/test.MOV base")
        return
    
    video_path = sys.argv[1]
    model_size = sys.argv[2] if len(sys.argv) > 2 else "base"
    
    print(f"🎬 Simple Video Transcription")
    print(f"Video: {video_path}")
    print(f"Model: {model_size}")
    print("=" * 40)
    
    transcribe_video_simple(video_path, model_size)

if __name__ == "__main__":
    main() 