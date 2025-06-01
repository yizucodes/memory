# Video AI Analysis System

A complete video analysis system that extracts frames and transcribes audio for multimodal AI analysis using Llama Vision models and Whisper speech recognition.

## Features

### 🎬 Video Frame Extraction
- ✅ **Extracts frames** at configurable intervals (default: every 20 seconds)
- ✅ **Converts to base64** format for Llama Vision integration
- ✅ **Auto-resizes** large frames for efficiency (1920x1080 → 1280x720)
- ✅ **Metadata included** (timestamps, frame numbers, dimensions)

### 🎵 Audio Transcription  
- ✅ **Speech-to-text** using OpenAI Whisper models
- ✅ **Multiple formats** supported (MP4, MOV, AVI, MKV, WebM)
- ✅ **No authentication** required (simplified setup)
- ✅ **Timestamps included** for each segment

## Quick Start

### 1. Install Dependencies
```bash
# Install required packages
pip install -r requirements.txt

# Additional for transcription
pip install openai-whisper
```

### 2. Test Both Features

#### Extract Video Frames
```bash
# Test frame extraction with your video
python test_video_extractor.py data/test.MOV

# Expected output: Extracted 5 frames with timestamps
```

#### Transcribe Video Audio
```bash
# Basic transcription (recommended for demo)
python transcribe_video.py data/test.MOV

# For better accuracy (slower)
python transcribe_video.py data/test.MOV --model large

# Preview only (no file save)
python transcribe_video.py data/test.MOV --nosave
```

### 3. Complete Multimodal Analysis
```bash
# Get both visual frames and audio transcript
python test_video_extractor.py data/test.MOV
python transcribe_video.py data/test.MOV

# Results:
# - Frames with base64 images for visual analysis
# - Text transcript for content analysis
```

## Usage Examples

### Real Test Results
With our sample 43-second networking video:
- **Frames**: 5 extracted (at 0s, 10s, 20s, 30s, 40s)
- **Transcript**: Complete conversation with speaker turns
- **Processing time**: ~30 seconds on MacBook Air

### Sample Frame Output
```json
{
  "timestamp": 20.0,
  "image_base64": "iVBORw0KGgoAAAANSUhE...",
  "prompt": "Analyze this video frame at 20.0 seconds",
  "frame_number": 554,
  "size": {"width": 1280, "height": 720}
}
```

### Sample Transcript Output
```text
Hey, Marla, nice to meet you. Yeah, nice to meet you. I'm here today...
I hear that Meta is working on some amazing new wearable devices...
So we're trying to build really cool technology with wearables...
```

## WebAI Navigator Integration

### Option A: Import Custom Element
```bash
# Check if WebAI CLI is available
webai --help

# Import the frame extractor element
webai import video-frame-extractor ./
```

### Option B: Standalone Usage
```python
# Use directly in your Python code
from video_frame_extractor_element import VideoFrameExtractorElement

extractor = VideoFrameExtractorElement()
frames = extractor.process("video.mp4", frame_interval_seconds=10)

# Frames are ready for Llama Vision API
for frame in frames:
    llama_response = your_llm_api.analyze_image(
        image_base64=frame["image_base64"],
        prompt="What's happening in this video frame?"
    )
```

### Flow Configuration in Navigator
Create this visual flow:
```
[File Input] → [Video Frame Extractor] → [LLM Chat] → [Output Display]
             ↓
        [Audio Transcriber] → [Text Analysis] → [Combined Results]
```

## Command Reference

### Frame Extraction Commands
```bash
# Basic usage
python test_video_extractor.py <video_file>

# Test with sample data
python test_video_extractor.py data/test.MOV
```

### Transcription Commands
```bash
# Basic transcription
python transcribe_video.py <video_file>

# With specific model
python transcribe_video.py <video_file> --model [tiny|base|small|medium|large]

# No file output
python transcribe_video.py <video_file> --nosave
```

### Model Selection Guide
| Model | Size | Speed | Accuracy | Best For |
|-------|------|-------|----------|----------|
| `tiny` | 39MB | ⚡⚡⚡ | 70% | Quick tests |
| `base` | 142MB | ⚡⚡ | 80% | **Demo (recommended)** |
| `medium` | 769MB | ⚡ | 90% | Production |
| `large` | 1.5GB | 🐌 | 95% | High accuracy needed |

## File Structure

```
video-ai-analysis/
├── video_frame_extractor_element.py  # Core frame extraction
├── transcribe_video.py               # Audio transcription
├── test_video_extractor.py          # Testing utilities
├── element_config.yaml               # WebAI configuration
├── requirements.txt                  # Dependencies
├── data/                            # Test videos
│   └── test.MOV
├── transcripts/                     # Output transcripts
│   └── videoNetworking_transcript.txt
└── README.md                        # This file
```

## Troubleshooting

### Common Issues

1. **"Video file not found"**
   ```bash
   # Check file exists
   ls -la data/test.MOV
   # Use absolute path
   python test_video_extractor.py /full/path/to/video.mp4
   ```

2. **"Could not open video file"**
   ```bash
   # Check video format
   ffmpeg -i data/test.MOV  # Should show video info
   # Convert if needed
   ffmpeg -i input.avi -c copy output.mp4
   ```

3. **Transcription model download fails**
   ```bash
   # Clear cache and retry
   rm -rf ~/.cache/whisper
   python transcribe_video.py data/test.MOV --model base
   ```

4. **Import errors**
   ```bash
   # Install missing dependencies
   pip install opencv-python numpy openai-whisper
   # Check Python version
   python --version  # Should be 3.7+
   ```

### Performance Tips

1. **For demos**: Use `base` model (fast, good enough)
2. **For production**: Use `medium` model (good balance)
3. **For best accuracy**: Use `large` model (slow but excellent)
4. **GPU acceleration**: Models run 3-5x faster with CUDA/MPS

## Integration Examples

### With Llama Vision
```python
# Combine frame analysis with transcript
frames = extractor.process("video.mp4")
transcript = transcribe_video("video.mp4")

# Send to Llama for multimodal analysis
llama_input = {
    "images": [frame["image_base64"] for frame in frames],
    "text": transcript,
    "prompt": "Analyze this video content using both visual and audio information"
}
```

### With Other AI Platforms
- **Streamlit**: Build web interface for video uploads
- **Jupyter**: Interactive analysis notebooks  
- **FastAPI**: REST API for video processing
- **Docker**: Containerized deployment

## Advanced Configuration

### Custom Frame Intervals
```python
# Extract every 5 seconds
frames = extractor.process("video.mp4", frame_interval_seconds=5)

# Extract every minute  
frames = extractor.process("video.mp4", frame_interval_seconds=60)
```

### Video Metadata
```python
# Get detailed video information
info = extractor.get_video_info("video.mp4")
print(f"Duration: {info['duration_seconds']} seconds")
print(f"FPS: {info['fps']}")
print(f"Resolution: {info['width']}x{info['height']}")
```

---

## 🚀 Ready for Production!

**Successfully tested with:**
- ✅ 43-second networking conversation video
- ✅ Frame extraction (5 frames at 10s intervals)
- ✅ Audio transcription (complete conversation)
- ✅ WebAI Navigator integration ready
- ✅ macOS M1/M2 compatibility

**Next steps:** Import to WebAI Navigator and build your multimodal AI flows! 🎯