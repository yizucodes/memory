# Video Frame Extractor for WebAI Navigator

A custom WebAI Navigator element that extracts frames from videos for AI analysis using Llama Vision models.

## Features

- ✅ **Extracts frames** at configurable intervals (default: every 20 seconds)
- ✅ **Converts to base64** format for Llama Vision integration
- ✅ **Auto-resizes** large frames for efficiency
- ✅ **Metadata included** (timestamps, frame numbers, dimensions)
- ✅ **Multiple formats** supported (MP4, AVI, MOV, MKV, WebM)

## Quick Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Test the Element
```bash
# Test with your video file
python test_video_extractor.py path/to/your/video.mp4

# Or place a video file named 'sample.mp4' in this directory and run:
python test_video_extractor.py
```

### 3. Import to WebAI Navigator

**Option A: Using WebAI CLI (if available)**
```bash
# Check if WebAI CLI is installed
webai --help

# Import the element
webai import video-frame-extractor ./
```

**Option B: Manual Import in Navigator**
1. Open WebAI Navigator
2. Look for "Import Element" or "Add Custom Element" in the elements drawer
3. Select this directory or upload the files

## Usage in WebAI Navigator

### Flow Configuration
Create this visual flow in Navigator:

```
[File Input] → [Video Frame Extractor] → [LLM Chat] → [Output Display]
```

### Element Settings
- **Video File**: Path to your video file
- **Frame Interval**: Seconds between extracted frames (default: 20)

### Output Format
Each extracted frame provides:
```json
{
  "timestamp": 20.0,
  "image_base64": "iVBORw0KGgoAAAANSUhE...",
  "prompt": "Analyze this video frame at 20.0 seconds",
  "frame_number": 600,
  "size": {"width": 1280, "height": 720}
}
```

## LLM Integration

The frames are ready for Llama Vision models. Example prompts:
- "What's happening in this video frame?"
- "Describe the objects and people in this scene"
- "What changes between these video frames?"

## File Structure

```
├── video_frame_extractor_element.py  # Main element code
├── element_config.yaml               # WebAI element configuration
├── test_video_extractor.py          # Test script
├── requirements.txt                  # Python dependencies
└── README.md                        # This file
```

## Troubleshooting

### Common Issues

1. **"Video file not found"**
   - Check the file path is correct
   - Ensure the video file exists and is readable

2. **"Could not open video file"**
   - Video format might not be supported
   - Try converting to MP4 format

3. **Import errors**
   - Run: `pip install opencv-python numpy`
   - Ensure Python 3.7+ is installed

### Dependencies
- Python 3.7+
- OpenCV (cv2) for video processing
- NumPy for array operations
- Base64 (built-in) for encoding

## Advanced Configuration

### Custom Frame Intervals
```python
# Extract every 5 seconds
extractor.process("video.mp4", frame_interval_seconds=5)

# Extract every minute
extractor.process("video.mp4", frame_interval_seconds=60)
```

### Video Information
```python
# Get video metadata
info = extractor.get_video_info("video.mp4")
print(f"Duration: {info['duration_seconds']} seconds")
print(f"Resolution: {info['width']}x{info['height']}")
```

## WebAI Navigator Integration

This element is designed to work seamlessly with:
- **File Input elements** for video upload
- **LLM Chat elements** for vision analysis
- **Display elements** for results visualization

The extracted frames can be sent directly to Llama Vision models for analysis without additional processing.

---

**Ready to use with WebAI Navigator!** 🎥✨