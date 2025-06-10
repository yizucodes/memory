# 🎬 CLI Video Analysis System with Llama 4

A command-line multimodal AI video analysis tool that extracts frames, transcribes audio, and performs intelligent analysis using Llama 4 models. Features a two-step workflow: analyze videos then chat interactively about the results.

## 🚀 Features

- **🖼️ Frame Extraction**: Extract video frames at custom intervals using OpenCV
- **🎵 Audio Transcription**: Transcribe video audio using OpenAI Whisper
- **🤖 Multimodal AI Analysis**: Analyze both visual and audio content with Llama 4
- **💬 Interactive Chat**: Natural language querying of analysis results
- **📊 Multiple Analysis Modes**: Comprehensive, overview, frames-only, or transcript-only
- **🔒 Secure API Management**: Environment-based API key configuration
- **📁 Dual Output**: Human-readable text + machine-readable JSON results
- **⚡ CLI Interface**: Simple command-line tools with flexible options

## 🛠️ Installation

### Prerequisites

1. **Python 3.8+**
2. **FFmpeg** (required for Whisper audio processing):
   ```bash
   # macOS
   brew install ffmpeg
   
   # Ubuntu/Debian
   sudo apt install ffmpeg
   
   # Windows: Download from https://ffmpeg.org/
   ```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## ⚙️ Setup

### 1. Configure API Key

Create your environment file:
```bash
cp .env.example .env
```

Edit `.env` and add your Llama API key:
```
LLAMA4_API_KEY=your_api_key_here
```

### 2. Test API Connection

```bash
python -c "
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(
    api_key=os.getenv('LLAMA4_API_KEY'),
    base_url='https://api.llama.com/compat/v1/'
)

response = client.chat.completions.create(
    model='Llama-4-Maverick-17B-128E-Instruct-FP8',
    messages=[{'role': 'user', 'content': 'Hello!'}]
)
print('✅ API connection successful!')
"
```

## 🚀 Quick Demo (No Setup Required)

Try the interactive chat with real example data:

### **Instant Demo - No Video Processing Needed**

```bash
# Test the interactive chat interface immediately
python interactive_video_chat.py examples/videoNetworking_llama_analysis.json
```

This uses a pre-processed networking conversation analysis, so you can:
- ✅ **Test the chat interface** without API setup
- ✅ **See sample questions** and responses  
- ✅ **Understand output format** before processing your own videos
- ✅ **Demo the system** to others instantly

### **Sample Chat Session**

```bash
$ python interactive_video_chat.py examples/videoNetworking_llama_analysis.json

🎬 Video Analysis Chat - Ask me anything about the video!
Commands: 'quit', 'exit', 'clear', 'context', 'help'
============================================================

💬 You: What were the main topics discussed?
🤖 Llama: [Response based on the networking conversation analysis...]

💬 You: What networking advice would you give?
🤖 Llama: [Insights about the conversation effectiveness...]

💬 You: help
📚 Available Commands:
- quit/exit/q: End the chat
- clear: Clear conversation history  
- context: Show video details
- help: Show this help

💡 Example Questions:
- "What were the main topics discussed?"
- "How did the participants' body language change?"
- "What networking advice would you give?"
- "Summarize the key insights"
```

### **Example Files Included**

- **`examples/videoNetworking_llama_analysis.json`** - Complete analysis data for chat interface
- **`examples/videoNetworking_llama_analysis.txt`** - Human-readable analysis results  
- **`examples/videoNetworking_transcript.txt`** - Raw transcript for reference

### **Try These Example Questions**

```bash
# Start the demo
python interactive_video_chat.py examples/videoNetworking_llama_analysis.json

# Try asking:
"What were the main topics discussed?"
"How effective was this networking conversation?"
"What follow-up actions were mentioned?"
"What could have been improved?"
"Summarize the key insights from this conversation"
```

## 🎯 Command Line Usage

### Basic Commands

**Quick transcript analysis:**
```bash
python llama_video_analyzer.py data/your_video.MOV --mode transcript_only
```

**Visual frame analysis:**
```bash
python llama_video_analyzer.py data/your_video.MOV --mode frames_only
```

**Complete multimodal analysis:**
```bash
python llama_video_analyzer.py data/your_video.MOV --mode comprehensive
```

**Fast overview (recommended for demos):**
```bash
python llama_video_analyzer.py data/your_video.MOV --mode overview
```

### Command Line Options

```bash
python llama_video_analyzer.py <video_file> [options]

Required:
  video_file             Path to video file (MP4, MOV, AVI, etc.)

Options:
  --interval SECONDS     Frame extraction interval (default: 20)
  --whisper MODEL        Whisper model: tiny,base,small,medium,large (default: base)
  --mode MODE            Analysis mode: comprehensive,frames_only,transcript_only,overview
  --output FILE          Output file prefix

Examples:
  # High-quality analysis
  python llama_video_analyzer.py meeting.MOV --interval 10 --whisper medium
  
  # Quick demo mode
  python llama_video_analyzer.py presentation.MP4 --mode overview --interval 30
  
  # Custom output filename
  python llama_video_analyzer.py interview.MOV --output job_interview_analysis
  
  # Transcript only for fast text analysis
  python llama_video_analyzer.py call.MOV --mode transcript_only --whisper large
```

## 📊 Analysis Modes

| Mode | Speed | API Calls | Use Case |
|------|-------|-----------|----------|
| **transcript_only** | ⚡ Fast | 1 | Text analysis, quick insights |
| **overview** | 🚀 Medium | 1 | Demo-ready multimodal analysis |
| **frames_only** | ⏱️ Medium | N frames | Visual-focused analysis |
| **comprehensive** | 🔍 Detailed | N+2 calls | Complete research analysis |

## 📁 Output Files

Each analysis generates two files:

- **`filename_llama_analysis.txt`** - Human-readable results
- **`filename_llama_analysis.json`** - Machine-readable data

### Sample CLI Workflow

```bash
# 1. Analyze networking video
python llama_video_analyzer.py data/networking_call.MOV --mode comprehensive

# 2. View results
cat networking_call_llama_analysis.txt

# 3. Process JSON data
python -c "import json; data=json.load(open('networking_call_llama_analysis.json')); print(f'Frames: {data[\"frames_extracted\"]}, Transcript: {data[\"transcript_length\"]} chars')"
```

### Sample Output Structure

```json
{
  "video_path": "data/networking_video.MOV",
  "frames_extracted": 5,
  "transcript_length": 2196,
  "analysis": {
    "individual_frames": [...],
    "comprehensive": "...",
    "transcript_only": "..."
  }
}
```

## 🏗️ Architecture

```
CLI Command → Video Input → [Frame Extractor] → Base64 Images
                          ↓
                       [Whisper] → Transcript
                          ↓
                       [Llama 4] → Analysis
                          ↓
                      [Output] → .txt + .json files
```

## 🔧 Technical Details

### Frame Processing
- **Resolution**: Auto-resize 1920x1080 → 1280x720
- **Format**: JPEG with base64 encoding
- **Timestamps**: Precise frame timing metadata
- **Intervals**: Configurable extraction frequency

### Audio Processing  
- **Engine**: OpenAI Whisper (local processing)
- **Models**: tiny, base, small, medium, large
- **Formats**: Supports all major video formats
- **Quality**: Automatic audio extraction via FFmpeg

### AI Analysis
- **Context**: Full transcript provided to each frame analysis
- **Focus Areas**: Networking, meetings, professional communication
- **Output**: Structured insights on dynamics, body language, effectiveness

## 🚨 Troubleshooting

### Common CLI Issues

**401 Authentication Error:**
```bash
# Check API key is loaded
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('Key loaded:', bool(os.getenv('LLAMA4_API_KEY')))"
```

**500 Inference Error (too many frames):**
```bash
# Use fewer frames
python llama_video_analyzer.py video.MOV --mode overview --interval 60
```

**FFmpeg Not Found:**
```bash
# Install FFmpeg first
brew install ffmpeg  # macOS
sudo apt install ffmpeg  # Linux
```

**File Not Found:**
```bash
# Check video file path
ls -la data/your_video.MOV
```

**Permission Issues:**
```bash
# Make script executable
chmod +x llama_video_analyzer.py
```

## 📝 Example: CLI Analysis Workflow

```bash
# Step 1: Quick transcript check
python llama_video_analyzer.py data/meeting.MOV --mode transcript_only

# Step 2: If transcript looks good, run full analysis
python llama_video_analyzer.py data/meeting.MOV --mode comprehensive --whisper medium
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI Whisper** for speech-to-text capabilities
- **Llama 4** for multimodal AI analysis  
- **OpenCV** for video frame processing

---

**Pure CLI power for video analysis! 🚀**