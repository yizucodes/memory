# 🎬 Video Analysis System for WebAI Navigator

A multimodal AI video analysis system that extracts frames, transcribes audio, and performs intelligent analysis using Llama 4 models. Built for hackathon demos and WebAI Navigator integration.

## 🚀 Features

- **🖼️ Frame Extraction**: Extract video frames at custom intervals using OpenCV
- **🎵 Audio Transcription**: Transcribe video audio using OpenAI Whisper
- **🤖 Multimodal AI Analysis**: Analyze both visual and audio content with Llama 4
- **📊 Multiple Analysis Modes**: Comprehensive, overview, frames-only, or transcript-only
- **🔒 Secure API Management**: Environment-based API key configuration
- **📁 Dual Output**: Human-readable text + machine-readable JSON results
- **🔧 WebAI Navigator Ready**: Pre-configured for visual workflow integration

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

## 🎯 Usage

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

### Advanced Options

```bash
python llama_video_analyzer.py <video_file> [options]

Options:
  --interval SECONDS     Frame extraction interval (default: 20)
  --whisper MODEL        Whisper model: tiny,base,small,medium,large (default: base)
  --mode MODE            Analysis mode: comprehensive,frames_only,transcript_only,overview
  --output FILE          Output file prefix

Examples:
  # High-quality analysis
  python llama_video_analyzer.py video.MOV --interval 10 --whisper medium
  
  # Quick demo mode
  python llama_video_analyzer.py video.MOV --mode overview --interval 30
  
  # Custom output filename
  python llama_video_analyzer.py video.MOV --output meeting_analysis
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

## 🏆 Hackathon Integration

### Llama 4 Models Used
- **Multimodal Analysis**: `Llama-4-Maverick-17B-128E-Instruct-FP8`
- **Text Analysis**: `Llama-4-Maverick-17B-128E-Instruct-FP8`
- **Context Window**: 128K tokens
- **Capabilities**: Text + Image input, multimodal reasoning

### WebAI Navigator
- **Element Config**: `element_config.yaml` 
- **Visual Workflow**: Drag-and-drop video analysis
- **Integration**: Works standalone or in visual pipelines

## 🏗️ Architecture

```
Video Input → [Frame Extractor] → Base64 Images
            ↓
          [Whisper] → Transcript
            ↓
         [Llama 4] → Analysis
            ↓
    [Output] → .txt + .json
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

### Common Issues

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

**Whisper Model Download:**
```bash
# Models download automatically on first use
# Ensure stable internet connection
```

## 📝 Example: Networking Video Analysis

```bash
# Analyze a networking conversation
python llama_video_analyzer.py data/networking_call.MOV --mode comprehensive --whisper medium

# Output: networking_call_llama_analysis.txt
# - Conversation flow analysis
# - Body language insights  
# - Meeting effectiveness scores
# - Professional communication assessment
# - Networking outcome recommendations
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI Whisper** for speech-to-text capabilities
- **Llama 4** for multimodal AI analysis  
- **OpenCV** for video frame processing
- **WebAI Navigator** for visual workflow integration

---

**Built for hackathons, optimized for insights! 🚀**