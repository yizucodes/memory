# Memory Assistant Deployment Guide

This guide will help you deploy the unified Memory Assistant API on your Intel server and connect to it from your M1 device running WebAI Navigator.

## 1. Setup on Intel Server

### Prerequisites

Make sure you have the following installed on your Intel server:

- Python 3.11+ (avoid Python 3.13 if possible due to compatibility issues)
- pip (package manager)
- ffmpeg (for video processing)
- Whisper CLI (for transcription)

### Installation Steps

1. **Clone or copy the repository to your Intel server**

   Copy the entire `memory` directory to your Intel server.

2. **Install dependencies**

   ```bash
   cd memory
   pip install -r requirements.txt
   ```

   If you encounter any issues with the dependencies, install them individually:

   ```bash
   pip install fastapi uvicorn python-multipart requests
   ```

3. **Install Whisper if not already installed**

   ```bash
   pip install openai-whisper
   ```

4. **Install ffmpeg if not already installed**

   On Ubuntu/Debian:
   ```bash
   sudo apt-get update
   sudo apt-get install ffmpeg
   ```

   On CentOS/RHEL:
   ```bash
   sudo yum install ffmpeg
   ```

### Running the API Server

1. **Start the unified API server**

   ```bash
   cd memory
   python unified_api.py
   ```

   This will start the server on port 8000, accessible at `http://your-server-ip:8000`.

2. **Test the API server**

   You can test the API server using the `simple_unified_test.py` script:

   ```bash
   python simple_unified_test.py path/to/your/video.mp4
   ```

   This will run through the entire pipeline and verify that everything is working correctly.

3. **Make the server accessible on your network**

   By default, the server listens on all interfaces (`0.0.0.0`), so it should be accessible from other devices on your network. Make sure your firewall allows connections to port 8000.

## 2. Setup on M1 Device with WebAI Navigator

### Configuring WebAI Navigator

1. **Find your Intel server's IP address**

   On your Intel server, run:
   ```bash
   ifconfig
   ```
   or
   ```bash
   ip addr show
   ```

   Look for an IP address like `192.168.1.xxx` that's on your local network.

2. **Add API Elements in Navigator**

   In WebAI Navigator on your M1 device:

   a. **Add Transcription API Element**:
      - Element Type: API
      - URL: `http://your-intel-server-ip:8000/transcribe/`
      - Method: POST
      - Input Mapping:
        - Connect your video source to the `file` parameter
        - Optionally add a text input for `trigger_keywords`
      - Output Mapping:
        - Map `transcript` to wherever you need the text
        - Map `trigger_detected` to a conditional router

   b. **Add Conditional Router**:
      - Connect the `trigger_detected` output from the Transcription API to this router
      - For the `true` path, connect to the Memory Processing API
      - For the `false` path, connect to whatever you want to happen when no triggers are detected

   c. **Add Memory Processing API Element** (for the `true` path):
      - Element Type: API
      - URL: `http://your-intel-server-ip:8000/process_memory/`
      - Method: POST
      - Input Mapping:
        - Connect the `transcript` output from the Transcription API to the `transcript` parameter
        - Optionally connect the `frames` output to the `frames` parameter if you want to process extracted frames

## 3. Testing the Complete Pipeline

1. **Start the API server on your Intel machine**

   ```bash
   cd memory
   python unified_api.py
   ```

2. **Run a flow in WebAI Navigator**

   - Upload a video or connect a camera input
   - The video will be sent to your Intel server for transcription
   - If trigger words are detected, the transcript will be processed by the memory endpoint
   - The results will be returned to Navigator

## 4. Customizing the Pipeline

### Adding Your LLaMA Integration

To integrate LLaMA with the memory processing endpoint:

1. Open `unified_api.py` on your Intel server
2. Locate the `process_memory` function
3. Replace the placeholder code with your actual LLaMA API calls
4. Restart the API server

### Customizing Trigger Words

You can customize the trigger words in two ways:

1. **In the API**:
   - Modify the default keywords list in the `transcribe` function in `unified_api.py`

2. **From Navigator**:
   - Pass a comma-separated list of keywords through the `trigger_keywords` parameter

## 5. Troubleshooting

### API Server Won't Start

- Check that all dependencies are installed correctly
- Verify that you're using a compatible Python version
- Check for error messages in the console

### Can't Connect from Navigator

- Verify that the Intel server's IP address is correct
- Check that the server is running and listening on port 8000
- Ensure there are no firewall rules blocking the connection
- Try accessing the API from a browser on your M1 device: `http://your-intel-server-ip:8000/`

### Transcription Not Working

- Verify that Whisper is installed correctly
- Check that ffmpeg is installed and accessible
- Make sure the video file format is supported

### Frames Not Being Extracted

- Verify that ffmpeg is installed correctly
- Check the permissions on the temp directory
- Make sure the video contains trigger words

## 6. Next Steps

- Implement the actual LLaMA integration for memory processing
- Add authentication to the API if needed
- Consider adding a database to store processed memories
- Expand the frame extraction to include more sophisticated image analysis
