# WebAI Navigator Setup Guide

## Using the API Key Field for Trigger Keywords

Since WebAI Navigator only provides an API key field for configuration, I've created a special version of the API (`navigator_friendly_api.py`) that accepts trigger keywords through the API key header. Here's how to set it up:

## 1. Start the Navigator-Friendly API on Your Intel Server

```bash
cd memory
python navigator_friendly_api.py
```

This will start the server on port 8000, making it accessible at `http://your-intel-server-ip:8000`.

## 2. Configure WebAI Navigator on Your M1 Device

### Option A: Using the Combined Endpoint (Simplest)

1. **Add a Single API Element**:
   - Element Type: API
   - URL: `http://your-intel-server-ip:8000/process/`
   - Method: POST
   - **API Key**: `remember,important,note` (your comma-separated trigger words)
   - Input Mapping:
     - Connect your video source to the `file` parameter
   - Output Mapping:
     - Map `transcript` to wherever you need the text
     - Map `trigger_detected` to display or use in your flow
     - Map `memory_result` to display or use in your flow

   This single endpoint handles everything - transcription, trigger detection, and memory processing.

### Option B: Using Separate Endpoints (More Flexible)

1. **Add Transcription API Element**:
   - Element Type: API
   - URL: `http://your-intel-server-ip:8000/transcribe/`
   - Method: POST
   - **API Key**: `remember,important,note` (your comma-separated trigger words)
   - Input Mapping:
     - Connect your video source to the `file` parameter
   - Output Mapping:
     - Map `transcript` to wherever you need the text
     - Map `trigger_detected` to a conditional router

2. **Add Conditional Router**:
   - Connect the `trigger_detected` output from the Transcription API to this router
   - For the `true` path, connect to the Memory Processing API
   - For the `false` path, connect to whatever you want to happen when no triggers are detected

3. **Add Memory Processing API Element** (for the `true` path):
   - Element Type: API
   - URL: `http://your-intel-server-ip:8000/process_memory/`
   - Method: POST
   - Input Mapping:
     - Connect the `transcript` output from the Transcription API to the `transcript` parameter

## How This Works

The Navigator-friendly API is designed to work with WebAI Navigator's limitations:

1. **API Key as Trigger Words**: 
   - The API accepts the API key header as a comma-separated list of trigger words
   - Example: If you set the API key to `remember,important,note`, the API will look for these words in the transcript

2. **Combined Endpoint**:
   - The `/process/` endpoint handles both transcription and memory processing in one call
   - This simplifies your Navigator flow to a single API element

3. **Separate Endpoints**:
   - If you prefer more control, you can still use the separate `/transcribe/` and `/process_memory/` endpoints
   - This allows for more complex flows with conditional logic

## Testing Your Setup

1. Start the API server on your Intel machine
2. In Navigator, create a simple flow with the API element
3. Connect a video source (upload or camera)
4. Run the flow and check the results

## Customizing Trigger Words

To change the trigger words, simply update the API key in the Navigator API element settings:

- For general memories: `remember,important,note`
- For task-related memories: `todo,task,deadline`
- For personal memories: `family,friend,birthday`

You can use any comma-separated list of words that are meaningful for your use case.
