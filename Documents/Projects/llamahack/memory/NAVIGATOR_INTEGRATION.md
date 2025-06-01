# Integrating Whisper Transcription with WebAI Navigator

This guide explains how to set up the Whisper transcription API with WebAI Navigator to create a memory assistant pipeline.

## Step 1: Set Up the API Server

1. **Install Dependencies on M1 Device**:
   ```bash
   pip install fastapi uvicorn openai-whisper python-multipart
   ```

2. **Clone the Repository**:
   ```bash
   git clone https://github.com/yizucodes/memory.git
   cd memory
   ```

3. **Start the API Server**:
   ```bash
   python simple_whisper_api.py
   ```
   This will start the server at `http://localhost:8000`

## Step 2: Configure WebAI Navigator

1. **Add API Element**:
   - Add a new API element to your Navigator flow
   - Set the URL to `http://localhost:8000/transcribe/`
   - Configure it as a POST request
   - Set the Content-Type to `multipart/form-data`

2. **Configure Input**:
   - Connect your video source to the API element
   - Map the video file to the `file` parameter
   - Optionally, add a text input for custom trigger keywords

3. **Add Conditional Logic**:
   - Add a Condition element after the API element
   - Set the condition to check `response.trigger_detected == true`
   - Connect the "True" path to your LLaMA API element
   - Connect the "False" path to skip LLaMA processing

## Step 3: Testing the Flow

1. **Test with Sample Video**:
   - Upload a test video with known trigger words
   - Verify that the transcription is accurate
   - Check that trigger detection works correctly

2. **Monitor API Responses**:
   - The API returns a JSON object with:
     - `transcript`: The full text transcription
     - `trigger_detected`: Boolean flag for LLaMA triggering
     - `matched_keywords`: List of detected trigger words (if any)
     - `message`: Human-readable explanation

## Step 4: Customizing Trigger Words

The default trigger words are:
- "remember"
- "important"
- "note"
- "don't forget"
- "remind me"
- "save this"
- "record this"

To customize these, pass a comma-separated list of words to the `trigger_keywords` parameter in your API request.

## Step 5: Integrating with LLaMA

When a trigger is detected, the API will set `trigger_detected` to `true`. Your Navigator flow should:

1. Extract the transcript from the response
2. Pass it to the LLaMA API with appropriate prompting
3. Process and store the LLaMA response as a memory

## Troubleshooting

- **API Not Responding**: Ensure the server is running and accessible from Navigator
- **Transcription Issues**: Check that ffmpeg and Whisper are properly installed
- **Trigger Not Detecting**: Verify the keywords and check case sensitivity
- **File Upload Problems**: Ensure the file is properly mapped in the multipart form
