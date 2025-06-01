# Video Audio Processor Element

A comprehensive WebAI Navigator element that processes videos and images with audio transcription and frame extraction capabilities. Supports both single video files and batch processing of multiple videos from a directory.

## Features

- **Audio Transcription**: Uses OpenAI Whisper to transcribe audio from video files
- **Frame Extraction**: Extracts frames from videos at configurable intervals
- **Image Processing**: Loads and processes images from directories
- **Batch Video Processing**: Process multiple video files from a directory
- **Llama Integration**: Automatically feeds processed data to Llama models for AI analysis

## Settings

### Media Input
- **Video File**: Path to a single video file (.mp4, .avi, .mov, .mkv, .wmv, .flv, .webm)
- **Video Directory**: Path to a directory containing multiple video files for batch processing
- **Image Directory**: Path to a directory containing images (.jpg, .png, .jpeg)

### Processing Options
- **Enable Audio Transcription**: Toggle audio transcription using Whisper
- **Enable Frame Extraction**: Toggle video frame extraction
- **Whisper Model Size**: Choose from tiny, base, small, medium, or large models
- **Frame Interval (seconds)**: Time interval between extracted frames (default: 20 seconds)

### Performance
- **Output Frame Rate**: Rate at which to output processed data (0 = as fast as possible)
- **Delay Between Videos (seconds)**: Wait time between processing different video files in batch mode
- **Stay Alive**: Keep element running indefinitely after processing

## Output Format

The element outputs Frame objects with different `media_type` values in `other_data`:

### Video Transcription
```python
{
    "media_type": "video_transcription",
    "transcript": "Full transcribed text...",
    "source_file": "/path/to/video.mp4",
    "video_name": "video.mp4",
    "video_index": 0,
    "total_videos": 3,
    "whisper_model": "base"
}
```

### Video Frames
```python
{
    "media_type": "video_frame",
    "frame_data": {
        "timestamp": 20.0,
        "image_base64": "base64_encoded_image...",
        "frame_number": 600,
        "size": {"width": 1280, "height": 720}
    },
    "source_file": "/path/to/video.mp4",
    "video_name": "video.mp4",
    "video_index": 0,
    "total_videos": 3,
    "frame_index": 0,
    "total_frames": 5
}
```

### Images
```python
{
    "media_type": "image_file",
    "image_base64": "base64_encoded_image...",
    "source_directory": "/path/to/images/",
    "image_index": 0
}
```

## Processing Modes

### Single Video Mode
Set the **Video File** path to process a single video file.

### Batch Video Mode
Set the **Video Directory** path to process all video files in a directory. The element will:
- Automatically discover all supported video files in the directory
- Process them in alphabetical order
- Include video indexing information in the output
- Apply optional delays between video processing

### Image Mode
Set the **Image Directory** path to process static images.

## Integration with Llama

The modified Llama element automatically:
1. Ingests media data without generating responses
2. Stores transcripts, frames, and images in context with video identification
3. Includes relevant media context when responding to API queries
4. Only generates responses for user text queries from the API element

## Usage Workflow

1. **Connect Elements**: Video Audio Processor → Llama → API
2. **Configure Settings**: Set video directory/file path and processing options
3. **Process Media**: Element automatically transcribes audio and extracts frames from all videos
4. **Query via API**: Send text queries through the API element
5. **Receive Enhanced Responses**: Llama uses ingested media context from all processed videos

## Supported Video Formats

- MP4 (.mp4)
- AVI (.avi)
- MOV (.mov)
- MKV (.mkv)
- WMV (.wmv)
- FLV (.flv)
- WebM (.webm)

## Dependencies

- OpenAI Whisper (for audio transcription)
- OpenCV (for video/image processing)
- NumPy (for array operations)
- WebAI Element SDK (for Navigator integration)

## Example Use Cases

- **Lecture Series Analysis**: Process multiple lecture videos for comprehensive analysis
- **Content Library Processing**: Batch transcribe and analyze video content libraries
- **Educational Course Processing**: Extract key information from entire course video sets
- **Video Surveillance Analysis**: Process multiple surveillance videos with timestamps
- **Documentary Analysis**: Analyze documentary series with both audio and visual context
- **Training Material Processing**: Process corporate training videos for Q&A systems

## Batch Processing Benefits

- **Efficiency**: Process entire video libraries automatically
- **Consistency**: Uniform processing settings across all videos
- **Organization**: Maintains video identification and ordering
- **Scalability**: Handle large collections of video content
- **Context Preservation**: Llama can reference content across multiple videos in responses 