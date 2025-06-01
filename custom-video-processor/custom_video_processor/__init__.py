import asyncio
import base64
import os
import time
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List
from uuid import UUID

import cv2
import numpy as np
import whisper
from webai_element_sdk.comms.messages import ColorFormat, Frame
from webai_element_sdk.element import Context, Element
from webai_element_sdk.element.settings import (
    BoolSetting,
    ElementSettings,
    NumberSetting,
    TextSetting,
)
from webai_element_sdk.element.variables import ElementOutputs, Output


from webai_element_utils.logs import setup_element_logger

logger = setup_element_logger("VideoAudioProcessor")


class Settings(ElementSettings):
    video_file = TextSetting(
        name="video_file",
        display_name="Video File",
        description="The path to a single video file to be processed.",
        default="",
        hints=["file_path"],
        required=False,
    )
    video_directory = TextSetting(
        name="video_directory",
        display_name="Video Directory",
        description="The path to a directory containing multiple video files to be processed.",
        default="",
        hints=["folder_path"],
        required=False,
    )
    image_directory = TextSetting(
        name="image_directory",
        display_name="Image Directory",
        description="The path to the image directory to be loaded.",
        default="",
        hints=["folder_path"],
        required=False,
    )
    frame_extraction_rate = NumberSetting[float](
        name="frame_extraction_rate",
        display_name="Frame Extraction Rate (per second)",
        description="Number of frames to extract per second of video (0.1 = one frame every 10 seconds).",
        default=0.1,
        min_value=0.01,
        max_value=30,
        hints=["advanced"],
    )
    whisper_model_size = TextSetting(
        name="whisper_model_size",
        display_name="Whisper Model Size",
        description="Size of the Whisper model for transcription.",
        default="base",
        valid_values=["tiny", "base", "small", "medium", "large"],
        required=True,
        hints=["dropdown"],
    )
    enable_transcription = BoolSetting(
        name="enable_transcription",
        display_name="Enable Audio Transcription",
        description="Enable audio transcription using Whisper.",
        default=True,
    )
    enable_frame_extraction = BoolSetting(
        name="enable_frame_extraction",
        display_name="Enable Frame Extraction",
        description="Enable extraction of video frames.",
        default=True,
    )
    frame_rate = NumberSetting[int](
        name="frame_rate",
        display_name="Output Frame Rate",
        description="The frame rate for outputting processed data (0 = process as fast as possible).",
        default=1,
        min_value=0,
        hints=["advanced"],
    )
    video_delay_seconds = NumberSetting[float](
        name="video_delay_seconds",
        display_name="Delay Between Videos (seconds)",
        description="Time delay between processing different video files.",
        default=1.0,
        min_value=0.0,
        hints=["advanced"],
    )
    stay_alive = BoolSetting(
        name="stay_alive",
        display_name="Stay Alive",
        description="Keep element running indefinitely after processing completes.",
        default=False,
        hints=["advanced"],
    )


class Outputs(ElementOutputs):
    default = Output[Frame]()


element = Element(
    id=UUID("8e6ecf76-4de7-46d5-8f0c-3053b65a8db3"),
    name="custom_video_processor",
    display_name="Video Audio Processor",
    description="Processes videos and images with audio transcription and frame extraction for AI analysis",
    version="2.3.0",
    settings=Settings(),
    outputs=Outputs(),
)


def _transcribe_video(video_path: str, model_size: str = "base") -> str:
    """Transcribe video audio using OpenAI Whisper"""
    try:
        logger.info(f"Loading Whisper model '{model_size}'...")
        model = whisper.load_model(model_size)
        
        logger.info(f"Transcribing audio from: {video_path}")
        result = model.transcribe(video_path)
        
        transcript = result["text"].strip()
        logger.info(f"Transcription complete: {len(transcript)} characters")
        return transcript
        
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return ""


def _extract_video_frames(video_path: str, frames_per_second: int = 1) -> List[Dict[str, Any]]:
    """
    Extract frames from video at specified rate per second
    frames_per_second=1 means extract 1 frame every second (every fps-th frame)
    frames_per_second=2 means extract 2 frames every second (every fps/2-th frame)
    """
    frames = []
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video file: {video_path}")
            return frames
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"Video info: {fps:.2f} FPS, {total_frames} frames, {duration:.2f} seconds")
        
        if fps <= 0:
            logger.error("Invalid FPS detected, using fallback of 30 FPS")
            fps = 30.0
        
        # Calculate frame interval: extract every N frames where N = fps / frames_per_second
        frame_interval = max(1, int(fps / frames_per_second))
        expected_frames = int(duration * frames_per_second)
        
        logger.info(f"Extracting {frames_per_second} frame(s) per second (every {frame_interval} frames)")
        logger.info(f"Expected to extract approximately {expected_frames} frames")
        
        frame_count = 0
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract frame at calculated intervals
            if frame_count % frame_interval == 0:
                timestamp = frame_count / fps
                
                try:
                    # Resize frame for efficiency
                    height, width = frame.shape[:2]
                    if width > 1280:
                        scale = 1280 / width
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        frame = cv2.resize(frame, (new_width, new_height))
                    
                    # Convert to base64
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    base64_image = base64.b64encode(buffer).decode('utf-8')
                    
                    frame_object = {
                        "timestamp": round(timestamp, 1),
                        "image_base64": base64_image,
                        "frame_number": frame_count,
                        "second": int(timestamp),
                        "size": {"width": frame.shape[1], "height": frame.shape[0]}
                    }
                    
                    frames.append(frame_object)
                    extracted_count += 1
                    
                    if extracted_count % 10 == 0:  # Log every 10th extracted frame
                        logger.info(f"Extracted frame at {timestamp:.1f}s (frame {frame_count}, total extracted: {extracted_count})")
                    
                except Exception as e:
                    logger.error(f"Error processing frame {frame_count}: {e}")
                    continue
            
            frame_count += 1
        
        cap.release()
        logger.info(f"Frame extraction complete: {len(frames)} frames extracted from {duration:.1f}s video")
        logger.info(f"Actual extraction rate: {len(frames)/duration:.2f} frames per second")
        
    except Exception as e:
        logger.error(f"Frame extraction failed: {e}")
    
    return frames


def _get_video_files_from_directory(directory_path: Path) -> List[Path]:
    """Get all video files from a directory"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
    video_files = []
    
    try:
        for file_path in directory_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in video_extensions:
                video_files.append(file_path)
        
        # Sort files for consistent processing order
        video_files.sort()
        logger.info(f"Found {len(video_files)} video files in {directory_path}")
        
    except Exception as e:
        logger.error(f"Error scanning directory {directory_path}: {e}")
    
    return video_files


def _load_images_from_directory(filepath: Path) -> AsyncIterator[np.ndarray]:
    """Load images from directory"""
    for file in filepath.iterdir():
        if file.is_file() and file.suffix.lower() in [".jpg", ".png", ".jpeg", ".npy", ".raw"]:
            time.sleep(0.1)  # Small delay to prevent overwhelming
            img = cv2.imread(str(file))
            if img is not None:
                yield img


async def _process_single_video(ctx: Context[None, Outputs, Settings], video_path: Path, video_index: int, total_videos: int) -> AsyncIterator[Any]:
    """Process a single video file"""
    logger.info(f"Processing video {video_index + 1}/{total_videos}: {video_path.name}")
    
    enable_transcription = ctx.settings.enable_transcription.value
    enable_frame_extraction = ctx.settings.enable_frame_extraction.value
    frame_extraction_rate = ctx.settings.frame_extraction_rate.value
    whisper_model = ctx.settings.whisper_model_size.value
    frame_rate = ctx.settings.frame_rate.value
    
    # Transcribe audio if enabled
    if enable_transcription:
        logger.info(f"Starting audio transcription for {video_path.name}...")
        transcript = _transcribe_video(str(video_path), whisper_model)
        
        # Yield transcription result
        if transcript:
            yield ctx.outputs.default(
                Frame(
                    ndframe=None,
                    rois=[],
                    color_space=ColorFormat.RGB,
                    other_data={
                        "media_type": "video_transcription",
                        "transcript": transcript,
                        "source_file": str(video_path),
                        "video_name": video_path.name,
                        "video_index": video_index,
                        "total_videos": total_videos,
                        "whisper_model": whisper_model
                    }
                )
            )
            logger.info(f"Transcription data yielded for {video_path.name}")
    
    # Extract frames if enabled
    if enable_frame_extraction:
        logger.info(f"Starting frame extraction for {video_path.name} at {frame_extraction_rate} frame(s) per second...")
        frames = _extract_video_frames(str(video_path), frame_extraction_rate)
        
        # Yield each frame
        for i, frame_data in enumerate(frames):
            yield ctx.outputs.default(
                Frame(
                    ndframe=None,
                    rois=[],
                    color_space=ColorFormat.RGB,
                    other_data={
                        "media_type": "video_frame",
                        "frame_data": frame_data,
                        "source_file": str(video_path),
                        "video_name": video_path.name,
                        "video_index": video_index,
                        "total_videos": total_videos,
                        "frame_index": i,
                        "total_frames": len(frames),
                        "extraction_rate": frame_extraction_rate
                    }
                )
            )
            
            if i % 10 == 0:  # Log every 10th frame to reduce noise
                logger.info(f"Frame {i+1}/{len(frames)} yielded for {video_path.name}")
            
            # Rate limiting
            if frame_rate > 0:
                await asyncio.sleep(1.0 / frame_rate)


@element.executor
async def run(ctx: Context[None, Outputs, Settings]) -> AsyncIterator[Any]:
    """Main execution function"""
    
    # Get settings
    video_file = ctx.settings.video_file.value
    video_directory = ctx.settings.video_directory.value
    image_directory = ctx.settings.image_directory.value
    video_delay = ctx.settings.video_delay_seconds.value
    frame_rate = ctx.settings.frame_rate.value
    
    # Determine media path and type
    video_files = []
    
    if video_file != "":
        # Single video file
        video_path = Path(video_file).resolve()
        if video_path.exists() and video_path.is_file():
            video_files = [video_path]
        else:
            raise ValueError(f"Video file does not exist: {video_file}")
    
    elif video_directory != "":
        # Multiple videos from directory
        video_dir_path = Path(video_directory).resolve()
        if video_dir_path.exists() and video_dir_path.is_dir():
            video_files = _get_video_files_from_directory(video_dir_path)
            if not video_files:
                raise ValueError(f"No video files found in directory: {video_directory}")
        else:
            raise ValueError(f"Video directory does not exist: {video_directory}")
    
    elif image_directory != "":
        # Process image directory
        media_path_obj = Path(image_directory).resolve()
        if not media_path_obj.exists():
            raise ValueError(f"Image directory does not exist: {image_directory}")
        
        logger.info(f"Processing image directory: {media_path_obj}")
        
        image_count = 0
        async for img in _load_images_from_directory(media_path_obj):
            if img is None:
                continue
            
            # Convert image to RGB
            image_rgb = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
            
            # Convert to base64 for consistency
            _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            base64_image = base64.b64encode(buffer).decode('utf-8')
            
            yield ctx.outputs.default(
                Frame(
                    ndframe=np.asarray(image_rgb),
                    rois=[],
                    color_space=ColorFormat.BGR,
                    other_data={
                        "media_type": "image_file",
                        "image_base64": base64_image,
                        "source_directory": str(media_path_obj),
                        "image_index": image_count
                    }
                )
            )
            
            image_count += 1
            logger.info(f"Image {image_count} processed")
            
            # Rate limiting
            if frame_rate > 0:
                await asyncio.sleep(1.0 / frame_rate)
        
        logger.info("Image processing complete")
        
    else:
        raise ValueError("No media path provided. Please specify video file, video directory, or image directory.")
    
    # Process video files
    if video_files:
        logger.info(f"Starting processing of {len(video_files)} video file(s)")
        
        for i, video_path in enumerate(video_files):
            logger.info(f"Processing video {i + 1}/{len(video_files)}: {video_path.name}")
            
            # Process the video
            async for frame in _process_single_video(ctx, video_path, i, len(video_files)):
                yield frame
            
            # Delay between videos (except for the last one)
            if i < len(video_files) - 1 and video_delay > 0:
                logger.info(f"Waiting {video_delay} seconds before next video...")
                await asyncio.sleep(video_delay)
        
        logger.info("All video processing complete")
    
    # Stay alive if configured
    while ctx.settings.stay_alive.value:
        await asyncio.sleep(1.0)
    