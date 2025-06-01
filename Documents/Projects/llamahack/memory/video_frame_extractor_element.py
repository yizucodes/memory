import cv2
import base64
import numpy as np
from typing import List, Dict, Any
import os

class VideoFrameExtractorElement:
    def __init__(self):
        self.name = "video-frame-extractor"
        self.version = "1.0.0"
        self.description = "Extract frames from video for AI analysis"
        
    def process(self, video_file_path: str, frame_interval_seconds: int = 20) -> List[Dict[str, Any]]:
        """
        Extract frames from video and return base64 objects for Llama
        
        Args:
            video_file_path: Path to the video file
            frame_interval_seconds: Interval between extracted frames (default: 20 seconds)
            
        Returns:
            List of frame objects with timestamp and base64 image data
        """
        if not os.path.exists(video_file_path):
            raise FileNotFoundError(f"Video file not found: {video_file_path}")
            
        frames = []
        cap = cv2.VideoCapture(video_file_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_file_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"Video info: {fps:.2f} FPS, {total_frames} frames, {duration:.2f} seconds")
        
        frame_count = 0
        frames_to_extract = int(fps * frame_interval_seconds) if fps > 0 else 600  # fallback to 600 frames
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Extract frame at specified intervals
            if frame_count % frames_to_extract == 0:
                timestamp = frame_count / fps if fps > 0 else frame_count / 30  # fallback fps
                
                try:
                    # Resize frame for efficiency (optional)
                    height, width = frame.shape[:2]
                    if width > 1280:  # Resize large frames
                        scale = 1280 / width
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        frame = cv2.resize(frame, (new_width, new_height))
                    
                    # Convert to base64
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    base64_image = base64.b64encode(buffer).decode('utf-8')
                    
                    # Create frame object for WebAI/Llama
                    frame_object = {
                        "timestamp": round(timestamp, 1),
                        "image_base64": base64_image,
                        "prompt": f"Analyze this video frame at {timestamp:.1f} seconds",
                        "frame_number": frame_count,
                        "size": {
                            "width": frame.shape[1],
                            "height": frame.shape[0]
                        }
                    }
                    
                    frames.append(frame_object)
                    print(f"Extracted frame at {timestamp:.1f}s (frame {frame_count})")
                    
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {e}")
                    continue
            
            frame_count += 1
        
        cap.release()
        print(f"Extraction complete: {len(frames)} frames extracted")
        return frames
    
    def get_video_info(self, video_file_path: str) -> Dict[str, Any]:
        """Get basic information about the video file"""
        cap = cv2.VideoCapture(video_file_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_file_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        cap.release()
        
        return {
            "fps": fps,
            "total_frames": total_frames,
            "width": width,
            "height": height,
            "duration_seconds": duration,
            "file_path": video_file_path
        }

# WebAI Navigator integration functions
def webai_process_video(inputs):
    """WebAI Navigator integration function"""
    extractor = VideoFrameExtractorElement()
    
    video_path = inputs.get('video_file_path')
    interval = inputs.get('frame_interval_seconds', 20)
    
    return extractor.process(video_path, interval)

def webai_get_element_info():
    """Return element metadata for WebAI Navigator"""
    return {
        "name": "video-frame-extractor",
        "version": "1.0.0",
        "description": "Extract frames from video for AI analysis",
        "inputs": {
            "video_file_path": {"type": "string", "required": True, "description": "Path to video file"},
            "frame_interval_seconds": {"type": "integer", "default": 20, "description": "Seconds between extracted frames"}
        },
        "outputs": {
            "frames": {"type": "array", "description": "Array of frame objects with timestamps and base64 images"}
        }
    } 