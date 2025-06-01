#!/usr/bin/env python3
"""
Test script for VideoFrameExtractorElement
"""

import os
import sys
from video_frame_extractor_element import VideoFrameExtractorElement

def test_extractor(video_path: str):
    """Test the video frame extractor with a sample video"""
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return False
    
    try:
        # Create extractor instance
        extractor = VideoFrameExtractorElement()
        
        # Get video info first
        print("Getting video information...")
        video_info = extractor.get_video_info(video_path)
        print(f"Video Info: {video_info}")
        
        # Extract frames (using shorter interval for testing)
        print("\nExtracting frames...")
        frames = extractor.process(video_path, frame_interval_seconds=10)
        
        print(f"\n✅ Success! Extracted {len(frames)} frames")
        
        # Display frame info
        for i, frame in enumerate(frames[:3]):  # Show first 3 frames
            print(f"Frame {i+1}:")
            print(f"  - Timestamp: {frame['timestamp']}s")
            print(f"  - Size: {frame['size']['width']}x{frame['size']['height']}")
            print(f"  - Base64 length: {len(frame['image_base64'])} characters")
            print(f"  - Prompt: {frame['prompt']}")
        
        if len(frames) > 3:
            print(f"... and {len(frames) - 3} more frames")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🎥 Video Frame Extractor Test")
    print("=" * 40)
    
    # Check if video path provided
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        # Look for sample videos in common locations
        test_videos = [
            "sample.mp4",
            "test.mp4",
            "video.mp4",
            "/Users/yizu/Desktop/sample.mp4"  # Adjust path as needed
        ]
        
        video_path = None
        for path in test_videos:
            if os.path.exists(path):
                video_path = path
                break
        
        if not video_path:
            print("No test video found. Usage:")
            print(f"python {sys.argv[0]} <path_to_video_file>")
            print("\nSupported formats: mp4, avi, mov, mkv, webm")
            return
    
    print(f"Testing with video: {video_path}")
    print("-" * 40)
    
    success = test_extractor(video_path)
    
    if success:
        print("\n🎉 Test completed successfully!")
        print("The VideoFrameExtractorElement is ready for WebAI Navigator!")
    else:
        print("\n💥 Test failed. Please check the error messages above.")

if __name__ == "__main__":
    main() 