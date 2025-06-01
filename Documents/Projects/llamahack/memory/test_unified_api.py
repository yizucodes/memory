import requests
import os
import subprocess
import time
import sys

def start_api_server():
    """Start the API server as a background process"""
    print("Starting unified API server...")
    process = subprocess.Popen(
        [sys.executable, "unified_api.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    # Give it a moment to start
    time.sleep(5)  # Increased wait time
    
    # Check if the server is running
    max_retries = 3
    for i in range(max_retries):
        try:
            # Try to connect to the server's root endpoint
            response = requests.get("http://localhost:8000/")
            if response.status_code == 200:
                print("Server started successfully!")
                return process
        except requests.exceptions.ConnectionError:
            if i < max_retries - 1:
                print(f"Server not ready yet, retrying in 2 seconds... (attempt {i+1}/{max_retries})")
                time.sleep(2)
            else:
                print("Failed to connect to server after multiple attempts")
                # Check if there's any error output from the server
                stderr_output = process.stderr.read().decode('utf-8', errors='ignore')
                if stderr_output:
                    print(f"Server error output: {stderr_output}")
    return process

def test_transcription(video_path):
    """Test the transcription API with a video file"""
    url = "http://localhost:8000/transcribe/"
    
    # Check if the file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} not found")
        return None
    
    print(f"Testing transcription with {video_path}...")
    
    # Prepare the file for upload
    files = {
        'file': (os.path.basename(video_path), open(video_path, 'rb'), 'video/quicktime')
    }
    
    # Add some trigger words to test the conditional logic
    data = {
        'trigger_keywords': 'remember,important,note,match'
    }
    
    try:
        # Send the request
        response = requests.post(url, files=files, data=data)
        
        # Check if successful
        if response.status_code == 200:
            result = response.json()
            print("\nTranscription successful!")
            print(f"\nTranscript: {result['transcript']}")
            print(f"Trigger detected: {result['trigger_detected']}")
            
            if result['trigger_detected']:
                print(f"Matched keywords: {result.get('matched_keywords', [])}")
                if 'frames' in result and result['frames']:
                    print(f"Extracted {len(result['frames'])} frames")
            
            print(f"Message: {result['message']}")
            return result
        else:
            print(f"Error: API returned status code {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"Error connecting to API: {str(e)}")
        return None

def test_memory_processing(transcript, frames=[]):
    """Test the memory processing endpoint"""
    url = "http://localhost:8000/process_memory/"
    
    data = {
        'transcript': transcript,
        'frames': frames
    }
    
    try:
        # Send the request
        response = requests.post(url, data=data)
        
        # Check if successful
        if response.status_code == 200:
            result = response.json()
            print("\nMemory processing successful!")
            print(f"Memory ID: {result['memory_id']}")
            print(f"Summary: {result['summary']}")
            print(f"Processed frames: {result['processed_frames']}")
            print(f"Status: {result['status']}")
            return result
        else:
            print(f"Error: API returned status code {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"Error connecting to API: {str(e)}")
        return None

def main():
    # Path to the video file (update this to your video file path)
    video_path = "../IMG_4032.MOV"
    
    # Start the API server
    server_process = start_api_server()
    
    try:
        # Test the transcription
        transcription_result = test_transcription(video_path)
        
        # If transcription was successful and trigger was detected, test memory processing
        if transcription_result and transcription_result.get('trigger_detected', False):
            test_memory_processing(transcription_result['transcript'], transcription_result.get('frames', []))
        elif transcription_result:
            print("\nNo trigger words detected, skipping memory processing test.")
            print("To test memory processing, add a trigger word to your video or modify the trigger_keywords in the test script.")
            
            # Let's force a test of memory processing anyway
            print("\nForcing memory processing test with the transcript...")
            test_memory_processing(transcription_result['transcript'])
    finally:
        # Clean up: terminate the server process
        print("\nShutting down API server...")
        server_process.terminate()

if __name__ == "__main__":
    main()
