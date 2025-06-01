import requests
import os
import subprocess
import time

def start_api_server():
    """Start the API server as a background process"""
    print("Starting API server...")
    process = subprocess.Popen(
        ["python", "simple_whisper_api.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    # Give it a moment to start
    time.sleep(3)
    return process

def test_transcription(video_path):
    """Test the transcription API with a video file"""
    url = "http://localhost:8000/transcribe/"
    
    # Check if the file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} not found")
        return
    
    print(f"Testing transcription with {video_path}...")
    
    # Prepare the file for upload
    files = {
        'file': (os.path.basename(video_path), open(video_path, 'rb'), 'video/quicktime')
    }
    
    # Add some trigger words to test the conditional logic
    data = {
        'trigger_keywords': 'remember,important,note'
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
            
            print(f"Message: {result['message']}")
        else:
            print(f"Error: API returned status code {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Error connecting to API: {str(e)}")

def main():
    # Path to the video file (update this to your video file path)
    video_path = "../IMG_4032.MOV"
    
    # Start the API server
    server_process = start_api_server()
    
    try:
        # Test the transcription
        test_transcription(video_path)
    finally:
        # Clean up: terminate the server process
        print("\nShutting down API server...")
        server_process.terminate()

if __name__ == "__main__":
    main()
