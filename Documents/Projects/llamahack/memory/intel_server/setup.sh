#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Install ffmpeg if not already installed
if ! command -v ffmpeg &> /dev/null; then
    echo "Installing ffmpeg..."
    # For Ubuntu/Debian
    if command -v apt-get &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y ffmpeg
    # For CentOS/RHEL
    elif command -v yum &> /dev/null; then
        sudo yum install -y ffmpeg
    # For macOS
    elif command -v brew &> /dev/null; then
        brew install ffmpeg
    else
        echo "Could not install ffmpeg automatically. Please install it manually."
    fi
fi

# Create temp directory
mkdir -p temp

# Make script executable
chmod +x start_api.sh

echo "Setup complete! Run ./start_api.sh to start the API server."
