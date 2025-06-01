#!/bin/bash

# This script transfers the necessary files to your Intel server

# Replace these with your actual values
INTEL_USERNAME="your-username"
INTEL_SERVER_IP="your-intel-server-ip"

# Create the remote directory
ssh $INTEL_USERNAME@$INTEL_SERVER_IP "mkdir -p ~/memory_assistant"

# Transfer all files
scp -r /Users/christy/Documents/Projects/llamahack/memory/intel_server/* $INTEL_USERNAME@$INTEL_SERVER_IP:~/memory_assistant/

echo "Files transferred successfully to $INTEL_SERVER_IP:~/memory_assistant/"
echo "Next steps:"
echo "1. SSH into your Intel server: ssh $INTEL_USERNAME@$INTEL_SERVER_IP"
echo "2. Navigate to the directory: cd ~/memory_assistant"
echo "3. Run setup: ./setup.sh"
echo "4. Start the API: ./start_api.sh"
