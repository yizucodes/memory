#!/bin/bash

# This script sets up an SSH tunnel from your M1 to your Intel server

# Replace these with your actual values
INTEL_USERNAME="your-username"
INTEL_SERVER_IP="your-intel-server-ip"

# Create the SSH tunnel
echo "Setting up SSH tunnel to $INTEL_SERVER_IP..."
ssh -L 8000:localhost:8000 $INTEL_USERNAME@$INTEL_SERVER_IP

# Note: This will keep running until you press Ctrl+C or close the terminal
# The tunnel will forward requests from localhost:8000 on your M1 to localhost:8000 on the Intel server
