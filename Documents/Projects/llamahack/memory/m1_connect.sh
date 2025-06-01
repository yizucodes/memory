#!/bin/bash

# This script creates an SSH tunnel from your M1 device to your Intel server

# Intel server details
INTEL_USERNAME="christy"  # Your username on the Intel server
INTEL_SERVER_IP="192.168.1.177"  # Your Intel server's IP address

# Create the SSH tunnel
echo "Setting up SSH tunnel to $INTEL_SERVER_IP..."
echo "This will forward requests from localhost:8000 on your M1 to $INTEL_SERVER_IP:8000"
echo "Keep this terminal window open to maintain the tunnel"
echo ""
ssh -L 8000:localhost:8000 $INTEL_USERNAME@$INTEL_SERVER_IP

# Note: This will keep running until you press Ctrl+C or close the terminal
