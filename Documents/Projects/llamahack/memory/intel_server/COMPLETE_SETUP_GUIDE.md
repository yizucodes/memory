# Complete Memory Assistant Setup Guide

## Step 1: Transfer Files to Intel Server

1. **Package the files** (already done for you in the `intel_server` folder)

2. **Transfer to Intel server** using scp:
   ```bash
   scp -r /Users/christy/Documents/Projects/llamahack/memory/intel_server username@intel-server-ip:~/memory_assistant
   ```
   Replace `username` and `intel-server-ip` with your actual values

## Step 2: Set Up the Intel Server

1. **SSH into your Intel server**:
   ```bash
   ssh username@intel-server-ip
   ```

2. **Navigate to the transferred directory**:
   ```bash
   cd ~/memory_assistant
   ```

3. **Make scripts executable**:
   ```bash
   chmod +x setup.sh start_api.sh
   ```

4. **Run the setup script**:
   ```bash
   ./setup.sh
   ```
   This will install all required dependencies

5. **Start the API server**:
   ```bash
   ./start_api.sh
   ```
   The server will start on port 8000

## Step 3: Set Up SSH Tunnel from M1 to Intel Server

1. **Open a new terminal on your M1 device**

2. **Create the SSH tunnel**:
   ```bash
   ssh -L 8000:localhost:8000 username@intel-server-ip
   ```
   Replace `username` and `intel-server-ip` with your actual values

   This will forward requests from port 8000 on your M1 to port 8000 on the Intel server

## Step 4: Configure WebAI Navigator on M1

1. **Open WebAI Navigator** on your M1 device

2. **Create a new flow with these elements**:
   - File Input element (for video)
   - API element
   - Display element

3. **Configure the API element**:
   - URL: `http://localhost:8000/process/`
   - API Key: `remember,important,note` (your trigger words)

4. **Connect elements**:
   - Connect File Input to API element
   - Connect API element to Display element

## Step 5: Test the Complete Flow

1. **Run your flow in Navigator**

2. **Upload a test video**:
   - For positive test: Video containing words like "remember" or "important"
   - For negative test: Video without trigger words

3. **Check the results**:
   - You should see the transcript
   - For videos with trigger words, you should see memory processing results

## Troubleshooting

- **API server won't start**: Check that all dependencies are installed correctly
- **Can't connect through SSH**: Verify your username and IP address
- **Navigator can't connect**: Make sure the SSH tunnel is running
- **No transcription**: Check that Whisper and ffmpeg are installed on the Intel server
