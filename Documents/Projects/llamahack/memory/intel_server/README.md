# Memory Assistant API

This package contains everything you need to run the Memory Assistant API on your Intel server.

## Quick Setup

1. Run the setup script:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

2. Start the API server:
   ```bash
   ./start_api.sh
   ```

## Connecting from M1 Device

Once the server is running, you can connect from your M1 device using SSH tunneling:

```bash
ssh -L 8000:localhost:8000 your-username@intel-server-ip
```

Replace `your-username` and `intel-server-ip` with your actual values.

## Using with WebAI Navigator

In WebAI Navigator on your M1 device:

1. Configure the API element to use:
   - URL: `http://localhost:8000/process/`
   - API Key: `remember,important,note` (or your custom trigger words)

2. Connect your video source to the API element

3. Display the results from the API response

## Customizing Trigger Words

To use different trigger words, simply change the API Key in Navigator to a comma-separated list of words, such as:
- `remember,important,note`
- `task,deadline,project`
- `family,birthday,event`
