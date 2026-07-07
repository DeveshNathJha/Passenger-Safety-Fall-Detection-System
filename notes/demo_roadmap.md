# Demo Roadmap: How to Run & Connect

This guide explains how to run the passenger fall detection system on your laptop and stream live sensor data from your smartphone to test it.

## 1. Environment Setup

Always ensure you are using the project's virtual environment before running any commands. This keeps dependencies isolated.

**Activate the virtual environment:**
```bash
# On Linux/macOS
source .venv/bin/activate

# On Windows (Command Prompt)
.venv\Scripts\activate.bat
```
*(You will see `(.venv)` appear at the beginning of your terminal prompt).*

## 2. Starting the Server (Laptop)

The laptop acts as the "Edge Server" that holds the AI model and processes the data.

1. Ensure your laptop and smartphone are connected to the **same Wi-Fi network**.
2. Find your laptop's Local IP Address:
   - Linux/Mac: Run `ip addr | grep "inet "` or `ifconfig` (look for `192.168.x.x`).
   - Windows: Run `ipconfig` (look for "IPv4 Address").
3. Start the FastAPI server:
   ```bash
   uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
   ```
   *(Leave this terminal window open).*

## 3. Connecting the Phone (Client)

You **do not need to install any app** on your phone! The web browser handles the sensors.

1. Open **Chrome, Safari, or Firefox** on your smartphone.
2. In the URL bar, type your laptop's IP address and the port, followed by `/demo`. 
   - *Example:* `http://192.168.1.5:8000/demo` (Replace `192.168.1.5` with your actual IP).
3. Tap **"Connect & Stream"**.
   - *Note for iPhone users (iOS 13+):* You will see a prompt asking for "Sensor Permission". Tap it and click "Allow" so the browser can read the gyroscope and accelerometer.
4. **Test it:** Hold the phone steady, then make a sudden shake or dropping motion to simulate a fall. The screen will flash red when a fall is detected.

---

## 4. Where to Make Changes

If you want to modify how the system works, here is where you look:

### Changing the Window Size or Overlap?
- **File:** `config/config.yaml`
- Modify `window_size` (default: 128) or `overlap` (default: 0.5).

### Changing the AI Model?
1. **File:** `src/train.py`
2. Look at the `MODEL_REGISTRY` and functions like `build_cnn_model`. You can modify the layers here.
3. Retrain by running: `python src/train.py`
4. Convert the new model: `python src/tflite_converter.py`

### Changing the Web Server Output or Logic?
- **File:** `src/app.py`
- Look at the `@app.websocket("/ws/predict")` function to change how the server receives data or what it sends back to the phone.

### Changing the Web UI / Dashboard?
- **File:** `src/demo_phone.html`
- This file contains all the HTML, CSS, and JavaScript for the phone screen. You can change colors, add graphs, or modify the sensor mapping here.

---
'''
# How to connect your Phone (No GitHub needed!)
You do not need to upload anything to GitHub to test this. Your laptop is acting as a local web server, and your phone can talk to it directly over your home Wi-Fi.

## Step 1: Get your laptop's Local IP Address

Open a new terminal on your laptop.
Run this command: ip addr | grep "inet "
Look for an IP address that usually starts with 192.168... (for example, 192.168.1.5 or 192.168.29.141). This is your laptop's "address" on your home Wi-Fi.
## Step 2: Open the page on your Phone

Make sure your phone and laptop are connected to the SAME Wi-Fi network.
Open Chrome or Safari on your phone.
Type the address exactly like this: http://YOUR_LAPTOP_IP:8000/demo (Example: http://192.168.1.5:8000/demo)

-- for Devesh Linux :http://192.168.29.91:8000/demo

## Step 3: What to do on the phone to test the Fall Detection
Once the page loads on your phone:

Tap the "Connect & Stream" button.
(If you are on an iPhone, it might ask for "Sensor Permission". Click Allow).
You will immediately see the numbers on the screen changing rapidly. That is your phone's actual Gyroscope and Accelerometer sending data live to the AI model on your laptop!
To see the change (Normal Activity): Just hold the phone in your hand or leave it flat on a table. The screen will say Normal.
To simulate a fall: Drop the phone onto a soft bed, or make a sudden, sharp shaking motion.
The AI model on your laptop will instantly process that movement and the phone screen will flash red with FALL DETECTED!

### Since you are testing locally on your Wi-Fi, there is a quick developer trick to bypass this security block on your Android phone:

Do this on your Android Phone:

Open Chrome.
Type this exactly into the URL bar and hit enter: chrome://flags/#unsafely-treat-insecure-origin-as-secure
You will see a yellow highlighted setting called "Insecure origins treated as secure".
Tap the "Disabled" button and change it to "Enabled".
In the text box right below it, type your laptop's URL exactly like this: http://192.168.29.91:8000
Click the blue "Relaunch" button that appears at the bottom of the screen to restart Chrome.

'''