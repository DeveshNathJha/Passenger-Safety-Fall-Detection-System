# How to Run & Connect Phone

Follow these instructions to extract the project, set up the environment, run the server, and connect your phone to test the real-time IMU-based fall detection system.

---

## 1. Extracting the Project
If you received the project as a `.zip` archive:
* **Using Command Line (Linux/macOS):**
  ```bash
  unzip project.zip -d Passenger-Safety-Fall-Detection
  cd Passenger-Safety-Fall-Detection
  ```
* **Using GUI:**
  Right-click `project.zip` and select **Extract Here** (or equivalent on Windows/macOS) and enter the directory.

---

## 2. Setting Up the Python Environment
This project requires Python **3.10 or higher**. Since the `env` folder was excluded to keep the package clean, you must recreate the virtual environment:

1. **Create a virtual environment:**
   ```bash
   python3 -m venv .venv
   ```
2. **Activate the environment:**
   * **Linux/macOS:**
     ```bash
     source .venv/bin/activate
     ```
   * **Windows (Command Prompt):**
     ```cmd
     .venv\Scripts\activate
     ```
   * **Windows (PowerShell):**
     ```powershell
     .venv\Scripts\Activate.ps1
     ```
3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 3. Finding Your PC's Local IP Address
To connect your phone, both your PC and your phone must be on the **same Wi-Fi network**. You need to find your PC's local IP address:

* **On Linux:**
  Open a terminal and run:
  ```bash
  ip route show | grep src
  # Look for "src 192.168.X.X" (your local IP)
  ```
  *Alternatively, run `ip a` and check the `inet` address under your wireless adapter (usually `wlan0` or `wlp...`).*
* **On macOS:**
  Open Terminal and run:
  ```bash
  ipconfig getifaddr en0
  # Or go to System Settings -> Wi-Fi -> Details to see your IP address.
  ```
* **On Windows:**
  Open Command Prompt (`cmd`) and run:
  ```cmd
  ipconfig
  # Look for "IPv4 Address" under your Wireless LAN adapter (e.g., 192.168.1.XX)
  ```

Let's assume your PC's IP address is `192.168.1.15` for the rest of this guide.

---

## 4. Starting the FastAPI Server
Run the FastAPI application with `uvicorn`. The `--host 0.0.0.0` option is **mandatory** because it opens the server to incoming connections from your local network (i.e., your phone):

```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```

---

## 5. Connecting and Testing from Your Phone

1. **Open the Demo Page on Phone:**
   Open the browser on your smartphone (Safari on iOS, Chrome/Firefox on Android) and go to:
   ```
   http://<YOUR_PC_IP>:8000/demo
   ```
   *(e.g., `http://192.168.1.15:8000/demo`)*

2. **Grant Sensor Permissions:**
   * **Android:** Sensors are accessed automatically. No prompts should appear.
   * **iOS (iPhone):** iOS requires explicit user interaction to read gyroscope and accelerometer data. 
     * Scroll to the bottom of the page.
     * Tap **"Request Sensor Permission (iOS)"**.
     * Tap **"Allow"** on the popup dialog.

3. **Establish WebSocket Connection:**
   * Under **Target AI Model**, select either:
     * **Human Fall Detection** (loads the 15 KB MobiFall-trained model)
     * **Vehicle Crash Detection** (loads the 15 KB physics-simulated crash model)
   * Verify the WebSocket URL box shows `ws://<YOUR_PC_IP>:8000/ws/predict`.
   * Tap **"Connect & Stream"**.

4. **Verify Sensor Data Flow:**
   * The status dot will turn **blue (Streaming)**.
   * The "Live Sensor Readings" grid will start updating rapidly (at 50 Hz).
   * The **Sliding Window Buffer** bar will fill up. It takes 128 samples (~2.5 seconds) to make the first prediction, and updates every 64 samples (~1.2 seconds) thereafter.

5. **Simulate Actions:**
   * **Normal Activity:** Lay the phone flat or hold it steady. Label will show **Normal** (Acc Y ≈ 9.8, Gyro ≈ 0).
   * **Fall/Crash:** Shake or flip the phone aggressively. The label will jump to **FALL!** and trigger a red warning banner on the screen.

---

## 6. Accessing Journal and Conference Papers
All academic documents are included in the project root:

* **Journal Paper (`/Journal`):**
  * `fall_detection_journal.tex`: LaTeX source file containing the complete paper.
  * `fall_detection_journal.pdf`: Ready-to-read pre-compiled PDF document.
  * `Bibliography.bib`: Reference sources.
  * `figures/`: Diagrams used inside the journal paper.
* **Conference Paper (`/Conference Paper`):**
  * `Passenger_Safety_Fall_Detection_IEEE.docx`: Complete, formatted Word document adhering to the IEEE template.
  * `Conference_Paper.md`: Markdown version of the text.
