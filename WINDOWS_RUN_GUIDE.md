# Windows Run Guide & Demo Instructions

This guide explains how to run the Passenger Safety & Fall Detection System on Windows and connect a smartphone, adjust detection sensitivity, and use the automated startup script.

---

## 1. Quick Start (One-Click)

We have created a helper script named [run_app.bat](file:///d:/Passenger-Safety-Fall-Detection-System/run_app.bat) in the project root. To run it:

1. Double-click [run_app.bat](file:///d:/Passenger-Safety-Fall-Detection-System/run_app.bat) to launch the HTTPS server.
2. The terminal will print a link resembling:
   `https://<YOUR_WI_FI_IP>:8000/demo`
3. Connect your smartphone and laptop to the **same Wi-Fi network**.
4. Open the printed link on your phone.
5. Tap **Advanced -> Proceed** (or "Visit this website" on iOS) to bypass the self-signed SSL certificate warning.
6. Press **Connect & Stream** on the demo page.

---

## 2. Adjusting Fall Detection Sensitivity (Classroom Demo Mode)

Currently, the system is configured in **Demonstration Mode**. Shaking or tilting the phone slightly will trigger a **FALL!** alert. 

### Why is sensitivity set high?
For classroom/lab demonstrations, it is impractical to throw the smartphone onto the floor or drop it heavily to simulate a real fall. Setting the sensitivity high allows you to trigger predictions easily by shaking or tilting the phone in your hand.

### How to change the sensitivity / threshold:
To make the detection more strict (less sensitive), open [config/config.yaml](file:///d:/Passenger-Safety-Fall-Detection-System/config/config.yaml) and modify these parameters under the `model` section:

```yaml
model:
  # 1. Classification threshold (0.0 to 1.0)
  # Increase this to require higher model confidence for a Fall.
  # Set to 0.8 or 0.85 for realistic fall detection.
  prediction_threshold: 0.85

  # 2. Stationary Phone Guardrail
  # Increase this to prevent slight hand shakes from overriding "Normal" state.
  # Set to 0.15 or 0.20 to ignore minor hand movements.
  static_variance_threshold: 0.15
```

---

## 3. Manual Step-by-Step Instructions

If you prefer to run it manually via PowerShell:

1. **Activate the environment:**
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
   .\fall\Scripts\Activate.ps1
   ```

2. **Run tests to verify:**
   ```powershell
   python -m pytest tests/ -v
   ```

3. **Start the secure server:**
   ```powershell
   uvicorn src.app:app --host 0.0.0.0 --port 8000 --ssl-keyfile key.pem --ssl-certfile cert.pem --reload
   ```

---

## 4. Connecting Phone (Troubleshooting)

If the page on your phone fails to load, verify the following:
* **Same Wi-Fi network:** Check that both your phone and PC are connected to the same Wi-Fi.
* **HTTPS Protocol:** Ensure the URL starts with `https://` (not `http://`).
* **Firewall Block:** Windows Defender Firewall may block port 8000. Try disabling the firewall temporarily or allowing port 8000 if it continues to timeout.
