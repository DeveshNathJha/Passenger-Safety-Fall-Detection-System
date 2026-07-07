# Notes: `src/demo_phone.html`

## Purpose
A single-file mobile web application that uses the phone's built-in **accelerometer and gyroscope** (DeviceMotion API) to stream real-time IMU data to the FastAPI WebSocket server. No native app or installation required — just open the URL in the phone browser.

---

## How It Works

```
Phone Browser                     FastAPI Server
      │                                 │
      │  Open http://<PC_IP>:8000/demo  │
      │◄────────────────────────────────│ Serves demo_phone.html
      │                                 │
      │  Connect to /ws/predict WS      │
      │─────────────────────────────────►│
      │                                 │
      │  DeviceMotion event (50 Hz)     │
      │  → send JSON sensor reading     │
      │─────────────────────────────────►│ push_sample()
      │                                 │
      │◄─── {"event": "buffering"} ─────│ (while buffer fills)
      │◄─── {"event": "prediction"} ────│ (every 64 new samples)
      │                                 │
      │  Update UI: label, confidence   │
      │  Show  ALERT if fall detected │
```

---

## Browser Sensor Access

### `window.addEventListener('devicemotion', handler)`
- `event.accelerationIncludingGravity` → `{x, y, z}` in m/s² (includes gravity)
- `event.rotationRate` → `{alpha, beta, gamma}` in °/s (gyroscope)

### iOS 13+ Permission
Apple requires an explicit user gesture to access DeviceMotion. The page detects this:
```javascript
if (typeof DeviceMotionEvent.requestPermission === 'function') {
    // Show "Request Permission" button
}
```
After the user taps the button, `requestPermission()` is called.

### Android / Chrome
DeviceMotion works automatically with no permission prompt.

---

## UI Components

| Section | Shows |
|---|---|
| **Status dot** | Disconnected / Connected / Streaming (animated) |
| **Live IMU grid** | 6 values updated in real-time at device rate |
| **Sliding Window Buffer** | Progress bar filling from 0→128 samples |
| **Prediction card** | Label + confidence bar (color changes green/red) |
| **FALL ALERT banner** | Red banner + shake animation when fall detected |
| **Session stats** | Samples sent / windows predicted / falls detected / actual Hz |

---

## Data Flow Validation
The page **clamps** sensor values to valid API ranges before sending:
```javascript
acc_x: Math.max(-20, Math.min(20, imu.ax))
gyro_x: Math.max(-600, Math.min(600, imu.gx))
```
This prevents Pydantic validation errors on the server.

---

## Demo Tips
- **Normal reading:** Hold phone still on a flat surface. Acc Y ≈ 9.8 (gravity). Gyro ≈ 0.
- **Simulate fall:** Quickly flip or shake the phone — gyro values spike, acc magnitude changes.
- The model needs **128 samples** to make its first prediction (~2.56 seconds at 50 Hz).
- After the first prediction, a new one arrives every **64 samples** (~1.28 seconds).
