# Notes: `src/app.py`

## Purpose
FastAPI server that exposes the `FallDetectionEngine` over HTTP and WebSocket. Designed for real-time 50 Hz sensor ingestion from a smartphone.

---

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Server liveness + engine stats |
| `GET` | `/demo` | Phone demo page (browser → WebSocket) |
| `POST` | `/predict/single` | REST alternative (one sample per call) |
| `WebSocket` | `/ws/predict` | Full-duplex real-time streaming |
| `GET` | `/docs` | Swagger auto-docs |

---

## Pydantic Schema: `SensorReading`
```python
class SensorReading(BaseModel):
    acc_x: float = Field(..., ge=-20.0, le=20.0)
    acc_y: float = Field(..., ge=-20.0, le=20.0)
    acc_z: float = Field(..., ge=-20.0, le=20.0)
    gyro_x: float = Field(..., ge=-600.0, le=600.0)
    gyro_y: float = Field(..., ge=-600.0, le=600.0)
    gyro_z: float = Field(..., ge=-600.0, le=600.0)
```
**Physical bounds:**
- Accelerometer: ±2g MPU6050 range ≈ ±~20 m/s²
- Gyroscope: ±500 °/s MPU6050 range → ±600 with headroom

If the phone sends values outside these bounds, the server returns a Pydantic validation error automatically.

---

## Lifespan (startup/shutdown)
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    _engine = FallDetectionEngine(...)   # loaded ONCE at startup
    yield
    # shutdown cleanup here
```
The TFLite model is loaded **once** when the server starts and reused for all connections. This avoids re-loading the model per request (which would be ~100 ms overhead).

---

## WebSocket Protocol

### Phone → Server (50 Hz):
```json
{"acc_x": 0.1, "acc_y": 9.8, "acc_z": 0.3,
 "gyro_x": 0.01, "gyro_y": 0.02, "gyro_z": 0.0}
```

### Server → Phone (between predictions):
```json
{"event": "buffering", "buffer_fill": 45, "window_size": 128}
```

### Server → Phone (every 64 samples once full window is ready):
```json
{"event": "prediction", "label": "Normal Activity",
 "confidence": 0.9312, "latency_ms": 2.1,
 "total_inferences": 42, "fall_count": 0}
```

### Server → Phone (on invalid JSON/value):
```json
{"event": "error", "detail": "Invalid payload: ..."}
```

---

## Start Server
```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```
- `--host 0.0.0.0` → accessible from phone on same WiFi
- `--port 8000` → default port
- `--reload` → auto-restarts on code changes (dev only)

## Open Demo on Phone
1. Connect phone and PC to **same WiFi**.
2. Find your PC's LAN IP: `ip addr | grep "inet "` (typically `192.168.x.y`)
3. Open on phone: `http://192.168.x.y:8000/demo`
4. Tap "Connect & Stream" → shake phone → see predictions.
