"""
src/app.py
----------
FastAPI WebSocket server for real-time fall detection from smartphone/device.

Endpoints:
    GET  /health        – Liveness check + engine metadata
    WS   /ws/predict    – Accepts 50 Hz IMU JSON, returns predictions

Start server:
    uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload

WebSocket message format (send):
    {"acc_x": 0.1, "acc_y": 9.8, "acc_z": 0.3,
     "gyro_x": 0.01, "gyro_y": 0.02, "gyro_z": 0.0}

WebSocket message format (receive when prediction ready):
    {"event": "prediction", "label": "Normal Activity",
     "confidence": 0.9312, "latency_ms": 2.1, ...}
"""

import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import FallDetectionEngine
from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.notifier import notifier

logger = get_logger(__name__)
config = load_config()


# ---------------------------------------------------------------------------
# Pydantic IMU Schema
# ---------------------------------------------------------------------------

class SensorReading(BaseModel):
    """
    A single 6-DOF IMU measurement from an MPU6050 or equivalent sensor.
    Ranges enforce realistic physical bounds:
      - Accelerometer : ±500 m/s² (supports up to ~50g crash impacts)
      - Gyroscope     : ±2000 °/s  (supports extreme rotational events)
    """
    acc_x: float = Field(..., ge=-500.0, le=500.0, description="Accelerometer X (m/s²)")
    acc_y: float = Field(..., ge=-500.0, le=500.0, description="Accelerometer Y (m/s²)")
    acc_z: float = Field(..., ge=-500.0, le=500.0, description="Accelerometer Z (m/s²)")
    gyro_x: float = Field(..., ge=-2000.0, le=2000.0, description="Gyroscope X (°/s)")
    gyro_y: float = Field(..., ge=-2000.0, le=2000.0, description="Gyroscope Y (°/s)")
    gyro_z: float = Field(..., ge=-2000.0, le=2000.0, description="Gyroscope Z (°/s)")
    
    # Optional GPS fusion data
    latitude: Optional[float] = Field(None, description="GPS Latitude")
    longitude: Optional[float] = Field(None, description="GPS Longitude")
    speed_kmh: Optional[float] = Field(None, ge=0.0, description="GPS Speed (km/h)")

    def to_sample(self) -> list[float]:
        """Return values in the model's expected column order."""
        return [self.acc_x, self.acc_y, self.acc_z,
                self.gyro_x, self.gyro_y, self.gyro_z]


# ---------------------------------------------------------------------------
# Application Lifespan (loads engine on startup)
# ---------------------------------------------------------------------------

_engine: Optional[FallDetectionEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the TFLite models once at startup; release on shutdown."""
    global _engine
    
    tflite_paths = {
        "human": config.paths.model_tflite_human,
        "car": config.paths.model_tflite_car
    }
    logger.info(f"Loading TFLite models: {tflite_paths}")
    
    _engine = FallDetectionEngine(
        tflite_paths=tflite_paths,
        window_size=config.windowing.window_size,
        step=int(config.windowing.window_size * (1 - config.windowing.overlap)),
        threshold=config.model.prediction_threshold,
        labels={int(k): v for k, v in config.model.labels.items()},
        static_variance_threshold=config.model.static_variance_threshold,
    )
    logger.info("FallDetectionEngine initialized. Server ready.")
    yield
    logger.info("Shutting down server.")


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Passenger Fall Detection API",
    description=(
        "Real-time 6-axis IMU fall detection system for in-vehicle passenger safety. "
        "Send sensor readings at 50 Hz over WebSocket to get continuous predictions."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", summary="Health check & engine status")
async def health_check():
    """
    Returns server status and current inference engine statistics.
    """
    if _engine is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "detail": "Engine not initialized."},
        )
    return {
        "status": "healthy",
        "active_mode": _engine.active_mode,
        "model": _engine.tflite_paths[_engine.active_mode],
        "window_size": _engine.window_size,
        "step": _engine.step,
        "buffer_fill": _engine.buffer_fill,
        "total_inferences": _engine.total_inferences,
        "fall_count": _engine.fall_count,
    }


@app.get("/demo", response_class=HTMLResponse, summary="Phone sensor demo page")
async def phone_demo():
    """
    Serve the phone WebSocket demo page.
    Open on your smartphone: http://<YOUR_PC_LAN_IP>:8000/demo
    The page uses the DeviceMotion API to stream gyroscope/accelerometer
    data directly to the WebSocket inference endpoint.
    """
    demo_path = Path(__file__).parent / "demo_phone.html"
    return HTMLResponse(content=demo_path.read_text(encoding="utf-8"), status_code=200)



@app.post("/predict/single", summary="Single-window inference (REST)")
async def predict_single(reading: SensorReading):
    """
    Push one sensor reading into the sliding-window buffer.
    Returns a prediction dict only when the buffer is full (every `step` samples),
    otherwise returns a buffering acknowledgement.
    """
    if _engine is None:
        return JSONResponse(status_code=503, content={"error": "Engine not ready."})

    result = _engine.push_sample(reading.to_sample())
    if result:
        return {"event": "prediction", **result}
    return {
        "event": "buffering",
        "buffer_fill": _engine.buffer_fill,
        "window_size": _engine.window_size,
    }


# ---------------------------------------------------------------------------
# WebSocket Endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket, model: str = "human"):
    """
    Persistent WebSocket connection for real-time streaming inference.
    Query Param: `?model=human` or `?model=car` (swaps the active TFLite engine).
    """
    await websocket.accept()
    client = websocket.client
    logger.info(f"WebSocket connected: {client} | Mode: {model}")

    if _engine is None:
        await websocket.send_json({"error": "Engine not initialized."})
        await websocket.close(code=1011)
        return
        
    try:
        _engine.set_mode(model)
    except ValueError as e:
        await websocket.send_json({"error": str(e)})
        await websocket.close(code=4000)
        return

    try:
        while True:
            raw = await websocket.receive_text()

            # --- Parse & validate ---
            try:
                data = json.loads(raw)
                reading = SensorReading(**data)
            except (json.JSONDecodeError, ValueError) as exc:
                await websocket.send_json(
                    {"event": "error", "detail": f"Invalid payload: {exc}"}
                )
                continue

            # --- Inference ---
            result = _engine.push_sample(reading.to_sample())

            if result:
                payload = {"event": "prediction", **result}
                if result["label"] != "Normal Activity":
                    logger.warning(
                        f"CRASH/FALL DETECTED from {client} — "
                        f"confidence: {result['confidence']:.1%} | "
                        f"raw_score: {result.get('raw_score', '?')} | "
                        f"acc_mag_std: {result.get('acc_mag_std', '?')}"
                    )
                    # Phase 14: SOS System triggered if speed dropped (or if GPS speed is unavailable as a fail-safe)
                    if model == "car" and result["confidence"] > 0.90:
                        if reading.speed_kmh is None or reading.speed_kmh < 10.0:
                            speed_str = f"{reading.speed_kmh:.1f} km/h" if reading.speed_kmh is not None else "UNKNOWN"
                            logger.critical(f"SOS INITIATED! Final Speed: {speed_str}")
                            notifier.send_sos_sms(reading.latitude, reading.longitude, reading.speed_kmh)
                            
                await websocket.send_json(payload)
            else:
                await websocket.send_json(
                    {
                        "event": "buffering",
                        "buffer_fill": _engine.buffer_fill,
                        "window_size": _engine.window_size,
                    }
                )

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {client}")
    except Exception as exc:
        logger.exception(f"Unexpected WebSocket error from {client}: {exc}")
        await websocket.close(code=1011)
