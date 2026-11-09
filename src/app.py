"""
src/app.py
----------
FastAPI server for real-time fall / crash detection from a smartphone.

Endpoints:
    GET  /health            – liveness + loaded models + active sessions
    GET  /models            – model metadata (threshold, size, test metrics)
    GET  /metrics           – counters and inference latency percentiles
    GET  /demo              – phone sensor demo page
    POST /predict/window    – stateless scoring of one full 128×6 window
    WS   /ws/predict        – streaming inference + alert workflow

Start server:
    uvicorn src.app:app --host 0.0.0.0 --port 8000

WebSocket protocol (client → server), one JSON object per message:
    {"acc_x": .., "acc_y": .., "acc_z": .., "gyro_x": .., "gyro_y": .., "gyro_z": ..,
     "t": 1712.5, "latitude": .., "longitude": .., "speed_kmh": ..}   # t (ms) optional
    {"type": "batch", "samples": [<sample>, ...]}
    {"type": "cancel_alert", "alert_id": 3}

Server → client events:
    session_started | buffering | prediction | confirmation | alert_pending |
    alert_suppressed | alert_cancelled | alert_dispatched | alert_failed | error

Post-fall check (config inference.post_fall_check): a positive window arrives as
a `prediction` with "candidate": true / "checking": true. A few seconds later a
`confirmation` event says whether the person stayed still ("is_event": true →
the alert workflow starts) or kept moving ("is_event": false → rejected).
With the check disabled, a positive `prediction` has "is_event": true directly.

Auth: if FDS_API_KEY is set, clients must send it as the `X-API-Key` header
(REST) or `?api_key=` query parameter (WebSocket – browsers cannot set headers).
"""

import asyncio
import hmac
import json
import os
import sys
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field, ValidationError, field_validator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.alerts import Alert, AlertManager
from src.inference import ModelRegistry, StreamSession, session_from_config
from src.signal_processing import clip_to_sensor_range
from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.notifier import build_notifier, build_sos_message

logger = get_logger(__name__)

WS_POLICY_VIOLATION = 1008
WS_UNKNOWN_MODE = 4000
WS_SERVER_FULL = 4003


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class SensorReading(BaseModel):
    """
    One 6-DOF IMU sample. Accelerometer in m/s² (gravity included),
    gyroscope in °/s. Bounds reject garbage; values beyond a phone's
    full-scale range are clipped server-side, exactly as in training.
    """
    acc_x: float = Field(..., ge=-500.0, le=500.0)
    acc_y: float = Field(..., ge=-500.0, le=500.0)
    acc_z: float = Field(..., ge=-500.0, le=500.0)
    gyro_x: float = Field(..., ge=-5000.0, le=5000.0)
    gyro_y: float = Field(..., ge=-5000.0, le=5000.0)
    gyro_z: float = Field(..., ge=-5000.0, le=5000.0)
    t: Optional[float] = Field(None, description="Client timestamp in ms (enables resampling)")

    latitude: Optional[float] = Field(None, ge=-90.0, le=90.0)
    longitude: Optional[float] = Field(None, ge=-180.0, le=180.0)
    speed_kmh: Optional[float] = Field(None, ge=0.0, le=500.0)

    def to_sample(self) -> list[float]:
        return [self.acc_x, self.acc_y, self.acc_z, self.gyro_x, self.gyro_y, self.gyro_z]


class WindowRequest(BaseModel):
    mode: Literal["human", "car"] = "human"
    window: list[list[float]] = Field(..., description="window_size rows × 6 columns")

    @field_validator("window")
    @classmethod
    def _six_columns(cls, v):
        if not v or any(len(row) != 6 for row in v):
            raise ValueError("every row must have 6 values (acc_x..gyro_z)")
        return v


# ---------------------------------------------------------------------------
# Server state
# ---------------------------------------------------------------------------

class ServerState:
    def __init__(self):
        self.registry: Optional[ModelRegistry] = None
        self.notifier = None
        self.sessions: dict[str, dict] = {}
        self.started_at = time.time()
        self.counters = {"connections": 0, "inferences": 0, "events": 0,
                         "candidates_confirmed": 0, "candidates_rejected": 0,
                         "alerts_pending": 0, "alerts_dispatched": 0,
                         "alerts_cancelled": 0, "alerts_failed": 0, "alerts_suppressed": 0}
        self.latencies: deque = deque(maxlen=2000)


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app(config=None, registry: Optional[ModelRegistry] = None, notifier=None) -> FastAPI:
    config = config or load_config()
    state = ServerState()
    api_key = os.environ.get("FDS_API_KEY", "")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state.registry = registry or ModelRegistry.from_config(config)
        state.notifier = notifier or build_notifier()
        if not api_key:
            logger.warning("FDS_API_KEY not set – API is open (fine for a LAN demo, not for production).")
        logger.info("Server ready.")
        yield
        logger.info("Shutting down server.")

    app = FastAPI(
        title="Passenger Safety & Fall Detection API",
        description="Real-time 6-axis IMU fall and crash detection with a cancellable SOS workflow.",
        version="2.0.0",
        lifespan=lifespan,
    )
    app.state.fds = state

    def _key_ok(provided: Optional[str]) -> bool:
        return not api_key or (provided is not None and hmac.compare_digest(provided, api_key))

    def require_api_key(x_api_key: Optional[str] = Header(None)):
        if not _key_ok(x_api_key):
            raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")

    async def dispatch(alert: Alert) -> bool:
        body = build_sos_message(alert.label, alert.confidence, alert.latitude,
                                 alert.longitude, alert.speed_kmh)
        ok = await asyncio.to_thread(state.notifier.send, body)
        state.counters["alerts_dispatched" if ok else "alerts_failed"] += 1
        return ok

    # -----------------------------------------------------------------------
    # REST
    # -----------------------------------------------------------------------

    @app.get("/", include_in_schema=False)
    async def root():
        return RedirectResponse("/demo")

    @app.get("/health", summary="Liveness check")
    async def health():
        ready = state.registry is not None
        return {
            "status": "healthy" if ready else "starting",
            "models": list(state.registry.models) if ready else [],
            "active_sessions": len(state.sessions),
            "notifier": getattr(state.notifier, "name", None),
            "auth_enabled": bool(api_key),
            "uptime_s": round(time.time() - state.started_at, 1),
        }

    @app.get("/models", summary="Loaded models and their evaluation metrics",
             dependencies=[Depends(require_api_key)])
    async def models_info():
        return {mode: m.info() for mode, m in state.registry.models.items()}

    @app.get("/metrics", summary="Server counters and latency percentiles",
             dependencies=[Depends(require_api_key)])
    async def metrics():
        lat = np.asarray(state.latencies) if state.latencies else np.zeros(1)
        return {
            **state.counters,
            "active_sessions": len(state.sessions),
            "inference_latency_ms": {
                "p50": round(float(np.percentile(lat, 50)), 3),
                "p95": round(float(np.percentile(lat, 95)), 3),
                "p99": round(float(np.percentile(lat, 99)), 3),
            },
        }

    @app.get("/demo", response_class=HTMLResponse, summary="Phone sensor demo page")
    async def phone_demo():
        demo_path = Path(__file__).parent / "demo_phone.html"
        return HTMLResponse(demo_path.read_text(encoding="utf-8"))

    @app.post("/predict/window", summary="Score one full window (stateless)",
              dependencies=[Depends(require_api_key)])
    async def predict_window(req: WindowRequest):
        model = state.registry.get(req.mode)
        window = np.asarray(req.window, dtype=np.float32)
        if window.shape != (model.window_size, 6):
            raise HTTPException(422, f"window must be {model.window_size}×6, got {window.shape[0]}×6")
        window = clip_to_sensor_range(window, config.signal.acc_range, config.signal.gyro_range)
        score, latency_ms = model.score(window)
        is_event = score >= model.threshold
        state.counters["inferences"] += 1
        state.latencies.append(latency_ms)
        return {
            "mode": req.mode,
            "label": model.labels[int(is_event)],
            "is_event": is_event,
            "score": round(score, 4),
            "threshold": round(model.threshold, 4),
            "latency_ms": round(latency_ms, 3),
        }

    # -----------------------------------------------------------------------
    # WebSocket
    # -----------------------------------------------------------------------

    @app.websocket("/ws/predict")
    async def ws_predict(websocket: WebSocket, model: str = "human",
                         api_key_q: Optional[str] = Query(None, alias="api_key")):
        await websocket.accept()

        if not _key_ok(api_key_q or websocket.headers.get("x-api-key")):
            await websocket.send_json({"event": "error", "detail": "Invalid or missing api_key"})
            await websocket.close(code=WS_POLICY_VIOLATION)
            return
        if model not in state.registry:
            await websocket.send_json({"event": "error",
                                       "detail": f"Unknown model '{model}'. Available: {list(state.registry.models)}"})
            await websocket.close(code=WS_UNKNOWN_MODE)
            return
        if len(state.sessions) >= config.api.max_sessions:
            await websocket.send_json({"event": "error", "detail": "Server at capacity"})
            await websocket.close(code=WS_SERVER_FULL)
            return

        session_id = uuid.uuid4().hex[:8]
        session: StreamSession = session_from_config(state.registry.get(model), config)
        car_cfg = config.alerts.get("car", {}) if model == "car" else {}
        alerts = AlertManager(
            mode=model,
            dispatcher=dispatch,
            countdown_s=config.alerts.countdown_s,
            cooldown_s=config.alerts.cooldown_s,
            min_recent_speed_kmh=car_cfg.get("min_recent_speed_kmh"),
            speed_lookback_s=car_cfg.get("speed_lookback_s", 30),
            dispatch_on_disconnect=config.alerts.dispatch_on_disconnect,
        )
        state.sessions[session_id] = {"mode": model, "client": str(websocket.client),
                                      "session": session, "alerts": alerts}
        state.counters["connections"] += 1
        logger.info(f"[{session_id}] connected: {websocket.client} | mode={model}")

        await websocket.send_json({"event": "session_started", "session_id": session_id,
                                   "mode": model, "model": session.model.info(),
                                   "countdown_s": config.alerts.countdown_s})

        async def handle_reading(reading: SensorReading) -> None:
            alerts.observe(reading.speed_kmh)
            sample = clip_to_sensor_range(np.asarray(reading.to_sample()),
                                          config.signal.acc_range, config.signal.gyro_range)
            results = session.push(sample, reading.t)

            if not results and session.buffer_fill < session.window_size and session.total_samples % 8 == 0:
                await websocket.send_json({"event": "buffering", "buffer_fill": session.buffer_fill,
                                           "window_size": session.window_size})

            for result in results:
                if result.get("kind") == "confirmation":
                    state.counters["candidates_confirmed" if result["is_event"] else "candidates_rejected"] += 1
                    await websocket.send_json({"event": "confirmation", **result})
                else:
                    state.counters["inferences"] += 1
                    state.latencies.append(result["latency_ms"])
                    await websocket.send_json({"event": "prediction", **result})
                if not result["is_event"]:
                    continue

                state.counters["events"] += 1
                logger.warning(f"[{session_id}] {result['label']} score={result['raw_score']:.3f}")
                alert, reason = alerts.on_event(result["label"], result["raw_score"],
                                                reading.latitude, reading.longitude, reading.speed_kmh)
                if alert is None:
                    state.counters["alerts_suppressed"] += 1
                    await websocket.send_json({"event": "alert_suppressed", "reason": reason})
                else:
                    state.counters["alerts_pending"] += 1
                    alerts.start_countdown(sink=websocket.send_json)
                    await websocket.send_json({"event": "alert_pending", "alert": alert.to_dict(),
                                               "countdown_s": config.alerts.countdown_s})

        try:
            while True:
                raw = await websocket.receive_text()
                if len(raw) > config.api.max_message_bytes:
                    await websocket.send_json({"event": "error", "detail": "Message too large"})
                    continue
                try:
                    msg = json.loads(raw)
                    if not isinstance(msg, dict):
                        raise ValueError("expected a JSON object")
                    kind = msg.get("type", "sample")

                    if kind == "cancel_alert":
                        cancelled = alerts.cancel(msg.get("alert_id"))
                        if cancelled:
                            state.counters["alerts_cancelled"] += 1
                            await websocket.send_json({"event": "alert_cancelled",
                                                       "alert": cancelled.to_dict()})
                        else:
                            await websocket.send_json({"event": "error", "detail": "No pending alert"})
                    elif kind == "batch":
                        for item in msg.get("samples", []):
                            await handle_reading(SensorReading(**item))
                    elif kind == "sample":
                        await handle_reading(SensorReading(**msg))
                    else:
                        raise ValueError(f"unknown message type '{kind}'")
                except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as exc:
                    await websocket.send_json({"event": "error", "detail": f"Invalid payload: {exc}"})

        except WebSocketDisconnect:
            logger.info(f"[{session_id}] disconnected")
        except Exception as exc:
            logger.exception(f"[{session_id}] unexpected error: {exc}")
            await websocket.close(code=1011)
        finally:
            alerts.detach()  # a pending alert still fires if dispatch_on_disconnect
            state.sessions.pop(session_id, None)

    return app


app = create_app()
