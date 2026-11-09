"""
Integration tests for the FastAPI app: REST endpoints, WebSocket streaming,
per-session isolation, the cancellable SOS workflow and API-key auth.
"""

import numpy as np
import pytest
from fastapi import WebSocketDisconnect
from fastapi.testclient import TestClient

from src.app import create_app
from tests.conftest import as_reading, jogging, walking


@pytest.fixture
def client(config, fake_registry, notifier):
    app = create_app(config, registry=fake_registry, notifier=notifier)
    with TestClient(app) as c:
        yield c


def stream(ws, rows, **extra):
    """Send rows as one batch and collect every reply until a prediction arrives."""
    ws.send_json({"type": "batch", "samples": [as_reading(r, **extra) for r in rows]})


def receive_until(ws, event, limit=200):
    for _ in range(limit):
        msg = ws.receive_json()
        if msg["event"] == event:
            return msg
    raise AssertionError(f"no '{event}' event received")


def fall_rows():
    rows = walking(128)
    rows[100, 0:3] = [35, 30, 10]
    return rows


def still_rows(n: int = 150, seed: int = 0):
    """Phone lying still on the floor (what follows a real fall)."""
    rng = np.random.default_rng(seed)
    rows = np.tile(np.array([0.0, 0.0, 9.81, 0, 0, 0], dtype=np.float32), (n, 1))
    rows[:, 0:3] += rng.normal(0, 0.05, (n, 3))
    rows[:, 3:6] += rng.normal(0, 0.5, (n, 3))
    return rows


def fall_then_still():
    """Spike, then lying still long enough for the post-fall check to confirm."""
    return np.concatenate([fall_rows(), still_rows()])


# ---------------------------------------------------------------------------
# REST
# ---------------------------------------------------------------------------

def test_health(client):
    data = client.get("/health").json()
    assert data["status"] == "healthy"
    assert data["models"] == ["human", "car"]
    assert data["active_sessions"] == 0 and data["notifier"] == "mock"


def test_demo_page(client):
    r = client.get("/demo")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert "WebSocket" in r.text and "cancel_alert" in r.text


def test_models_and_metrics(client):
    assert set(client.get("/models").json()) == {"human", "car"}
    m = client.get("/metrics").json()
    assert "inference_latency_ms" in m and m["connections"] == 0


def test_predict_window(client):
    r = client.post("/predict/window", json={"mode": "human", "window": fall_rows().tolist()})
    assert r.status_code == 200
    assert r.json()["is_event"] is True and r.json()["label"] == "EVENT"

    r = client.post("/predict/window", json={"mode": "car", "window": walking(128).tolist()})
    assert r.json()["is_event"] is False


@pytest.mark.parametrize("payload", [
    {"mode": "human", "window": walking(100).tolist()},          # wrong length
    {"mode": "human", "window": [[1, 2, 3]] * 128},              # wrong width
    {"mode": "plane", "window": walking(128).tolist()},          # unknown mode
])
def test_predict_window_validation(client, payload):
    assert client.post("/predict/window", json=payload).status_code == 422


# ---------------------------------------------------------------------------
# WebSocket streaming
# ---------------------------------------------------------------------------

def test_ws_session_started_and_prediction(client):
    with client.websocket_connect("/ws/predict?model=human") as ws:
        start = ws.receive_json()
        assert start["event"] == "session_started" and start["mode"] == "human"
        stream(ws, walking(128))
        pred = receive_until(ws, "prediction")
        assert pred["is_event"] is False and pred["total_inferences"] == 1


def test_ws_single_sample_messages_and_buffering(client):
    with client.websocket_connect("/ws/predict") as ws:
        ws.receive_json()
        for row in walking(8):
            ws.send_json(as_reading(row))
        msg = ws.receive_json()
        assert msg == {"event": "buffering", "buffer_fill": 8, "window_size": 128}


def test_ws_invalid_model_closes_4000(client):
    with client.websocket_connect("/ws/predict?model=bicycle") as ws:
        assert ws.receive_json()["event"] == "error"
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_json()
        assert exc.value.code == 4000


def test_ws_bad_payload_keeps_connection(client):
    with client.websocket_connect("/ws/predict") as ws:
        ws.receive_json()
        ws.send_text("not json")
        assert ws.receive_json()["event"] == "error"
        ws.send_json({"acc_x": 999})  # missing fields / out of range
        assert ws.receive_json()["event"] == "error"
        stream(ws, walking(128))
        assert receive_until(ws, "prediction")["event"] == "prediction"


def test_ws_sessions_are_isolated(client):
    """Two phones, two modes, two buffers (v1 shared one global engine)."""
    with client.websocket_connect("/ws/predict?model=human") as a, \
            client.websocket_connect("/ws/predict?model=car") as b:
        assert a.receive_json()["mode"] == "human"
        assert b.receive_json()["mode"] == "car"
        assert client.get("/health").json()["active_sessions"] == 2
        stream(a, walking(100))
        stream(b, walking(30))
        receive_until(a, "buffering")
        stream(b, walking(98))
        pred = receive_until(b, "prediction")
        assert pred["total_inferences"] == 1  # b needed its own 128 samples


# ---------------------------------------------------------------------------
# Alert workflow
# ---------------------------------------------------------------------------

def test_fall_alert_dispatched_after_countdown(client, notifier):
    with client.websocket_connect("/ws/predict?model=human") as ws:
        ws.receive_json()
        stream(ws, fall_then_still(), latitude=12.97, longitude=77.59)
        pending = receive_until(ws, "alert_pending")
        assert pending["alert"]["label"] == "EVENT"
        done = receive_until(ws, "alert_dispatched")
        assert done["alert"]["id"] == pending["alert"]["id"]
    assert len(notifier.sent) == 1 and "maps.google.com/?q=12.97" in notifier.sent[0]


def test_alert_cancel(client, notifier, config):
    config.alerts.countdown_s = 5
    with client.websocket_connect("/ws/predict?model=human") as ws:
        ws.receive_json()
        stream(ws, fall_then_still())
        alert_id = receive_until(ws, "alert_pending")["alert"]["id"]
        ws.send_json({"type": "cancel_alert", "alert_id": alert_id})
        assert receive_until(ws, "alert_cancelled")["alert"]["state"] == "cancelled"
        # a second event inside the cooldown is suppressed, not re-alerted
        stream(ws, walking(64))
        stream(ws, fall_then_still()[64:])
        assert receive_until(ws, "alert_suppressed")["reason"] == "cooldown"
    assert notifier.sent == []


def test_pending_alert_survives_disconnect(client, notifier, config):
    config.alerts.countdown_s = 0.2
    with client.websocket_connect("/ws/predict?model=human") as ws:
        ws.receive_json()
        stream(ws, fall_then_still())
        receive_until(ws, "alert_pending")
    # phone gone (e.g. destroyed in the crash) – SOS must still go out
    import time
    for _ in range(50):
        if notifier.sent:
            break
        time.sleep(0.05)
    assert len(notifier.sent) == 1


def test_car_crash_suppressed_when_vehicle_never_moved(client, notifier):
    with client.websocket_connect("/ws/predict?model=car") as ws:
        ws.receive_json()
        stream(ws, fall_then_still(), speed_kmh=0.0)
        assert receive_until(ws, "alert_suppressed")["reason"] == "vehicle_not_moving"


def test_car_crash_after_driving_alerts(client):
    with client.websocket_connect("/ws/predict?model=car") as ws:
        ws.receive_json()
        stream(ws, walking(64), speed_kmh=55.0)
        stream(ws, fall_then_still()[64:], speed_kmh=3.0)
        assert receive_until(ws, "alert_pending")["alert"]["max_recent_speed_kmh"] == 55.0


def test_post_fall_check_confirms_before_alerting(client):
    with client.websocket_connect("/ws/predict?model=human") as ws:
        ws.receive_json()
        stream(ws, fall_then_still())
        cand = receive_until(ws, "prediction")
        assert cand["candidate"] and cand["checking"] and not cand["is_event"]
        conf = receive_until(ws, "confirmation")
        assert conf["is_event"] and conf["stillness"]["acc_std"] < 1.5
        assert receive_until(ws, "alert_pending")["alert"]["label"] == "EVENT"


def test_spike_followed_by_movement_is_rejected(client, notifier):
    """Jumping / jogging / getting into a car: a spike, but the person keeps moving."""
    with client.websocket_connect("/ws/predict?model=human") as ws:
        ws.receive_json()
        stream(ws, np.concatenate([fall_rows(), jogging(150, seed=3)]))
        conf = receive_until(ws, "confirmation")
        assert conf["is_event"] is False
        assert client.get("/metrics").json()["candidates_rejected"] == 1
        assert client.get("/metrics").json()["alerts_pending"] == 0
    assert notifier.sent == []


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def test_api_key_required_when_configured(monkeypatch, config, fake_registry, notifier):
    monkeypatch.setenv("FDS_API_KEY", "s3cret")
    app = create_app(config, registry=fake_registry, notifier=notifier)
    with TestClient(app) as c:
        assert c.get("/health").status_code == 200  # liveness stays open
        assert c.get("/models").status_code == 401
        assert c.get("/models", headers={"X-API-Key": "s3cret"}).status_code == 200

        with c.websocket_connect("/ws/predict") as ws:
            assert ws.receive_json()["event"] == "error"
            with pytest.raises(WebSocketDisconnect) as exc:
                ws.receive_json()
            assert exc.value.code == 1008

        with c.websocket_connect("/ws/predict?api_key=s3cret") as ws:
            assert ws.receive_json()["event"] == "session_started"


def test_sensor_values_beyond_phone_range_are_clipped(client, fake_registry):
    seen = []
    model = fake_registry.get("human")
    original = model.score
    model.score = lambda w: (seen.append(np.abs(w[:, :3]).max()), original(w))[1]
    with client.websocket_connect("/ws/predict") as ws:
        ws.receive_json()
        rows = walking(128)
        rows[50, 0] = 400.0  # > ±16 g
        stream(ws, rows)
        receive_until(ws, "prediction")
    assert seen[0] == pytest.approx(156.9, rel=1e-4)
