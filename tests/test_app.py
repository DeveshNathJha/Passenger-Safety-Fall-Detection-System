"""
tests/test_app.py
-----------------
Integration tests for the FastAPI application.
Covers REST API endpoints (/health, /demo, /predict/single) and the WebSocket endpoint (/ws/predict).
Run with: python -m pytest tests/test_app.py -v
"""

import pytest
import sys
from pathlib import Path
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.app import app
from src.utils.config_loader import load_config

# Load config to verify values
config = load_config()


def test_health_endpoint():
    """Verify that the health check endpoint returns 200 and correct structure."""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "active_mode" in data
        assert "model" in data
        assert "total_inferences" in data


def test_demo_endpoint():
    """Verify that the demo endpoint returns HTML content."""
    with TestClient(app) as client:
        response = client.get("/demo")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "WebSocket" in response.text or "websocket" in response.text.lower()


def test_predict_single_endpoint_valid():
    """Verify that sending a valid IMU reading returns a buffering/prediction event."""
    with TestClient(app) as client:
        payload = {
            "acc_x": 0.1,
            "acc_y": 9.8,
            "acc_z": 0.3,
            "gyro_x": 0.01,
            "gyro_y": 0.02,
            "gyro_z": 0.0,
            "latitude": 37.7749,
            "longitude": -122.4194,
            "speed_kmh": 45.0
        }
        response = client.post("/predict/single", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "event" in data
        assert data["event"] in ["buffering", "prediction"]


def test_predict_single_endpoint_invalid_bounds():
    """Verify that values outside Pydantic validation bounds return 422 Unprocessable Entity."""
    with TestClient(app) as client:
        # acc_x is 600.0, which is > 500.0 limit
        payload = {
            "acc_x": 600.0,
            "acc_y": 9.8,
            "acc_z": 0.3,
            "gyro_x": 0.01,
            "gyro_y": 0.02,
            "gyro_z": 0.0
        }
        response = client.post("/predict/single", json=payload)
        assert response.status_code == 422


def test_websocket_predict_human_model():
    """Verify that WebSocket connection is established and streams data using human model."""
    with TestClient(app) as client:
        with client.websocket_connect("/ws/predict?model=human") as websocket:
            # Send one reading
            reading = {
                "acc_x": 0.1,
                "acc_y": 9.8,
                "acc_z": 0.3,
                "gyro_x": 0.01,
                "gyro_y": 0.02,
                "gyro_z": 0.0
            }
            websocket.send_json(reading)
            data = websocket.receive_json()
            assert "event" in data
            assert data["event"] in ["buffering", "prediction"]


def test_websocket_predict_car_model():
    """Verify that WebSocket connection works with car model."""
    with TestClient(app) as client:
        with client.websocket_connect("/ws/predict?model=car") as websocket:
            reading = {
                "acc_x": 0.1,
                "acc_y": 9.8,
                "acc_z": 0.3,
                "gyro_x": 0.01,
                "gyro_y": 0.02,
                "gyro_z": 0.0,
                "speed_kmh": 0.0
            }
            websocket.send_json(reading)
            data = websocket.receive_json()
            assert "event" in data
            assert data["event"] in ["buffering", "prediction"]


def test_websocket_invalid_model():
    """Verify that WebSocket closes with custom code 4000 when model is invalid."""
    with TestClient(app) as client:
        with client.websocket_connect("/ws/predict?model=invalid_model_name") as websocket:
            # First it should send the error payload
            data = websocket.receive_json()
            assert "error" in data
            # Then the subsequent read/action should raise WebSocketDisconnect with code 4000
            from fastapi import WebSocketDisconnect
            with pytest.raises(WebSocketDisconnect) as exc_info:
                websocket.receive_json()
            assert exc_info.value.code == 4000


def test_websocket_predict_car_model_sos_trigger_low_speed(monkeypatch):
    """Verify that WebSocket triggers Twilio SOS when a crash is detected at low speed (< 10 km/h)."""
    import src.app as app_mod
    
    dummy_prediction = {
        "label": "FALL DETECTED",
        "confidence": 0.95,
        "raw_score": 0.95,
        "latency_ms": 1.5,
        "total_inferences": 1,
        "fall_count": 1
    }
    
    sos_calls = []
    def mock_send_sos_sms(lat, lon, speed):
        sos_calls.append((lat, lon, speed))
        return True
        
    with TestClient(app) as client:
        # After entering client context, lifespan has run and app_mod._engine is initialized.
        monkeypatch.setattr(app_mod._engine, "push_sample", lambda sample: dummy_prediction)
        monkeypatch.setattr(app_mod._engine, "set_mode", lambda mode: None)
        
        from src.utils.notifier import notifier
        monkeypatch.setattr(notifier, "send_sos_sms", mock_send_sos_sms)
        
        with client.websocket_connect("/ws/predict?model=car") as websocket:
            reading = {
                "acc_x": 0.1,
                "acc_y": 9.8,
                "acc_z": 0.3,
                "gyro_x": 0.01,
                "gyro_y": 0.02,
                "gyro_z": 0.0,
                "latitude": 37.7749,
                "longitude": -122.4194,
                "speed_kmh": 5.0
            }
            websocket.send_json(reading)
            resp = websocket.receive_json()
            assert resp["event"] == "prediction"
            assert resp["label"] == "FALL DETECTED"
            
            # SOS should be triggered
            assert len(sos_calls) == 1
            assert sos_calls[0] == (37.7749, -122.4194, 5.0)


def test_websocket_predict_car_model_sos_trigger_no_gps(monkeypatch):
    """Verify that WebSocket triggers Twilio SOS when a crash is detected and speed is None (GPS speed unavailable)."""
    import src.app as app_mod
    
    dummy_prediction = {
        "label": "FALL DETECTED",
        "confidence": 0.95,
        "raw_score": 0.95,
        "latency_ms": 1.5,
        "total_inferences": 1,
        "fall_count": 1
    }
    
    sos_calls = []
    def mock_send_sos_sms(lat, lon, speed):
        sos_calls.append((lat, lon, speed))
        return True
        
    with TestClient(app) as client:
        # After entering client context, lifespan has run and app_mod._engine is initialized.
        monkeypatch.setattr(app_mod._engine, "push_sample", lambda sample: dummy_prediction)
        monkeypatch.setattr(app_mod._engine, "set_mode", lambda mode: None)
        
        from src.utils.notifier import notifier
        monkeypatch.setattr(notifier, "send_sos_sms", mock_send_sos_sms)
        
        with client.websocket_connect("/ws/predict?model=car") as websocket:
            reading = {
                "acc_x": 0.1,
                "acc_y": 9.8,
                "acc_z": 0.3,
                "gyro_x": 0.01,
                "gyro_y": 0.02,
                "gyro_z": 0.0,
                # No GPS fields provided
            }
            websocket.send_json(reading)
            resp = websocket.receive_json()
            assert resp["event"] == "prediction"
            
            # SOS should be triggered (speed_kmh is None, latitude is None, longitude is None)
            assert len(sos_calls) == 1
            assert sos_calls[0] == (None, None, None)


def test_websocket_predict_car_model_no_sos_on_high_speed(monkeypatch):
    """Verify that WebSocket does NOT trigger Twilio SOS when a crash is detected but speed is high (>= 10 km/h)."""
    import src.app as app_mod
    
    dummy_prediction = {
        "label": "FALL DETECTED",
        "confidence": 0.95,
        "raw_score": 0.95,
        "latency_ms": 1.5,
        "total_inferences": 1,
        "fall_count": 1
    }
    
    sos_calls = []
    def mock_send_sos_sms(lat, lon, speed):
        sos_calls.append((lat, lon, speed))
        return True
        
    with TestClient(app) as client:
        # After entering client context, lifespan has run and app_mod._engine is initialized.
        monkeypatch.setattr(app_mod._engine, "push_sample", lambda sample: dummy_prediction)
        monkeypatch.setattr(app_mod._engine, "set_mode", lambda mode: None)
        
        from src.utils.notifier import notifier
        monkeypatch.setattr(notifier, "send_sos_sms", mock_send_sos_sms)
        
        with client.websocket_connect("/ws/predict?model=car") as websocket:
            reading = {
                "acc_x": 0.1,
                "acc_y": 9.8,
                "acc_z": 0.3,
                "gyro_x": 0.01,
                "gyro_y": 0.02,
                "gyro_z": 0.0,
                "latitude": 37.7749,
                "longitude": -122.4194,
                "speed_kmh": 65.0  # High speed, vehicle has not stopped yet
            }
            websocket.send_json(reading)
            resp = websocket.receive_json()
            assert resp["event"] == "prediction"
            
            # SOS should NOT be triggered
            assert len(sos_calls) == 0



