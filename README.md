<div align="center">

#  Passenger Safety & Fall Detection System

**A production-ready Edge AI pipeline for real-time in-vehicle passenger fall detection using 6-axis IMU data.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?logo=tensorflow)](https://tensorflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![TFLite](https://img.shields.io/badge/TFLite-Edge%20Optimized-blueviolet)](https://tensorflow.org/lite)
[![Accuracy](https://img.shields.io/badge/Accuracy-97.04%25-brightgreen)](#model-performance)

</div>

---

##  Overview

This system features a **Dual-Model Architecture** to detect both **Human Passenger Falls** and **High-Impact Vehicle Crashes**. It uses **1D-CNN** models optimized for edge deployment. 

Key features include:
* **Human Fall Detection**: Trained on the MobiFall v2.0 Dataset.
* **Car Crash Detection**: Trained on a physics-based synthetic dataset simulating high-G impacts and rollovers.
* **Sensor Fusion**: Integrates 6-axis IMU (Accelerometer + Gyroscope) with live GPS Coordinates and Speed.
* **Automated SOS**: Triggers emergency alerts via Twilio SMS when a crash is detected at speed.
* **Mobile Dashboard**: A real-time HTML5 interface for sensor streaming and visualization.

---

##  Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA PIPELINE                           │
│                                                             │
│  MobiFall v2.0 ──► parse_txt_file() ──► create_windows()   │
│  (Kaggle/Local)    (6-col IMU rows)    (128 samples, 50%)   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   TRAINING PIPELINE                         │
│                                                             │
│  train.py ──► 1D-CNN (selected) ──► ModelCheckpoint        │
│              LSTM / CNN-LSTM         EarlyStopping          │
│              Transformer             mobifall_model.h5      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 EDGE OPTIMIZATION                           │
│                                                             │
│  .h5 ──► tflite_converter.py ──► mobifall_edge_model.tflite │
│           (Dynamic Quantization)   (15 KB – 90% smaller)   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 REAL-TIME INFERENCE (EDGE)                     │
│                                                             │
│  Smartphone ──► WebSocket ──► FallDetectionEngine           │
│  (50 Hz IMU)   /ws/predict    Hot-Swappable Model Dict       │
│  + GPS/Speed                  {"human": ..., "car": ...}    │
│                                           │                 │
│                                           ▼                 │
│                                  {"label": "CRASH",         │
│                                   "confidence": 0.98}      │
│                                           │                 │
│                                           ▼                 │
│                                  Twilio Notifier (SOS)      │
└─────────────────────────────────────────────────────────────┘
```

---

##  Project Structure

```
Passenger-Safety-Fall-Detection-System/
├── config/
│   └── config.yaml              # Central configuration (hyperparams, paths)
├── data/
│   ├── raw/                     # Downloaded MobiFall data (gitignored)
│   └── processed/               # Windowed .npy arrays (gitignored)
├── logs/                        # Rotating log files (gitignored)
├── models/
│   ├── mobifall_edge_model.tflite # Quantized Human Fall Model
│   └── car_crash_model.tflite    # Quantized Car Crash Model
├── scripts/
│   ├── generate_car_crash_data.py # Synthetic crash data generator
│   └── train_car_crash.py        # Vehicle crash model training
├── src/
│   ├── preprocess.py            # Data loading, parsing, sliding window
│   ├── train.py                 # Model architectures + training loop
│   ├── tflite_converter.py      # H5 → TFLite quantized conversion
│   ├── inference.py             # Dual-model stateful inference engine
│   ├── app.py                   # FastAPI WebSocket server (Fusion ready)
│   └── utils/
│       ├── logger.py            # RotatingFileHandler logger
│       ├── config_loader.py     # YAML → AttrDict loader
│       └── notifier.py          # Twilio SOS & Mock Alerts
├── tests/
│   ├── test_preprocess.py       # Sliding window unit tests
│   └── test_inference.py        # Dual-engine smoke tests
├── .gitignore
├── requirements.txt
└── README.md
```

---

##  Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/DeveshNathJha/Passenger-Safety-Fall-Detection-System.git
cd Passenger-Safety-Fall-Detection-System
pip install -r requirements.txt
```

### 2. Preprocess Data

```bash
# Download MobiFall dataset and generate sliding windows:
python src/preprocess.py

# OR use local data:
python src/preprocess.py --local-data /path/to/mobifall

# Test the pipeline with dummy data (no download needed):
python src/preprocess.py --dry-run
```

### 3. Train the Model

```bash
# Train with default config (1D-CNN, 10 epochs):
python src/train.py

# Train a specific architecture:
python src/train.py --model CNN-LSTM --epochs 20

# Dry run (dummy data, no preprocessing required):
python src/train.py --dry-run
```

### 4. Convert to TFLite

```bash
python src/tflite_converter.py
# Output: models/mobifall_edge_model.tflite
```

### 5. Run Inference Simulation

```bash
# Simulate 50 Hz streaming on a dummy CSV:
python src/inference.py --generate-dummy

# Run on your own CSV file:
python src/inference.py --data path/to/sensor_data.csv

# Run as fast as possible (no rate limiting):
python src/inference.py --generate-dummy --no-realtime
```

### 6. Start the FastAPI Server

```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```

Visit **http://localhost:8000/docs** for the interactive Swagger UI.

---

##  Screenshots

*(Add screenshots from your phone here to demonstrate the working system.)*

---

##  API Reference

### `GET /health`

Returns server status and inference engine stats.

```json
{
  "status": "healthy",
  "model": "models/mobifall_edge_model.tflite",
  "window_size": 128,
  "total_inferences": 42,
  "fall_count": 3
}
```

### `WebSocket /ws/predict`

**Send** (one JSON per sample, at ~50 Hz):
```json
{"acc_x": 0.1, "acc_y": 9.8, "acc_z": 0.3, "gyro_x": 0.01, "gyro_y": 0.02, "gyro_z": 0.0}
```

**Receive – buffering** (while accumulating samples):
```json
{"event": "buffering", "buffer_fill": 45, "window_size": 128}
```

**Receive – prediction** (every 64 samples after buffer is full):
```json
{"event": "prediction", "label": "Normal Activity", "confidence": 0.9312, "latency_ms": 2.1}
```

### `POST /predict/single`

REST alternative to WebSocket for single sample push.

---

##  Model Performance

| Architecture | Accuracy | Precision (Fall) | Recall (Fall) | F1-Score (Fall) | Parameters | Training Time | Deployment |
|---|---|---|---|---|---|---|---|
| **1D-CNN**   | 96.65% | 95.0% | 99.0% | 97.0% | **8,481** | Fastest | **Selected (Edge)** |
| CNN-LSTM     | 97.04% | 95.5% | 99.1% | 97.2% | 36,353 | ~147s | High accuracy |
| LSTM         | 96.06% | 94.8% | 98.7% | 96.7% | 20,289 | ~167s | Moderate |
| Transformer  | 95.90% | 94.5% | 98.4% | 96.4% | 1,557 | ~184s | Ultra-compact |

**Why 1D-CNN for edge?** It is **4× smaller** than CNN-LSTM (8K vs 36K params), converges fastest, and achieves only 0.39% lower accuracy — a favourable tradeoff for Raspberry Pi / Android deployment.

| Format | Size | Reduction |
|---|---|---|
| Human Fall (TFLite) | **~15 KB** | 89% |
| Car Crash (TFLite) | **~15 KB** | 89% |

### Core Sensor Data
The system currently utilizes a **6-axis sensor stream** (Accelerometer + Gyroscope) at 50 Hz. Both models were trained on 6-axis data to ensure maximum robustness against device rotation and vehicle tilt.

* **Accelerometer (X, Y, Z)**: Detects impact forces and sudden braking.
* **Gyroscope (X, Y, Z)**: Detects vehicle rolling, spinning, and tumbling during a collision.
* **GPS (Lat, Lon, Speed)**: Provides context for the SOS system (detecting the stop after an impact).

---

##  Configuration

All hyperparameters live in `config/config.yaml`. No hardcoded magic numbers in source files.

```yaml
windowing:
  window_size: 128    # samples per window
  overlap: 0.5        # 50% → step = 64 samples

model:
  selected_architecture: "1D-CNN"
  epochs: 10
  batch_size: 64

api:
  port: 8000
  sensor_frequency_hz: 50
```


---

##  Testing

```bash
python -m pytest tests/ -v
```

Test coverage:
-  Sliding window correctness (shape, count, overlap, dtype)
-  Edge cases (input shorter than window, exact fit)
-  TFLite engine initialization and buffer state machine
-  Prediction output schema validation
-  Error handling (invalid sample size, missing model file)

---

##  Edge Deployment (Raspberry Pi)

1. Copy `models/mobifall_edge_model.tflite` to the device.
2. Install lightweight runtime: `pip install tflite-runtime`
3. Comment out `tensorflow` in `requirements.txt`, uncomment `tflite-runtime`.
4. Run inference: `python src/inference.py --model models/mobifall_edge_model.tflite`

The engine auto-detects `tflite_runtime` vs `tensorflow.lite` at import time.

---

##  Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| ML Framework | TensorFlow 2.15 / Keras |
| Edge Runtime | TFLite / tflite-runtime |
| API Server | FastAPI + Uvicorn |
| Data Validation | Pydantic v2 |
| Configuration | PyYAML |
| Dataset | MobiFall v2.0 (Kaggle) |
| Target Hardware | Raspberry Pi 4 / Android (MPU6050) |
| Testing | pytest |

---

##  License

MIT License – see [LICENSE](LICENSE) for details.
