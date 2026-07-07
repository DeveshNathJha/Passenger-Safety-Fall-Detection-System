# Car Crash Training (`scripts/train_car_crash.py`)

## Purpose
An isolated training script explicitly designed to process the `synthetic_car_crash` dataset into a functional 1D-CNN binary classifier without interfering with the existing `src/train.py` human-fall routines.

## Mechanics
* **Data Ingestion**: Iterates through collision and normal driving CSVs, segmenting them into `(128, 6)` arrays with a 50% overlap.
* **Model Overfit Protection**: Compiles the Keras Sequential model utilizing binary cross-entropy, tracking validation loss carefully through EarlyStopping and ModelCheckpoint. 
* **TFLite Conversion**: Immediately converts the most accurate epoch into a compressed `models/car_crash_model.tflite` compatible with the exact I/O shape of `models/mobifall_edge_model.tflite` for instantaneous hot-swapping in `src/inference.py`.
