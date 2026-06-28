import os
import glob
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Import the existing model architecture and converter
from src.train import build_cnn_model
from src.tflite_converter import convert_to_tflite

def create_windows_from_csv(csv_path, label_int, window_size=128, step=64):
    df = pd.read_csv(csv_path)
    data = df[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']].values
    
    windows = []
    labels = []
    for i in range(0, len(data) - window_size + 1, step):
        window = data[i:i + window_size]
        windows.append(window)
        labels.append(label_int)
        
    return windows, labels

def load_synthetic_data(data_dir="data/raw/synthetic_car_crash", window_size=128, step=64):
    all_X = []
    all_y = []
    
    # Load Normal (class 0)
    normal_files = glob.glob(os.path.join(data_dir, "NORMAL", "*.csv"))
    for f in normal_files:
        X, y = create_windows_from_csv(f, 0, window_size, step)
        all_X.extend(X)
        all_y.extend(y)
        
    # Load Crash (class 1)
    crash_files = glob.glob(os.path.join(data_dir, "CRASH", "*.csv"))
    for f in crash_files:
        X, y = create_windows_from_csv(f, 1, window_size, step)
        all_X.extend(X)
        all_y.extend(y)
        
    return np.array(all_X, dtype=np.float32), np.array(all_y)

if __name__ == "__main__":
    print("Loading synthetic car crash data...")
    X, y = load_synthetic_data()
    print(f"Total windows: {X.shape}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Building 1D-CNN Model...")
    model = build_cnn_model(input_shape=(128, 6))
    
    # Train
    h5_path = "models/car_crash_model.h5"
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint(h5_path, save_best_only=True)
    ]
    
    print("Training Car Crash Model...")
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=15,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Validation Accuracy: {acc:.4f}")
    
    # Convert to TFLite
    print("Converting to TFLite...")
    tflite_path = "models/car_crash_model.tflite"
    tflite_metrics = convert_to_tflite(h5_path, tflite_path)
    print(f"Successfully saved TFLite model -> {tflite_path} ({tflite_metrics['tflite_size_kb']:.1f} KB)")
