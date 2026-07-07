# Notes: `src/train.py`

## Purpose
Defines all 4 model architectures and the training loop. Extracted from Notebook Cells 2–4.

## Model Registry
```python
MODEL_REGISTRY = {
    "1D-CNN": build_cnn_model,
    "LSTM": build_lstm_model,
    "CNN-LSTM": build_cnn_lstm_model,
    "Transformer": build_transformer_model,
}
```
To add a new architecture, just add a builder function and register it here.

## Model Architectures

### 1D-CNN (Selected for Edge Deployment)
```
Conv1D(64, 3, relu) → MaxPool1D(2) → Dropout(0.3)
→ Conv1D(32, 3, relu) → GlobalMaxPool1D
→ Dense(32, relu) → Dense(1, sigmoid)
```
- **8,481 parameters** | Fastest inference | ~15 KB TFLite
- Best for Raspberry Pi, Android → **use this for deployment**

### LSTM
```
LSTM(64) → Dropout(0.4) → Dense(32, relu) → Dense(1, sigmoid)
```
- **20,289 parameters** | Good for sequential patterns

### CNN-LSTM (Hybrid)
```
Conv1D(64) → MaxPool1D(2) → Dropout(0.3)
→ LSTM(64) → Dropout(0.3) → Dense(32) → Dense(1)
```
- **36,353 parameters** | Highest accuracy (97.04%)

### Transformer
```
MultiHeadAttention(2 heads, key_dim=32) → GlobalAvgPool1D
→ Dense(32) → Dense(1)
```
- **1,557 parameters** | Tiny but lower accuracy (95.90%)

## Function: `train(X, y, config, model_name=None) → keras.Model`

### Training Loop
1. **Splits** data 80/20 stratified.
2. **Builds** the model via `MODEL_REGISTRY[model_name]`.
3. **Callbacks:**
   - `EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)` — stops early if validation loss stops improving
   - `ModelCheckpoint(monitor="val_accuracy", save_best_only=True)` — saves the best weights to `models/mobifall_model.h5`
4. **Evaluates** on test set and logs classification report.

### Why EarlyStopping?
The notebook used a fixed 10 epochs. EarlyStopping automatically finds the best checkpoint without overfitting, often finishing in 6–8 epochs.

## CLI Usage
```bash
python src/train.py                       # Train 1D-CNN (default)
python src/train.py --model CNN-LSTM      # Train CNN-LSTM
python src/train.py --epochs 20           # Override epochs
python src/train.py --dry-run             # Dummy data, no dataset needed
```

## Output
- `models/mobifall_model.h5` — best model (saved by ModelCheckpoint)
- Console + `logs/app.log` — training progress + final metrics
