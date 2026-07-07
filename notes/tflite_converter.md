# Notes: `src/tflite_converter.py`

## Purpose
Converts the trained Keras `.h5` model to a quantized `.tflite` model for edge deployment. Extracted from Notebook Cell 5.

## Function: `convert_to_tflite(h5_path, tflite_path, quantize=True) → dict`

### What is Quantization?
**Dynamic Range Quantization (DEFAULT):** TensorFlow reduces weight precision from `float32` → `int8` during conversion. This:
- Reduces model size by ~4× (138 KB → ~15 KB)
- Speeds up inference on CPU/ARM hardware
- Negligible accuracy loss (<0.1%)

```
converter.optimizations = [tf.lite.Optimize.DEFAULT]
```

### Outputs
```python
{
    "original_size_kb": 134.94,
    "tflite_size_kb": 15.06,
    "reduction_pct": 88.8,
}
```

## CLI Usage
```bash
# Convert with default paths from config:
python src/tflite_converter.py

# Custom paths:
python src/tflite_converter.py \
    --input models/mobifall_model.h5 \
    --output models/mobifall_edge_model.tflite

# Convert without quantization (keep float32):
python src/tflite_converter.py --no-quantize
```

## Edge Device Notes
On **Raspberry Pi / Android**, replace `tensorflow` with `tflite-runtime`:
```bash
pip install tflite-runtime
```
The `inference.py` module auto-detects which runtime is available.

## Size Results (from this project)
| Format | Size |
|---|---|
| `mobifall_model.h5` | ~138 KB |
| `mobifall_edge_model.tflite` | ~15 KB |
| **Reduction** | **~89%** |
