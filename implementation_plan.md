# Implementation Plan — Final (After Deep Code Analysis)

## What I Found (Real Root Causes)

After reading every file and running data analysis on the actual `.npy` files and model, I found the definitive causes.

---

## ROOT CAUSE #1 — Model Is Trained on DUMMY RANDOM DATA (Most Critical)

The file `data/processed/X_windows.npy` — used to train the saved model — is **random uniform [0,1] data** from `preprocess.py --dry-run`, NOT the actual MobiFall dataset.

**Proof from inspecting the file:**
```
X_windows.npy values:
[[0.77  0.44  0.86  0.70  0.09  0.98]  <- all between 0 and 1
 [0.76  0.79  0.13  0.45  0.37  0.93]  <- no physical meaning]

Subjects: sub01..sub10 (only 10 mock subjects, 200 total windows)
```

**Real IMU data looks like (from test_dummy.csv which uses physical units):**
```
Normal: acc_y mean=9.788 (gravity), std=0.203
Fall:   acc_y mean=-0.275, std=10.6 (chaotic)
```

A model trained on random noise cannot learn any fall pattern. When it sees your phone's real sensor data (gravity at ~9.8 m/s²), the data is completely out-of-distribution — predictions are essentially random.

> [!CAUTION]
> The MobiFall dataset was never used for training. `data/processed/` only has dry-run random data. `mobifall_model.h5` and `mobifall_edge_model.tflite` were trained on random noise — this is the primary reason predictions fail.

---

## ROOT CAUSE #2 — Static Variance Guardrail Threshold is Wrong

In `inference.py` lines 246–249, the guardrail forces "Normal" when `std(acc_mag) < 0.5`.

**But from real data analysis:**
```
NORMAL walking:  acc_mag std = 0.2027  <- BELOW 0.5 -> forced Normal (overrides model)
FALL events:     acc_mag std = 6.35    <- ABOVE 0.5 -> model output used (which is junk)
```

So currently: normal walking is always overridden to Normal by the guardrail (accidentally correct), but FALL events bypass the guardrail and go to the untrained model — which outputs garbage. This is why FALL shows as not-fall.

**After retraining on real data**, this guardrail needs to be set to `0.05` (only fire for a truly stationary/resting phone) so it doesn't interfere with normal activity classification.

---

## ROOT CAUSE #3 — Phone Gravity Axis Mismatch

MobiFall was collected with **phone in trouser pocket** — gravity on **Y-axis** (`acc_y ~9.8`).

Phone `DeviceMotionEvent` returns `accelerationIncludingGravity`. The axis where gravity appears depends on how you hold the phone:
- Portrait upright → gravity on Y ✓ (matches training)
- Landscape / flat → gravity on X or Z ✗ (mismatch)

**This matters after retraining** — once model is trained correctly, wrong axis = wrong predictions.

---

## Summary Table

| # | Root Cause | Effect | Fix |
|---|---|---|---|
| 1 | Model trained on dummy random data | Predictions meaningless | Retrain on real MobiFall |
| 2 | Guardrail threshold 0.5 vs walking std 0.2 | Falls not suppressed but also not predicted right | Lower threshold to 0.05 |
| 3 | Phone gravity axis mismatch | After fix #1, still wrong axis | JS axis auto-alignment + calibration |

---

## Proposed Changes

---

### STEP 1 — Fix Config (Quick Fix, Do First)

#### [MODIFY] [config.yaml](file:///d:/Passenger-Safety-Fall-Detection-System/config/config.yaml)

```yaml
# Change:
static_variance_threshold: 0.5
# To:
static_variance_threshold: 0.05  # Real walking std ~0.2, stationary ~0.02
```

---

### STEP 2 — Fix Inference Engine

#### [MODIFY] [inference.py](file:///d:/Passenger-Safety-Fall-Detection-System/src/inference.py)

1. Change default `static_variance_threshold` from `0.5` to `0.05`
2. Add `acc_mag_std` and `guardrail_fired` fields to the result dict for debugging
3. Add detailed debug logging: log `raw_score`, `acc_mag_std`, and whether guardrail fired on every window
4. Add a `detect_gravity_axis()` helper method (prep for calibration feature)

---

### STEP 3 — Fix Phone Demo UI

#### [MODIFY] [demo_phone.html](file:///d:/Passenger-Safety-Fall-Detection-System/src/demo_phone.html)

1. **Add `raw_score` display** — show the raw model output (0.0–1.0) so you can debug predictions in real time
2. **Add orientation warning** — detect if `|acc.y|` is NOT the largest axis and show: "📱 Hold phone upright for best accuracy"
3. **Add auto gravity axis alignment in JS** — before sending payload, detect which axis has largest absolute value (~gravity) and swap axes so `acc_y` always has the gravity component
4. **Add "Calibrate" button** — samples 3 seconds of resting data, sends to server to detect gravity axis
5. **Fix gyro axis mapping** — verify `rotationRate.alpha/beta/gamma` maps to `gyro_x/y/z` correctly per MobiFall convention

---

### STEP 4 — Add Gravity Alignment Utility

#### [NEW] [src/utils/gravity_align.py](file:///d:/Passenger-Safety-Fall-Detection-System/src/utils/gravity_align.py)

```python
def detect_gravity_axis(baseline_window: np.ndarray) -> int:
    """Detect which axis (0=X,1=Y,2=Z) has the ~9.8 gravity component."""
    mean_abs = np.abs(np.mean(baseline_window[:, 0:3], axis=0))
    return int(np.argmax(mean_abs))

def reorient_to_y_gravity(window: np.ndarray, gravity_axis: int) -> np.ndarray:
    """Rotate axes so gravity is always on Y (index 1), matching MobiFall."""
    ...
```

---

### STEP 5 — Fix WebSocket Server

#### [MODIFY] [app.py](file:///d:/Passenger-Safety-Fall-Detection-System/src/app.py)

1. Add `raw_score`, `acc_mag_std`, `guardrail_fired` to the WebSocket prediction payload
2. Add per-client gravity axis calibration state
3. Add server-side axis reorientation using `gravity_align.py`

---

### STEP 6 — Retrain Model on Real MobiFall Data (Core Fix)

> [!IMPORTANT]
> This is the most critical step. The Kaggle API key must be configured. Run:
> ```bash
> python src/preprocess.py        # downloads MobiFall, creates X_windows.npy with real data
> python src/train.py             # trains on real data
> python src/tflite_converter.py  # converts to TFLite
> ```
> The `preprocess.py` code is already correct — it just needs to be run without `--dry-run`.

---

### STEP 7 — Add 5-fold Cross-Validation to train.py

#### [MODIFY] [train.py](file:///d:/Passenger-Safety-Fall-Detection-System/src/train.py)

Add `run_kfold_cross_validation()` function that:
- Runs 5-fold stratified CV
- Reports Mean +/- Std accuracy
- Required by journal reviewer for statistical significance

---

### STEP 8 — Journal LaTeX Improvements

#### [MODIFY] [Journal/fall_detection_journal.tex](file:///d:/Passenger-Safety-Fall-Detection-System/Journal/fall_detection_journal.tex)

Address all review points:

1. **Add "Research Novelty" bullets** at end of Introduction
2. **Add crash model performance table** in Section VI (Accuracy, Precision, Recall, F1)
3. **Add "Validation of Synthetic Crash Dataset" subsection** in Section III
4. **Add 5-fold CV results** (Mean +/- Std) to Table I
5. **Expand Transformer underperformance** discussion in Section VI
6. **Fix author block** — change "Member 1,2,3,4" to "Student Member, IEEE" style
7. **Shorten title** as suggested by reviewer
8. **Add GitHub URL** in conclusion

#### [MODIFY] [Journal/Bibliography.bib](file:///d:/Passenger-Safety-Fall-Detection-System/Journal/Bibliography.bib)

Add 6 new references (2023-2025):
- TinyML edge deployment paper (2023)
- Smartphone crash detection with ML (2024)
- Transformer for HAR with limited data (2023)
- Edge AI for safety-critical systems survey (2024)
- NHTSA / real crash dataset reference
- Recent fall detection benchmark (2024/2025)

---

## Verification Plan

### After all code fixes:
```bash
# 1. Download real data and preprocess
python src/preprocess.py   # needs Kaggle API key

# 2. Retrain on real data
python src/train.py --model 1D-CNN

# 3. Convert to TFLite
python src/tflite_converter.py

# 4. Run server
uvicorn src.app:app --host 0.0.0.0 --port 8000

# 5. Open phone demo at http://YOUR_IP:8000/demo
#    - Hold phone upright (portrait)
#    - Walk around -> should show Normal Activity
#    - Shake phone hard + drop movement -> should show FALL DETECTED
```

### What to watch on phone UI:
- `raw_score` value — should be ~0.1 for normal, ~0.8+ for falls
- Orientation warning — should disappear when holding phone upright
- `acc_mag_std` — should show ~0.2 for walking, >1.0 for falls

---

## Priority Order

| Priority | Task | Files Changed |
|---|---|---|
| P0 (do first) | Lower guardrail threshold 0.5 -> 0.05 | `config.yaml` |
| P0 | Add raw_score + debug fields to payload | `inference.py` |
| P0 | Add raw_score + orientation warning to phone UI | `demo_phone.html` |
| P0 | Auto gravity axis alignment in JS | `demo_phone.html` |
| P1 | Retrain on real MobiFall data | Run pipeline |
| P1 | Add gravity_align.py utility | New file |
| P1 | Server-side axis calibration | `app.py` |
| P2 | 5-fold CV in train.py | `train.py` |
| P2 | Journal LaTeX + Bibliography updates | `.tex`, `.bib` |
