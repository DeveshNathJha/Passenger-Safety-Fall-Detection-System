# Design Notes

This document explains *why* the system is built the way it is, and what
changed between the first prototype (v1) and the current version (v2).

## 1. Data pipeline

### MobiFall v2.0 is not a 6-column table
MobiFall stores the accelerometer and gyroscope in **separate files**, each with
its own nanosecond timestamps and its own irregular rate (≈ 80–90 Hz acc,
≈ 200 Hz gyro, different start times).

| | v1 (notebook) | v2 (`src/preprocess.py`) |
|---|---|---|
| Acc/gyro alignment | rows paired **by index** (`acc[:n]`, `gyro[:n]`) → the gyro channel covered a different, ~2.5× shorter time span than the acc channel | both streams interpolated onto the **same 50 Hz time grid** over their common span |
| Sampling rate | "128 samples" meant ~1.5 s of acc but ~0.6 s of gyro; the phone streams at 50 Hz → train/serve mismatch | every window is exactly 2.56 s at 50 Hz in training **and** serving |
| Gyro units | rad/s in training, °/s from the browser (57× mismatch) | converted to °/s at load time |
| Labels | every window of a fall trial labelled "fall" – including the standing-before and lying-after parts | only **impact-centred** windows are "fall"; ambiguous pre/post-fall windows are dropped |
| `src/preprocess.py` | expected 6 values per line, so it could not parse real MobiFall files at all; results came from a Colab notebook | the committed pipeline is the one that produced the committed models |

### Synthetic vehicle-crash data
There is no public, labelled smartphone crash dataset, so the car model is
trained on physics-informed synthetic data (`scripts/generate_car_crash_data.py`).
v2 makes it far less trivial than v1:

* the crash pulse is clipped to a **phone's ±16 g range** (v1 used 20–50 g spikes
  that a phone can never report, *and* the demo page clamped input to ±2 g, so the
  deployed crash model could never fire);
* **hard negatives**: potholes, emergency braking, the passenger handling the
  phone, and the phone being dropped (40 % in a parked car, so it ends up still –
  just like after a crash);
* 30 % low-speed (5–10 g) collisions that overlap with hard drops;
* random phone orientation per recording;
* windows are labelled relative to the impact, and train/val/test are split
  **by sequence**.

Even so, a model evaluated on data from the same generator learns that
generator; the car metrics validate the pipeline, **not** real-world crash
performance. That needs real crash-test or naturalistic driving recordings.

## 2. Evaluation protocol

* **Subject-wise split.** Windows overlap by 50 %. A random window split (v1)
  puts near-duplicate windows of the same recording into both train and test.
  v2 splits by subject (human) / sequence (car); `tests/test_models.py` asserts
  that no group is shared.
* **Stratified by subject type.** In MobiFall v2.0 only 9 of 24 subjects
  recorded normal activities; 15 recorded falls only. A plain random pick of
  validation subjects can therefore contain (almost) no normal windows, and the
  threshold tuned on it is far too low. This actually happened in the first
  LOSO run: one fold produced 332 false positives and pooled precision was
  0.51. With stratified picking (`stratified_group_pick`) pooled precision is
  0.785 and that fold has zero false positives.
* **Three splits.** v1 used the test set for early stopping and checkpoint
  selection. v2 uses a validation split for early stopping, checkpointing and
  threshold tuning; the test split is used once.
* **Leave-one-subject-out CV** (`--loso`) over all 24 subjects gives a variance
  estimate instead of one lucky split.
* **Metrics that matter for an imbalanced safety problem:** precision, recall,
  F1 and PR-AUC on the fall class, plus false alarms per hour of normal activity.
  Accuracy alone is misleading when 93 % of windows are "normal".
* **Orientation robustness:** the test set is re-evaluated with every window
  randomly rotated, simulating a phone held differently from MobiFall's
  trouser-pocket placement.

## 3. Model

* 1D-CNN (≈ 8.5 k parameters) chosen for the edge; LSTM, CNN-LSTM and a small
  Transformer are benchmarked on the same split (`--compare`).
* **Normalisation is a layer inside the model**, adapted on the training split,
  so the `.tflite` file consumes raw m/s² and °/s and no preprocessing
  constants can drift between training and serving.
* **Rotation-invariant inputs (v2.1, deployed).** The first layer
  (`GravityAlignedFeatures`) estimates the gravity direction u as the window's
  mean acceleration and emits |a|, a·u, |a − (a·u)u| and the same three for the
  gyroscope. Rotating the phone rotates a, ω and u together, so these values
  cannot change: the model is invariant by construction. The layer is exported
  into the TFLite file with built-in ops, so the server still sends raw values.
  v2 used **rotation augmentation** instead (a random rotation applied to acc
  and gyro together); that replaced v1's client-side "auto-alignment", which
  swapped accelerometer but not gyroscope axes. Augmentation only *encourages*
  invariance: rotated-phone F1 was 0.62 for the MobiFall-only model and 0.92 for
  the combined one, versus an unchanged 0.974 with invariant inputs.
* **Class weighting** for the ~1:14 imbalance.
* **Threshold tuned on validation** (max F1) and shipped in the model's
  metadata JSON, so the server and the model can never disagree on it.
* **TFLite dynamic-range quantization**, then a parity check: Keras and TFLite
  decisions are compared on 512 test windows and the result is stored in the
  metadata.
* **Fixed batch size 1 for conversion.** With a dynamic batch dimension, Keras
  LSTM layers lower to TensorList ops that need the Flex delegate, which
  LiteRT / tflite-runtime don't ship. Converting with batch 1 (the server
  scores one window at a time) maps them to the built-in LSTM kernel, so all
  four architectures run on the lightweight runtime.

## 4. Serving

```
phone ──WS batch (100 ms, timestamped)──► StreamSession (per connection)
                                            ├─ StreamResampler → 50 Hz
                                            ├─ clip to ±16 g / ±2000 °/s
                                            ├─ ring buffer (128) → TFLiteModel (shared, locked)
                                            ├─ stationary guardrail, k-of-n confirmation
                                            └─ AlertManager (per connection)
                                                  ├─ countdown, user cancel
                                                  ├─ cooldown, car speed gating
                                                  └─ dispatch → notifier (worker thread)
```

| Problem in v1 | Fix in v2 |
|---|---|
| One global engine: every phone wrote into the **same buffer**, and connecting with `?model=car` switched every other user to the car model | `ModelRegistry` (shared, read-only, thread-locked) + one `StreamSession` and one `AlertManager` per connection |
| SOS sent on **every** window above 0.9 → an SMS every 1.3 s | `AlertManager` state machine: one pending alert, user-cancellable countdown, cooldown |
| Fall detection never alerted anyone | both modes alert |
| No GPS speed was sent as `0` km/h, which satisfied the "vehicle stopped" rule | `null` when unknown; crash requires the vehicle to have been **moving recently** (phone dropped in a parked car is suppressed) |
| Phone disconnects mid-incident | a pending alert is still dispatched (the phone may be destroyed) |
| Twilio call blocked the event loop | `asyncio.to_thread` |
| Browser sent the latest sample on a 20 ms timer (duplicates / drops) | browser sends every native event with its timestamp; the server resamples |
| No auth | optional API key (`FDS_API_KEY`), constant-time comparison |
| Absolute server paths returned to clients | file names only |

## 5. v2.1: second dataset, event-level evaluation, post-fall check

### Choosing the second dataset
SisFall (Sucerquia et al., 2017): free, 38 people incl. 15 aged 60–75, a different
device (waist-worn ADXL345 + ITG3200, 200 Hz) and a different body position.
MobiAct would have matched the phone-in-pocket setup more closely, but it requires a
signed usage agreement.
Raw values are converted with the dataset's own formula (value = 2·range / 2^bits · raw)
and downsampled 200 → 50 Hz by averaging blocks of 4 samples. That average is a box
anti-aliasing filter, so impact spikes aren't aliased the way plain sample-dropping would alias them.

### Cross-dataset test: a domain-shift lesson
A model trained only on MobiFall was tested on all of SisFall. Raw axes generalised better
(F1 0.839 vs 0.568). The invariant model flagged vigorous waist movements, such as fast
jogging (67 % of windows), as falls. The likely reason: with the orientation cue
removed, the model leans on impact magnitude, and jogging impacts at the waist look
different from those at the thigh. Pooling both datasets fixed it
(SisFall F1 0.98 in 10-fold group CV). With both placements in training, the remaining
raw-vs-invariant gap (0.977 vs 0.971 CV F1) is smaller than the raw model's loss on
rotated phones (0.986 → 0.921), so invariant inputs are deployed.

### Per-dataset CV
`train.py --cv 10` pools the confusion matrix over folds and also reports it per
dataset (MobiFall / SisFall young / SisFall elderly). Splits stay stratified by subject
type; SisFall elderly subjects (ADLs only, apart from one) form their own stratum.

### Event-level evaluation
Window metrics don't say how many alarms a user gets. `src/evaluate_events.py` replays
whole recordings sample by sample through the real `StreamSession`. A fall counts as
detected if an event's impact is within ±2.56 s of the true impact, and events less
than one window apart count as one incident. `--loso` repeats this with a fresh fold
model for each held-out subject, exported to quantized TFLite like the deployed one.

### Post-fall stillness check
A positive window becomes a candidate; the impact is the peak |acc| in that window.
The check looks at [impact + 1.0 s, impact + 2.5 s]: if std(|acc|) ≤ 3.0 m/s² and
mean |ω| ≤ 60 °/s, the event is confirmed, otherwise rejected. This adds ~2.5 s
before the 15 s countdown starts. Parameters were grid-searched on the validation
subjects only (objective: fewest false alarms with recall within 2 points of
check-off). The stricter defaults (1.5 m/s², 30 °/s) missed backward falls onto a
chair, where people keep moving a little.

Measured effect: under domain shift (MobiFall-only model on SisFall) false alarms
dropped 1,616 → 486 (invariant) and 178 → 33 (raw) for 0.5–1.4 points of recall.
With the deployed combined model it changes little (15 vs 16 false alarms on SisFall
test subjects), because the model itself no longer fires on jogging. It is kept as a
safety net. False alarms that remain come from activities that end still
(MobiFall CSI car step-in: 11 of 12 remaining in event-level LOSO; SisFall rolling over
in bed, collapsing into a chair).

## 6. Known limitations

* Activities that **end still** — sitting down in a car, lying down quickly, rolling over,
  collapsing into a chair — are the main remaining false alarms; a stillness check cannot
  separate them. Next: posture-change features or a sequence model over the minute after
  a candidate.
* Both datasets contain **simulated** falls on mats; only one elderly participant fell.
  Real falls of older adults (e.g. FARSEEING) are the most important missing validation.
* Training placements are trouser pocket and waist; other placements rely on the
  invariance guarantee only.
* The gravity estimate is the window mean; during a fall the window mixes upright and
  lying postures, so "vertical" is approximate (still invariant, but less informative).
* The vehicle model is trained only on synthetic data.
* The post-fall check plus the countdown means an SOS is sent ~17.5 s after the event at the earliest.
