"""
src/cross_dataset.py
--------------------
Cross-dataset evaluation: train on MobiFall v2.0 only, test on SisFall.

Why: a subject-wise split inside one dataset still shares the device, the body
position and the recording protocol. SisFall changes all three – a waist-worn
ADXL345 + ITG3200 instead of a phone in a trouser pocket, a different lab, and
15 elderly adults (60–75) among its 38 subjects – so it measures how far the
model generalises beyond the data it was built on.

Protocol (identical training to src/train.py):
  1. MobiFall, all 24 subjects. A stratified subset of subjects
     (stratified_group_pick, config.model.val_size) is held out as validation
     for early stopping and threshold tuning (max F1). No SisFall data is used
     for training, normalisation statistics or the threshold.
  2. Window-level evaluation on every SisFall window: overall, young adults
     (SA*), elderly (SE*), recall per fall code (F01–F15) and false positives
     per activity code (D01–D19).

Standalone usage:
    python src/cross_dataset.py --features invariant
    python src/cross_dataset.py --features raw
    python src/cross_dataset.py --features invariant --export-tflite reports/cross_dataset_invariant.tflite

Output:
    reports/cross_dataset_<features>.json

Event-level evaluation (whole SisFall trials replayed through the streaming
pipeline) can be run on the exported model with src/evaluate_events.py, if
present – see that script's --help.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import Recording, load_processed, load_recordings  # noqa: E402
from src.signal_processing import clip_to_sensor_range, find_impact_index, sliding_windows  # noqa: E402
from src.train import compute_metrics, fit_model, stratified_group_pick, tune_threshold  # noqa: E402
from src.utils.config_loader import load_config, resolve_path  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Windowing with activity codes
# ---------------------------------------------------------------------------

def windows_with_codes(recordings: list[Recording], config):
    """
    Same windowing as preprocess.windows_from_recordings (clip, impact-centred
    fall windows), but also returns the activity code of every window.

    Returns (X, y, subjects, codes).
    """
    ws = config.windowing.window_size
    step = int(ws * (1 - config.windowing.overlap))
    margin = config.data.impact_margin
    all_X, all_y, all_s, all_c = [], [], [], []
    for rec in recordings:
        data = clip_to_sensor_range(rec.data, config.signal.acc_range, config.signal.gyro_range)
        if len(data) < ws:
            continue
        impact = find_impact_index(data) if rec.label == 1 else None
        X, y = sliding_windows(data, rec.label, ws, step, impact_idx=impact, impact_margin=margin)
        if len(y) == 0:
            continue
        all_X.append(X)
        all_y.append(y)
        all_s.append(np.full(len(y), rec.subject))
        all_c.append(np.full(len(y), rec.code))
    if not all_X:
        raise RuntimeError("No SisFall windows produced.")
    return (np.concatenate(all_X), np.concatenate(all_y),
            np.concatenate(all_s), np.concatenate(all_c))


# ---------------------------------------------------------------------------
# Per-code tables
# ---------------------------------------------------------------------------

def per_code_table(codes: np.ndarray, y: np.ndarray, pred: np.ndarray) -> dict:
    """Falls: recall per code. ADLs: false-positive count and rate per code."""
    table = {}
    for code in sorted(set(codes.tolist())):
        m = codes == code
        n = int(m.sum())
        if (y[m] == 1).all():
            tp = int(pred[m].sum())
            table[code] = {"type": "fall", "n": n, "detected": tp, "recall": round(tp / max(n, 1), 4)}
        else:
            fp = int(pred[m].sum())
            table[code] = {"type": "adl", "n": n, "false_positives": fp, "fp_rate": round(fp / max(n, 1), 4)}
    return table


# ---------------------------------------------------------------------------
# Main protocol
# ---------------------------------------------------------------------------

def run_cross_dataset(
    config,
    features: str = "invariant",
    epochs: Optional[int] = None,
    train_data: Optional[tuple] = None,
    test_recordings: Optional[list[Recording]] = None,
    export_tflite: Optional[str] = None,
    write_report: bool = True,
    verbose: int = 0,
) -> dict:
    """
    Train on MobiFall, evaluate on SisFall.

    Parameters
    ----------
    features : "raw" | "invariant"
        Model input (sets config.model.input_features).
    train_data : (X, y, groups) | None
        Training windows; default: data/processed/human_windows.npz.
    test_recordings : list[Recording] | None
        Test trials; default: data/processed/sisfall_recordings.npz.
    export_tflite : str | None
        Also write the trained model as a quantized TFLite file (+ metadata JSON
        next to it) for event-level replay.
    """
    config.model.input_features = features
    if epochs:
        config.model.epochs = epochs
    name = config.model.selected_architecture
    step_s = config.windowing.window_size * (1 - config.windowing.overlap) / config.signal.sample_rate_hz

    X, y, groups = train_data if train_data is not None else load_processed(config, "human")
    rng = np.random.default_rng(config.model.random_state)
    val_groups = stratified_group_pick(groups, y, config.model.val_size, rng)
    va = np.isin(groups, val_groups)
    logger.info(f"[cross-dataset/{features}] train on MobiFall: {(~va).sum():,} windows, "
                f"validation subjects {sorted(val_groups.tolist())} ({va.sum():,} windows)")

    model, history, train_time = fit_model(name, X[~va], y[~va], X[va], y[va], config, verbose=verbose)
    val_scores = model.predict(X[va], verbose=0, batch_size=512).ravel()
    threshold = tune_threshold(y[va], val_scores)

    recs = test_recordings if test_recordings is not None else load_recordings(config, "sisfall")
    Xt, yt, subj, codes = windows_with_codes(recs, config)
    scores = model.predict(Xt, verbose=0, batch_size=512).ravel()
    pred = (scores >= threshold).astype(int)

    young = np.char.startswith(subj.astype(str), "SA")
    elderly = np.char.startswith(subj.astype(str), "SE")

    def subset(mask):
        return compute_metrics(yt[mask], scores[mask], threshold, step_s) if mask.any() else None

    result = {
        "protocol": "train MobiFall v2.0 (24 subjects) → test SisFall (38 subjects), window level",
        "architecture": name,
        "input_features": features,
        "threshold": float(threshold),
        "epochs_run": len(history.history["loss"]),
        "train_time_s": round(train_time, 1),
        "mobifall_validation_subjects": sorted(val_groups.tolist()),
        "mobifall_validation_metrics": compute_metrics(y[va], val_scores, threshold, step_s),
        "sisfall": {
            "n_subjects": int(len(np.unique(subj))),
            "overall": subset(np.ones(len(yt), dtype=bool)),
            "young_adults": subset(young),
            "elderly": subset(elderly),
        },
        "per_code": per_code_table(codes, yt, pred),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    if export_tflite:
        from src.tflite_converter import convert_to_tflite
        conv = convert_to_tflite(model, export_tflite, quantize=True)
        meta = {"task": "human", "architecture": name, "input_features": features,
                "dataset": "MobiFall v2.0 (cross-dataset model)", "threshold": float(threshold),
                "labels": {"0": "Normal Activity", "1": "FALL DETECTED"},
                "window_size": config.windowing.window_size,
                "sample_rate_hz": config.signal.sample_rate_hz, **conv}
        # Metadata next to the model (TFLiteModel reads <model>.json). If that path is
        # also the report path, the report below carries the same keys instead.
        meta_path = Path(export_tflite).with_suffix(".json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        result.update({"exported_tflite": str(export_tflite), "labels": meta["labels"],
                       "window_size": meta["window_size"], "sample_rate_hz": meta["sample_rate_hz"]})

    o = result["sisfall"]["overall"]
    logger.info(f"[cross-dataset/{features}] SisFall overall: acc={o['accuracy']} P={o['precision']} "
                f"R={o['recall']} F1={o['f1']} PR-AUC={o['pr_auc']} (thr={threshold:.3f})")
    if result["sisfall"]["elderly"]:
        e = result["sisfall"]["elderly"]
        logger.info(f"[cross-dataset/{features}] SisFall elderly: specificity={e['specificity']} "
                    f"FP={e['confusion_matrix']['fp']} / {e['n'] - e['n_events']} normal windows")

    if write_report:
        reports = resolve_path(config.paths.reports_dir)
        reports.mkdir(parents=True, exist_ok=True)
        path = reports / f"cross_dataset_{features}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Report → {path}")
    return result


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="Train on MobiFall, evaluate on SisFall.")
    parser.add_argument("--features", choices=["raw", "invariant"], default="invariant")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--export-tflite", type=str, default=None, metavar="PATH",
                        help="Also save the trained model as TFLite (+ .json metadata).")
    parser.add_argument("--config", type=str, default=None)
    return parser.parse_args()


def main():
    args = _parse_args()
    config = load_config(args.config)
    run_cross_dataset(config, features=args.features, epochs=args.epochs,
                      export_tflite=args.export_tflite, verbose=1)


if __name__ == "__main__":
    main()
