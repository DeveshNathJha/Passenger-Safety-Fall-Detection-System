"""
src/evaluate_events.py
----------------------
EVENT-level evaluation: replay whole recordings through the real streaming
pipeline (StreamSession: windows, guardrail, k-of-n, post-fall check) and
count what a user would experience.

Window-level metrics (src/train.py) answer "is this 2.56 s window a fall?".
This script answers the questions that matter in deployment:

  - fall recall   : fraction of fall recordings with >= 1 event whose impact
                    lies within ±impact_tol_s of the recording's true impact
  - false alarms  : events in normal-activity (ADL) recordings – total,
                    per hour of ADL data, and per activity code. Events less
                    than one window (2.56 s) apart count as one incident.

It compares the post-fall stillness check OFF vs ON, and can grid-search the
check's parameters. Tuning objective (documented choice): minimise false
alarms subject to fall recall >= recall_without_check - 0.02, i.e. the check
may cost at most 2 points of recall. Ties: higher recall, then the shortest
decision delay (settle_s + still_s).

Usage:
    python src/evaluate_events.py --task human --subjects test
    python src/evaluate_events.py --task human --subjects val --grid --json reports/post_fall_grid.json
    python src/evaluate_events.py --task sisfall --model models/human_fall.tflite --prefix SE
"""

import argparse
import itertools
import json
import os
import sys
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import PostFallCheck, StreamSession, TFLiteModel, post_fall_check_from_config
from src.preprocess import Recording, load_recordings
from src.signal_processing import clip_to_sensor_range, find_impact_index
from src.utils.config_loader import load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_GRID = {
    # people often keep moving 1–2.5 s after the impact (rolling, trying to get
    # up), so settle times up to 3 s are included; MobiFall fall trials have
    # ~7 s of data after the impact, so settle + still must stay below that
    "settle_s": [1.0, 2.0, 3.0],
    "still_s": [1.0, 1.5, 2.0],
    "still_acc_std": [1.0, 1.5, 2.0, 3.0],
    "still_gyro_dps": [20.0, 30.0, 45.0, 60.0],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class CachedModel:
    """
    Wraps a model and memoises window scores, so replaying the same recordings
    with different post-fall-check parameters does not re-run inference.
    """

    def __init__(self, model):
        self.model = model
        self.name, self.threshold = model.name, model.threshold
        self.labels, self.window_size = model.labels, model.window_size
        self._cache: dict[bytes, tuple[float, float]] = {}

    def score(self, window: np.ndarray):
        key = window.tobytes()
        if key not in self._cache:
            self._cache[key] = self.model.score(window)
        return self._cache[key]


def session_factory(model, config, check: Optional[PostFallCheck]) -> Callable[[], StreamSession]:
    step = int(config.windowing.window_size * (1 - config.windowing.overlap))

    def make() -> StreamSession:
        return StreamSession(model, step=step, sample_rate_hz=config.signal.sample_rate_hz,
                             confirm_k=config.inference.confirm_k, confirm_n=config.inference.confirm_n,
                             static_variance_threshold=config.inference.static_variance_threshold,
                             post_fall_check=check)
    return make


def filter_recordings(recordings: Iterable[Recording], subjects: Optional[Iterable[str]] = None,
                      prefix: Optional[str] = None) -> list[Recording]:
    """Keep recordings of the given subjects and/or whose subject id starts with prefix
    (e.g. SisFall 'SA' = young adults, 'SE' = elderly)."""
    subjects = set(subjects) if subjects else None
    return [r for r in recordings
            if (subjects is None or r.subject in subjects) and (prefix is None or r.subject.startswith(prefix))]


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

def replay(make_session: Callable[[], StreamSession], recordings: list[Recording], config) -> list[dict]:
    """
    Stream every recording sample by sample (50 Hz, no timestamps) through a
    fresh session. Returns one dict per recording with the detected events
    (each with the absolute impact sample) and the true impact for falls.
    """
    acc_range, gyro_range = config.signal.acc_range, config.signal.gyro_range
    out = []
    for rec in recordings:
        data = clip_to_sensor_range(rec.data, acc_range, gyro_range)
        session = make_session()
        events, candidates = [], 0
        for i, sample in enumerate(data):
            for res in session.push(sample):
                if res.get("candidate"):
                    candidates += 1
                if not res["is_event"]:
                    continue
                if res.get("kind") == "confirmation":
                    impact = res["impact_sample"]
                else:  # check disabled: impact = peak |acc| of the event window
                    w = data[i + 1 - session.window_size:i + 1]
                    impact = i + 1 - session.window_size + int(np.argmax(np.linalg.norm(w[:, :3], axis=1)))
                # one incident → one alarm (the app's AlertManager also merges them):
                # an event within one window length of the previous one is the same incident
                if events and impact - events[-1] < session.window_size:
                    continue
                events.append(int(impact))
        out.append({
            "subject": rec.subject, "code": rec.code, "trial": rec.trial, "label": rec.label,
            "n_samples": len(data),
            "true_impact": find_impact_index(data) if rec.label == 1 else None,
            "events": events, "candidates": candidates,
        })
    return out


def summarize(results: list[dict], sample_rate_hz: float, impact_tol_s: float = 2.56) -> dict:
    tol = impact_tol_s * sample_rate_hz
    falls = [r for r in results if r["label"] == 1]
    adls = [r for r in results if r["label"] == 0]
    detected = sum(any(abs(e - r["true_impact"]) <= tol for e in r["events"]) for r in falls)
    off_target = sum(sum(abs(e - r["true_impact"]) > tol for e in r["events"]) for r in falls)
    adl_hours = sum(r["n_samples"] for r in adls) / sample_rate_hz / 3600
    fa = sum(len(r["events"]) for r in adls)

    per_code: dict[str, dict] = {}
    for r in adls:
        d = per_code.setdefault(r["code"], {"recordings": 0, "false_alarms": 0, "minutes": 0.0})
        d["recordings"] += 1
        d["false_alarms"] += len(r["events"])
        d["minutes"] += r["n_samples"] / sample_rate_hz / 60
    for d in per_code.values():
        d["minutes"] = round(d["minutes"], 1)

    missed = Counter(r["code"] for r in falls
                     if not any(abs(e - r["true_impact"]) <= tol for e in r["events"]))
    return {
        "fall_recordings": len(falls),
        "falls_detected": int(detected),
        "fall_recall": round(detected / len(falls), 4) if falls else None,
        "missed_falls_by_code": dict(sorted(missed.items())),
        "off_target_events_in_fall_recordings": int(off_target),
        "adl_recordings": len(adls),
        "adl_hours": round(adl_hours, 3),
        "false_alarms": int(fa),
        "false_alarms_per_hour": round(fa / adl_hours, 2) if adl_hours else None,
        "adl_recordings_with_false_alarm": int(sum(bool(r["events"]) for r in adls)),
        "false_alarms_by_code": dict(sorted(per_code.items())),
    }


def evaluate(model, recordings: list[Recording], config, check: Optional[PostFallCheck]) -> dict:
    return summarize(replay(session_factory(model, config, check), recordings, config),
                     config.signal.sample_rate_hz)


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def grid_search(model, recordings: list[Recording], config, grid: Optional[dict] = None,
                max_recall_drop: float = 0.02) -> dict:
    """
    Try every combination in `grid`. Best = fewest false alarms among settings
    whose fall recall >= recall_off - max_recall_drop (ties: higher recall,
    then shorter settle_s + still_s).
    """
    grid = grid or DEFAULT_GRID
    model = model if isinstance(model, CachedModel) else CachedModel(model)
    baseline = evaluate(model, recordings, config, None)
    floor = (baseline["fall_recall"] or 0) - max_recall_drop
    rows = []
    for values in itertools.product(*grid.values()):
        params = dict(zip(grid.keys(), values))
        m = evaluate(model, recordings, config, PostFallCheck(**params))
        rows.append({**params, "fall_recall": m["fall_recall"], "false_alarms": m["false_alarms"],
                     "false_alarms_per_hour": m["false_alarms_per_hour"]})
    feasible = [r for r in rows if r["fall_recall"] is not None and r["fall_recall"] >= floor - 1e-9]
    best = min(feasible, key=lambda r: (r["false_alarms"], -r["fall_recall"], r["settle_s"] + r["still_s"]),
               default=None)
    return {"objective": f"min false alarms s.t. recall >= recall_off - {max_recall_drop}",
            "baseline_off": baseline, "recall_floor": round(floor, 4), "best": best, "rows": rows}


# ---------------------------------------------------------------------------
# Event-level leave-one-subject-out
# ---------------------------------------------------------------------------

def loso_events(config, checks: dict[str, Optional[PostFallCheck]], task: str = "human") -> dict:
    """
    For every subject: train the deployed architecture WITHOUT that subject
    (same pipeline as `train.py --loso`: stratified validation subjects for
    early stopping + threshold), export it to quantized TFLite, and replay the
    held-out subject's whole recordings once per entry in `checks`.
    Per-recording results are pooled over all folds and summarised.
    """
    import tempfile

    from src.preprocess import load_processed
    from src.tflite_converter import convert_to_tflite
    from src.train import fit_model, stratified_group_pick, tune_threshold

    X, y, groups = load_processed(config, task)
    recordings = load_recordings(config, task)
    name = config.model.selected_architecture
    pooled: dict[str, list] = {k: [] for k in checks}
    folds = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, subject in enumerate(sorted(set(groups.tolist())), start=1):
            trv = groups != subject
            rng = np.random.default_rng(config.model.random_state + i)
            val_g = stratified_group_pick(groups[trv], y[trv], config.model.val_size, rng)
            va = trv & np.isin(groups, val_g)
            tr = trv & ~va
            model, _, _ = fit_model(name, X[tr], y[tr], X[va], y[va], config, verbose=0)
            thr = tune_threshold(y[va], model.predict(X[va], verbose=0, batch_size=512).ravel())

            path = os.path.join(tmp, f"fold_{subject}.tflite")
            convert_to_tflite(model, path, quantize=True)
            tfl = TFLiteModel(subject, path)
            tfl.threshold = thr
            cached = CachedModel(tfl)
            recs = filter_recordings(recordings, subjects=[subject])
            fold = {"subject": subject, "threshold": round(float(thr), 4)}
            for key, check in checks.items():
                res = replay(session_factory(cached, config, check), recs, config)
                pooled[key].extend(res)
                s = summarize(res, config.signal.sample_rate_hz)
                fold[key] = {k: s[k] for k in ("fall_recordings", "falls_detected", "adl_hours", "false_alarms")}
            folds.append(fold)
            logger.info(f"Fold {i} {subject}: " + " | ".join(
                f"{k}: falls {fold[k]['falls_detected']}/{fold[k]['fall_recordings']}, FA {fold[k]['false_alarms']}"
                for k in checks))

    return {
        "protocol": "event-level leave-one-subject-out (fold models exported to quantized TFLite)",
        "task": task, "architecture": name,
        "input_features": config.model.get("input_features", "raw"),
        "checks": {k: (None if c is None else c.__dict__) for k, c in checks.items()},
        "pooled": {k: summarize(v, config.signal.sample_rate_hz) for k, v in pooled.items()},
        "folds": folds,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_table(title: str, off: dict, on: dict) -> None:
    keys = ["fall_recordings", "falls_detected", "fall_recall", "adl_recordings", "adl_hours",
            "false_alarms", "false_alarms_per_hour", "adl_recordings_with_false_alarm",
            "off_target_events_in_fall_recordings"]
    print(f"\n{title}")
    print(f"{'metric':<40}{'check OFF':>12}{'check ON':>12}")
    for k in keys:
        print(f"{k:<40}{off[k]!s:>12}{on[k]!s:>12}")
    codes = sorted(set(off["false_alarms_by_code"]) | set(on["false_alarms_by_code"]))
    print("false alarms by activity (OFF -> ON):  " + ", ".join(
        f"{c} {off['false_alarms_by_code'].get(c, {}).get('false_alarms', 0)}->"
        f"{on['false_alarms_by_code'].get(c, {}).get('false_alarms', 0)}" for c in codes))
    print(f"missed falls OFF {off['missed_falls_by_code']} | ON {on['missed_falls_by_code']}")


def _resolve_subjects(arg: Optional[str], metadata: dict) -> Optional[list[str]]:
    if not arg:
        return None
    if arg in ("test", "val", "train"):
        return metadata["split"][f"{arg}_groups"]
    return [s.strip() for s in arg.split(",") if s.strip()]


def main():
    p = argparse.ArgumentParser(description="Event-level evaluation with / without the post-fall check.")
    p.add_argument("--task", default="human", help="recordings to replay: human (MobiFall) | sisfall")
    p.add_argument("--model", default=None, help="TFLite model (default: the human model from config)")
    p.add_argument("--subjects", default=None, help="comma list, or test | val | train (from model metadata)")
    p.add_argument("--prefix", default=None, help="subject-id prefix filter, e.g. SA / SE for SisFall")
    p.add_argument("--grid", action="store_true", help="grid-search the check parameters on these recordings")
    p.add_argument("--loso", action="store_true",
                   help="event-level leave-one-subject-out: retrain per subject (slow)")
    p.add_argument("--json", default=None, help="write results to this JSON file")
    p.add_argument("--config", default=None)
    a = p.parse_args()
    config = load_config(a.config)

    if a.loso:
        checks = {"check_off": None,
                  "check_default": PostFallCheck(),
                  "check_configured": post_fall_check_from_config(config, "human") or PostFallCheck()}
        out = loso_events(config, checks, a.task)
        for key, s in out["pooled"].items():
            by_code = {c: d["false_alarms"] for c, d in s["false_alarms_by_code"].items() if d["false_alarms"]}
            print(f"{key:<17} falls {s['falls_detected']}/{s['fall_recordings']} (recall {s['fall_recall']}) | "
                  f"false alarms {s['false_alarms']} in {s['adl_hours']} h ({s['false_alarms_per_hour']}/h) | "
                  f"by code {by_code}")
        path = a.json or str(resolve_path(config.paths.reports_dir) / f"{a.task}_events_loso.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        return

    spec = config.models.human
    tflite = a.model or str(resolve_path(spec.tflite))
    meta_path = os.path.splitext(tflite)[0] + ".json"
    model = CachedModel(TFLiteModel("human", tflite, meta_path, labels=spec.labels))

    recordings = filter_recordings(load_recordings(config, a.task),
                                   _resolve_subjects(a.subjects, model.model.metadata), a.prefix)
    logger.info(f"Replaying {len(recordings)} recordings ({sum(r.label for r in recordings)} falls)")

    check = post_fall_check_from_config(config, "human") or PostFallCheck()
    off = evaluate(model, recordings, config, None)
    on = evaluate(model, recordings, config, check)
    _print_table(f"{a.task} {a.subjects or ''} {a.prefix or ''} – post-fall check {check}", off, on)
    out = {"task": a.task, "subjects": a.subjects, "prefix": a.prefix, "model": os.path.basename(tflite),
           "threshold": model.threshold, "check_params": check.__dict__, "check_off": off, "check_on": on}

    if a.grid:
        gs = grid_search(model, recordings, config)
        print(f"\nGrid search ({len(gs['rows'])} settings), recall floor {gs['recall_floor']}: best = {gs['best']}")
        out["grid_search"] = gs

    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
