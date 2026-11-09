"""
src/train.py
------------
Model definitions (1D-CNN, LSTM, CNN-LSTM, Transformer) and a leakage-free
training / evaluation pipeline.

Evaluation protocol:
  - Subject-wise (group-wise) split into train / validation / test. Windows
    overlap by 50 %, so a random window split leaks near-duplicate windows
    from the same recording into the test set and inflates accuracy.
  - Early stopping, checkpointing and threshold tuning use VALIDATION only;
    the test split is touched exactly once, at the end.
  - Per-channel normalisation is a layer inside the model (statistics from
    the training split), so the exported TFLite file is self-contained.
  - Optional random-rotation augmentation for phone-orientation robustness;
    the test set is additionally evaluated under random rotations.

Standalone usage:
    python src/train.py --task human                     # train + export selected architecture
    python src/train.py --task car
    python src/train.py --task human --compare           # benchmark all 4 architectures
    python src/train.py --task human --cv 5              # 5-fold subject-wise cross-validation
    python src/train.py --task human --loso              # leave-one-subject-out
    python src/train.py --dry-run                        # tiny synthetic data (CI smoke test)

Outputs:
    models/<task>.keras / .tflite / .json   (model, edge model, metadata incl. threshold)
    reports/<task>_metrics.json, reports/<task>_*.png
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402
from tensorflow.keras import callbacks, layers, models  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model_layers import GravityAlignedFeatures  # noqa: E402
from src.preprocess import generate_dummy_data, load_processed  # noqa: E402
from src.signal_processing import augment_rotation, invariant_features  # noqa: E402
from src.tflite_converter import convert_to_tflite, tflite_parity  # noqa: E402
from src.utils.config_loader import load_config, resolve_path  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

TASK_DATASET = {
    "human": "MobiFall v2.0 (24 subjects, smartphone in trouser pocket)",
    "human+sisfall": ("MobiFall v2.0 (24 subjects, phone in trouser pocket) + SisFall "
                      "(38 subjects incl. 15 aged 60–75, waist-worn IMU)"),
    "car": "Synthetic physics-informed vehicle crash dataset (scripts/generate_car_crash_data.py)",
}


def load_training_data(config, task: str, data: str = "mobifall"):
    """
    Windows for a task. For the human model, `data="mobifall+sisfall"` pools both
    datasets; subject ids stay distinct (MobiFall subNN, SisFall SA/SE NN), so
    every split remains subject-wise.
    """
    X, y, groups = load_processed(config, task)
    if task == "human" and data == "mobifall+sisfall":
        Xs, ys, gs = load_processed(config, "sisfall")
        X, y, groups = np.concatenate([X, Xs]), np.concatenate([y, ys]), np.concatenate([groups, gs])
    return X, y, groups


# ---------------------------------------------------------------------------
# Model Architectures
# ---------------------------------------------------------------------------
# Every builder takes a pre-adapted Normalization layer so that the deployed
# model consumes raw sensor units (m/s², °/s) directly.

def build_cnn_model(input_shape: tuple, norm: layers.Layer) -> tf.keras.Model:
    """1D-CNN – lightweight, selected for edge deployment."""
    inputs = layers.Input(shape=input_shape)
    x = norm(inputs)
    x = layers.Conv1D(64, 3, activation="relu")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv1D(32, 3, activation="relu")(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return models.Model(inputs, outputs, name="1D-CNN")


def build_lstm_model(input_shape: tuple, norm: layers.Layer) -> tf.keras.Model:
    """LSTM – long-range temporal modelling."""
    inputs = layers.Input(shape=input_shape)
    x = norm(inputs)
    x = layers.LSTM(64)(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return models.Model(inputs, outputs, name="LSTM")


def build_cnn_lstm_model(input_shape: tuple, norm: layers.Layer) -> tf.keras.Model:
    """CNN-LSTM hybrid."""
    inputs = layers.Input(shape=input_shape)
    x = norm(inputs)
    x = layers.Conv1D(64, 3, activation="relu")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(64)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return models.Model(inputs, outputs, name="CNN-LSTM")


def build_transformer_model(input_shape: tuple, norm: layers.Layer) -> tf.keras.Model:
    """Lightweight Transformer encoder block."""
    inputs = layers.Input(shape=input_shape)
    x = norm(inputs)
    attn = layers.MultiHeadAttention(num_heads=2, key_dim=16)(x, x)
    x = layers.LayerNormalization()(x + attn)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return models.Model(inputs, outputs, name="Transformer")


MODEL_REGISTRY: dict[str, callable] = {
    "1D-CNN": build_cnn_model,
    "LSTM": build_lstm_model,
    "CNN-LSTM": build_cnn_lstm_model,
    "Transformer": build_transformer_model,
}


def build_model(name: str, X_train: np.ndarray, config) -> tf.keras.Model:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Valid options: {list(MODEL_REGISTRY)}")
    norm = layers.Normalization(axis=-1, name="input_norm")
    if config.model.get("input_features", "raw") == "invariant":
        # raw sensor values in → rotation-invariant channels → normalisation
        norm.adapt(invariant_features(X_train).reshape(-1, X_train.shape[-1]))
        prep = models.Sequential([GravityAlignedFeatures(name="gravity_aligned"), norm], name="preprocess")
    else:
        norm.adapt(X_train.reshape(-1, X_train.shape[-1]))
        prep = norm
    model = MODEL_REGISTRY[name](tuple(config.model.input_shape), prep)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.model.learning_rate),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(curve="PR", name="pr_auc"), "accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def _group_strata(groups: np.ndarray, y: np.ndarray) -> dict[str, list]:
    """
    Bucket groups by which classes they contain. MobiFall matters here: 15 of
    24 subjects recorded falls only, so an unstratified pick can produce a
    validation set with (almost) no normal windows and a useless threshold.
    """
    strata: dict[str, list] = {}
    for g in sorted(set(groups.tolist())):
        labels = set(np.unique(y[groups == g]).tolist())
        strata.setdefault("both" if labels == {0, 1} else f"only_{labels.pop()}", []).append(g)
    return strata


def stratified_group_pick(groups: np.ndarray, y: np.ndarray, frac: float,
                          rng: np.random.Generator) -> np.ndarray:
    """Pick ~frac of the groups from every stratum (at least one from strata with ≥ 3 groups)."""
    picked = []
    for members in _group_strata(groups, y).values():
        members = np.array(members)
        rng.shuffle(members)
        n = round(len(members) * frac)
        if n == 0 and len(members) >= 3:
            n = 1
        picked.extend(members[:n].tolist())
    return np.array(picked)


def group_split(groups: np.ndarray, y: np.ndarray, test_size: float, val_size: float, seed: int):
    """
    Split groups (subjects / sequences) into train / val / test, stratified so
    that val and test both contain normal and event windows.
    Returns three boolean masks over the windows.
    """
    rng = np.random.default_rng(seed)
    test_g = stratified_group_pick(groups, y, test_size, rng)
    rest = ~np.isin(groups, test_g)
    val_g = stratified_group_pick(groups[rest], y[rest], val_size / (1 - test_size), rng)
    train = rest & ~np.isin(groups, val_g)
    if not train.any():
        raise ValueError(f"Not enough groups ({len(np.unique(groups))}) for a train/val/test split")
    return train, np.isin(groups, val_g), np.isin(groups, test_g)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def tune_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Threshold that maximises F1 on the given (validation) data."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    prec, rec, thr = precision_recall_curve(y_true, scores)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-9, None)
    return float(thr[int(np.argmax(f1))])


def compute_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float, step_s: float) -> dict:
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    both = len(np.unique(y_true)) == 2
    normal_hours = max((y_true == 0).sum() * step_s / 3600.0, 1e-9)
    return {
        "n": int(len(y_true)),
        "n_events": int(y_true.sum()),
        "accuracy": round(float((tp + tn) / len(y_true)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "specificity": round(float(tn / max(tn + fp, 1)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, scores)), 4) if both else None,
        "pr_auc": round(float(average_precision_score(y_true, scores)), 4) if both else None,
        "false_alarms_per_hour": round(float(fp / normal_hours), 2),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def class_weights(y: np.ndarray) -> dict:
    counts = np.bincount(y, minlength=2)
    return {c: float(len(y) / (2 * max(counts[c], 1))) for c in (0, 1)}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def fit_model(name: str, X_tr, y_tr, X_val, y_val, config, ckpt_path: Path | None = None, verbose=1):
    rng = np.random.default_rng(config.model.random_state)
    invariant = config.model.get("input_features", "raw") == "invariant"
    if config.model.augment_rotation and not invariant:  # invariant inputs make rotations a no-op
        aug = [augment_rotation(X_tr, rng) for _ in range(config.model.augment_copies)]
        X_tr = np.concatenate([X_tr, *aug])
        y_tr = np.concatenate([y_tr] * (1 + len(aug)))

    tf.keras.utils.set_random_seed(config.model.random_state)
    model = build_model(name, X_tr, config)
    cbs = [callbacks.EarlyStopping(monitor="val_loss", patience=config.model.early_stopping_patience,
                                   restore_best_weights=True, verbose=0)]
    if ckpt_path is not None:
        cbs.append(callbacks.ModelCheckpoint(str(ckpt_path), monitor="val_loss", save_best_only=True))

    t0 = time.time()
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=config.model.epochs,
        batch_size=config.model.batch_size,
        class_weight=class_weights(y_tr) if config.model.class_weighting else None,
        callbacks=cbs,
        shuffle=True,
        verbose=verbose,
    )
    return model, history, time.time() - t0


def evaluate_split(model, X, y, threshold, step_s, config) -> dict:
    scores = model.predict(X, verbose=0, batch_size=512).ravel()
    m = compute_metrics(y, scores, threshold, step_s)
    rotated = augment_rotation(X, np.random.default_rng(config.model.random_state + 1))
    m_rot = compute_metrics(y, model.predict(rotated, verbose=0, batch_size=512).ravel(), threshold, step_s)
    m["rotated_phone"] = {k: m_rot[k] for k in ("accuracy", "precision", "recall", "f1", "pr_auc")}
    return m, scores


def train(X, y, groups, config, task: str, model_name: str | None = None,
          export: bool = True, verbose: int = 1, dataset_key: str | None = None) -> tuple[dict, tf.keras.Model]:
    """
    Train one architecture with a subject-wise split, tune the threshold on
    validation, evaluate once on test, export Keras + TFLite + metadata.

    Returns (result dict, trained Keras model).
    """
    model_name = model_name or config.model.selected_architecture
    step_s = config.windowing.window_size * (1 - config.windowing.overlap) / config.signal.sample_rate_hz
    tr, va, te = group_split(groups, y, config.model.test_size, config.model.val_size, config.model.random_state)
    logger.info(f"[{task}/{model_name}] windows train {tr.sum():,} | val {va.sum():,} | test {te.sum():,} "
                f"(groups {len(np.unique(groups[tr]))}/{len(np.unique(groups[va]))}/{len(np.unique(groups[te]))})")

    spec = config.models[task]
    keras_path = resolve_path(spec.keras)
    keras_path.parent.mkdir(parents=True, exist_ok=True)
    model, history, train_time = fit_model(model_name, X[tr], y[tr], X[va], y[va], config,
                                           keras_path if export else None, verbose)

    val_scores = model.predict(X[va], verbose=0, batch_size=512).ravel()
    fixed = config.model.prediction_threshold
    threshold = float(fixed) if fixed is not None else tune_threshold(y[va], val_scores)
    val_metrics = compute_metrics(y[va], val_scores, threshold, step_s)
    test_metrics, test_scores = evaluate_split(model, X[te], y[te], threshold, step_s, config)

    result = {
        "task": task,
        "architecture": model_name,
        "input_features": config.model.get("input_features", "raw"),
        "dataset": TASK_DATASET.get(dataset_key or task, task),
        "params": int(model.count_params()),
        "train_time_s": round(train_time, 1),
        "epochs_run": len(history.history["loss"]),
        "threshold": float(threshold),  # full precision: a rounded 0.99996 → 1.0 would never fire
        "split": {
            "method": "group-wise (subjects / sequences never shared across splits)",
            "train_groups": sorted(np.unique(groups[tr]).tolist()),
            "val_groups": sorted(np.unique(groups[va]).tolist()),
            "test_groups": sorted(np.unique(groups[te]).tolist()),
            "n_train": int(tr.sum()), "n_val": int(va.sum()), "n_test": int(te.sum()),
        },
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    logger.info(f"[{task}/{model_name}] TEST  acc={test_metrics['accuracy']:.4f} "
                f"P={test_metrics['precision']:.4f} R={test_metrics['recall']:.4f} "
                f"F1={test_metrics['f1']:.4f} PR-AUC={test_metrics['pr_auc']} "
                f"FA/h={test_metrics['false_alarms_per_hour']} (thr={threshold:.3f})")

    if export:
        model.save(keras_path)
        tflite_path = resolve_path(spec.tflite)
        conv = convert_to_tflite(model, str(tflite_path), quantize=True)
        parity = tflite_parity(model, str(tflite_path), X[te][:512], threshold)
        result.update({
            "keras_size_kb": round(os.path.getsize(keras_path) / 1024, 1),
            "tflite_size_kb": conv["tflite_size_kb"],
            "float32_tflite_size_kb": conv["float32_size_kb"],
            "tflite_parity": parity,
            "labels": {str(k): v for k, v in spec.labels.items()},
            "window_size": config.windowing.window_size,
            "sample_rate_hz": config.signal.sample_rate_hz,
            "channels": list(config.data.sensor_columns),
            "units": {"acc": config.signal.acc_unit, "gyro": config.signal.gyro_unit},
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tensorflow": tf.__version__,
        })
        with open(resolve_path(spec.metadata), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        _save_reports(task, result, history, y[te], test_scores, threshold, config)
        logger.info(f"Exported {keras_path.name}, {tflite_path.name} ({conv['tflite_size_kb']} KB), metadata")
    return result, model


def _save_reports(task, result, history, y_test, scores, threshold, config):
    reports = resolve_path(config.paths.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    with open(reports / f"{task}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    labels = list(config.models[task].labels.values())
    cm = result["test_metrics"]["confusion_matrix"]
    mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    ax.imshow(mat, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{mat[i, j]:,}", ha="center", va="center",
                    color="white" if mat[i, j] > mat.max() / 2 else "black", fontsize=12)
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Test confusion matrix (thr={threshold:.2f})")

    prec, rec, _ = precision_recall_curve(y_test, scores)
    axes[1].plot(rec, prec)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"Test PR curve (AP={result['test_metrics']['pr_auc']})")
    axes[1].grid(alpha=0.3)

    axes[2].plot(history.history["loss"], label="train")
    axes[2].plot(history.history["val_loss"], label="validation")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Loss")
    axes[2].set_title("Training curve")
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    fig.suptitle(f"{task} – {result['architecture']}")
    fig.tight_layout()
    fig.savefig(reports / f"{task}_evaluation.png", dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Architecture comparison & cross-validation
# ---------------------------------------------------------------------------

def compare_architectures(X, y, groups, config, task: str) -> list[dict]:
    """
    Train every architecture on the same split and measure the deployable
    artefact: quantized TFLite size and single-window latency.
    """
    import tempfile

    from src.inference import TFLiteModel

    out = []
    _, _, te = group_split(groups, y, config.model.test_size, config.model.val_size, config.model.random_state)
    with tempfile.TemporaryDirectory() as tmp:
        for name in MODEL_REGISTRY:
            res, model = train(X, y, groups, config, task, model_name=name, export=False, verbose=0)
            tfl_path = os.path.join(tmp, f"{name}.tflite")
            conv = convert_to_tflite(model, tfl_path, quantize=True)
            try:
                tfl = TFLiteModel(name, tfl_path)
                lat = [tfl.score(w)[1] for w in X[te][:300]]
            except RuntimeError as exc:  # e.g. needs the Flex delegate → not edge-deployable as is
                logger.warning(f"{name}: TFLite model not runnable with the builtin interpreter: {exc}")
                lat = [float("nan")]
            out.append({
                "architecture": name,
                "params": res["params"],
                "tflite_size_kb": conv["tflite_size_kb"],
                "latency_ms_p50": round(float(np.percentile(lat, 50)), 3),
                "train_time_s": res["train_time_s"],
                **{k: res["test_metrics"][k] for k in ("accuracy", "precision", "recall", "f1", "pr_auc",
                                                        "false_alarms_per_hour")},
                "rotated_f1": res["test_metrics"]["rotated_phone"]["f1"],
            })
    reports = resolve_path(config.paths.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    with open(reports / f"{task}_architecture_comparison.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    _log_table(f"Architecture comparison ({task}, subject-wise test split)", out)
    return out


def cross_validate(X, y, groups, config, task: str, n_splits: int | None = None,
                   tag_suffix: str = "") -> dict:
    """
    Group k-fold CV (n_splits=None → leave-one-group-out). In each fold one
    training group is held out for validation (early stopping + threshold).
    """
    name = config.model.selected_architecture
    step_s = config.windowing.window_size * (1 - config.windowing.overlap) / config.signal.sample_rate_hz
    unique = np.unique(groups)
    k = len(unique) if n_splits is None else n_splits
    folds = []
    for i, (trv_idx, te_idx) in enumerate(GroupKFold(n_splits=k).split(X, y, groups), start=1):
        rng = np.random.default_rng(config.model.random_state + i)
        val_groups = stratified_group_pick(groups[trv_idx], y[trv_idx], config.model.val_size, rng)
        va_mask = np.isin(groups[trv_idx], val_groups)
        tr_idx, va_idx = trv_idx[~va_mask], trv_idx[va_mask]

        model, _, _ = fit_model(name, X[tr_idx], y[tr_idx], X[va_idx], y[va_idx], config, verbose=0)
        thr = tune_threshold(y[va_idx], model.predict(X[va_idx], verbose=0, batch_size=512).ravel())
        te_scores = model.predict(X[te_idx], verbose=0, batch_size=512).ravel()
        m = compute_metrics(y[te_idx], te_scores, thr, step_s)
        m["test_groups"] = sorted(np.unique(groups[te_idx]).tolist())
        domains = np.array([_domain(g) for g in groups[te_idx]])
        if len(set(domains)) > 1 or domains[0] != "mobifall":
            m["by_domain"] = {d: compute_metrics(y[te_idx][domains == d], te_scores[domains == d], thr, step_s)
                              for d in sorted(set(domains))}
        m["threshold"] = round(thr, 4)
        folds.append(m)
        logger.info(f"Fold {i}/{k} {m['test_groups']}: acc={m['accuracy']:.4f} F1={m['f1']:.4f} "
                    f"R={m['recall']:.4f} FA/h={m['false_alarms_per_hour']}")

    summary = summarize_folds(folds, step_s)
    domains = sorted({d for f in folds for d in f.get("by_domain", {})})
    if domains:  # pooled metrics per dataset (e.g. MobiFall vs SisFall young vs SisFall elderly)
        summary["pooled_by_domain"] = {
            d: summarize_folds([f["by_domain"][d] for f in folds if d in f.get("by_domain", {})], step_s)["pooled"]
            for d in domains}
    features = config.model.get("input_features", "raw")
    summary = {"task": task, "architecture": name, "input_features": features,
               "protocol": "leave-one-subject-out" if n_splits is None else f"{k}-fold group CV",
               **summary}
    reports = resolve_path(config.paths.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    tag = "loso" if n_splits is None else f"cv{k}"
    with open(reports / f"{task}_{tag}_{features}{tag_suffix}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"{summary['protocol']} ({name}) pooled: " + " | ".join(
        f"{key}={summary['pooled'][key]}" for key in ("accuracy", "precision", "recall", "f1")))
    return summary


def _domain(group: str) -> str:
    """Dataset of a subject id: MobiFall subNN, SisFall SA (young) / SE (elderly)."""
    return {"SA": "sisfall_young", "SE": "sisfall_elderly"}.get(group[:2], "mobifall")


def summarize_folds(folds: list[dict], step_s: float) -> dict:
    """
    Aggregate CV folds two ways:
      pooled   – metrics of the summed confusion matrix (every window counts once;
                 robust to folds with few or no windows of one class)
      per_fold – mean ± std of per-fold metrics, over folds that contain both classes
    """
    cm = {k: sum(f["confusion_matrix"][k] for f in folds) for k in ("tn", "fp", "fn", "tp")}
    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    pooled = {
        "accuracy": round((tp + tn) / max(tn + fp + fn + tp, 1), 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "specificity": round(tn / max(tn + fp, 1), 4),
        "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4),
        "false_alarms_per_hour": round(fp / max((tn + fp) * step_s / 3600, 1e-9), 2),
        "confusion_matrix": cm,
    }

    both = [f for f in folds if 0 < f["n_events"] < f["n"]]
    per_fold = {"n_folds_with_both_classes": len(both)}
    for key in ("accuracy", "precision", "recall", "f1", "pr_auc"):
        vals = np.array([f[key] for f in both if f[key] is not None], dtype=float)
        if len(vals):
            per_fold[key] = {"mean": round(float(vals.mean()), 4), "std": round(float(vals.std()), 4),
                             "median": round(float(np.median(vals)), 4)}
    return {"pooled": pooled, "per_fold": per_fold, "folds": folds}


def _log_table(title: str, rows: list[dict]) -> None:
    if not rows:
        return
    cols = list(rows[0])
    lines = [title, " | ".join(cols)] + [" | ".join(str(r[c]) for c in cols) for r in rows]
    logger.info("\n" + "\n".join(lines))


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="Train a fall / crash detection model.")
    parser.add_argument("--task", choices=["human", "car"], default="human")
    parser.add_argument("--data", choices=["mobifall", "mobifall+sisfall"], default=None,
                        help="Training data for the human model (default: config data.human_training).")
    parser.add_argument("--model", choices=list(MODEL_REGISTRY), default=None,
                        help="Architecture (default from config).")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--compare", action="store_true", help="Benchmark all architectures.")
    parser.add_argument("--cv", type=int, default=None, metavar="K", help="K-fold subject-wise CV.")
    parser.add_argument("--loso", action="store_true", help="Leave-one-subject-out CV.")
    parser.add_argument("--features", choices=["raw", "invariant"], default=None,
                        help="Model input: raw phone-frame axes or rotation-invariant features.")
    parser.add_argument("--dry-run", action="store_true", help="Tiny synthetic data, no export.")
    parser.add_argument("--config", type=str, default=None)
    return parser.parse_args()


def main():
    args = _parse_args()
    config = load_config(args.config)
    if args.epochs:
        config.model.epochs = args.epochs
    if args.batch_size:
        config.model.batch_size = args.batch_size
    if args.model:
        config.model.selected_architecture = args.model
    if args.features:
        config.model.input_features = args.features

    if args.dry_run:
        logger.info("DRY RUN — synthetic data, 2 epochs, no export.")
        config.model.epochs = 2
        X, y, groups = generate_dummy_data(config)
        train(X, y, groups, config, args.task, export=False, verbose=0)
        return

    data = args.data or config.data.get("human_training", "mobifall")
    X, y, groups = load_training_data(config, args.task, data)
    key = "human+sisfall" if args.task == "human" and data == "mobifall+sisfall" else args.task
    if args.compare:
        compare_architectures(X, y, groups, config, args.task)
    elif args.loso:
        cross_validate(X, y, groups, config, args.task, n_splits=None, tag_suffix=_data_tag(key))
    elif args.cv:
        cross_validate(X, y, groups, config, args.task, n_splits=args.cv, tag_suffix=_data_tag(key))
    else:
        train(X, y, groups, config, args.task, dataset_key=key)


def _data_tag(key: str) -> str:
    return "_combined" if key == "human+sisfall" else ""


if __name__ == "__main__":
    main()
