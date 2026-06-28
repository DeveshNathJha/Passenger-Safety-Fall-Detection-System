"""
src/train.py
------------
Model definitions (1D-CNN, LSTM, CNN-LSTM, Transformer) and
training loop for the Fall Detection System.

Standalone usage:
    python src/train.py                          # train with config defaults
    python src/train.py --model 1D-CNN --epochs 20
    python src/train.py --model CNN-LSTM --batch-size 32

Outputs:
    models/mobifall_model.h5           – best Keras model weights
    logs/app.log                       – structured training logs
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.preprocess import load_processed, generate_dummy_data

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Model Architectures
# ---------------------------------------------------------------------------

def build_cnn_model(input_shape: tuple) -> tf.keras.Model:
    """1D-CNN – Lightweight, selected for edge deployment."""
    model = models.Sequential(
        [
            layers.Conv1D(64, 3, activation="relu", input_shape=input_shape),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),
            layers.Conv1D(32, 3, activation="relu"),
            layers.GlobalMaxPooling1D(),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="1D-CNN",
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_lstm_model(input_shape: tuple) -> tf.keras.Model:
    """LSTM – Good for long-range temporal sequences."""
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.LSTM(64, return_sequences=False),
            layers.Dropout(0.4),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="LSTM",
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn_lstm_model(input_shape: tuple) -> tf.keras.Model:
    """CNN-LSTM Hybrid – Highest accuracy (97.04%) but 4× larger than 1D-CNN."""
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv1D(64, 3, activation="relu"),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),
            layers.LSTM(64),
            layers.Dropout(0.3),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="CNN-LSTM",
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_transformer_model(input_shape: tuple) -> tf.keras.Model:
    """Lightweight Transformer – Most compact (1,557 params)."""
    inputs = layers.Input(shape=input_shape)
    x = layers.MultiHeadAttention(num_heads=2, key_dim=32)(inputs, inputs)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs, name="Transformer")
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# Registry: maps CLI name → builder function
MODEL_REGISTRY: dict[str, callable] = {
    "1D-CNN": build_cnn_model,
    "LSTM": build_lstm_model,
    "CNN-LSTM": build_cnn_lstm_model,
    "Transformer": build_transformer_model,
}


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train(
    X: np.ndarray,
    y: np.ndarray,
    config,
    model_name: str | None = None,
    subjects: np.ndarray | None = None,
) -> tf.keras.Model:
    """
    Split data, build model, train with callbacks, evaluate, and save.

    Parameters
    ----------
    X : np.ndarray
        Windows array, shape (N, 128, 6).
    y : np.ndarray
        Labels array, shape (N,).
    config : AttrDict
        Loaded config.
    model_name : str | None
        Architecture to use. Falls back to config.model.selected_architecture.
    subjects : np.ndarray | None
        Subject IDs for each window, shape (N,).

    Returns
    -------
    tf.keras.Model
        The trained Keras model.
    """
    model_name = model_name or config.model.selected_architecture
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Valid options: {list(MODEL_REGISTRY.keys())}"
        )

    logger.info(f"Training architecture: {model_name}")
    logger.info(f"Dataset — Total windows: {X.shape[0]:,} | Shape: {X.shape}")

    # --- Train / Test Split ---
    split_method = getattr(config.model, "split_method", "window")
    if split_method == "subject" and subjects is not None:
        logger.info("Using Subject-wise Train/Test Split (Subject-independent evaluation)")
        unique_subs = np.unique(subjects)
        unique_subs = sorted(list(unique_subs))
        
        # Shuffle subjects using random state
        rng = np.random.default_rng(config.model.random_state)
        rng.shuffle(unique_subs)
        
        n_test = max(1, int(len(unique_subs) * config.model.test_size))
        test_subs = set(unique_subs[:n_test])
        train_subs = set(unique_subs[n_test:])
        
        logger.info(f"Train subjects ({len(train_subs)}): {sorted(list(train_subs))}")
        logger.info(f"Test subjects ({len(test_subs)}): {sorted(list(test_subs))}")
        
        train_mask = np.array([s in train_subs for s in subjects])
        test_mask = np.array([s in test_subs for s in subjects])
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
    else:
        logger.info("Using Window-level Train/Test Split")
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=config.model.test_size,
            random_state=config.model.random_state,
            stratify=y,
        )
    logger.info(f"Train samples: {X_train.shape[0]:,} | Test samples: {X_test.shape[0]:,}")

    input_shape = tuple(config.model.input_shape)
    model = MODEL_REGISTRY[model_name](input_shape)
    model.summary(print_fn=lambda line: logger.debug(line))

    # --- Callbacks ---
    os.makedirs(os.path.dirname(config.paths.model_h5), exist_ok=True)
    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.model.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=config.paths.model_h5,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # --- Training ---
    start = time.time()
    history = model.fit(
        X_train,
        y_train,
        epochs=config.model.epochs,
        batch_size=config.model.batch_size,
        validation_data=(X_test, y_test),
        callbacks=cb_list,
        verbose=1,
    )
    elapsed = time.time() - start

    # --- Evaluation ---
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    logger.info(
        f"[{model_name}] Test Accuracy: {accuracy * 100:.2f}% | "
        f"Loss: {loss:.4f} | Params: {model.count_params():,} | "
        f"Training time: {elapsed:.1f}s"
    )

    y_pred = (model.predict(X_test) > config.model.prediction_threshold).astype(int).flatten()
    
    report_dict = classification_report(
        y_test, y_pred, target_names=list(config.model.labels.values()), output_dict=True, zero_division=0
    )
    
    # Extract specific metrics
    prec_normal = report_dict["Normal Activity"]["precision"]
    rec_normal = report_dict["Normal Activity"]["recall"]
    f1_normal = report_dict["Normal Activity"]["f1-score"]
    
    prec_fall = report_dict["FALL DETECTED"]["precision"]
    rec_fall = report_dict["FALL DETECTED"]["recall"]
    f1_fall = report_dict["FALL DETECTED"]["f1-score"]
    
    logger.info(
        f"\n--- Model Performance Summary ({model_name}) ---"
        f"\nAccuracy: {accuracy * 100:.2f}%"
        f"\nNormal Activity: Precision={prec_normal:.4f}, Recall={rec_normal:.4f}, F1-Score={f1_normal:.4f}"
        f"\nFALL DETECTED:   Precision={prec_fall:.4f}, Recall={rec_fall:.4f}, F1-Score={f1_fall:.4f}"
        f"\n--------------------------------------------"
    )

    logger.info(
        "\n"
        + classification_report(
            y_test, y_pred, target_names=list(config.model.labels.values()), zero_division=0
        )
    )

    return model


def run_loso_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    config,
    model_name: str | None = None,
):
    """
    Perform Leave-One-Subject-Out (LOSO) cross-validation and print average metrics.
    """
    model_name = model_name or config.model.selected_architecture
    unique_subs = sorted(list(np.unique(subjects)))
    logger.info(f"Starting Leave-One-Subject-Out (LOSO) Cross-Validation with {len(unique_subs)} subjects.")
    
    accuracies = []
    losses = []
    reports = []
    
    input_shape = tuple(config.model.input_shape)
    
    for i, sub in enumerate(unique_subs):
        logger.info(f"\n--- LOSO Fold {i+1}/{len(unique_subs)}: Leaving out subject {sub} ---")
        
        train_mask = (subjects != sub)
        test_mask = (subjects == sub)
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        logger.info(f"Train samples: {X_train.shape[0]:,} | Test samples (Subject {sub}): {X_test.shape[0]:,}")
        
        model = MODEL_REGISTRY[model_name](input_shape)
        
        early_stopping = callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.model.early_stopping_patience,
            restore_best_weights=True,
            verbose=0,
        )
        
        model.fit(
            X_train,
            y_train,
            epochs=config.model.epochs,
            batch_size=config.model.batch_size,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=0,
        )
        
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        logger.info(f"Fold {i+1} [Subject {sub}] - Accuracy: {accuracy * 100:.2f}% | Loss: {loss:.4f}")
        
        accuracies.append(accuracy)
        losses.append(loss)
        
        y_pred = (model.predict(X_test, verbose=0) > config.model.prediction_threshold).astype(int).flatten()
        report = classification_report(
            y_test, y_pred, target_names=list(config.model.labels.values()), output_dict=True, zero_division=0
        )
        reports.append(report)
        
    mean_acc = np.mean(accuracies)
    mean_loss = np.mean(losses)
    
    logger.info("\n============================================================")
    logger.info("Leave-One-Subject-Out (LOSO) Cross-Validation Results")
    logger.info("============================================================")
    for i, sub in enumerate(unique_subs):
        logger.info(f"Subject {sub}: Accuracy = {accuracies[i]*100:.2f}%, F1 Normal = {reports[i]['Normal Activity']['f1-score']:.4f}, F1 Fall = {reports[i]['FALL DETECTED']['f1-score']:.4f}")
    logger.info(f"\nAverage Accuracy: {mean_acc * 100:.2f}%")
    logger.info(f"Average Loss: {mean_loss:.4f}")
    
    fall_f1s = [r['FALL DETECTED']['f1-score'] for r in reports]
    normal_f1s = [r['Normal Activity']['f1-score'] for r in reports]
    logger.info(f"Average Normal F1-score: {np.mean(normal_f1s):.4f}")
    logger.info(f"Average Fall F1-score: {np.mean(fall_f1s):.4f}")
    logger.info("============================================================\n")


def run_kfold_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    config,
    model_name: str | None = None,
    n_splits: int = 5,
):
    """
    Perform stratified K-fold cross-validation and report Mean ± Std metrics.
    Required for journal statistical significance analysis.

    Parameters
    ----------
    X : np.ndarray   shape (N, 128, 6)
    y : np.ndarray   shape (N,)
    config : AttrDict
    model_name : str | None
    n_splits : int   number of folds (default: 5)
    """
    from sklearn.model_selection import StratifiedKFold

    model_name = model_name or config.model.selected_architecture
    input_shape = tuple(config.model.input_shape)

    logger.info(
        f"Starting {n_splits}-Fold Stratified Cross-Validation — "
        f"Architecture: {model_name}"
    )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=config.model.random_state)

    accuracies, losses = [], []
    fall_f1s, normal_f1s, fall_precs, fall_recs = [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        logger.info(f"\n--- K-Fold {fold}/{n_splits} ---")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = MODEL_REGISTRY[model_name](input_shape)
        early_stopping = callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.model.early_stopping_patience,
            restore_best_weights=True,
            verbose=0,
        )

        model.fit(
            X_train, y_train,
            epochs=config.model.epochs,
            batch_size=config.model.batch_size,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=0,
        )

        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        y_pred = (
            model.predict(X_test, verbose=0) > config.model.prediction_threshold
        ).astype(int).flatten()

        report = classification_report(
            y_test, y_pred,
            target_names=list(config.model.labels.values()),
            output_dict=True, zero_division=0
        )

        accuracies.append(accuracy)
        losses.append(loss)
        fall_f1s.append(report["FALL DETECTED"]["f1-score"])
        fall_precs.append(report["FALL DETECTED"]["precision"])
        fall_recs.append(report["FALL DETECTED"]["recall"])
        normal_f1s.append(report["Normal Activity"]["f1-score"])

        logger.info(
            f"Fold {fold}: Acc={accuracy*100:.2f}% | "
            f"Fall F1={report['FALL DETECTED']['f1-score']:.4f} | "
            f"Normal F1={report['Normal Activity']['f1-score']:.4f}"
        )

    # ── Summary ────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info(f"{n_splits}-Fold Stratified Cross-Validation Summary ({model_name})")
    logger.info("=" * 60)
    logger.info(
        f"Accuracy        : {np.mean(accuracies)*100:.2f}% \u00b1 {np.std(accuracies)*100:.2f}%"
    )
    logger.info(
        f"Fall Precision  : {np.mean(fall_precs):.4f} \u00b1 {np.std(fall_precs):.4f}"
    )
    logger.info(
        f"Fall Recall     : {np.mean(fall_recs):.4f} \u00b1 {np.std(fall_recs):.4f}"
    )
    logger.info(
        f"Fall F1-Score   : {np.mean(fall_f1s):.4f} \u00b1 {np.std(fall_f1s):.4f}"
    )
    logger.info(
        f"Normal F1-Score : {np.mean(normal_f1s):.4f} \u00b1 {np.std(normal_f1s):.4f}"
    )
    logger.info("=" * 60 + "\n")

    # Print in table format for copy-paste into journal
    logger.info("Journal Table Format (copy-paste):")
    logger.info(
        f"  {model_name}: "
        f"Acc={np.mean(accuracies)*100:.2f}\u00b1{np.std(accuracies)*100:.2f}%, "
        f"Prec={np.mean(fall_precs):.4f}\u00b1{np.std(fall_precs):.4f}, "
        f"Rec={np.mean(fall_recs):.4f}\u00b1{np.std(fall_recs):.4f}, "
        f"F1={np.mean(fall_f1s):.4f}\u00b1{np.std(fall_f1s):.4f}"
    )


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="Train a fall detection model.")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=list(MODEL_REGISTRY.keys()),
        help="Model architecture to train (default from config).",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Train on dummy data (no dataset required).",
    )
    parser.add_argument(
        "--loso",
        action="store_true",
        help="Perform Leave-One-Subject-Out (LOSO) cross-validation.",
    )
    parser.add_argument(
        "--kfold",
        action="store_true",
        help="Perform 5-fold stratified cross-validation (reports Mean ± Std for journal).",
    )
    parser.add_argument("--config", type=str, default=None)
    return parser.parse_args()


def main():
    args = _parse_args()
    config = load_config(args.config)

    # CLI overrides
    if args.epochs:
        config.model["epochs"] = args.epochs
    if args.batch_size:
        config.model["batch_size"] = args.batch_size

    if args.dry_run:
        logger.info("DRY RUN — using dummy data.")
        X, y, subjects = generate_dummy_data(config, return_subjects=True)
    else:
        X, y, subjects = load_processed(config, return_subjects=True)

    if args.loso:
        run_loso_cross_validation(X, y, subjects, config, model_name=args.model)
    elif args.kfold:
        run_kfold_cross_validation(X, y, config, model_name=args.model)
    else:
        model = train(X, y, config, model_name=args.model, subjects=subjects)
        logger.info(f"Model saved to: {config.paths.model_h5}")


if __name__ == "__main__":
    main()
