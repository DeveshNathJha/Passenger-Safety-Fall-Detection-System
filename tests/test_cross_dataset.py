"""Smoke test for the MobiFall → SisFall cross-dataset protocol on tiny synthetic data."""

import copy
import importlib.util

import numpy as np
import pytest

from src.preprocess import Recording
from src.utils.config_loader import AttrDict

pytestmark = pytest.mark.skipif(importlib.util.find_spec("tensorflow") is None,
                                reason="tensorflow not installed")


def make_train_data(n_subjects=8, per=20, seed=0):
    rng = np.random.default_rng(seed)
    X, y, g = [], [], []
    for s in range(n_subjects):
        for i in range(per):
            w = rng.normal(0, 1, (128, 6)).astype(np.float32)
            w[:, 1] += 9.81
            label = int(i % 4 == 0)
            if label:
                w[64, 0:3] += 40
            X.append(w)
            y.append(label)
            g.append(f"sub{s:02d}")
    return np.stack(X), np.array(y, dtype=np.int32), np.array(g)


def make_test_recordings(seed=1):
    rng = np.random.default_rng(seed)
    recs = []
    for subj in ("SA01", "SE02"):
        for code, label in (("D01", 0), ("D17", 0), ("F01", 1)):
            if subj.startswith("SE") and label:
                continue  # elderly: ADLs only, like SisFall
            data = rng.normal(0, 1, (400, 6)).astype(np.float32)
            data[:, 1] += 9.81
            if label:
                data[200, 0:3] += 40
            recs.append(Recording(code, subj, "R01", label, data))
    return recs


def test_run_cross_dataset_structure(base_config, tmp_path):
    from src.cross_dataset import run_cross_dataset

    cfg = AttrDict(copy.deepcopy(dict(base_config)))
    cfg.paths.reports_dir = str(tmp_path)
    out = run_cross_dataset(cfg, features="invariant", epochs=1,
                            train_data=make_train_data(), test_recordings=make_test_recordings(),
                            export_tflite=str(tmp_path / "cd.tflite"), write_report=True)

    assert out["input_features"] == "invariant"
    assert 0.0 <= out["threshold"] <= 1.0
    assert out["mobifall_validation_subjects"]
    sis = out["sisfall"]
    assert sis["n_subjects"] == 2
    for part in ("overall", "young_adults", "elderly"):
        assert {"accuracy", "precision", "recall", "f1", "confusion_matrix"} <= set(sis[part])
    assert sis["elderly"]["n_events"] == 0
    assert out["per_code"]["F01"]["type"] == "fall" and "recall" in out["per_code"]["F01"]
    assert out["per_code"]["D17"]["type"] == "adl" and "fp_rate" in out["per_code"]["D17"]
    assert (tmp_path / "cross_dataset_invariant.json").exists()
    assert (tmp_path / "cd.tflite").exists() and (tmp_path / "cd.json").exists()


def test_windows_with_codes_keeps_activity_codes(base_config):
    from src.cross_dataset import windows_with_codes

    X, y, subj, codes = windows_with_codes(make_test_recordings(), base_config)
    assert len(X) == len(y) == len(subj) == len(codes)
    assert set(codes) == {"D01", "D17", "F01"}
    assert (y[codes == "F01"] == 1).all() and (y[codes != "F01"] == 0).all()
