"""Regression checks for the trained root-cause classifier."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))

from intervention_map import INTERVENTION_MAP  # noqa: E402
from train_classifier import MODEL_PATH, load_bundle  # noqa: E402

HOLDOUT_PATH = ROOT / "data" / "raw" / "holdout_set.csv"
TARGET = "true_root_cause_category"
MIN_HOLDOUT_ACCURACY = 0.80
MIN_RISK_BLOCKED_RECALL = 0.5


@pytest.fixture(scope="module")
def bundle():
    return load_bundle()


@pytest.fixture(scope="module")
def holdout():
    return pd.read_csv(HOLDOUT_PATH)


@pytest.fixture(scope="module")
def predictions(bundle, holdout):
    X = bundle["transformer"].transform(holdout)
    encoded = bundle["model"].predict(X)
    return pd.Series(
        bundle["label_encoder"].inverse_transform(encoded),
        index=holdout.index,
        name="predicted_root_cause",
    )


def test_classifier_pkl_loads_without_error(bundle):
    assert MODEL_PATH.is_file()
    assert bundle["model"] is not None
    assert bundle["transformer"] is not None
    assert bundle["label_encoder"] is not None


def test_holdout_predictions_one_per_row_valid_labels(bundle, holdout, predictions):
    train_labels = set(bundle["label_encoder"].classes_)
    assert len(predictions) == len(holdout)
    assert predictions.notna().all()
    assert not predictions.eq("").any()
    unexpected = set(predictions.unique()) - train_labels
    assert not unexpected, f"unexpected predicted classes: {unexpected}"


def test_holdout_accuracy_at_least_0_80(holdout, predictions):
    acc = accuracy_score(holdout[TARGET], predictions)
    assert acc >= MIN_HOLDOUT_ACCURACY, (
        f"holdout accuracy {acc:.3f} dropped below {MIN_HOLDOUT_ACCURACY:.2f}"
    )


def test_risk_blocked_recall_at_least_0_5(holdout, predictions):
    y_true = holdout[TARGET]
    n_true = int((y_true == "RISK_BLOCKED").sum())
    assert n_true > 0, "holdout has no RISK_BLOCKED rows; cannot measure recall"
    recall = recall_score(
        y_true,
        predictions,
        labels=["RISK_BLOCKED"],
        average="micro",
        zero_division=0,
    )
    assert recall >= MIN_RISK_BLOCKED_RECALL, (
        f"RISK_BLOCKED recall {recall:.3f} dropped below {MIN_RISK_BLOCKED_RECALL:.1f}"
    )


def test_every_predicted_class_has_intervention(predictions):
    missing = sorted(set(predictions.unique()) - set(INTERVENTION_MAP))
    assert not missing, (
        f"predicted labels with no intervention_map entry (would crash Phase 3): {missing}"
    )
