"""Confidence-aware prediction utilities for the root-cause classifier.

Public API
----------
CONFIDENCE_THRESHOLD : float
    Minimum top-class probability required for the model prediction to be
    trusted without human review.  Anything below this triggers the
    low-confidence safety gate in agent/policies.py.

predict_proba_full(frame) -> (labels, top_confs, proba_dicts)
    Return the top predicted label, the top-class probability, *and* the
    full per-class probability breakdown for every row in *frame*.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_MODEL_DIR = Path(__file__).resolve().parent
if str(_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_MODEL_DIR))

from train_classifier import load_bundle  # noqa: E402

# ── Safety threshold ──────────────────────────────────────────────────────────
#
# If the classifier's top-class probability is below this value, the agent's
# confidence gate (agent/policies.py) will force ESCALATE_HUMAN regardless of
# what the intervention map recommends.
#
# Start at 0.55.  Tune upward to be more conservative; downward to let the
# model act more autonomously.
CONFIDENCE_THRESHOLD: float = 0.55


# ── Core prediction function ──────────────────────────────────────────────────

def predict_proba_full(
    frame: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, list[dict[str, float]]]:
    """Predict root causes with full probability breakdown for each row.

    Parameters
    ----------
    frame : pd.DataFrame
        Raw payment-failure rows (same schema as the holdout set).

    Returns
    -------
    labels : pd.Series
        Top predicted class label for each row.
    top_confs : pd.Series
        Probability of the top class (0–1) for each row.
    proba_dicts : list[dict[str, float]]
        One dict per row mapping every class label → probability (rounded to
        4 decimal places).  Order matches *frame*.
    """
    bundle = load_bundle()
    clf = bundle["model"]
    transformer = bundle["transformer"]
    y_enc = bundle["label_encoder"]
    class_labels: list[str] = list(y_enc.classes_)

    X = transformer.transform(frame)
    proba: np.ndarray = clf.predict_proba(X)  # shape (n_rows, n_classes)

    top_idx = proba.argmax(axis=1)
    labels = pd.Series(
        y_enc.inverse_transform(top_idx),
        index=frame.index,
        name="predicted_root_cause",
    )
    top_confs = pd.Series(
        proba.max(axis=1),
        index=frame.index,
        name="classifier_confidence",
    )
    proba_dicts: list[dict[str, float]] = [
        {cls: round(float(p), 4) for cls, p in zip(class_labels, row_proba)}
        for row_proba in proba
    ]
    return labels, top_confs, proba_dicts


def predict_one_full(
    row: dict[str, Any],
) -> tuple[str, float, dict[str, float]]:
    """Convenience wrapper for a single transaction dict.

    Returns
    -------
    (label, top_confidence, proba_dict)
    """
    frame = pd.DataFrame([row])
    labels, top_confs, proba_dicts = predict_proba_full(frame)
    return str(labels.iloc[0]), float(top_confs.iloc[0]), proba_dicts[0]
