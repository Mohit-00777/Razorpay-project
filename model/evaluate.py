"""Score the holdout set: report, confusion matrix, rupee error costs."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_classifier import MODEL_PATH, load_bundle  # noqa: E402
from intervention_map import recommend_action  # noqa: E402

HOLDOUT_PATH = ROOT / "data" / "raw" / "holdout_set.csv"
TARGET = "true_root_cause_category"


def main() -> None:
    bundle = load_bundle()
    holdout = pd.read_csv(HOLDOUT_PATH)
    transformer = bundle["transformer"]
    clf = bundle["model"]
    y_encoder = bundle["label_encoder"]
    labels = list(y_encoder.classes_)

    X = transformer.transform(holdout)
    y_true = holdout[TARGET]
    y_pred = pd.Series(
        y_encoder.inverse_transform(clf.predict(X)),
        index=holdout.index,
        name="predicted_root_cause",
    )

    print("=== Classification report ===")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            digits=3,
            zero_division=0,
        )
    )
    acc = accuracy_score(y_true, y_pred)
    print(f"Overall accuracy: {acc:.3f}  ({(y_true == y_pred).sum()}/{len(holdout)})")

    print("\n=== Confusion matrix (rows=true, cols=predicted) ===")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df.to_string())

    errors = holdout.copy()
    errors["predicted_root_cause"] = y_pred
    errors = errors.loc[errors[TARGET] != errors["predicted_root_cause"]].copy()
    errors["recommended_action"] = errors["predicted_root_cause"].map(recommend_action)
    errors["fp_fn_cost_inr"] = errors["amount"]

    print("\n=== Misclassified rows (each row's amount is the FP/FN rupee cost) ===")
    if errors.empty:
        print("None.")
    else:
        cols = [
            "transaction_id",
            "amount",
            "failure_code",
            TARGET,
            "predicted_root_cause",
            "recommended_action",
            "fp_fn_cost_inr",
        ]
        print(errors[cols].to_string(index=False))

    total_error_inr = float(errors["amount"].sum()) if len(errors) else 0.0
    print("\n=== Rupee costs ===")
    print(f"Misclassified rows: {len(errors)}")
    print(
        f"FALSE-POSITIVE / FALSE-NEGATIVE COST (sum of amount on errors): "
        f"INR {total_error_inr:,.2f}"
    )

    print("\nPer-class false-negative cost (true class missed; sum of amount):")
    fn_rows = []
    for cls in labels:
        mask = (y_true == cls) & (y_pred != cls)
        fn_rows.append({"class": cls, "n": int(mask.sum()), "inr": float(holdout.loc[mask, "amount"].sum())})
    print(pd.DataFrame(fn_rows).to_string(index=False))

    print("\nPer-class false-positive cost (predicted this class incorrectly; sum of amount):")
    fp_rows = []
    for cls in labels:
        mask = (y_pred == cls) & (y_true != cls)
        fp_rows.append({"class": cls, "n": int(mask.sum()), "inr": float(holdout.loc[mask, "amount"].sum())})
    print(pd.DataFrame(fp_rows).to_string(index=False))

    gave_up = (y_pred == "PERMANENT_FAILURE") & (y_true != "PERMANENT_FAILURE")
    print(
        "\nRecoverable rupees written off "
        "(predicted PERMANENT_FAILURE but true class was recoverable): "
        f"INR {float(holdout.loc[gave_up, 'amount'].sum()):,.2f}"
    )


if __name__ == "__main__":
    if not MODEL_PATH.exists():
        raise SystemExit(f"Missing {MODEL_PATH}. Run train_classifier.py first.")
    if not HOLDOUT_PATH.exists():
        raise SystemExit(f"Missing {HOLDOUT_PATH}. Run train_classifier.py first.")
    main()
