"""Train an XGBoost root-cause classifier and persist the holdout set."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import FeatureTransformer  # noqa: E402

DATA_PATH = ROOT / "data" / "raw" / "failures_batch.csv"
HOLDOUT_PATH = ROOT / "data" / "raw" / "holdout_set.csv"
MODEL_PATH = Path(__file__).resolve().parent / "classifier.pkl"
TARGET = "true_root_cause_category"
RANDOM_SEED = 42
TEST_SIZE = 0.20


def load_bundle():
    return joblib.load(MODEL_PATH)


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    if TARGET not in df.columns:
        raise KeyError(f"{DATA_PATH} missing {TARGET}")

    train_df, holdout_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=df[TARGET],
    )
    train_df = train_df.reset_index(drop=True)
    holdout_df = holdout_df.reset_index(drop=True)

    transformer = FeatureTransformer().fit(train_df)
    X_train = transformer.transform(train_df)
    y_encoder = LabelEncoder()
    y_train = y_encoder.fit_transform(train_df[TARGET])

    clf = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=len(y_encoder.classes_),
        n_estimators=200,
        max_depth=3,
        learning_rate=0.08,
        min_child_weight=2,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=RANDOM_SEED,
        n_jobs=1,
    )
    clf.fit(X_train, y_train)

    HOLDOUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    holdout_df.to_csv(HOLDOUT_PATH, index=False)

    bundle = {
        "model": clf,
        "transformer": transformer,
        "label_encoder": y_encoder,
        "feature_names": list(X_train.columns),
        "target": TARGET,
        "random_seed": RANDOM_SEED,
    }
    joblib.dump(bundle, MODEL_PATH)

    train_pred = y_encoder.inverse_transform(clf.predict(X_train))
    train_acc = (train_pred == train_df[TARGET]).mean()
    print(f"Train rows: {len(train_df)}  Holdout rows: {len(holdout_df)}")
    print(f"Train accuracy: {train_acc:.3f}")
    print(f"Saved model -> {MODEL_PATH}")
    print(f"Saved holdout -> {HOLDOUT_PATH}")
    print("Holdout class counts:")
    print(holdout_df[TARGET].value_counts().to_string())


if __name__ == "__main__":
    main()
