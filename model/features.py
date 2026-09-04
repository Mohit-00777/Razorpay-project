"""Turn raw payment-failure rows into model-ready numeric features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

CATEGORICAL_COLUMNS = ["failure_code", "payment_method", "bank_name"]
AMOUNT_BIN_EDGES = [0, 500, 2000, 5000, 15000, np.inf]
AMOUNT_BIN_LABELS = ["0-500", "500-2000", "2000-5000", "5000-15000", "15000+"]
AMOUNT_BUCKET_COL = "amount_bucket"


def bucket_amount(amount: pd.Series) -> pd.Series:
    return pd.cut(
        amount,
        bins=AMOUNT_BIN_EDGES,
        labels=AMOUNT_BIN_LABELS,
        right=True,
        include_lowest=True,
    ).astype(str)


@dataclass
class FeatureTransformer:
    """Fit one-hot encoders on train only; reuse on holdout."""

    encoder: OneHotEncoder | None = None
    feature_names_: list[str] | None = None

    def fit(self, df: pd.DataFrame) -> FeatureTransformer:
        cats = self._categorical_frame(df)
        self.encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            dtype=np.float64,
        )
        self.encoder.fit(cats)
        ohe_names = list(self.encoder.get_feature_names_out(cats.columns))
        self.feature_names_ = ohe_names + ["retry_count_so_far"]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.encoder is None or self.feature_names_ is None:
            raise RuntimeError("FeatureTransformer must be fit() before transform().")
        cats = self._categorical_frame(df)
        encoded = self.encoder.transform(cats)
        ohe_names = list(self.encoder.get_feature_names_out(cats.columns))
        out = pd.DataFrame(encoded, columns=ohe_names, index=df.index)
        out["retry_count_so_far"] = pd.to_numeric(
            df["retry_count_so_far"], errors="raise"
        ).astype(float)
        return out[self.feature_names_]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    @staticmethod
    def _categorical_frame(df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in CATEGORICAL_COLUMNS + ["amount"] if c not in df.columns]
        if missing:
            raise KeyError(f"Missing columns for features: {missing}")
        frame = df[CATEGORICAL_COLUMNS].astype(str).copy()
        frame[AMOUNT_BUCKET_COL] = bucket_amount(df["amount"])
        return frame
