"""Challenger model: gradient-boosted trees (XGBoost) on raw (un-binned) features.

Purpose in the validation framework: quantify how much discriminatory power the
constrained, additive scorecard leaves on the table (non-linearities, interactions).
If the lift is small the simpler champion is preferred; if large it is evidence the
scorecard should be redeveloped.
"""
from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = dict(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=4,            # shallow trees: captures 2-4 way interactions, resists overfit
    min_child_weight=50,    # large leaves -> stabler probabilities on a 300K+ sample
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    tree_method="hist",
)


def _safe(name: str) -> str:
    # XGBoost rejects '[', ']' and '<' in feature names.
    return re.sub(r"[\[\]<>,]", "_", name)


class XGBChallenger:
    def __init__(self, numeric: list[str], categorical: list[str], params: dict | None = None,
                 early_stopping_rounds: int = 50, valid_size: float = 0.2,
                 random_state: int = 42):
        self.numeric, self.categorical = list(numeric), list(categorical)
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.early_stopping_rounds = early_stopping_rounds
        self.valid_size, self.random_state = valid_size, random_state

    def _encode(self, X: pd.DataFrame) -> pd.DataFrame:
        """Numerics passed through with NaN intact (XGBoost learns a default split
        direction for missing values - no imputation needed). Categoricals are one-hot
        encoded against the category set learned at fit time, so unseen levels map to
        all-zeros. One-hot (rather than native categorical splits) keeps SHAP support
        simple and robust across library versions."""
        parts = [X[self.numeric].apply(pd.to_numeric, errors="coerce").astype(float)]
        for c in self.categorical:
            cat = pd.Categorical(X[c].astype("string").fillna("Missing"),
                                 categories=self.categories_[c])
            d = pd.get_dummies(cat, prefix=c, prefix_sep="=", dtype=float)
            d.index = X.index
            parts.append(d)
        enc = pd.concat(parts, axis=1)
        enc.columns = [_safe(c) for c in enc.columns]
        return enc

    @property
    def feature_groups(self) -> dict[str, str]:
        """Encoded column -> original feature (to aggregate one-hot SHAP values)."""
        groups = {_safe(c): c for c in self.numeric}
        for c in self.categorical:
            for level in self.categories_[c]:
                groups[_safe(f"{c}={level}")] = c
        return groups

    def fit(self, X: pd.DataFrame, y) -> "XGBChallenger":
        from xgboost import XGBClassifier

        self.categories_ = {c: sorted(X[c].astype("string").fillna("Missing").unique().tolist())
                            for c in self.categorical}
        Xe, y = self._encode(X), np.asarray(y)
        # Early-stopping set is carved out of the TRAIN window only; the OOT 2016
        # sample is never touched during fitting or tuning.
        X_tr, X_va, y_tr, y_va = train_test_split(Xe, y, test_size=self.valid_size,
                                                  stratify=y, random_state=self.random_state)
        # No scale_pos_weight / class re-weighting: we need calibrated PDs for EL and
        # capital, not just a good ranking.
        self.model_ = XGBClassifier(**self.params, objective="binary:logistic",
                                    eval_metric="auc",
                                    early_stopping_rounds=self.early_stopping_rounds,
                                    random_state=self.random_state, n_jobs=-1)
        self.model_.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        self.columns_ = list(Xe.columns)
        logger.info("XGBoost best iteration: %s", getattr(self.model_, "best_iteration", "n/a"))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._encode(X).reindex(columns=self.columns_, fill_value=0.0)

    def predict_pd(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(self.transform(X))[:, 1]
