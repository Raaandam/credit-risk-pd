"""Champion model: logistic regression on WoE features, scaled to scorecard points.

Why logistic regression on WoE?
* WoE linearises each driver's relationship with log-odds, so one coefficient per
  feature is enough and the model is additive -> every applicant's score decomposes
  into per-attribute points (the standard retail-credit / adverse-action format).
* Coefficients are few and stable, which matters for SR 11-7 conceptual soundness.

Scaling: score = offset + factor * ln(good:bad odds), factor = PDO / ln 2,
offset = base_score - factor * ln(base_odds). With PDO 20 / 600 / 50:1, a score of 600
means 50:1 odds and every +20 points doubles the odds.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .features import WoEBinner

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScorecardScaling:
    pdo: float = 20.0
    base_score: float = 600.0
    base_odds: float = 50.0

    @property
    def factor(self) -> float:
        return self.pdo / np.log(2.0)

    @property
    def offset(self) -> float:
        return self.base_score - self.factor * np.log(self.base_odds)

    def score_from_pd(self, pd_: np.ndarray) -> np.ndarray:
        pd_ = np.clip(np.asarray(pd_, dtype=float), 1e-12, 1 - 1e-12)
        return self.offset + self.factor * np.log((1 - pd_) / pd_)

    def pd_from_score(self, score: np.ndarray) -> np.ndarray:
        odds = np.exp((np.asarray(score, dtype=float) - self.offset) / self.factor)
        return 1.0 / (1.0 + odds)


class LogisticScorecard:
    def __init__(self, binner: WoEBinner, scaling: ScorecardScaling = ScorecardScaling(),
                 C: float = 1.0, corr_max: float = 0.7, max_iter: int = 1000):
        self.binner, self.scaling = binner, scaling
        self.C, self.corr_max, self.max_iter = C, corr_max, max_iter
        self.dropped_: dict[str, str] = {}

    # ------------------------------------------------------------------ selection
    def _correlation_filter(self, woe: pd.DataFrame, iv: pd.Series) -> list[str]:
        """Among highly correlated WoE pairs keep the higher-IV one: collinearity makes
        coefficients unstable and can flip their sign."""
        keep: list[str] = []
        corr = woe.corr().abs()
        for f in iv.sort_values(ascending=False).index:
            clash = [k for k in keep if corr.loc[f, k] > self.corr_max]
            if clash:
                self.dropped_[f] = f"|corr| > {self.corr_max} with {clash[0]}"
            else:
                keep.append(f)
        return keep

    def fit(self, X: pd.DataFrame, y) -> "LogisticScorecard":
        y = np.asarray(y)
        candidates = self.binner.selected_features
        woe = self.binner.transform(X, candidates)
        iv = self.binner.iv_table().set_index("feature")["iv"].loc[candidates]
        features = self._correlation_filter(woe, iv)

        # Sign check: each coefficient must have the same sign as the feature's
        # univariate relation with default; otherwise points would reward risk in a
        # way no business user could defend. Drop the offender and refit.
        univariate_sign = {f: np.sign(np.corrcoef(woe[f], y)[0, 1]) for f in features}
        while True:
            model = LogisticRegression(C=self.C, max_iter=self.max_iter, solver="lbfgs")
            model.fit(woe[features], y)
            coefs = dict(zip(features, model.coef_[0]))
            wrong = [f for f in features if np.sign(coefs[f]) != univariate_sign[f]]
            if not wrong or len(features) == 1:
                break
            worst = min(wrong, key=lambda f: iv[f])
            self.dropped_[worst] = "coefficient sign reversed vs univariate WoE relation"
            features.remove(worst)

        self.features_, self.model_ = features, model
        self.intercept_ = float(model.intercept_[0])
        self.coef_ = pd.Series(model.coef_[0], index=features)
        logger.info("Scorecard fitted on %d features (dropped %d)", len(features), len(self.dropped_))
        return self

    # ------------------------------------------------------------------ scoring
    def _log_odds_bad(self, X: pd.DataFrame) -> np.ndarray:
        woe = self.binner.transform(X, self.features_)
        return self.intercept_ + woe.to_numpy() @ self.coef_.to_numpy()

    def predict_pd(self, X: pd.DataFrame) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self._log_odds_bad(X)))

    def score(self, X: pd.DataFrame) -> np.ndarray:
        # ln(good:bad) = -ln(bad:good) = -logit(PD)
        return self.scaling.offset - self.scaling.factor * self._log_odds_bad(X)

    def _attribute_points(self, woe: np.ndarray, feature: str) -> np.ndarray:
        """Points for one attribute: intercept and offset are spread evenly across features
        so that the points of an applicant's attributes sum exactly to the score."""
        m = len(self.features_)
        s = self.scaling
        return (-s.factor * self.coef_[feature] * woe
                + (s.offset - s.factor * self.intercept_) / m)

    def points(self, X: pd.DataFrame) -> pd.DataFrame:
        woe = self.binner.transform(X, self.features_)
        return pd.DataFrame({f: self._attribute_points(woe[f].to_numpy(), f)
                             for f in self.features_}, index=X.index)

    def points_table(self) -> pd.DataFrame:
        rows = []
        for f in self.features_:
            t = self.binner.binning_table(f)
            t.insert(0, "feature", f)
            t["coefficient"] = self.coef_[f]
            t["points"] = self._attribute_points(t["woe"].to_numpy(), f).round(1)
            rows.append(t[["feature", "bin", "count", "event_rate", "woe", "coefficient", "points"]])
        return pd.concat(rows, ignore_index=True)

    def coefficients(self) -> pd.DataFrame:
        iv = self.binner.iv_table().set_index("feature")["iv"]
        return pd.DataFrame({"feature": ["(intercept)"] + self.features_,
                             "coefficient": [self.intercept_] + self.coef_.tolist(),
                             "iv": [np.nan] + [iv[f] for f in self.features_]})

    def reason_codes(self, X: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
        """Adverse-action style reasons: features where the applicant lost the most points
        versus the best attainable attribute for that feature."""
        pts = self.points(X)
        best = self.points_table().groupby("feature")["points"].max()
        shortfall = best[pts.columns].to_numpy() - pts.to_numpy()
        order = np.argsort(-shortfall, axis=1)[:, :top_n]
        cols = np.array(pts.columns)
        return pd.DataFrame({f"reason_{i + 1}": cols[order[:, i]] for i in range(order.shape[1])},
                            index=X.index)
