"""Weight-of-Evidence (WoE) / Information Value (IV) binning.

Primary backend: optbinning.OptimalBinning (mixed-integer optimal binning with
monotonicity constraints). A lightweight quantile + monotone-merge backend is used
only if optbinning is not installed, so tests and smoke runs work everywhere.

Sign convention (same as optbinning / Siddiqi): WoE = ln(%goods / %bads), so bins with
a HIGHER default rate have a LOWER (negative) WoE.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MISSING_LABEL = "Missing"
OTHER_LABEL = "Other"


def iv_strength(iv: float) -> str:
    """Rule-of-thumb IV bands (Siddiqi, 2006). IV > 0.5 is flagged because in
    application data it usually signals leakage rather than a genuinely strong driver."""
    if iv < 0.02:
        return "unpredictive"
    if iv < 0.1:
        return "weak"
    if iv < 0.3:
        return "medium"
    if iv < 0.5:
        return "strong"
    return "suspicious (check leakage)"


def _woe_table(labels: np.ndarray, y: np.ndarray, order: list[str],
               smoothing: float = 0.5) -> pd.DataFrame:
    """Per-bin counts, event rate, WoE and IV. Additive smoothing avoids infinite WoE
    for bins with zero goods or bads."""
    df = pd.DataFrame({"bin": labels, "y": y})
    g = df.groupby("bin", observed=True)["y"].agg(["count", "sum"]).reindex(order).fillna(0)
    events, count = g["sum"].to_numpy(float), g["count"].to_numpy(float)
    nonevents = count - events
    k = len(order)
    p_bad = (events + smoothing) / (events.sum() + smoothing * k)
    p_good = (nonevents + smoothing) / (nonevents.sum() + smoothing * k)
    woe = np.log(p_good / p_bad)
    return pd.DataFrame({
        "bin": order,
        "count": count.astype(int),
        "count_pct": count / max(count.sum(), 1),
        "events": events.astype(int),
        "event_rate": np.divide(events, count, out=np.zeros_like(events), where=count > 0),
        "woe": woe,
        "iv": (p_good - p_bad) * woe,
    })


# --------------------------------------------------------------------------- backends
class _QuantileWoE:
    """Fallback binner. Numeric: quantile pre-bins, then adjacent bins are merged until
    every bin holds >= min_bin_size of the sample and event rates are monotone (the same
    constraints optbinning enforces, solved greedily rather than optimally).
    Categorical: one bin per category, rare categories pooled into 'Other'."""

    def __init__(self, name: str, dtype: str, max_n_bins: int = 8, min_bin_size: float = 0.05):
        self.name, self.dtype = name, dtype
        self.max_n_bins, self.min_bin_size = max_n_bins, min_bin_size

    # numeric helpers
    def _num_labels(self, x: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(self.splits_, x, side="right")
        bounds = np.concatenate([[-np.inf], self.splits_, [np.inf]])
        names = np.array([f"[{bounds[i]:.8g}, {bounds[i + 1]:.8g})" for i in range(len(bounds) - 1)]
                         + [MISSING_LABEL], dtype=object)
        idx = np.where(np.isnan(x), len(names) - 1, idx)
        return names[idx]

    def _fit_splits(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        qs = np.linspace(0, 1, self.max_n_bins + 1)[1:-1]
        edges = list(np.unique(np.quantile(x, qs)))
        min_count = self.min_bin_size * len(x)
        while edges:
            idx = np.searchsorted(edges, x, side="right")
            n = len(edges) + 1
            cnt = np.bincount(idx, minlength=n).astype(float)
            ev = np.bincount(idx, weights=y, minlength=n)
            small = np.flatnonzero(cnt < min_count)
            if small.size:
                i = small[0]
                edges.pop(i if i < n - 1 else i - 1)  # merge into right (or left if last)
                continue
            rate = ev / cnt
            trend = np.sign(rate[-1] - rate[0]) or 1.0
            viol = np.flatnonzero(np.diff(rate) * trend < 0)
            if not viol.size:
                break
            edges.pop(viol[0])  # merge the first non-monotone adjacent pair
        return np.asarray(edges, dtype=float)

    def fit(self, x: pd.Series, y: np.ndarray) -> "_QuantileWoE":
        y = np.asarray(y, dtype=float)
        if self.dtype == "numerical":
            xv = pd.to_numeric(x, errors="coerce").to_numpy(float)
            ok = ~np.isnan(xv)
            self.splits_ = self._fit_splits(xv[ok], y[ok]) if ok.any() else np.array([])
            labels = self._num_labels(xv)
            order = list(pd.unique(self._num_labels(np.sort(xv[ok])))) if ok.any() else []
            if (~ok).any():
                order.append(MISSING_LABEL)
        else:
            xs = x.astype("string").fillna(MISSING_LABEL)
            freq = xs.value_counts(normalize=True)
            self.categories_ = sorted(freq[freq >= self.min_bin_size].index.tolist())
            labels = xs.where(xs.isin(self.categories_), OTHER_LABEL).to_numpy(object)
            order = self.categories_ + ([OTHER_LABEL] if (labels == OTHER_LABEL).any() else [])
        self.table_ = _woe_table(labels, y, order)
        self._woe_map = dict(zip(self.table_["bin"], self.table_["woe"]))
        return self

    def transform(self, x: pd.Series, metric: str = "woe") -> np.ndarray:
        if self.dtype == "numerical":
            labels = self._num_labels(pd.to_numeric(x, errors="coerce").to_numpy(float))
        else:
            xs = x.astype("string").fillna(MISSING_LABEL)
            labels = xs.where(xs.isin(self.categories_), OTHER_LABEL).to_numpy(object)
        if metric == "bins":
            return labels
        # Unseen bins (e.g. Missing never observed in training) get WoE 0 = neutral evidence.
        return pd.Series(labels).map(self._woe_map).fillna(0.0).to_numpy(float)

    @property
    def table(self) -> pd.DataFrame:
        return self.table_


class _OptbinningWoE:
    """Thin adapter over optbinning.OptimalBinning exposing the same interface."""

    def __init__(self, name: str, dtype: str, max_n_bins: int = 8, min_bin_size: float = 0.05,
                 solver: str = "cp"):
        from optbinning import OptimalBinning

        # auto_asc_desc: numeric bins are forced monotone (ascending or descending,
        # whichever fits better) - required for a defensible, explainable scorecard.
        trend = "auto_asc_desc" if dtype == "numerical" else None
        self.name, self.dtype = name, dtype
        self._ob = OptimalBinning(name=name, dtype=dtype, solver=solver, max_n_bins=max_n_bins,
                                  min_bin_size=min_bin_size, monotonic_trend=trend,
                                  cat_cutoff=0.01 if dtype == "categorical" else None)

    def fit(self, x: pd.Series, y: np.ndarray) -> "_OptbinningWoE":
        xv = x.astype("string").fillna(MISSING_LABEL).to_numpy(object) \
            if self.dtype == "categorical" else pd.to_numeric(x, errors="coerce").to_numpy(float)
        self._ob.fit(xv, np.asarray(y, dtype=int))
        bt = self._ob.binning_table.build()
        bt = bt.iloc[:-1]  # drop "Totals"
        self.table_ = pd.DataFrame({
            "bin": bt["Bin"].astype(str).to_numpy(),
            "count": bt["Count"].astype(int).to_numpy(),
            "count_pct": bt["Count (%)"].astype(float).to_numpy(),
            "events": bt["Event"].astype(int).to_numpy(),
            "event_rate": bt["Event rate"].astype(float).to_numpy(),
            "woe": bt["WoE"].astype(float).to_numpy(),
            "iv": bt["IV"].astype(float).to_numpy(),
        })
        # Empty Special/Missing rows carry no information; drop them from the report table.
        self.table_ = self.table_[self.table_["count"] > 0].reset_index(drop=True)
        return self

    def transform(self, x: pd.Series, metric: str = "woe") -> np.ndarray:
        xv = x.astype("string").fillna(MISSING_LABEL).to_numpy(object) \
            if self.dtype == "categorical" else pd.to_numeric(x, errors="coerce").to_numpy(float)
        if metric == "bins":
            return np.asarray(self._ob.transform(xv, metric="bins"), dtype=object)
        return np.asarray(self._ob.transform(xv, metric="woe"), dtype=float)

    @property
    def table(self) -> pd.DataFrame:
        return self.table_


def _optbinning_available() -> bool:
    try:
        import optbinning  # noqa: F401
        return True
    except ImportError:
        return False


# --------------------------------------------------------------------------- public API
@dataclass
class WoEBinner:
    """Fit one WoE binning per feature, rank by IV, select IV > iv_min."""

    numeric: list[str]
    categorical: list[str]
    backend: str = "optbinning"
    max_n_bins: int = 8
    min_bin_size: float = 0.05
    iv_min: float = 0.02
    binnings_: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        if self.backend == "optbinning" and not _optbinning_available():
            logger.warning("optbinning not installed - falling back to quantile WoE binning. "
                           "Install optbinning for optimal (MIP/CP) binning.")
            self.backend = "quantile"

    def _make(self, name: str, dtype: str):
        cls = _OptbinningWoE if self.backend == "optbinning" else _QuantileWoE
        return cls(name, dtype, max_n_bins=self.max_n_bins, min_bin_size=self.min_bin_size)

    def fit(self, X: pd.DataFrame, y) -> "WoEBinner":
        y = np.asarray(y)
        for name in self.numeric + self.categorical:
            dtype = "numerical" if name in self.numeric else "categorical"
            try:
                self.binnings_[name] = self._make(name, dtype).fit(X[name], y)
            except Exception as exc:  # one bad column should not kill the run
                logger.warning("Binning failed for %s (%s) - feature skipped", name, exc)
        return self

    def iv_table(self) -> pd.DataFrame:
        rows = [{"feature": n, "iv": float(b.table["iv"].sum()), "n_bins": len(b.table)}
                for n, b in self.binnings_.items()]
        t = pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)
        t["strength"] = t["iv"].map(iv_strength)
        t["selected"] = t["iv"] > self.iv_min
        return t

    @property
    def selected_features(self) -> list[str]:
        t = self.iv_table()
        return t.loc[t["selected"], "feature"].tolist()

    def binning_table(self, feature: str) -> pd.DataFrame:
        return self.binnings_[feature].table.copy()

    def transform(self, X: pd.DataFrame, features: list[str] | None = None,
                  metric: str = "woe") -> pd.DataFrame:
        features = features or self.selected_features
        return pd.DataFrame({f: self.binnings_[f].transform(X[f], metric=metric)
                             for f in features}, index=X.index)
