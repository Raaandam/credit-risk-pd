"""Discrimination, calibration and stability metrics used in model validation.

* AUC / Gini (= 2*AUC - 1): rank-ordering power. Gini is the industry convention.
* KS: max separation between the score CDFs of defaulters and non-defaulters.
* Brier score + calibration table: are predicted PDs right in level, not just rank?
* PSI: population stability of the score distribution (train vs OOT / monitoring).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

_EPS = 1e-6


def auc(y, p) -> float:
    return float(roc_auc_score(y, p))


def gini(y, p) -> float:
    return 2.0 * auc(y, p) - 1.0


def ks_statistic(y, p) -> float:
    """KS = max_t |F_bad(t) - F_good(t)| = max(TPR - FPR) along the ROC curve."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def brier(y, p) -> float:
    return float(brier_score_loss(y, p))


def calibration_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed default rate by PD quantile bin (reliability diagram data)."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), n_bins, labels=False) + 1
    t = df.groupby("bin").agg(n=("y", "size"), mean_pd=("p", "mean"), observed_dr=("y", "mean"))
    t["ratio_obs_to_pred"] = t["observed_dr"] / t["mean_pd"]
    return t.reset_index()


def performance_summary(y, p) -> dict:
    return {"n": int(len(y)), "default_rate": float(np.mean(y)), "mean_pd": float(np.mean(p)),
            "auc": auc(y, p), "gini": gini(y, p), "ks": ks_statistic(y, p), "brier": brier(y, p)}


# --------------------------------------------------------------------------- stability
def psi_from_distributions(expected_pct, actual_pct) -> float:
    e = np.clip(np.asarray(expected_pct, dtype=float), _EPS, None)
    a = np.clip(np.asarray(actual_pct, dtype=float), _EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi(expected, actual, n_bins: int = 10) -> float:
    """Population Stability Index with bins = deciles of the EXPECTED (development)
    distribution. Rule of thumb: <0.10 stable, 0.10-0.25 monitor, >0.25 significant shift.
    Quantile bins make PSI invariant to monotone transforms (PD vs points give same PSI)."""
    expected, actual = np.asarray(expected, dtype=float), np.asarray(actual, dtype=float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)[1:-1]))
    e_idx = np.searchsorted(edges, expected, side="right")
    a_idx = np.searchsorted(edges, actual, side="right")
    k = len(edges) + 1
    e_pct = np.bincount(e_idx, minlength=k) / len(expected)
    a_pct = np.bincount(a_idx, minlength=k) / len(actual)
    return psi_from_distributions(e_pct, a_pct)


def psi_categorical(expected, actual) -> float:
    """PSI over discrete labels (e.g. WoE bins) - used for characteristic stability (CSI)."""
    e = pd.Series(expected).astype(str).value_counts(normalize=True)
    a = pd.Series(actual).astype(str).value_counts(normalize=True)
    idx = e.index.union(a.index)
    return psi_from_distributions(e.reindex(idx, fill_value=0), a.reindex(idx, fill_value=0))


def psi_status(value: float, warn: float = 0.10, trigger: float = 0.25) -> str:
    return "RED" if value > trigger else ("AMBER" if value > warn else "GREEN")


# --------------------------------------------------------------------------- comparison
def bootstrap_gini_difference(y, p_champion, p_challenger, n_boot: int = 200,
                              seed: int = 42) -> dict:
    """Paired bootstrap of Gini(challenger) - Gini(champion) on the same resampled loans,
    so we can say whether the challenger lift is statistically distinguishable from 0."""
    y, a, b = map(np.asarray, (y, p_champion, p_challenger))
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        diffs[i] = gini(y[idx], b[idx]) - gini(y[idx], a[idx])
    return {"gini_diff": gini(y, b) - gini(y, a), "ci_low": float(np.quantile(diffs, 0.025)),
            "ci_high": float(np.quantile(diffs, 0.975)), "n_boot": n_boot}


def model_comparison(results: dict[str, dict]) -> pd.DataFrame:
    """results: {"Scorecard / OOT": performance_summary(...), ...} -> tidy table."""
    return pd.DataFrame([{"model / sample": k, **v} for k, v in results.items()])


def monitoring_status(psi_value: float, gini_dev: float, gini_oot: float,
                      psi_warn: float = 0.10, psi_trigger: float = 0.25,
                      gini_drop_trigger_pts: float = 5.0) -> pd.DataFrame:
    drop_pts = (gini_dev - gini_oot) * 100
    return pd.DataFrame([
        {"metric": "Score PSI (train vs OOT)", "value": round(psi_value, 4),
         "trigger": f"> {psi_trigger} (amber > {psi_warn})",
         "status": psi_status(psi_value, psi_warn, psi_trigger)},
        {"metric": "Gini drop dev -> OOT (pts)", "value": round(drop_pts, 2),
         "trigger": f"> {gini_drop_trigger_pts} pts",
         "status": "RED" if drop_pts > gini_drop_trigger_pts else "GREEN"},
    ])
