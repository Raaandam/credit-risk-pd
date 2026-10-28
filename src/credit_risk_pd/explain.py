"""SHAP explanations for the challenger, plus responsible-AI checks.

SHAP (TreeSHAP) gives an exact additive decomposition of each prediction's log-odds into
per-feature contributions, so the same quantity serves as a global importance
(mean |SHAP|) and a local explanation (one applicant's contributions).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .validation import auc

logger = logging.getLogger(__name__)

# Fields deliberately kept OUT of both models, with the rationale documented for review.
EXCLUDED_FEATURES = {
    "addr_state": "Geography is a well-known proxy for race/ethnicity (redlining risk); "
                  "kept only as a monitoring dimension for disparate-impact checks.",
    "zip_code": "Same proxy concern as addr_state, at finer granularity.",
    "emp_title": "Free text; can encode gender/ethnicity and is not consistently validated.",
    "desc / title": "Borrower free text; proxy risk and not available for all applicants.",
    "sub_grade": "Redundant with grade/int_rate; would double-count the lender's own view.",
    "post-origination fields": "Payments, recoveries, last_fico_*, hardship/settlement "
                               "flags: unknown at application time (target leakage).",
}

# US Census regions, used to aggregate addr_state into groups large enough to test.
_REGIONS = {
    "Northeast": "CT ME MA NH RI VT NJ NY PA",
    "Midwest": "IL IN MI OH WI IA KS MN MO NE ND SD",
    "South": "DE FL GA MD NC SC VA DC WV AL KY MS TN AR LA OK TX",
    "West": "AZ CO ID MT NV NM UT WY AK CA HI OR WA",
}
STATE_TO_REGION = {s: r for r, states in _REGIONS.items() for s in states.split()}


# --------------------------------------------------------------------------- SHAP
def shap_values(challenger, X: pd.DataFrame, sample_size: int = 5000,
                seed: int = 42) -> tuple[pd.DataFrame, float, pd.DataFrame]:
    """Return (SHAP values per ORIGINAL feature, base value, sampled raw rows).

    Uses the shap library when available; otherwise XGBoost's built-in TreeSHAP
    (pred_contribs=True), which computes the same quantity."""
    Xs = X.sample(min(sample_size, len(X)), random_state=seed)
    Xe = challenger.transform(Xs)
    try:
        import shap

        explainer = shap.TreeExplainer(challenger.model_)
        vals = np.asarray(explainer.shap_values(Xe))
        base = float(np.ravel(explainer.expected_value)[0])
    except Exception as exc:  # ImportError or a shap/xgboost version mismatch
        logger.warning("shap unavailable (%s); using XGBoost pred_contribs TreeSHAP", exc)
        import xgboost as xgb

        contrib = challenger.model_.get_booster().predict(xgb.DMatrix(Xe), pred_contribs=True)
        vals, base = contrib[:, :-1], float(contrib[0, -1])
    enc = pd.DataFrame(vals, columns=Xe.columns, index=Xs.index)
    # SHAP values are additive, so one-hot columns can be summed back to their feature.
    agg = enc.T.groupby(pd.Series(challenger.feature_groups)).sum().T
    return agg, base, Xs


def global_importance(shap_df: pd.DataFrame) -> pd.DataFrame:
    imp = shap_df.abs().mean().sort_values(ascending=False)
    # Direction summary that also works for categoricals: share of loans for which the
    # feature pushed predicted risk UP.
    up = (shap_df > 0).mean()
    return pd.DataFrame({"feature": imp.index, "mean_abs_shap": imp.values,
                         "share_increasing_risk": up[imp.index].values})


def local_explanations(shap_df: pd.DataFrame, X: pd.DataFrame, pd_pred: np.ndarray,
                       top_n: int = 5) -> pd.DataFrame:
    """Top drivers for three reference applicants: highest, median and lowest PD."""
    s = pd.Series(pd_pred, index=shap_df.index).sort_values()
    picks = {"highest PD": s.index[-1], "median PD": s.index[len(s) // 2], "lowest PD": s.index[0]}
    rows = []
    for label, idx in picks.items():
        contrib = shap_df.loc[idx].reindex(shap_df.loc[idx].abs().sort_values(ascending=False).index)
        for f, v in contrib.head(top_n).items():
            rows.append({"applicant": label, "pd": float(s[idx]), "feature": f,
                         "value": X.loc[idx, f] if f in X.columns else None,
                         "shap_log_odds": float(v)})
    return pd.DataFrame(rows)


def save_shap_summary_plot(shap_df: pd.DataFrame, path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    imp = shap_df.abs().mean().sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(imp.index, imp.values, color="#3b6ea5")
    ax.set_xlabel("mean |SHAP| (log-odds)")
    ax.set_title("XGBoost challenger - global SHAP importance")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------- fairness
def fairness_by_group(df: pd.DataFrame, group: pd.Series, y, pd_pred,
                      approval_rate: float = 0.8) -> pd.DataFrame:
    """Per-group calibration, discrimination and approval outcomes at a common cut-off.

    The cut-off approves the lowest-PD `approval_rate` share of the whole population.
    Adverse impact ratio = group approval rate / highest group approval rate; < 0.8 fails
    the 'four-fifths' screening rule and warrants investigation (not proof of bias)."""
    d = pd.DataFrame({"group": np.asarray(group), "y": np.asarray(y), "p": np.asarray(pd_pred)})
    cutoff = np.quantile(d["p"], approval_rate)
    d["approved"] = d["p"] <= cutoff
    rows = []
    for g, sub in d.groupby("group"):
        rows.append({"group": g, "n": len(sub), "observed_dr": sub["y"].mean(),
                     "mean_pd": sub["p"].mean(),
                     "auc": auc(sub["y"], sub["p"]) if sub["y"].nunique() == 2 else np.nan,
                     "approval_rate": sub["approved"].mean()})
    t = pd.DataFrame(rows)
    t["calibration_ratio"] = t["mean_pd"] / t["observed_dr"]
    t["adverse_impact_ratio"] = t["approval_rate"] / t["approval_rate"].max()
    t["four_fifths_flag"] = t["adverse_impact_ratio"] < 0.8
    return t.sort_values("n", ascending=False).reset_index(drop=True)


def responsible_ai_note(fairness_tables: dict[str, pd.DataFrame]) -> str:
    lines = ["**Deliberately excluded features**", ""]
    lines += [f"- `{k}`: {v}" for k, v in EXCLUDED_FEATURES.items()]
    lines += ["", "**Fairness / proxy checks** (Lending Club data carries no protected "
              "attributes, so checks use available proxies; a production review would use "
              "BISG-style inference or self-reported data):", ""]
    for name, t in fairness_tables.items():
        flagged = t.loc[t["four_fifths_flag"], "group"].astype(str).tolist()
        cal = t["calibration_ratio"]
        lines.append(f"- By {name}: groups failing four-fifths rule: "
                     f"{', '.join(flagged) if flagged else 'none'}; calibration ratio "
                     f"(mean PD / observed DR) ranges {cal.min():.2f}-{cal.max():.2f}.")
    lines += ["", "**Explainability**: every scorecard decision decomposes into attribute "
              "points (reason codes = largest point shortfalls); challenger decisions are "
              "explained with TreeSHAP. The challenger is a benchmark and is not used for "
              "decisions without a separate explainability sign-off."]
    return "\n".join(lines)
