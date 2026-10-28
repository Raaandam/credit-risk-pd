"""Back-testing: predicted PD vs observed default rate by score band on the OOT sample.

Tests (one-sided, H0: the band's PD is correct, H1: PD is UNDER-estimated, which is the
direction that understates capital):
* Binomial test: p = P(D >= d | n, PD).
* Jeffreys test (ECB/EBA IRB validation guidance): posterior Beta(d + 1/2, n - d + 1/2)
  under a Jeffreys prior; p = P(true DR <= PD | data). Less conservative than the exact
  binomial for small bands and gives a natural credible interval.

Caveat: both assume independent defaults. With asset correlation the true dispersion of
default rates is larger, so these tests reject too often in a downturn year - a red band
should trigger investigation, not automatic recalibration.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import beta, binom


def binomial_test(n: int, d: int, pd_: float) -> float:
    return float(binom.sf(d - 1, n, pd_))


def jeffreys_test(n: int, d: int, pd_: float) -> float:
    return float(beta.cdf(pd_, d + 0.5, n - d + 0.5))


def jeffreys_interval(n: int, d: int, level: float = 0.95) -> tuple[float, float]:
    a = (1 - level) / 2
    return float(beta.ppf(a, d + 0.5, n - d + 0.5)), float(beta.ppf(1 - a, d + 0.5, n - d + 0.5))


def traffic_light(p_value: float, amber: float = 0.05, red: float = 0.01) -> str:
    return "RED" if p_value < red else ("AMBER" if p_value < amber else "GREEN")


def band_edges(dev_scores, n_bands: int = 10) -> np.ndarray:
    """Score-band cut points fixed on the DEVELOPMENT sample, as in production, so the
    OOT back-test is not re-optimised on the data it is testing."""
    return np.unique(np.quantile(np.asarray(dev_scores), np.linspace(0, 1, n_bands + 1)[1:-1]))


def backtest_by_band(y, pd_pred, scores, edges) -> pd.DataFrame:
    d = pd.DataFrame({"y": np.asarray(y), "pd": np.asarray(pd_pred), "score": np.asarray(scores)})
    # Band 1 = lowest score = riskiest.
    d["band"] = np.searchsorted(edges, d["score"], side="right") + 1
    bounds = np.concatenate([[-np.inf], edges, [np.inf]])
    rows = []
    groups = list(d.groupby("band")) + [("All", d)]
    for band, g in groups:
        n, dft, p = len(g), int(g["y"].sum()), float(g["pd"].mean())
        lo, hi = jeffreys_interval(n, dft)
        p_bin, p_jef = binomial_test(n, dft, p), jeffreys_test(n, dft, p)
        rng = "all" if band == "All" else f"[{bounds[band - 1]:.0f}, {bounds[band]:.0f})"
        rows.append({"band": band, "score_range": rng, "n": n, "defaults": dft,
                     "predicted_pd": p, "observed_dr": dft / n, "jeffreys_ci_low": lo,
                     "jeffreys_ci_high": hi, "binomial_p": p_bin, "jeffreys_p": p_jef,
                     "result": traffic_light(p_jef)})
    return pd.DataFrame(rows)


def backtest_summary(t: pd.DataFrame) -> dict:
    bands = t[t["band"] != "All"]
    return {"n_bands": len(bands), "green": int((bands["result"] == "GREEN").sum()),
            "amber": int((bands["result"] == "AMBER").sum()),
            "red": int((bands["result"] == "RED").sum()),
            "portfolio_result": t.loc[t["band"] == "All", "result"].iloc[0]}
