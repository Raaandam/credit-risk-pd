"""LGD, EAD and Expected Loss for the loan book.

EL_i = PD_i x LGD_i x EAD_i, aggregated by grade and by purpose (purpose plays the role
of 'sector' for concentration and stress analysis).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import TARGET


def empirical_lgd(defaults: pd.DataFrame, net_of_fees: bool = False) -> pd.DataFrame:
    """Workout LGD per defaulted loan: 1 - recoveries / outstanding principal at default.

    Outstanding at default = funded_amnt - principal repaid before charge-off. Accrued
    interest and collection costs are ignored (principal-only LGD); net_of_fees subtracts
    the collection agency fee from recoveries for an economic-LGD variant. LGD is
    clipped to [0, 1] (recoveries can marginally exceed principal due to fees/interest)."""
    d = defaults.loc[defaults[TARGET] == 1].copy()
    d["ead_at_default"] = d["funded_amnt"] - d["outcome_total_rec_prncp"]
    d = d.loc[d["ead_at_default"] > 0]
    rec = d["outcome_recoveries"].fillna(0.0)
    if net_of_fees:
        rec = rec - d["outcome_collection_recovery_fee"].fillna(0.0)
    d["lgd"] = (1.0 - rec / d["ead_at_default"]).clip(0.0, 1.0)
    return d[["grade", "purpose", "ead_at_default", "lgd"]]


def lgd_by_grade(lgd_loans: pd.DataFrame) -> pd.DataFrame:
    """Mean LGD per grade (simple and EAD-weighted). A grade-level segmentation is
    coarse but stable; loan-level LGD models are out of scope (see Limitations)."""
    g = lgd_loans.groupby("grade").apply(
        lambda x: pd.Series({"n_defaults": len(x), "mean_lgd": x["lgd"].mean(),
                             "ead_weighted_lgd": np.average(x["lgd"], weights=x["ead_at_default"]),
                             "ead_at_default": x["ead_at_default"].sum()}),
        include_groups=False)
    g["n_defaults"] = g["n_defaults"].astype(int)
    return g.reset_index()


def assign_lgd(df: pd.DataFrame, lgd_table: pd.DataFrame, column: str = "mean_lgd") -> pd.Series:
    m = lgd_table.set_index("grade")[column]
    overall = float(np.average(lgd_table[column], weights=lgd_table["n_defaults"]))
    return df["grade"].map(m).fillna(overall).astype(float)


def amortised_balance(principal, annual_rate_pct, term_months, months_elapsed):
    """Outstanding balance of an annuity loan after k payments."""
    r = np.asarray(annual_rate_pct, dtype=float) / 100.0 / 12.0
    n, k = float(term_months), np.asarray(months_elapsed, dtype=float)
    growth_n, growth_k = (1 + r) ** n, (1 + r) ** k
    return np.asarray(principal, dtype=float) * (growth_n - growth_k) / (growth_n - 1)


def exposure_at_default(df: pd.DataFrame, method: str = "funded", term_months: int = 36) -> pd.Series:
    """EAD = outstanding principal. 'funded' uses the balance at origination (conservative:
    ignores amortisation); 'avg_12m' uses the average scheduled balance over the 1-year
    risk horizon. Fixed-term instalment loans have no undrawn line, so no CCF is needed."""
    if method == "funded":
        return df["funded_amnt"].astype(float)
    if method == "avg_12m":
        bal = np.mean([amortised_balance(df["funded_amnt"], df["int_rate"], term_months, k)
                       for k in range(13)], axis=0)
        return pd.Series(bal, index=df.index)
    raise ValueError(f"unknown EAD method {method!r}")


def annualise_pd(pd_lifetime, term_months: int = 36) -> np.ndarray:
    """Convert a lifetime (term) PD to a 1-year PD assuming a constant default hazard.
    Credit VaR and Basel IRB are defined on a 1-year horizon."""
    p = np.clip(np.asarray(pd_lifetime, dtype=float), 0.0, 1.0 - 1e-12)
    return 1.0 - (1.0 - p) ** (12.0 / term_months)


def build_portfolio(df: pd.DataFrame, pd_lifetime, lgd_table: pd.DataFrame,
                    ead_method: str = "funded", term_months: int = 36,
                    horizon: str = "1y") -> pd.DataFrame:
    pd_used = annualise_pd(pd_lifetime, term_months) if horizon == "1y" else np.asarray(pd_lifetime)
    port = pd.DataFrame({
        "id": df["id"].to_numpy() if "id" in df else np.arange(len(df)),
        "grade": df["grade"].to_numpy(),
        "sector": df["purpose"].astype(str).to_numpy(),
        "pd": pd_used,
        "lgd": assign_lgd(df, lgd_table).to_numpy(),
        "ead": exposure_at_default(df, ead_method, term_months).to_numpy(),
    })
    port["el"] = port["pd"] * port["lgd"] * port["ead"]
    return port


def el_summary(port: pd.DataFrame, by: str) -> pd.DataFrame:
    g = port.groupby(by).agg(loans=("ead", "size"), ead=("ead", "sum"), el=("el", "sum"),
                             mean_pd=("pd", "mean"), mean_lgd=("lgd", "mean"))
    g["ead_weighted_pd"] = port.groupby(by).apply(
        lambda x: np.average(x["pd"], weights=x["ead"]), include_groups=False)
    g["el_pct_ead"] = g["el"] / g["ead"]
    g["share_of_el"] = g["el"] / g["el"].sum()
    total = pd.DataFrame([{"loans": len(port), "ead": port["ead"].sum(), "el": port["el"].sum(),
                           "mean_pd": port["pd"].mean(), "mean_lgd": port["lgd"].mean(),
                           "ead_weighted_pd": np.average(port["pd"], weights=port["ead"]),
                           "el_pct_ead": port["el"].sum() / port["ead"].sum(), "share_of_el": 1.0}],
                         index=["TOTAL"])
    out = pd.concat([g.sort_values("el", ascending=False), total])
    out.index.name = by
    return out.reset_index()
