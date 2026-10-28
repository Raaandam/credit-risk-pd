"""Synthetic Lending-Club-shaped data for smoke tests (NOT for results).

Columns, string formats (' 36 months', 'Dec-2015', '10+ years') and status values mimic
the Kaggle accepted_2007_to_2018Q4.csv so the full pipeline, including parsing, the
population filter, unresolved-loan exclusion and leakage drop, runs end to end.
Default is driven by a latent credit-quality factor plus a few non-linear effects.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_GRADES = list("ABCDEFG")
_GRADE_SHARE = [0.18, 0.29, 0.28, 0.15, 0.07, 0.02, 0.01]
_GRADE_RATE = dict(zip(_GRADES, [7.0, 10.5, 13.5, 17.0, 20.5, 24.5, 28.0]))
_PURPOSES = {
    "debt_consolidation": 0.58, "credit_card": 0.22, "home_improvement": 0.06, "other": 0.05,
    "major_purchase": 0.025, "small_business": 0.012, "car": 0.012, "medical": 0.011,
    "moving": 0.007, "vacation": 0.006, "house": 0.005, "wedding": 0.002,
    "renewable_energy": 0.001, "educational": 0.001,
}
_STATES = ["CA", "NY", "TX", "FL", "IL", "NJ", "PA", "OH", "GA", "VA", "NC", "MI", "MA", "AZ",
           "WA", "MD", "CO", "MN", "MO", "IN", "TN", "CT", "NV", "WI", "OR", "LA", "SC", "AL"]
_EMP = ["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years", "6 years",
        "7 years", "8 years", "9 years", "10+ years"]


def _month_str(year: np.ndarray, month: np.ndarray) -> np.ndarray:
    return np.array([f"{_MONTHS[m - 1]}-{y}" for y, m in zip(year, month)], dtype=object)


def make_synthetic_lending_club(n: int = 20_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)  # latent credit quality (higher = safer)

    year = rng.choice(np.arange(2012, 2018), n, p=[0.06, 0.12, 0.18, 0.24, 0.24, 0.16])
    month = rng.integers(1, 13, n)
    term = np.where(rng.random(n) < 0.75, 36, 60)

    risk = -z + 0.5 * rng.standard_normal(n)
    grade = np.array(_GRADES)[np.searchsorted(np.quantile(risk, np.cumsum(_GRADE_SHARE)[:-1]), risk)]
    int_rate = np.round(np.vectorize(_GRADE_RATE.get)(grade) + rng.normal(0, 0.8, n)
                        + 2.0 * (term == 60), 2)

    loan_amnt = np.clip(np.round(rng.lognormal(9.4, 0.6, n) / 25) * 25, 1000, 40000)
    r = int_rate / 1200
    installment = np.round(loan_amnt * r / (1 - (1 + r) ** -term), 2)
    annual_inc = np.round(rng.lognormal(11.0 + 0.15 * z, 0.5), -2)
    dti = np.round(np.clip(18 - 3 * z + rng.normal(0, 7, n), 0, 45), 2)
    fico_low = np.clip(np.round((700 + 30 * z + rng.normal(0, 15, n)) / 5) * 5, 660, 845)
    revol_util = np.round(np.clip(50 - 12 * z + rng.normal(0, 20, n), 0, 120), 1)
    inq = rng.poisson(np.clip(0.8 - 0.3 * z, 0.05, None))
    delinq = rng.poisson(np.clip(0.3 - 0.1 * z, 0.02, None))
    open_acc = rng.poisson(11, n) + 1
    pub_rec = rng.poisson(np.clip(0.2 - 0.08 * z, 0.01, None))
    home = rng.choice(["MORTGAGE", "RENT", "OWN"], n, p=[0.49, 0.40, 0.11])
    home = np.where((z > 1.0) & (rng.random(n) < 0.3), "MORTGAGE", home)
    purpose = rng.choice(list(_PURPOSES), n, p=np.array(list(_PURPOSES.values())) / sum(_PURPOSES.values()))
    hist_months = np.clip(rng.normal(190 + 40 * z, 80), 36, 600).astype(int)
    total_months = year * 12 + (month - 1) - hist_months
    ecl = _month_str(total_months // 12, total_months % 12 + 1)
    emp = rng.choice(_EMP + [None], n, p=[0.08, 0.065, 0.09, 0.08, 0.06, 0.065, 0.045, 0.04,
                                         0.045, 0.04, 0.33, 0.06])

    # ---- outcome: logistic in risk drivers + non-linear terms the scorecard cannot fit
    logit = (-2.05 - 0.75 * z + 0.035 * (dti - 18) + 0.12 * inq
             + 0.45 * (purpose == "small_business") + 0.55 * (term == 60)
             + 0.6 * ((dti > 30) & (revol_util > 80))          # interaction
             + 0.3 * (np.log(loan_amnt / annual_inc + 1e-9) > -1.2)
             + 0.25 * (home == "RENT"))
    is_default = rng.random(n) < 1 / (1 + np.exp(-logit))

    # Recent vintages are partly unresolved at the data cut-off.
    unresolved_p = np.select([year == 2016, year == 2017], [0.15, 0.55], 0.02)
    unresolved = rng.random(n) < unresolved_p
    status = np.where(is_default, np.where(rng.random(n) < 0.03, "Default", "Charged Off"), "Fully Paid")
    status = np.where(unresolved, rng.choice(["Current", "Late (31-120 days)", "In Grace Period"],
                                             n, p=[0.9, 0.07, 0.03]), status)

    # ---- post-origination fields (leakage) and recoveries for LGD
    months_paid = np.where(is_default, rng.integers(2, 30, n), term)
    g_n = (1 + r) ** term
    bal = loan_amnt * (g_n - (1 + r) ** np.minimum(months_paid, term)) / (g_n - 1)
    total_rec_prncp = np.round(loan_amnt - bal, 2)
    rec_rate = rng.beta(1.5, 12, n) * (1.2 - 0.05 * np.vectorize(_GRADES.index)(grade))
    recoveries = np.where(is_default, np.round(bal * rec_rate, 2), 0.0)
    total_pymnt = np.round(installment * months_paid + recoveries, 2)
    last_fico = np.where(is_default, rng.integers(500, 600, n), rng.integers(650, 800, n))

    df = pd.DataFrame({
        "id": np.arange(10_000_000, 10_000_000 + n).astype(str),
        "loan_amnt": loan_amnt, "funded_amnt": loan_amnt,
        "funded_amnt_inv": np.round(loan_amnt * rng.uniform(0.95, 1.0, n), 2),
        "term": np.array([f" {t} months" for t in term], dtype=object),
        "int_rate": int_rate, "installment": installment, "grade": grade,
        "sub_grade": [f"{g}{k}" for g, k in zip(grade, rng.integers(1, 6, n))],
        "emp_title": rng.choice(["Teacher", "Manager", "Nurse", "Driver", "Engineer", None], n),
        "emp_length": emp, "home_ownership": home, "annual_inc": annual_inc,
        "verification_status": rng.choice(["Not Verified", "Source Verified", "Verified"], n,
                                          p=[0.3, 0.4, 0.3]),
        "issue_d": _month_str(year, month), "loan_status": status, "purpose": purpose,
        "title": purpose, "zip_code": [f"{k:03d}xx" for k in rng.integers(100, 999, n)],
        "addr_state": rng.choice(_STATES, n), "dti": dti, "delinq_2yrs": delinq,
        "earliest_cr_line": ecl, "fico_range_low": fico_low, "fico_range_high": fico_low + 4,
        "inq_last_6mths": inq,
        "mths_since_last_delinq": np.where(delinq > 0, rng.integers(1, 80, n), np.nan),
        "open_acc": open_acc, "pub_rec": pub_rec,
        "revol_bal": np.round(rng.lognormal(9.3, 0.9, n)), "revol_util": revol_util,
        "total_acc": open_acc + rng.poisson(12, n),
        "initial_list_status": rng.choice(["w", "f"], n, p=[0.6, 0.4]),
        "application_type": rng.choice(["Individual", "Joint App"], n, p=[0.97, 0.03]),
        "mort_acc": np.where(home == "MORTGAGE", rng.poisson(2.5, n), rng.poisson(0.3, n)),
        "pub_rec_bankruptcies": np.minimum(pub_rec, rng.poisson(0.1, n)),
        "acc_open_past_24mths": rng.poisson(np.clip(4.5 - 0.8 * z, 0.5, None)),
        "bc_util": np.round(np.clip(revol_util + rng.normal(0, 10, n), 0, 130), 1),
        "num_actv_rev_tl": rng.poisson(5.5, n),
        "tot_cur_bal": np.round(rng.lognormal(11 + 0.5 * (home == "MORTGAGE"), 1.0)),
        "total_rev_hi_lim": np.round(rng.lognormal(10.1 + 0.2 * z, 0.7)),
        "percent_bc_gt_75": np.round(np.clip(45 - 20 * z + rng.normal(0, 25, n), 0, 100), 1),
        # post-origination / leakage
        "out_prncp": np.where(unresolved, np.round(bal, 2), 0.0),
        "total_pymnt": total_pymnt, "total_rec_prncp": total_rec_prncp,
        "total_rec_int": np.round(np.maximum(total_pymnt - total_rec_prncp - recoveries, 0), 2),
        "recoveries": recoveries, "collection_recovery_fee": np.round(recoveries * 0.18, 2),
        "last_pymnt_amnt": installment, "last_fico_range_high": last_fico + 4,
        "last_fico_range_low": last_fico,
        "debt_settlement_flag": np.where(is_default & (rng.random(n) < 0.1), "Y", "N"),
    })
    # Sprinkle realistic missingness in bureau fields.
    for col, rate in (("mort_acc", 0.02), ("bc_util", 0.01), ("revol_util", 0.001),
                      ("percent_bc_gt_75", 0.01), ("dti", 0.0005)):
        df.loc[rng.random(n) < rate, col] = np.nan
    return df
