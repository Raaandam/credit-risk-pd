"""Load, clean, filter and split the Lending Club accepted-loans file.

Population: 36-month loans issued 2013-2016, resolved outcomes only.
Target:     default = 1 if loan_status in {Charged Off, Default}, 0 if Fully Paid.
Split:      out-of-time (train 2013-2015, OOT test 2016).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TARGET = "default"

DEFAULT_STATUSES = {
    "Charged Off",
    "Default",
    "Does not meet the credit policy. Status:Charged Off",
}
GOOD_STATUSES = {
    "Fully Paid",
    "Does not meet the credit policy. Status:Fully Paid",
}

# Fields that are only populated / updated AFTER origination. Using any of them as a
# predictor leaks the outcome (e.g. recoveries > 0 only for charged-off loans,
# last_fico_* is pulled after delinquency). They are dropped before modelling.
LEAKAGE_COLUMNS = [
    "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv", "total_rec_prncp",
    "total_rec_int", "total_rec_late_fee", "recoveries", "collection_recovery_fee",
    "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d", "last_credit_pull_d",
    "last_fico_range_high", "last_fico_range_low", "pymnt_plan",
    "hardship_flag", "hardship_type", "hardship_reason", "hardship_status", "deferral_term",
    "hardship_amount", "hardship_start_date", "hardship_end_date", "payment_plan_start_date",
    "hardship_length", "hardship_dpd", "hardship_loan_status",
    "orig_projected_additional_accrued_interest", "hardship_payoff_balance_amount",
    "hardship_last_payment_amount", "debt_settlement_flag", "debt_settlement_flag_date",
    "settlement_status", "settlement_date", "settlement_amount", "settlement_percentage",
    "settlement_term",
]

# Post-origination fields we still need, but ONLY for LGD estimation (never as features).
# They are renamed with an "outcome_" prefix so they cannot slip into a feature list.
OUTCOME_COLUMNS = ["total_rec_prncp", "recoveries", "collection_recovery_fee"]

# Free-text / identifier / geographic fields excluded from the model on purpose
# (see explain.EXCLUDED_FEATURES for the responsible-AI rationale). addr_state is
# retained as a monitoring-only column for proxy fairness checks.
EXCLUDED_COLUMNS = ["emp_title", "desc", "title", "url", "zip_code", "member_id", "sub_grade"]
MONITORING_COLUMNS = ["addr_state"]

# Application-time features (allowlist). An allowlist is safer than a blocklist:
# a new post-origination column in a future data drop cannot silently become a feature.
NUMERIC_FEATURES = [
    "loan_amnt", "int_rate", "installment", "annual_inc", "dti", "fico_score",
    "emp_length_years", "credit_history_months", "delinq_2yrs", "inq_last_6mths",
    "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc", "mort_acc",
    "pub_rec_bankruptcies", "mths_since_last_delinq", "acc_open_past_24mths", "bc_util",
    "num_actv_rev_tl", "tot_cur_bal", "total_rev_hi_lim", "percent_bc_gt_75",
    "loan_to_income", "installment_to_income",
]
# grade / int_rate are Lending Club's own risk pricing. They are application-time
# (known at booking) so not leakage; keeping them means our models are benchmarked
# against the lender's existing view of risk.
CATEGORICAL_FEATURES = [
    "grade", "home_ownership", "verification_status", "purpose",
    "application_type", "initial_list_status",
]

# Raw columns needed from the 1.6 GB CSV (reading only these keeps memory manageable).
RAW_COLUMNS = sorted(set([
    "id", "loan_amnt", "funded_amnt", "term", "int_rate", "installment", "grade",
    "emp_length", "home_ownership", "annual_inc", "verification_status", "issue_d",
    "loan_status", "purpose", "addr_state", "dti", "earliest_cr_line", "fico_range_low",
    "fico_range_high", "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec", "revol_bal",
    "revol_util", "total_acc", "mort_acc", "pub_rec_bankruptcies", "mths_since_last_delinq",
    "acc_open_past_24mths", "bc_util", "num_actv_rev_tl", "tot_cur_bal", "total_rev_hi_lim",
    "percent_bc_gt_75", "application_type", "initial_list_status",
] + OUTCOME_COLUMNS))

_EMP_LENGTH_MAP = {
    "< 1 year": 0.0, "1 year": 1.0, "2 years": 2.0, "3 years": 3.0, "4 years": 4.0,
    "5 years": 5.0, "6 years": 6.0, "7 years": 7.0, "8 years": 8.0, "9 years": 9.0,
    "10+ years": 10.0,
}


# --------------------------------------------------------------------------- parsing
def _to_float(s: pd.Series) -> pd.Series:
    """Numeric parse that also handles '13.5%' strings (older LC file versions)."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    return pd.to_numeric(s.astype("string").str.replace("%", "", regex=False).str.strip(),
                         errors="coerce")


def _to_month(s: pd.Series) -> pd.Series:
    """'Dec-2015' -> Timestamp."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    return pd.to_datetime(s, format="%b-%Y", errors="coerce")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Type-parse raw LC columns and engineer a few application-time ratios."""
    df = df.copy()
    if "term" in df:
        df["term"] = pd.to_numeric(df["term"].astype("string").str.extract(r"(\d+)")[0],
                                   errors="coerce")
    for col in ("int_rate", "revol_util", "bc_util", "dti", "annual_inc", "loan_amnt",
                "funded_amnt", "installment"):
        if col in df:
            df[col] = _to_float(df[col])
    df["issue_d"] = _to_month(df["issue_d"])
    df["issue_year"] = df["issue_d"].dt.year

    if "emp_length" in df:
        df["emp_length_years"] = df["emp_length"].map(_EMP_LENGTH_MAP)
    if {"fico_range_low", "fico_range_high"} <= set(df.columns):
        df["fico_score"] = (df["fico_range_low"] + df["fico_range_high"]) / 2.0
    if "earliest_cr_line" in df:
        ecl = _to_month(df["earliest_cr_line"])
        df["credit_history_months"] = ((df["issue_d"].dt.year - ecl.dt.year) * 12
                                       + (df["issue_d"].dt.month - ecl.dt.month))
    # Affordability ratios: scale-free versions of loan size, which bin more stably
    # across income levels than raw loan_amnt.
    income = df["annual_inc"].where(df["annual_inc"] > 0)
    df["loan_to_income"] = df["loan_amnt"] / income
    df["installment_to_income"] = df["installment"] * 12.0 / income
    return df


# --------------------------------------------------------------------------- population
def filter_population(df: pd.DataFrame, term_months: int = 36,
                      year_min: int = 2013, year_max: int = 2016) -> pd.DataFrame:
    """Keep one product (36m) and one vintage window so the PD has a single horizon."""
    mask = (df["term"] == term_months) & df["issue_year"].between(year_min, year_max)
    return df.loc[mask].copy()


def define_target(df: pd.DataFrame, drop_unresolved: bool = True) -> pd.DataFrame:
    """default = Charged Off / Default. Loans still Current / Late are unresolved;
    we drop them rather than guess their outcome (see Limitations: this over-weights
    early outcomes in the most recent vintage)."""
    status = df["loan_status"].astype("string").str.strip()
    df = df.copy()
    df[TARGET] = np.where(status.isin(DEFAULT_STATUSES), 1,
                          np.where(status.isin(GOOD_STATUSES), 0, -1))
    if drop_unresolved:
        n_before = len(df)
        df = df.loc[df[TARGET] >= 0].copy()
        logger.info("Dropped %d unresolved loans (Current/Late/Grace)", n_before - len(df))
    df[TARGET] = df[TARGET].astype(int)
    return df


def drop_leakage(df: pd.DataFrame) -> pd.DataFrame:
    """Move LGD-only outcome fields to outcome_*, drop all other post-origination and
    excluded columns."""
    df = df.rename(columns={c: f"outcome_{c}" for c in OUTCOME_COLUMNS if c in df.columns})
    to_drop = [c for c in LEAKAGE_COLUMNS + EXCLUDED_COLUMNS if c in df.columns]
    return df.drop(columns=to_drop)


def assert_no_leakage(features: Iterable[str]) -> None:
    bad = [f for f in features if f in LEAKAGE_COLUMNS or f.startswith("outcome_")]
    if bad:
        raise ValueError(f"Post-origination columns in feature list: {bad}")


def available_features(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Allowlisted features present in this extract (some bureau fields are sparse in
    early vintages or absent in older file versions)."""
    num = [c for c in NUMERIC_FEATURES if c in df.columns]
    cat = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    missing = sorted(set(NUMERIC_FEATURES + CATEGORICAL_FEATURES) - set(num + cat))
    if missing:
        logger.warning("Features not found in data and skipped: %s", missing)
    assert_no_leakage(num + cat)
    return num, cat


# --------------------------------------------------------------------------- I/O
def prepare_dataset(raw: pd.DataFrame, term_months: int = 36, year_min: int = 2013,
                    year_max: int = 2016, drop_unresolved: bool = True) -> pd.DataFrame:
    """clean -> filter -> target -> leakage drop, for an in-memory raw frame."""
    df = filter_population(clean(raw), term_months, year_min, year_max)
    df = define_target(df, drop_unresolved)
    return drop_leakage(df).reset_index(drop=True)


def load_lending_club(path: str | Path, term_months: int = 36, year_min: int = 2013,
                      year_max: int = 2016, drop_unresolved: bool = True,
                      nrows: int | None = None, chunksize: int = 250_000) -> pd.DataFrame:
    """Read the Kaggle accepted_2007_to_2018Q4.csv in chunks, keeping only the needed
    columns and filtering each chunk to the target population (the full file does not
    fit comfortably in memory with all 151 columns)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download 'accepted_2007_to_2018Q4.csv' from the Kaggle "
            "Lending Club dataset into data/raw/ (or run with --synthetic N).")
    header = pd.read_csv(path, nrows=0).columns
    usecols = [c for c in RAW_COLUMNS if c in header]
    chunks = []
    reader = pd.read_csv(path, usecols=usecols, chunksize=chunksize, nrows=nrows,
                         low_memory=False, dtype={"id": "string"})
    for chunk in reader:
        # The Kaggle file ends with summary footer rows that have no issue_d.
        chunk = chunk.dropna(subset=["issue_d", "loan_amnt"])
        chunks.append(filter_population(clean(chunk), term_months, year_min, year_max))
    df = pd.concat(chunks, ignore_index=True)
    logger.info("Loaded %d loans in population from %s", len(df), path.name)
    df = define_target(df, drop_unresolved)
    return drop_leakage(df).reset_index(drop=True)


def out_of_time_split(df: pd.DataFrame, train_years: Iterable[int],
                      oot_years: Iterable[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Out-of-time rather than random split: the model is used on future vintages, so
    validation must show it survives a shift in time (macro cycle, underwriting policy)."""
    train = df.loc[df["issue_year"].isin(list(train_years))].reset_index(drop=True)
    oot = df.loc[df["issue_year"].isin(list(oot_years))].reset_index(drop=True)
    if train.empty or oot.empty:
        raise ValueError("Empty train or OOT sample - check years in config.")
    return train, oot


def population_summary(train: pd.DataFrame, oot: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, d in (("Train (dev)", train), ("OOT test", oot)):
        rows.append({
            "sample": name,
            "issue_years": f"{int(d['issue_year'].min())}-{int(d['issue_year'].max())}",
            "loans": len(d),
            "defaults": int(d[TARGET].sum()),
            "default_rate": d[TARGET].mean(),
            "total_funded": d["funded_amnt"].sum() if "funded_amnt" in d else np.nan,
        })
    return pd.DataFrame(rows)
