import numpy as np
import pandas as pd
import pytest

from credit_risk_pd import backtest as bt
from credit_risk_pd import portfolio as pf


def test_binomial_and_jeffreys_pvalues():
    # observed exactly at PD -> no rejection
    assert bt.binomial_test(1000, 50, 0.05) > 0.4
    assert bt.jeffreys_test(1000, 50, 0.05) > 0.4
    # observed double the PD -> strong rejection (PD underestimated)
    assert bt.binomial_test(1000, 100, 0.05) < 1e-6
    assert bt.jeffreys_test(1000, 100, 0.05) < 1e-6
    lo, hi = bt.jeffreys_interval(1000, 50)
    assert lo < 0.05 < hi


def test_backtest_by_band_well_calibrated():
    rng = np.random.default_rng(0)
    n = 50_000
    p = rng.uniform(0.02, 0.3, n)
    y = (rng.random(n) < p).astype(int)
    score = -np.log(p / (1 - p))
    t = bt.backtest_by_band(y, p, score, bt.band_edges(score, 10))
    assert t["n"].iloc[:-1].sum() == n
    s = bt.backtest_summary(t)
    assert s["n_bands"] == 10 and s["red"] <= 1
    # riskiest band (1 = lowest score) has the highest observed default rate
    bands = t[t["band"] != "All"]
    assert bands["observed_dr"].iloc[0] == bands["observed_dr"].max()


def test_backtest_flags_underestimation():
    rng = np.random.default_rng(1)
    p = np.full(20_000, 0.05)
    y = (rng.random(20_000) < 0.08).astype(int)
    t = bt.backtest_by_band(y, p, rng.random(20_000), np.array([0.5]))
    assert (t["result"] == "RED").all()


def _loans():
    return pd.DataFrame({
        "default": [1, 1, 1, 0], "grade": ["A", "A", "B", "B"], "purpose": ["x", "y", "x", "y"],
        "funded_amnt": [10_000.0, 10_000.0, 5_000.0, 5_000.0],
        "outcome_total_rec_prncp": [6_000.0, 2_000.0, 1_000.0, 5_000.0],
        "outcome_recoveries": [400.0, 9_000.0, 0.0, 0.0],
        "outcome_collection_recovery_fee": [72.0, 0.0, 0.0, 0.0],
        "int_rate": [10.0, 10.0, 15.0, 15.0], "id": ["1", "2", "3", "4"],
    })


def test_empirical_lgd_and_grade_table():
    lgd = pf.empirical_lgd(_loans())
    assert len(lgd) == 3                                   # non-defaulter excluded
    assert lgd["lgd"].tolist() == pytest.approx([0.9, 0.0, 1.0])  # 1-400/4000, clipped, 1-0
    t = pf.lgd_by_grade(lgd).set_index("grade")
    assert t.loc["A", "mean_lgd"] == pytest.approx(0.45)
    assert pf.empirical_lgd(_loans(), net_of_fees=True)["lgd"].iloc[0] == pytest.approx(1 - 328 / 4000)


def test_annualise_pd_and_amortisation():
    assert pf.annualise_pd(0.0) == 0.0
    p1 = pf.annualise_pd(np.array([0.27]), 36)[0]
    assert (1 - p1) ** 3 == pytest.approx(0.73)
    assert pf.amortised_balance(1000, 12, 36, 0) == pytest.approx(1000)
    assert pf.amortised_balance(1000, 12, 36, 36) == pytest.approx(0, abs=1e-8)


def test_el_aggregation_sums_to_total():
    loans = _loans()
    lgd_t = pf.lgd_by_grade(pf.empirical_lgd(loans))
    port = pf.build_portfolio(loans, np.array([0.1, 0.2, 0.3, 0.05]), lgd_t)
    assert np.allclose(port["el"], port["pd"] * port["lgd"] * port["ead"])
    by_grade = pf.el_summary(port, "grade")
    total = by_grade.loc[by_grade["grade"] == "TOTAL", "el"].iloc[0]
    assert by_grade.loc[by_grade["grade"] != "TOTAL", "el"].sum() == pytest.approx(total)
    assert total == pytest.approx(port["el"].sum())
