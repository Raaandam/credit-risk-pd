import numpy as np
import pytest
from scipy.stats import ks_2samp

from credit_risk_pd import validation as v


@pytest.fixture
def scored():
    rng = np.random.default_rng(0)
    n = 20_000
    x = rng.standard_normal(n)
    p = 1 / (1 + np.exp(-(-2 + 1.2 * x)))
    y = (rng.random(n) < p).astype(int)
    return y, p


def test_gini_perfect_and_inverted():
    y = np.array([0, 0, 1, 1])
    assert v.gini(y, [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    assert v.gini(y, [0.9, 0.8, 0.2, 0.1]) == pytest.approx(-1.0)


def test_gini_is_two_auc_minus_one(scored):
    y, p = scored
    assert v.gini(y, p) == pytest.approx(2 * v.auc(y, p) - 1)


def test_gini_random_scores_near_zero():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 50_000)
    assert abs(v.gini(y, rng.random(50_000))) < 0.02


def test_ks_matches_scipy_two_sample(scored):
    y, p = scored
    expected = ks_2samp(p[y == 1], p[y == 0]).statistic
    assert v.ks_statistic(y, p) == pytest.approx(expected, abs=1e-9)


def test_ks_bounds():
    y = np.array([0, 0, 1, 1])
    assert v.ks_statistic(y, [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)


def test_psi_identical_is_zero():
    x = np.random.default_rng(2).standard_normal(10_000)
    assert v.psi(x, x) == pytest.approx(0.0, abs=1e-12)


def test_psi_same_distribution_small_and_shift_large():
    rng = np.random.default_rng(3)
    a, b = rng.standard_normal(50_000), rng.standard_normal(50_000)
    assert v.psi(a, b) < 0.01
    assert v.psi(a, b + 1.0) > 0.25
    assert v.psi_status(v.psi(a, b + 1.0)) == "RED"


def test_psi_invariant_to_monotone_transform():
    rng = np.random.default_rng(4)
    a, b = rng.random(20_000), rng.random(20_000) ** 1.3
    logit = lambda u: np.log(u / (1 - u))  # noqa: E731
    assert v.psi(a, b) == pytest.approx(v.psi(logit(a), logit(b)), rel=1e-9)


def test_psi_hand_computed():
    # expected deciles uniform; actual puts 20% in first decile and 8.888..% in others
    e_pct = np.full(10, 0.1)
    a_pct = np.array([0.2] + [0.8 / 9] * 9)
    manual = np.sum((a_pct - e_pct) * np.log(a_pct / e_pct))
    assert v.psi_from_distributions(e_pct, a_pct) == pytest.approx(manual)


def test_psi_categorical_handles_new_levels():
    assert v.psi_categorical(["a"] * 50 + ["b"] * 50, ["a"] * 50 + ["b"] * 50) == pytest.approx(0)
    assert v.psi_categorical(["a"] * 100, ["a"] * 50 + ["c"] * 50) > 0.25


def test_calibration_table_and_brier(scored):
    y, p = scored
    t = v.calibration_table(y, p, 10)
    assert len(t) == 10 and t["n"].sum() == len(y)
    # well-specified model -> observed close to predicted in every decile
    assert np.allclose(t["observed_dr"], t["mean_pd"], atol=0.03)
    assert v.brier(y, p) < v.brier(y, np.full_like(p, 0.5))


def test_monitoring_triggers():
    t = v.monitoring_status(psi_value=0.30, gini_dev=0.45, gini_oot=0.38)
    assert t["status"].tolist() == ["RED", "RED"]
    t = v.monitoring_status(psi_value=0.05, gini_dev=0.45, gini_oot=0.43)
    assert t["status"].tolist() == ["GREEN", "GREEN"]


def test_bootstrap_gini_difference_sign(scored):
    y, p = scored
    noisy = p + np.random.default_rng(5).normal(0, 0.2, len(p))
    res = v.bootstrap_gini_difference(y, noisy, p, n_boot=30)
    assert res["gini_diff"] > 0 and res["ci_low"] > 0
