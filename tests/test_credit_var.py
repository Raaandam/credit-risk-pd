import numpy as np
import pandas as pd
import pytest

from credit_risk_pd import credit_var as cv
from credit_risk_pd.stress import run_stress_tests


@pytest.fixture(scope="module")
def book():
    rng = np.random.default_rng(0)
    n = 5_000
    return pd.DataFrame({"pd": rng.uniform(0.01, 0.12, n), "lgd": rng.uniform(0.6, 0.95, n),
                         "ead": rng.uniform(2_000, 30_000, n),
                         "sector": rng.choice(["debt_consolidation", "credit_card", "other"], n)})


def test_mc_el_converges_to_analytic(book):
    ld = cv.simulate_losses(book["pd"], book["lgd"], book["ead"], book["sector"],
                            rho=0.15, sector_rho=0.05, n_paths=100_000, seed=1)
    # Monte Carlo standard error of the mean, from the simulated distribution itself
    se = ld.losses.std(ddof=1) / np.sqrt(len(ld.losses))
    assert abs(ld.el_mc - ld.el_analytic) < 4 * se
    assert abs(ld.el_mc / ld.el_analytic - 1) < 0.01


def test_el_error_shrinks_with_paths(book):
    errs = []
    for n_paths in (2_000, 200_000):
        ld = cv.simulate_losses(book["pd"], book["lgd"], book["ead"], book["sector"],
                                n_paths=n_paths, seed=7, method="conditional")
        errs.append(ld.losses.std() / np.sqrt(n_paths))
    assert errs[1] < errs[0] / 5


def test_var_matches_vasicek_closed_form_for_homogeneous_pool():
    n = 10_000
    pd_, lgd, ead = np.full(n, 0.03), np.full(n, 0.8), np.full(n, 1_000.0)
    rho, q = 0.15, 0.999
    ld = cv.simulate_losses(pd_, lgd, ead, np.zeros(n), rho=rho, sector_rho=0.0,
                            n_paths=200_000, seed=3, method="conditional", n_pd_bands=1)
    closed = cv.vasicek_loss_quantile(pd_, lgd, ead, rho, q)
    assert ld.var(q) == pytest.approx(closed, rel=0.03)


def test_es_at_least_var_and_var_above_el(book):
    ld = cv.simulate_losses(book["pd"], book["lgd"], book["ead"], book["sector"],
                            n_paths=20_000, seed=2)
    assert ld.es(0.999) >= ld.var(0.999) > ld.el_analytic


def test_sector_factor_fattens_tail(book):
    kw = dict(n_paths=50_000, seed=4, rho=0.15)
    base = cv.simulate_losses(book["pd"], book["lgd"], book["ead"], book["sector"], sector_rho=0.0, **kw)
    sect = cv.simulate_losses(book["pd"], book["lgd"], book["ead"], book["sector"], sector_rho=0.10, **kw)
    assert sect.var(0.999) > base.var(0.999)
    assert sect.el_mc == pytest.approx(base.el_mc, rel=0.02)


def test_invalid_correlation_rejected(book):
    with pytest.raises(ValueError):
        cv.simulate_losses(book["pd"], book["lgd"], book["ead"], book["sector"],
                           rho=0.7, sector_rho=0.4, n_paths=10)


def test_basel_retail_correlation_range_and_irb_known_value():
    # limits: R -> 16% as PD -> 0 and R -> 3% for high PD
    r = cv.basel_retail_correlation(np.array([1e-9, 0.5]))
    assert r[0] == pytest.approx(0.16, abs=1e-6) and r[1] == pytest.approx(0.03, abs=1e-6)
    # Hand calculation for PD=1%, LGD=45%: R = 0.1216,
    # Phi((Phi^-1(0.01) + sqrt(R) * 3.0902) / sqrt(1-R)) = 0.0914 -> K = 0.45 * (0.0914 - 0.01)
    k = cv.irb_capital_k(np.array([0.01]), np.array([0.45]))[0]
    assert k == pytest.approx(0.0366, abs=0.001)


def test_stress_scenarios_move_in_right_direction(book):
    t = run_stress_tests(book, dict(rho=0.15, sector_rho=0.05, n_paths=20_000, seed=5),
                         [{"name": "PD x2", "pd_multiplier": 2.0},
                          {"name": "corr", "sector": "debt_consolidation", "sector_rho": 0.25}])
    base, pdx2, corr = t.iloc[0], t.iloc[1], t.iloc[2]
    assert pdx2["el"] == pytest.approx(2 * base["el"], rel=0.01)
    assert pdx2["var"] > base["var"]
    assert corr["el"] == pytest.approx(base["el"])
    assert corr["var"] > base["var"]
