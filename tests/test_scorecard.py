import numpy as np
import pandas as pd
import pytest

from credit_risk_pd.features import WoEBinner
from credit_risk_pd.scorecard import LogisticScorecard, ScorecardScaling


def test_scaling_anchor_and_pdo():
    s = ScorecardScaling(pdo=20, base_score=600, base_odds=50)
    pd_at_base = 1 / (1 + 50)  # 50:1 good:bad
    assert s.score_from_pd(pd_at_base) == pytest.approx(600)
    pd_double_odds = 1 / (1 + 100)
    assert s.score_from_pd(pd_double_odds) == pytest.approx(620)
    assert s.pd_from_score(600) == pytest.approx(pd_at_base)


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(1)
    n = 20_000
    df = pd.DataFrame({"x1": rng.standard_normal(n), "x2": rng.standard_normal(n),
                       "c": rng.choice(["a", "b", "c"], n)})
    logit = -2 + 0.8 * df["x1"] - 0.5 * df["x2"] + df["c"].map({"a": 0, "b": 0.4, "c": 0.9})
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).to_numpy().astype(int)
    b = WoEBinner(["x1", "x2"], ["c"], backend="quantile").fit(df, y)
    return LogisticScorecard(b).fit(df, y), df, y


def test_points_sum_to_score(fitted):
    sc, df, _ = fitted
    pts = sc.points(df.head(500))
    assert np.allclose(pts.sum(axis=1), sc.score(df.head(500)))


def test_score_and_pd_consistent(fitted):
    sc, df, _ = fitted
    assert np.allclose(sc.scaling.pd_from_score(sc.score(df.head(200))), sc.predict_pd(df.head(200)))


def test_higher_score_is_lower_risk(fitted):
    sc, df, y = fitted
    s = sc.score(df)
    assert np.corrcoef(s, y)[0, 1] < 0


def test_coefficient_signs_consistent(fitted):
    sc, df, y = fitted
    woe = sc.binner.transform(df, sc.features_)
    for f in sc.features_:
        assert np.sign(sc.coef_[f]) == np.sign(np.corrcoef(woe[f], y)[0, 1])


def test_points_table_and_reason_codes(fitted):
    sc, df, _ = fitted
    t = sc.points_table()
    assert set(t["feature"]) == set(sc.features_)
    rc = sc.reason_codes(df.head(10), top_n=2)
    assert rc.shape == (10, 2)
