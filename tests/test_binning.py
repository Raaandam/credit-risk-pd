import numpy as np
import pandas as pd
import pytest

from credit_risk_pd.features import WoEBinner, iv_strength


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    n = 30_000
    x = rng.standard_normal(n)
    cat = rng.choice(["A", "B", "C", "D"], n, p=[0.4, 0.3, 0.2, 0.1])
    cat_eff = pd.Series(cat).map({"A": -0.8, "B": 0.0, "C": 0.5, "D": 1.0}).to_numpy()
    logit = -2 + 0.9 * x + cat_eff
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame({"risky": x, "noise": rng.standard_normal(n), "cat": cat,
                       "rare_cat": rng.choice(["X", "Y", "Z"], n, p=[0.6, 0.39, 0.01])})
    df.loc[rng.random(n) < 0.05, "risky"] = np.nan
    return df, y


def _fit(df, y, backend):
    return WoEBinner(["risky", "noise"], ["cat", "rare_cat"], backend=backend).fit(df, y)


@pytest.fixture(params=["quantile", "optbinning"])
def backend(request):
    if request.param == "optbinning":
        pytest.importorskip("optbinning")
    return request.param


def test_iv_ranks_informative_above_noise(frame, backend):
    df, y = frame
    iv = _fit(df, y, backend).iv_table().set_index("feature")["iv"]
    assert iv["risky"] > 0.3
    assert iv["cat"] > 0.1
    assert iv["noise"] < 0.02


def test_selection_threshold(frame, backend):
    df, y = frame
    b = _fit(df, y, backend)
    sel = b.selected_features
    assert "risky" in sel and "cat" in sel and "noise" not in sel
    t = b.iv_table()
    assert (t.loc[t["selected"], "iv"] > 0.02).all()


def test_woe_sign_and_monotonicity(frame, backend):
    df, y = frame
    b = _fit(df, y, backend)
    t = b.binning_table("risky")
    t = t[~t["bin"].str.contains("Missing|Special")]
    # higher x -> higher default rate -> monotone DEcreasing WoE (WoE = ln(%good/%bad))
    assert np.all(np.diff(t["event_rate"].to_numpy()) >= -1e-12)
    assert np.all(np.diff(t["woe"].to_numpy()) <= 1e-12)
    # bins with above-average bad rate have negative WoE
    avg = y.mean()
    assert np.all(t.loc[t["event_rate"] > avg * 1.05, "woe"] < 0)


def test_transform_shapes_missing_and_unseen(frame, backend):
    df, y = frame
    b = _fit(df, y, backend)
    new = pd.DataFrame({"risky": [np.nan, 0.0, 5.0], "noise": [0.0, 0.0, 0.0],
                        "cat": ["A", "NEVER_SEEN", "D"], "rare_cat": ["X", "Y", "Z"]})
    w = b.transform(new, ["risky", "cat"])
    assert w.shape == (3, 2)
    assert np.isfinite(w.to_numpy()).all()
    assert w.loc[0, "cat"] > w.loc[2, "cat"]  # A is safer than D


def test_quantile_min_bin_size_respected(frame):
    df, y = frame
    b = WoEBinner(["risky"], [], backend="quantile", min_bin_size=0.1).fit(df, y)
    t = b.binning_table("risky")
    non_missing = t[t["bin"] != "Missing"]
    assert (non_missing["count"] >= 0.1 * df["risky"].notna().sum() - 1).all()


def test_quantile_rare_category_pooled(frame):
    df, y = frame
    b = WoEBinner([], ["rare_cat"], backend="quantile").fit(df, y)
    assert "Other" in b.binning_table("rare_cat")["bin"].tolist()


def test_iv_strength_bands():
    assert iv_strength(0.01) == "unpredictive"
    assert iv_strength(0.2) == "medium"
    assert iv_strength(0.7).startswith("suspicious")


def test_fallback_when_optbinning_missing(monkeypatch):
    import credit_risk_pd.features as f
    monkeypatch.setattr(f, "_optbinning_available", lambda: False)
    assert WoEBinner(["a"], [], backend="optbinning").backend == "quantile"
