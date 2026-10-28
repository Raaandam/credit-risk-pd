import numpy as np
import pytest

from credit_risk_pd import data, report
from credit_risk_pd.synthetic import make_synthetic_lending_club


@pytest.fixture(scope="module")
def prepared():
    raw = make_synthetic_lending_club(8_000, seed=3)
    return raw, data.prepare_dataset(raw)


def test_population_filter_and_target(prepared):
    raw, df = prepared
    assert set(df["term"].unique()) == {36}
    assert df["issue_year"].between(2013, 2016).all()
    assert set(df[data.TARGET].unique()) <= {0, 1}
    assert len(df) < len(raw)


def test_leakage_columns_removed(prepared):
    _, df = prepared
    assert not set(data.LEAKAGE_COLUMNS) & set(df.columns)
    assert {"outcome_recoveries", "outcome_total_rec_prncp"} <= set(df.columns)
    num, cat = data.available_features(df)
    assert not any(c.startswith("outcome_") for c in num + cat)
    with pytest.raises(ValueError):
        data.assert_no_leakage(["fico_score", "recoveries"])


def test_parsing(prepared):
    _, df = prepared
    assert df["emp_length_years"].dropna().between(0, 10).all()
    assert (df["fico_score"] >= 660).all()
    assert df["credit_history_months"].min() > 0


def test_oot_split(prepared):
    _, df = prepared
    tr, oot = data.out_of_time_split(df, [2013, 2014, 2015], [2016])
    assert tr["issue_year"].max() <= 2015 and (oot["issue_year"] == 2016).all()


def test_report_template_placeholders():
    text = report.render_placeholder()
    assert "<<RUNTIME: performance_table>>" in text
    assert "{{" not in text


def test_md_table():
    import pandas as pd
    t = report.md_table(pd.DataFrame({"a": [1, 2], "b": [0.5, np.nan]}))
    assert t.splitlines()[0] == "| a | b |" and len(t.splitlines()) == 4


def test_end_to_end_smoke(tmp_path):
    pytest.importorskip("xgboost")
    from credit_risk_pd.pipeline import load_config, run
    from pathlib import Path
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "config.yaml")
    cfg["validation"]["n_bootstrap"] = 10
    cfg["explain"]["shap_sample"] = 500
    cfg["xgboost"]["params"]["n_estimators"] = 50
    metrics = run(cfg, synthetic_n=12_000, output_dir=tmp_path, n_paths=5_000)
    assert (tmp_path / "model_validation_report.md").exists()
    text = (tmp_path / "model_validation_report.md").read_text(encoding="utf-8")
    assert "<<RUNTIME" not in text and "{{" not in text
    assert 0.0 < metrics["performance"]["Scorecard / OOT"]["auc"] < 1.0
