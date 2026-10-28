"""End-to-end pipeline: data -> WoE binning -> scorecard + XGBoost -> validation -> SHAP
-> LGD/EAD/EL -> credit VaR -> stress -> back-test -> SR 11-7 report.

Usage:
    python -m credit_risk_pd.pipeline --data data/raw/accepted_2007_to_2018Q4.csv
    python -m credit_risk_pd.pipeline --synthetic 20000 --output-dir reports/synthetic_smoke
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import backtest, credit_var, data, explain, portfolio, report, stress, validation
from .challenger import XGBChallenger
from .features import WoEBinner
from .scorecard import LogisticScorecard, ScorecardScaling
from .synthetic import make_synthetic_lending_club

logger = logging.getLogger("credit_risk_pd")


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _save(df: pd.DataFrame, out: Path, name: str) -> None:
    df.to_csv(out / "tables" / f"{name}.csv", index=False)


def run(cfg: dict, data_path: str | None = None, synthetic_n: int | None = None,
        output_dir: str | Path = "reports", nrows: int | None = None,
        n_paths: int | None = None, config_path: str = "configs/config.yaml") -> dict:
    out = Path(output_dir)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    seed = cfg.get("seed", 42)
    dc = cfg["data"]

    # ------------------------------------------------------------------ 1. data
    if synthetic_n:
        raw = make_synthetic_lending_club(synthetic_n, seed=seed)
        df = data.prepare_dataset(raw, dc["term_months"], dc["issue_year_min"],
                                  dc["issue_year_max"], dc["drop_unresolved"])
        source = f"SYNTHETIC sample ({synthetic_n:,} generated rows) - smoke test only, not real results"
    else:
        path = data_path or dc["raw_path"]
        df = data.load_lending_club(path, dc["term_months"], dc["issue_year_min"],
                                    dc["issue_year_max"], dc["drop_unresolved"], nrows=nrows)
        source = f"Lending Club accepted loans (`{Path(path).name}`)"
    train, oot = data.out_of_time_split(df, dc["train_years"], dc["oot_years"])
    y_tr, y_oot = train[data.TARGET].to_numpy(), oot[data.TARGET].to_numpy()
    pop = data.population_summary(train, oot)
    logger.info("Population:\n%s", pop.to_string(index=False))
    numeric, categorical = data.available_features(df)

    # ------------------------------------------------------------------ 2. WoE / IV
    bc = cfg["binning"]
    binner = WoEBinner(numeric, categorical, backend=bc["backend"], max_n_bins=bc["max_n_bins"],
                       min_bin_size=bc["min_bin_size"], iv_min=bc["iv_min"]).fit(train, y_tr)
    iv = binner.iv_table()
    print("\nInformation Value table (train):\n" + iv.to_string(index=False))
    _save(iv, out, "iv_table")

    # ------------------------------------------------------------------ 3. scorecard
    sc_cfg = cfg["scorecard"]
    scaling = ScorecardScaling(sc_cfg["pdo"], sc_cfg["base_score"], sc_cfg["base_odds"])
    scorecard = LogisticScorecard(binner, scaling, C=sc_cfg["C"],
                                  corr_max=sc_cfg["corr_max"]).fit(train, y_tr)
    _save(scorecard.points_table(), out, "scorecard_points")

    # ------------------------------------------------------------------ 4. challenger
    xc = cfg["xgboost"]
    xgb = XGBChallenger(numeric, categorical, params=xc["params"],
                        early_stopping_rounds=xc["early_stopping_rounds"],
                        valid_size=xc["valid_size"], random_state=seed).fit(train, y_tr)

    p_sc_tr, p_sc_oot = scorecard.predict_pd(train), scorecard.predict_pd(oot)
    p_xgb_tr, p_xgb_oot = xgb.predict_pd(train), xgb.predict_pd(oot)
    s_tr, s_oot = scorecard.score(train), scorecard.score(oot)

    # ------------------------------------------------------------------ 5. validation
    vc, mc = cfg["validation"], cfg["monitoring"]
    perf = {"Scorecard / train": validation.performance_summary(y_tr, p_sc_tr),
            "Scorecard / OOT": validation.performance_summary(y_oot, p_sc_oot),
            "XGBoost / train": validation.performance_summary(y_tr, p_xgb_tr),
            "XGBoost / OOT": validation.performance_summary(y_oot, p_xgb_oot)}
    perf_t = validation.model_comparison(perf)
    _save(perf_t, out, "performance")
    lift = validation.bootstrap_gini_difference(y_oot, p_sc_oot, p_xgb_oot,
                                                n_boot=vc["n_bootstrap"], seed=seed)
    calib = validation.calibration_table(y_oot, p_sc_oot, vc["calibration_bins"])
    _save(calib, out, "calibration_scorecard_oot")
    psi_rows = [{"model": "Scorecard (points)", "psi": validation.psi(s_tr, s_oot, vc["psi_bins"])},
                {"model": "XGBoost (PD)", "psi": validation.psi(p_xgb_tr, p_xgb_oot, vc["psi_bins"])}]
    psi_t = pd.DataFrame(psi_rows)
    psi_t["status"] = psi_t["psi"].map(lambda v: validation.psi_status(v, mc["psi_warn"], mc["psi_trigger"]))
    bins_tr = binner.transform(train, scorecard.features_, metric="bins")
    bins_oot = binner.transform(oot, scorecard.features_, metric="bins")
    csi_t = pd.DataFrame([{"feature": f, "csi": validation.psi_categorical(bins_tr[f], bins_oot[f])}
                          for f in scorecard.features_]).sort_values("csi", ascending=False)
    csi_t["status"] = csi_t["csi"].map(lambda v: validation.psi_status(v, mc["psi_warn"], mc["psi_trigger"]))
    monitor_t = validation.monitoring_status(psi_rows[0]["psi"], perf["Scorecard / train"]["gini"],
                                             perf["Scorecard / OOT"]["gini"], mc["psi_warn"],
                                             mc["psi_trigger"], mc["gini_drop_trigger_pts"])

    # ------------------------------------------------------------------ 6. explainability
    ec = cfg["explain"]
    shap_df, _, X_shap = explain.shap_values(xgb, oot, ec["shap_sample"], seed)
    shap_global = explain.global_importance(shap_df)
    _save(shap_global, out, "shap_global")
    shap_local = explain.local_explanations(shap_df, X_shap, xgb.predict_pd(X_shap))
    _save(shap_local, out, "shap_local")
    _save(scorecard.reason_codes(X_shap.head(50)).reset_index(), out, "reason_codes_sample")
    fair_home = explain.fairness_by_group(oot, oot["home_ownership"], y_oot, p_sc_oot,
                                          ec["fairness_approval_rate"])
    region = oot["addr_state"].map(explain.STATE_TO_REGION).fillna("Other") \
        if "addr_state" in oot else pd.Series("n/a", index=oot.index)
    fair_region = explain.fairness_by_group(oot, region, y_oot, p_sc_oot, ec["fairness_approval_rate"])
    rai_note = explain.responsible_ai_note({"home ownership": fair_home, "census region": fair_region})

    # ------------------------------------------------------------------ 7. LGD / EAD / EL
    pc = cfg["portfolio"]
    lgd_loans = portfolio.empirical_lgd(train, net_of_fees=pc["lgd_net_of_fees"])
    lgd_t = portfolio.lgd_by_grade(lgd_loans)
    pd_book = p_sc_oot if pc["champion"] == "scorecard" else p_xgb_oot
    port = portfolio.build_portfolio(oot, pd_book, lgd_t, pc["ead_method"], dc["term_months"],
                                     pc["pd_horizon"])
    el_grade, el_sector = portfolio.el_summary(port, "grade"), portfolio.el_summary(port, "sector")
    for name, t in (("lgd_by_grade", lgd_t), ("el_by_grade", el_grade), ("el_by_sector", el_sector)):
        _save(t, out, name)

    # ------------------------------------------------------------------ 8. credit VaR
    cv = cfg["credit_var"]
    q = cv["confidence"]
    var_params = dict(rho=cv["rho"], sector_rho=cv["sector_rho"], n_paths=n_paths or cv["n_paths"],
                      seed=seed, batch_size=cv["batch_size"], method=cv["method"],
                      n_pd_bands=cv["n_pd_bands"])
    loss = credit_var.simulate_losses(port["pd"], port["lgd"], port["ead"], port["sector"], **var_params)
    var_sum = loss.summary(q)
    irb_t = credit_var.irb_comparison(port, loss, cv["rho"], q)

    # ------------------------------------------------------------------ 9. stress
    stress_t = stress.run_stress_tests(port, var_params, cfg["stress"]["scenarios"], q)
    _save(stress_t, out, "stress_tests")

    # ------------------------------------------------------------------ 10. back-test
    edges = backtest.band_edges(s_tr, cfg["backtest"]["n_bands"])
    bt = backtest.backtest_by_band(y_oot, p_sc_oot, s_oot, edges)
    bt_sum = backtest.backtest_summary(bt)
    _save(bt, out, "backtest_oot")

    # ------------------------------------------------------------------ 11. report
    figs = report.save_figures(out / "figures", y_oot, p_sc_oot, p_xgb_oot, s_tr, s_oot,
                               loss.losses, loss.var(q), loss.el_analytic)
    if explain.save_shap_summary_plot(shap_df, out / "figures" / "shap_global.png"):
        figs.append("shap_global.png")

    gini_lift_pts = lift["gini_diff"] * 100
    psi_val = psi_rows[0]["psi"]
    red_flags = monitor_t.loc[monitor_t["status"] == "RED", "metric"].tolist()
    conclusion = "\n".join([
        f"- Champion OOT Gini {perf['Scorecard / OOT']['gini'] * 100:.1f} vs challenger "
        f"{perf['XGBoost / OOT']['gini'] * 100:.1f}; challenger lift {gini_lift_pts:+.1f} pts "
        f"(95% CI {lift['ci_low'] * 100:+.1f} to {lift['ci_high'] * 100:+.1f}).",
        f"- Score PSI train vs OOT = {psi_val:.3f} ({validation.psi_status(psi_val, mc['psi_warn'], mc['psi_trigger'])}).",
        f"- Back-test: {bt_sum['green']} green / {bt_sum['amber']} amber / {bt_sum['red']} red "
        f"bands; portfolio-level result {bt_sum['portfolio_result']}.",
        f"- 1-year EL {loss.el_analytic:,.0f}; {q:.1%} VaR {loss.var(q):,.0f}; ES {loss.es(q):,.0f} "
        f"on EAD {port['ead'].sum():,.0f}.",
        f"- Monitoring triggers breached: {', '.join(red_flags) if red_flags else 'none'}.",
        "- Final model-risk rating and approval are for the independent validator to record.",
    ])
    f2 = lambda t, ff=".4f", n=None: report.md_table(t, ff, n)  # noqa: E731
    ctx = {
        "report_status": "Rendered from pipeline run" + (" - SYNTHETIC DATA, numbers are not meaningful"
                                                         if synthetic_n else ""),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "data_source": source, "config_path": config_path,
        "population_table": f2(pop),
        "issue_window": f"{dc['issue_year_min']}-{dc['issue_year_max']}",
        "train_years": ", ".join(map(str, dc["train_years"])),
        "oot_years": ", ".join(map(str, dc["oot_years"])),
        "leakage_columns": ", ".join(f"`{c}`" for c in data.LEAKAGE_COLUMNS),
        "binning_backend": binner.backend, "max_n_bins": bc["max_n_bins"],
        "min_bin_size": bc["min_bin_size"], "iv_min": bc["iv_min"], "iv_table": f2(iv),
        "corr_max": sc_cfg["corr_max"], "pdo": scaling.pdo, "base_score": scaling.base_score,
        "base_odds": scaling.base_odds, "factor": f"{scaling.factor:.4f}",
        "offset": f"{scaling.offset:.4f}", "scorecard_coefficients": f2(scorecard.coefficients()),
        "dropped_features": "; ".join(f"`{k}` ({v})" for k, v in scorecard.dropped_.items()) or "none",
        "xgb_params": ", ".join(f"{k}={v}" for k, v in xgb.params.items()),
        "xgb_best_iteration": getattr(xgb.model_, "best_iteration", "n/a"),
        "ead_method": pc["ead_method"], "pd_horizon": pc["pd_horizon"],
        "n_paths": f"{var_params['n_paths']:,}", "rho": cv["rho"], "sector_rho": cv["sector_rho"],
        "performance_table": f2(perf_t),
        "gini_lift_pts": f"{gini_lift_pts:+.2f}",
        "gini_lift_ci": f"[{lift['ci_low'] * 100:+.2f}, {lift['ci_high'] * 100:+.2f}]",
        "n_boot": lift["n_boot"], "calibration_table": f2(calib), "psi_table": f2(psi_t),
        "csi_table": f2(csi_t),
        "figures": "\n\n".join(f"![{f}](figures/{f})" for f in figs),
        "shap_global_table": f2(shap_global, n=15), "shap_local_table": f2(shap_local),
        "responsible_ai_note": rai_note,
        "fairness_home_ownership": f2(fair_home), "fairness_region": f2(fair_region),
        "lgd_table": f2(lgd_t), "el_by_grade": f2(el_grade), "el_by_sector": f2(el_sector),
        "var_summary": f2(pd.DataFrame([var_sum])), "irb_table": f2(irb_t),
        "stress_table": f2(stress_t), "backtest_table": f2(bt),
        "backtest_summary": ", ".join(f"{k}={v}" for k, v in bt_sum.items()),
        "psi_warn": mc["psi_warn"], "psi_trigger": mc["psi_trigger"],
        "gini_drop_trigger": mc["gini_drop_trigger_pts"], "monitoring_table": f2(monitor_t),
        "conclusion": conclusion,
    }
    path = report.write_report(ctx, out / "model_validation_report.md")
    metrics = {"performance": perf, "gini_lift": lift, "psi": psi_rows, "credit_var": var_sum,
               "backtest": bt_sum}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")
    logger.info("Report written to %s", path)
    return metrics


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="path to accepted_2007_to_2018Q4.csv")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--output-dir", default="reports")
    ap.add_argument("--synthetic", type=int, metavar="N",
                    help="use N synthetic Lending-Club-shaped rows instead of real data")
    ap.add_argument("--nrows", type=int, help="read only the first N CSV rows (dev runs)")
    ap.add_argument("--n-paths", type=int, help="override Monte Carlo paths (default from config)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    run(cfg, data_path=args.data, synthetic_n=args.synthetic, output_dir=args.output_dir,
        nrows=args.nrows, n_paths=args.n_paths, config_path=args.config)


if __name__ == "__main__":
    main()
