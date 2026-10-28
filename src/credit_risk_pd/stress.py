"""Stress testing of the portfolio loss distribution.

Scenarios (config-driven):
* PD multipliers (x1.5, x2): a downturn where every obligor's PD rises proportionally.
* Sector correlation shock: raise the sector-factor loading for one purpose sector, i.e.
  defaults in that sector become more correlated (a concentrated sector shock).

All scenarios reuse the same random seed (common random numbers), so differences between
scenarios reflect the shock, not Monte Carlo noise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .credit_var import simulate_losses


def _run(port: pd.DataFrame, pd_mult: float, var_params: dict, overrides: dict | None):
    pd_ = np.clip(port["pd"].to_numpy() * pd_mult, 0.0, 0.999)
    return simulate_losses(pd_, port["lgd"], port["ead"], port["sector"],
                           sector_rho_overrides=overrides, **var_params)


def run_stress_tests(port: pd.DataFrame, var_params: dict, scenarios: list[dict],
                     q: float = 0.999) -> pd.DataFrame:
    """var_params: kwargs for simulate_losses (rho, sector_rho, n_paths, seed, ...).
    scenarios: [{"name":..., "pd_multiplier":1.5}, {"name":..., "sector":..., "sector_rho":0.3}]"""
    rows = []
    base = _run(port, 1.0, var_params, None)
    for sc in [{"name": "Base"}] + list(scenarios):
        if sc["name"] == "Base":
            ld = base
        else:
            overrides = {sc["sector"]: sc["sector_rho"]} if "sector" in sc else None
            ld = _run(port, sc.get("pd_multiplier", 1.0), var_params, overrides)
        rows.append({"scenario": sc["name"],
                     "pd_multiplier": sc.get("pd_multiplier", 1.0),
                     "shocked_sector": sc.get("sector", "-"),
                     "sector_rho": sc.get("sector_rho", var_params.get("sector_rho")),
                     "el": ld.el_analytic, "var": ld.var(q), "es": ld.es(q),
                     "unexpected_loss": ld.var(q) - ld.el_analytic})
    t = pd.DataFrame(rows)
    for col in ("el", "var", "es"):
        t[f"{col}_vs_base"] = t[col] / t.loc[0, col] - 1.0
    return t
