"""Portfolio credit VaR: one-factor Gaussian copula (Vasicek) Monte Carlo with a sector factor.

Latent asset return of obligor i in sector s:

    A_i = sqrt(rho) * Z  +  sqrt(rho_s) * Z_s  +  sqrt(1 - rho - rho_s) * eps_i

Z (economy), Z_s (sector) and eps_i are independent N(0,1). Obligor defaults if
A_i < Phi^-1(PD_i). Asset correlation is rho across sectors and rho + rho_s within a
sector, so concentration in one 'purpose' sector fattens the loss tail.

Conditional on (Z, Z_s) defaults are independent with probability

    p_i(Z, Z_s) = Phi( (Phi^-1(PD_i) - sqrt(rho) Z - sqrt(rho_s) Z_s) / sqrt(1 - rho - rho_s) )

Vectorisation: simulating 300K obligors x 100K paths is 3e10 draws, so loans are grouped
into homogeneous cells (sector x PD band). Per path we draw the factors, compute each
cell's conditional PD, then draw the cell's default COUNT ~ Binomial(n_cell, p_cell),
which keeps idiosyncratic (granularity) risk. method="conditional" uses the expected
conditional loss instead (the ASRF / infinitely-granular limit that Basel IRB assumes).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri


@dataclass
class LossDistribution:
    losses: np.ndarray
    el_analytic: float
    total_ead: float

    @property
    def el_mc(self) -> float:
        return float(self.losses.mean())

    def var(self, q: float = 0.999) -> float:
        return float(np.quantile(self.losses, q))

    def es(self, q: float = 0.999) -> float:
        """Expected Shortfall: mean loss in the tail beyond VaR (coherent, sub-additive)."""
        v = self.var(q)
        return float(self.losses[self.losses >= v].mean())

    def summary(self, q: float = 0.999) -> dict:
        v, e = self.var(q), self.es(q)
        return {"n_paths": len(self.losses), "total_ead": self.total_ead,
                "el_analytic": self.el_analytic, "el_mc": self.el_mc,
                f"var_{q}": v, f"es_{q}": e, "unexpected_loss": v - self.el_analytic,
                "var_pct_ead": v / self.total_ead, "es_pct_ead": e / self.total_ead}


def build_cells(pd_, lgd, ead, sector, n_pd_bands: int = 20) -> pd.DataFrame:
    """Group loans into (sector, PD-band) cells. Cell PD is the EAD*LGD-weighted mean PD,
    which makes the simulated EL equal the loan-level EL exactly."""
    d = pd.DataFrame({"pd": pd_, "w": np.asarray(lgd) * np.asarray(ead), "sector": sector})
    d["band"] = pd.qcut(d["pd"].rank(method="first"), min(n_pd_bands, len(d)), labels=False)
    d["pdw"] = d["pd"] * d["w"]
    c = d.groupby(["sector", "band"]).agg(n=("pd", "size"), w_total=("w", "sum"),
                                          pdw=("pdw", "sum"), pd_mean=("pd", "mean"))
    c["pd"] = np.where(c["w_total"] > 0, c["pdw"] / c["w_total"].where(c["w_total"] > 0, 1),
                       c["pd_mean"])
    c["w_mean"] = c["w_total"] / c["n"]
    return c.drop(columns=["pdw", "pd_mean"]).reset_index()


def simulate_losses(pd_, lgd, ead, sector, rho: float = 0.15, sector_rho: float = 0.05,
                    sector_rho_overrides: dict | None = None, n_paths: int = 100_000,
                    seed: int = 42, batch_size: int = 10_000, method: str = "binomial",
                    n_pd_bands: int = 20) -> LossDistribution:
    pd_ = np.clip(np.asarray(pd_, dtype=float), 1e-10, 1 - 1e-10)
    lgd, ead = np.asarray(lgd, dtype=float), np.asarray(ead, dtype=float)
    cells = build_cells(pd_, lgd, ead, np.asarray(sector), n_pd_bands)

    sectors = sorted(cells["sector"].unique())
    s_idx = cells["sector"].map({s: i for i, s in enumerate(sectors)}).to_numpy()
    rho_s_by_sector = np.array([(sector_rho_overrides or {}).get(s, sector_rho) for s in sectors])
    rho_s = rho_s_by_sector[s_idx]
    if np.any(rho + rho_s >= 1) or rho < 0 or np.any(rho_s < 0):
        raise ValueError("Need 0 <= rho, rho_s and rho + rho_s < 1")

    thresh = ndtri(cells["pd"].to_numpy())
    idio_sd = np.sqrt(1.0 - rho - rho_s)
    n_c = cells["n"].to_numpy()
    w_mean, w_total = cells["w_mean"].to_numpy(), cells["w_total"].to_numpy()

    rng = np.random.default_rng(seed)
    losses = np.empty(n_paths)
    for start in range(0, n_paths, batch_size):
        m = min(batch_size, n_paths - start)
        z = rng.standard_normal((m, 1))
        zs = rng.standard_normal((m, len(sectors)))[:, s_idx]          # (m, cells)
        p_cond = ndtr((thresh - np.sqrt(rho) * z - np.sqrt(rho_s) * zs) / idio_sd)
        if method == "binomial":
            defaults = rng.binomial(n_c, p_cond)                         # (m, cells)
            losses[start:start + m] = defaults @ w_mean
        elif method == "conditional":
            losses[start:start + m] = p_cond @ w_total
        else:
            raise ValueError(f"unknown method {method!r}")
    return LossDistribution(losses, float(np.sum(pd_ * lgd * ead)), float(ead.sum()))


# --------------------------------------------------------------------------- closed forms
def vasicek_loss_quantile(pd_, lgd, ead, rho: float, q: float = 0.999) -> float:
    """ASRF closed form: q-quantile of portfolio loss = sum_i LGD_i EAD_i p_i(Z = Phi^-1(1-q))."""
    pd_ = np.clip(np.asarray(pd_, dtype=float), 1e-10, 1 - 1e-10)
    p_q = ndtr((ndtri(pd_) + np.sqrt(rho) * ndtri(q)) / np.sqrt(1 - rho))
    return float(np.sum(np.asarray(lgd) * np.asarray(ead) * p_q))


def basel_retail_correlation(pd_) -> np.ndarray:
    """Basel II/III 'other retail' asset correlation: 3%-16%, decreasing in PD."""
    pd_ = np.asarray(pd_, dtype=float)
    w = (1 - np.exp(-35 * pd_)) / (1 - np.exp(-35))
    return 0.03 * w + 0.16 * (1 - w)


def irb_capital_k(pd_, lgd, correlation=None, q: float = 0.999) -> np.ndarray:
    """IRB capital requirement K per unit EAD (retail: no maturity adjustment):
    K = LGD * [Phi((Phi^-1(PD) + sqrt(R) Phi^-1(0.999)) / sqrt(1-R)) - PD]."""
    pd_ = np.clip(np.asarray(pd_, dtype=float), 0.0003, 1 - 1e-10)  # Basel 3bp PD floor
    R = basel_retail_correlation(pd_) if correlation is None else np.full_like(pd_, correlation)
    cond = ndtr((ndtri(pd_) + np.sqrt(R) * ndtri(q)) / np.sqrt(1 - R))
    return np.asarray(lgd, dtype=float) * (cond - pd_)


def irb_comparison(port: pd.DataFrame, mc: LossDistribution, rho: float,
                   q: float = 0.999) -> pd.DataFrame:
    """Compare MC unexpected loss (VaR - EL) with closed-form capital.

    Rows: (1) MC with sector factor, (2) Vasicek ASRF at the same rho (isolates the
    effect of sector concentration + granularity), (3) Basel IRB other-retail formula
    with regulatory correlations (what a regulator would require)."""
    pd_, lgd, ead = port["pd"].to_numpy(), port["lgd"].to_numpy(), port["ead"].to_numpy()
    el = float(np.sum(pd_ * lgd * ead))
    k_irb = irb_capital_k(pd_, lgd)
    rows = [
        {"method": "Monte Carlo (one-factor + sector factor)", "unexpected_loss": mc.var(q) - el},
        {"method": f"Vasicek ASRF closed form (rho={rho})",
         "unexpected_loss": vasicek_loss_quantile(pd_, lgd, ead, rho, q) - el},
        {"method": "Basel IRB other-retail (R=3%-16%)",
         "unexpected_loss": float(np.sum(k_irb * ead))},
    ]
    t = pd.DataFrame(rows)
    t["pct_of_ead"] = t["unexpected_loss"] / ead.sum()
    t.loc[2, "rwa"] = 12.5 * t.loc[2, "unexpected_loss"]
    return t
