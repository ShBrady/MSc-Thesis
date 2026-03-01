
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DES-Dovekie Hubble Diagram + MCMC Cosmology Fit
Compatible with NumPy 2.x
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import emcee
from scipy.stats import gaussian_kde

# =========================
# CONFIG
# =========================
DATA_FILE = "DES-Dovekie_HD.csv"

NWALKERS = 32
NSTEPS   = 2000
BURN     = 500
THIN     = 5
SEED     = 123

H0_PRIOR = (50.0, 90.0)
OM_PRIOR = (0.01, 1.5)
SIGMA_MCAL = 0.05

C_LIGHT = 299792.458  # km/s

plt.rcParams["figure.dpi"] = 130
plt.rcParams["axes.grid"] = True

# =========================
# DATA LOADER (DES FORMAT)
# =========================
def load_des_dovekie(path):
    rows = []
    varnames = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("VARNAMES:"):
                varnames = line.replace("VARNAMES:", "").split()
                continue

            if line.startswith("SN:"):
                vals = line.replace("SN:", "").split()
                if varnames is None:
                    continue

                row = {}
                row["SN"] = vals[0]
                for k, v in zip(varnames, vals[1:]):
                    try:
                        row[k] = float(v)
                    except ValueError:
                        row[k] = np.nan
                rows.append(row)

    return pd.DataFrame(rows)


# =========================
# COSMOLOGY (FAST & STABLE)
# =========================
def mu_theory(z, H0, Om):
    """
    Flat LCDM distance modulus (low-z safe)
    """
    Ez = np.sqrt(Om * (1 + z) ** 3 + (1 - Om))
    chi = np.cumsum(np.diff(np.insert(z, 0, 0)) / Ez)
    dL = (C_LIGHT / H0) * (1 + z) * chi
    return 5.0 * np.log10(dL) + 25.0


# =========================
# PROBABILITY MODEL
# =========================
def log_prior(theta):
    H0, Om, Mcal = theta
    if not (H0_PRIOR[0] < H0 < H0_PRIOR[1]):
        return -np.inf
    if not (OM_PRIOR[0] < Om < OM_PRIOR[1]):
        return -np.inf
    return -0.5 * (Mcal / SIGMA_MCAL) ** 2


def log_likelihood(theta, z, mu, muerr):
    H0, Om, Mcal = theta
    mu_model = mu_theory(z, H0, Om) + Mcal
    return -0.5 * np.sum(((mu - mu_model) / muerr) ** 2)


def log_posterior(theta, z, mu, muerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, z, mu, muerr)


# =========================
# KDE (NUMPY 2.x SAFE)
# =========================
def kde_1d(samples, grid):
    samples = samples[np.isfinite(samples)]
    if len(samples) < 10 or np.std(samples) == 0:
        return np.zeros_like(grid)

    kde = gaussian_kde(samples)
    y = kde(grid)
    area = np.trapezoid(y, grid)

    if not np.isfinite(area) or area <= 0:
        return np.zeros_like(grid)

    return y / area


# =========================
# MAIN
# =========================
def main():
    # -------- Load data --------
    df = load_des_dovekie(DATA_FILE)
    print("Loaded:", df.shape)

    # -------- Basic filtering --------

# ---- Basic filtering (robust to missing columns) ----
    mask = (
        (df["MUERR"] < 0.5) &
        np.isfinite(df["zHD"]) &
     np.isfinite(df["MU"])
    )

# Only apply BEAMS probability cut if column exists
    if "PROBIA_BEAMS" in df.columns:
        mask &= (df["PROBIA_BEAMS"] > 0.5)
    else:
        print("[INFO] PROBIA_BEAMS not found — skipping BEAMS cut")

    df = df[mask].copy()
    df.sort_values("zHD", inplace=True)

    df.sort_values("zHD", inplace=True)

    z = df["zHD"].to_numpy()
    mu = df["MU"].to_numpy()
    muerr = df["MUERR"].to_numpy()

    print("Using N =", len(z))

    # -------- Hubble diagram --------
    plt.figure(figsize=(6, 4))
    plt.errorbar(z, mu, yerr=muerr, fmt=".", alpha=0.5)
    plt.xlabel("zHD")
    plt.ylabel("Distance Modulus μ")
    plt.title("DES-Dovekie Hubble Diagram")
    plt.show()

    # -------- MCMC --------
    rng = np.random.default_rng(SEED)
    p0 = np.array([70.0, 0.3, 0.0])
    p0 = p0 + 1e-2 * rng.standard_normal((NWALKERS, 3))

    sampler = emcee.EnsembleSampler(
        NWALKERS, 3, log_posterior, args=(z, mu, muerr)
    )
    sampler.run_mcmc(p0, NSTEPS, progress=True)

    samples = sampler.get_chain(discard=BURN, thin=THIN, flat=True)

    # -------- Posterior plots --------
    H0_s = samples[:, 0]
    Om_s = samples[:, 1]

    H0_grid = np.linspace(55, 85, 300)
    Om_grid = np.linspace(0.05, 0.8, 300)

    plt.figure(figsize=(6, 4))
    plt.plot(H0_grid, kde_1d(H0_s, H0_grid))
    plt.xlabel(r"$H_0$")
    plt.ylabel("Posterior density")
    plt.title("Posterior of $H_0$")
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(kde_1d(Om_s, Om_grid), Om_grid)
    plt.ylabel(r"$\Omega_m$")
    plt.xlabel("Posterior density")
    plt.title("Posterior of $\Omega_m$")
    plt.show()

    # -------- Summary --------
    def summarize(x):
        q16, q50, q84 = np.percentile(x, [16, 50, 84])
        return q50, q84 - q50, q50 - q16

    H0_med, H0_p, H0_m = summarize(H0_s)
    Om_med, Om_p, Om_m = summarize(Om_s)

    print("\n===== RESULTS =====")
    print(f"H0 = {H0_med:.2f} +{H0_p:.2f} -{H0_m:.2f}")
    print(f"Om = {Om_med:.3f} +{Om_p:.3f} -{Om_m:.3f}")

    print("Mean acceptance fraction:",
          np.mean(sampler.acceptance_fraction))


if __name__ == "__main__":
    main()
