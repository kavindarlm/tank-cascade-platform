"""
scenario_generator.py
=====================
Generate K stochastic rainfall-inflow scenarios for the mass balance.

Under the agreed design (Option A), Module 3's risk probabilities DO NOT enter here.
Instead, inflow scenarios are produced directly from historical rainfall:

    1. Take the last RAINFALL_HISTORY_DAYS of daily rainfall (mm) per tank
       (from Open-Meteo, same API Module 3 uses).
    2. Convert rainfall -> catchment inflow (m3/day) via a lumped runoff coefficient:
           Q = C * rainfall_mm * catchment_km2 * 1000
       (1 mm over 1 km2 = 1000 m3).
    3. Fit a Gamma distribution per tank by moment matching:
           shape kappa = mu^2 / sigma^2 ,  scale theta = sigma^2 / mu
    4. Draw K samples per tank per horizon-day.

Why Gamma: rainfall/inflow is strictly non-negative and right-skewed; Gamma is the
standard parametric choice in statistical hydrology (Wilks, 2011).

Why a 90-day (climatological) history window rather than ~30 days: it decouples the
scenario mean from Module 3's short-term drought signal, so that Option A's risk
weighting is not double-counting the same recent-rainfall information. See explanation
.md, "Double counting" section.
"""

import numpy as np
from .config import Config, DEFAULT_CONFIG
from . import logging_utils as lg


def rainfall_to_inflow(rainfall_mm: np.ndarray,
                       catchment_km2: np.ndarray,
                       cfg: Config) -> np.ndarray:
    """
    Convert a rainfall history (mm/day) to catchment inflow (m3/day).

    Parameters
    ----------
    rainfall_mm   : (N, H)  daily rainfall per tank over H history days
    catchment_km2 : (N,)    catchment area per tank in km2
    cfg           : Config

    Returns
    -------
    inflow_m3 : (N, H)  historical inflow per tank per day
    """
    return (cfg.RUNOFF_COEFFICIENT *
            rainfall_mm *
            catchment_km2[:, np.newaxis] *
            1000.0)


def fit_gamma_params(inflow_history: np.ndarray):
    """
    Fit Gamma (shape, scale) per tank by moment matching on ALL days.

    Retained for the ablation (cfg.BERNOULLI_GAMMA = False). See
    fit_occurrence_intensity for why this is the wrong model for daily rainfall.

    Parameters
    ----------
    inflow_history : (N, H)  historical inflow per tank

    Returns
    -------
    shape : (N,)  Gamma shape parameter kappa (0 where undefined)
    scale : (N,)  Gamma scale parameter theta (0 where undefined)
    """
    N = inflow_history.shape[0]
    mu = inflow_history.mean(axis=1)                 # (N,)
    var = inflow_history.var(axis=1)                 # (N,)

    shape = np.zeros(N)
    scale = np.zeros(N)
    valid = (mu > 0) & (var > 0)
    shape[valid] = mu[valid] ** 2 / var[valid]
    scale[valid] = var[valid] / mu[valid]
    return shape, scale


def fit_occurrence_intensity(inflow_history: np.ndarray, thresh=0.0):
    """
    Bernoulli-Gamma (occurrence-intensity) fit - the standard model for daily
    rainfall.

    Daily dry-zone rainfall is ZERO-INFLATED: most days are dry, a few carry
    large totals. Moment-matching a single Gamma to that series preserves the
    mean but destroys the temporal structure - the fitted Gamma is continuous
    and strictly positive, so almost every sampled day is a wet day.

    Measured on synthetic dry-zone rainfall with a true 75% dry-day share:

        model              mean     max/mean   dry-day share
        history           2,164          43x          75.0%
        plain Gamma       2,115          53x           1.9%   <- structure gone
        Bernoulli-Gamma   2,314          63x          75.3%   <- preserved

    Note what this does and does NOT fix. The mean is essentially unchanged, and
    the heavy tail SURVIVES, because real rainfall genuinely is heavy-tailed. It
    fixes the thing that matters physically: a rain-fed cascade receives water in
    episodes, not as a constant daily drip. A constant drip lets every tank fill
    smoothly and makes spill look inevitable everywhere at once.

    Parameters
    ----------
    inflow_history : (N, H)  historical inflow per tank
    thresh         : float or (N,)  inflow above which a day counts as wet.
                     Pass a per-tank vector derived from a mm threshold; a
                     threshold of 0 counts reanalysis drizzle as rain.

    Returns
    -------
    p_wet : (N,)  probability a given day is wet
    shape : (N,)  Gamma shape fitted on WET DAYS ONLY
    scale : (N,)  Gamma scale fitted on WET DAYS ONLY
    """
    N, H = inflow_history.shape
    p_wet = np.zeros(N)
    shape = np.zeros(N)
    scale = np.zeros(N)
    thresh = np.broadcast_to(np.asarray(thresh, dtype=float).reshape(-1), (N,)) \
        if np.ndim(thresh) else np.full(N, float(thresh))

    for i in range(N):
        wet = inflow_history[i][inflow_history[i] > thresh[i]]
        if wet.size < 2:
            continue
        p_wet[i] = wet.size / H
        mu, var = wet.mean(), wet.var()
        if mu > 0 and var > 0:
            shape[i] = mu ** 2 / var
            scale[i] = var / mu
    return p_wet, shape, scale


def generate_scenarios(rainfall_mm: np.ndarray,
                       catchment_km2: np.ndarray,
                       cfg: Config = DEFAULT_CONFIG,
                       seed: int = None) -> np.ndarray:
    """
    Generate K inflow scenarios over the T-day horizon for all tanks.

    Parameters
    ----------
    rainfall_mm   : (N, H)  daily rainfall history per tank (mm/day)
    catchment_km2 : (N,)    catchment area per tank (km2)
    cfg           : Config
    seed          : optional int override for the RNG seed

    Returns
    -------
    scenarios : (K, N, T)  sampled inflow trajectories in m3/day
    """
    rng = np.random.default_rng(cfg.SCENARIO_SEED if seed is None else seed)
    N = rainfall_mm.shape[0]
    K, T = cfg.K, cfg.T

    inflow_hist = rainfall_to_inflow(rainfall_mm, catchment_km2, cfg)
    scenarios = np.zeros((K, N, T))

    if getattr(cfg, "BERNOULLI_GAMMA", True):
        # Occurrence-intensity: draw a wet/dry day, then an intensity if wet.
        #
        # The wet-day THRESHOLD matters. Reanalysis precipitation (Open-Meteo /
        # ERA5) reports trace drizzle of 0.05-0.5 mm on a large share of days
        # that are dry in any practical sense. With a threshold of 0 those days
        # count as wet, p_wet is inflated toward 1, and the episodic structure
        # the Bernoulli-Gamma exists to capture is lost again - the sampled
        # dry-day share collapses to match a history that is itself wrong.
        #
        # The standard meteorological definition of a wet day is >= 1 mm. The
        # threshold is converted to an inflow volume per tank using the same
        # runoff relation as the history: C * A_km2 * mm * 1000.
        thr_mm = getattr(cfg, "WET_DAY_THRESHOLD_MM", 1.0)
        thresh = (cfg.RUNOFF_COEFFICIENT * thr_mm
                  * np.asarray(catchment_km2, dtype=float) * 1000.0)
        p_wet, shape, scale = fit_occurrence_intensity(inflow_hist, thresh)
        for i in range(N):
            if shape[i] > 0 and scale[i] > 0 and p_wet[i] > 0:
                occ = rng.random((K, T)) < p_wet[i]
                intensity = rng.gamma(shape[i], scale[i], size=(K, T))
                scenarios[:, i, :] = np.where(occ, intensity, 0.0)
    else:
        shape, scale = fit_gamma_params(inflow_hist)
        for i in range(N):
            if shape[i] > 0 and scale[i] > 0:
                scenarios[:, i, :] = rng.gamma(shape[i], scale[i], size=(K, T))

    # Physical ceiling on a single day's catchment inflow. Guards against the
    # Gamma tail producing a draw no storm could deliver. C * A * max_mm.
    cap_mm = getattr(cfg, "MAX_DAILY_RAINFALL_MM", 0.0)
    if cap_mm > 0:
        cap = (cfg.RUNOFF_COEFFICIENT * cap_mm
               * np.asarray(catchment_km2, dtype=float) * 1000.0)
        scenarios = np.minimum(scenarios, cap[None, :, None])

    log_gamma_fit(inflow_hist, shape, scale, scenarios, cfg, catchment_km2)
    return scenarios


def log_gamma_fit(inflow_hist, shape, scale, scenarios, cfg: Config = DEFAULT_CONFIG,
                  catchment_km2=None):
    """
    Print a diagnostic summary of the Gamma fit and the resulting scenario
    tensor. Observability only - called after sampling, changes nothing.

    The fitted mean/variance RANGE across tanks is the useful signal here: a
    tank whose fit collapsed (zero recent rain) will show as a dry tank, and a
    huge variance range hints that one catchment dominates the inflow.
    """
    log = lg.Log(cfg)
    if not log.enabled or not getattr(cfg, "LOG_SCENARIOS", True):
        return

    # Gamma moments from the fitted parameters: mean = k*theta, var = k*theta^2
    g_mean = shape * scale
    g_var = shape * scale ** 2
    n_dry = int(np.sum((shape <= 0) | (scale <= 0)))
    K, N, T = scenarios.shape

    log.kv("history window", f"{inflow_hist.shape[1]} days × {N} tanks "
                             f"(C = {cfg.RUNOFF_COEFFICIENT})")
    log.kv("gamma mean range", f"{lg.fmt_m3(g_mean.min())} … {lg.fmt_m3(g_mean.max())} /day")
    log.kv("gamma std range", f"{lg.fmt_m3(np.sqrt(g_var).min())} … "
                              f"{lg.fmt_m3(np.sqrt(g_var).max())} /day")
    if n_dry:
        log.warn(f"{n_dry} tank(s) had no rainfall in the window → zero inflow assumed",
                 indent=2)
    log.kv("scenario tensor", f"K={K} × N={N} × T={T}")
    log.kv("mean inflow", f"{lg.fmt_m3(scenarios.mean())} /tank/day")
    log.kv("max sampled inflow", f"{lg.fmt_m3(scenarios.max())} /tank/day")
    # Dry-day share is the check that matters: a rain-fed cascade receives water
    # in episodes. If the sampled share is far below the historical share, the
    # generator has turned rainfall into a constant drip and the cascade will
    # fill smoothly and unphysically.
    # Both shares must be measured at the SAME threshold or the comparison is
    # meaningless: sampled dry days are exact zeros produced by the Bernoulli
    # draw, so the history must be judged against the same wet-day definition.
    thr_mm = getattr(cfg, "WET_DAY_THRESHOLD_MM", 1.0)
    if catchment_km2 is not None:
        thr_vol = (cfg.RUNOFF_COEFFICIENT * thr_mm
                   * np.asarray(catchment_km2, dtype=float) * 1000.0)[:, None]
        hist_dry = float(np.mean(inflow_hist <= thr_vol)) * 100.0
    else:
        hist_dry = float(np.mean(inflow_hist <= 1e-9)) * 100.0
    samp_dry = float(np.mean(scenarios <= 1e-9)) * 100.0
    log.kv("dry-day share", f"history {hist_dry:.1f}%  sampled {samp_dry:.1f}%  "
                            f"(wet-day threshold {thr_mm:g} mm)")
    if hist_dry < 40.0:
        log.warn(f"only {hist_dry:.1f}% of history days are dry. A dry-zone "
                 f"rainfall record should be 60-75% dry at a 1 mm threshold - "
                 f"check the fetched date range and WET_DAY_THRESHOLD_MM",
                 indent=2)
    if hist_dry - samp_dry > 20.0:
        log.warn("sampled rainfall has far fewer dry days than history - "
                 "episodic structure lost (set BERNOULLI_GAMMA=True)", indent=2)


def scenario_summary(scenarios: np.ndarray) -> dict:
    """Quick diagnostics on a scenario tensor (K, N, T)."""
    return {
        "shape": scenarios.shape,
        "mean_inflow_per_tank": scenarios.mean(axis=(0, 2)),   # (N,)
        "overall_mean": float(scenarios.mean()),
        "overall_max": float(scenarios.max()),
    }