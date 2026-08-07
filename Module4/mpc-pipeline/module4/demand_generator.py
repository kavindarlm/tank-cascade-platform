"""
demand_generator.py
===================
Paddy irrigation demand for the Mahakanumulla cascade, built from OFFICIAL
Sri Lankan government Crop Water Requirement (CWR) data.

SOURCE
------
Climate Smart Irrigated Agriculture Project (CSIAP), Ministry of Agriculture and
Plantation Industries / World Bank (2024). "District Wise Reference Crop
Evapotranspiration (ETo) and Crop Water Requirements (ETc)."
  - Anuradhapura District, Paddy, Maha and Yala seasons.
  - Tables 5.4 (135-day), 5.5 (105-day), 5.6 (90-day).
  - Values are ETc in mm per DECADE (10-day period), computed with CROPWAT 8.0
    using the FAO Penman-Monteith method on Anuradhapura meteorological data.
  - The CWR ALREADY INCLUDES land preparation (soaking 101.6 mm + tilling 76.2 mm
    = 177.8 mm) as its first two decades, and already accounts for the paddy crop
    coefficient through the season. It is a FIELD-level requirement.

WHY THIS REPLACES THE OLD FAO-56 BUILD-UP
-----------------------------------------
The previous generator computed ETc from an assumed Kc curve and a constant ET0,
then ADDED a separate percolation term and a separate land-prep term and divided
by an efficiency. That stacked assumptions and double-counted (land prep twice;
percolation on top of a CWR that already reflects field losses). The government
CWR is authoritative, Anuradhapura-specific, paddy-specific, season-specific, and
citable - so demand is taken directly from it.

TANK-RELEASE CONVERSION (Option A: conveyance efficiency only)
--------------------------------------------------------------
The CWR is the requirement AT THE FIELD. Water released from the tank travels
through channels and loses some to seepage/evaporation in transit, so the tank
must release MORE than the field needs:

    gross_at_tank (mm) = CWR_field (mm) / conveyance_efficiency

Conveyance efficiency for the Anuradhapura dry zone is taken as ~0.70, based on
measured canal conveyance losses of ~26.5% at Hurulu Wewa (an Anuradhapura dry-zone
scheme) reported by:
  Insights on the irrigation conveyance efficiency of three canal systems for
  improving agricultural sustainability in Sri Lanka (2022-2023 field study,
  float-method flow measurements; Maduru Oya 27.8%, Hurulu Wewa 26.5%,
  Minipe LBC 33.1%).

This is CONVEYANCE efficiency only (tank -> field edge). Field-APPLICATION losses
(field edge -> root zone) are outside the model boundary, consistent with the
model's scope ending at water delivery to the command area. State this clearly in
the thesis.

VOLUME
------
    D_i(t) [m3/day] = gross_at_tank_mm(t) * command_area_ha_i * 10
(1 mm depth over 1 ha = 10 m3.)
"""

import numpy as np
from .config import Config, DEFAULT_CONFIG


# ==============================================================
# OFFICIAL ANURADHAPURA PADDY CWR  (mm per 10-day decade)
# Source: CSIAP 2024, Tables 5.4 / 5.5 / 5.6.
# First two decades of every series are land preparation.
# ==============================================================
ANURADHAPURA_PADDY_CWR = {
    ("maha", 135): [101.6, 76.2, 39.2, 35.9, 34.6, 32.6, 31.1, 37.3,
                    37.1, 39.4, 45.6, 43.5, 45.5, 38.5, 48.6, 24.6],
    ("maha", 105): [101.6, 76.2, 39.2, 35.9, 34.6, 32.7, 31.1, 37.1,
                    36.7, 38.9, 43.5, 39.2, 11.9],
    ("maha", 90):  [101.6, 76.2, 39.2, 35.9, 34.6, 32.6, 31.0, 36.9,
                    36.5, 37.2, 33.0],
    ("yala", 135): [101.6, 76.2, 55.0, 54.9, 60.1, 54.8, 55.1, 56.0,
                    57.0, 57.6, 64.2, 59.2, 60.0, 63.6, 53.0, 10.0],
    ("yala", 105): [101.6, 76.2, 55.0, 54.9, 60.6, 55.7, 56.2, 56.8,
                    57.1, 57.5, 61.8, 53.8, 15.8],
    ("yala", 90):  [101.6, 76.2, 55.0, 54.9, 60.7, 55.9, 56.4, 56.8,
                    57.1, 55.0, 47.0],
}

# Official seasonal totals (Tables 3.1 / 3.2) for validation.
CWR_SEASONAL_TOTAL = {
    ("maha", 135): 711.3, ("maha", 105): 558.6, ("maha", 90): 494.7,
    ("yala", 135): 938.3, ("yala", 105): 763.0, ("yala", 90): 676.6,
}


def decade_to_daily(decade_mm):
    """
    Expand a per-decade CWR series (mm/decade) into a per-day series (mm/day).

    Each decade covers 10 days at a constant daily rate (decade / 10). The last
    decade may be a partial period in the official tables; it is still treated as
    up to 10 days at its stated daily rate, which is the standard CROPWAT reading.

    Returns
    -------
    np.ndarray of daily CWR in mm/day (length = 10 * number_of_decades)
    """
    daily = []
    for dec in decade_mm:
        daily.extend([dec / 10.0] * 10)
    return np.array(daily)


def get_cwr_daily(cfg: Config, duration_days: int = None):
    """
    Return the daily FIELD CWR curve (mm/day) for the configured season.

    Parameters
    ----------
    cfg           : Config  (uses cfg.SEASON)
    duration_days : 135, 105, or 90. If None, chosen from cfg (default 135).

    Returns
    -------
    daily_cwr_mm : (season_days,)  field-level CWR in mm/day
    duration     : int  the duration actually used
    """
    season = cfg.SEASON
    duration = duration_days or getattr(cfg, "PADDY_DURATION_DAYS", 135)
    key = (season, duration)
    if key not in ANURADHAPURA_PADDY_CWR:
        raise ValueError(
            f"No Anuradhapura CWR data for season='{season}', duration={duration}. "
            f"Available: {sorted(ANURADHAPURA_PADDY_CWR.keys())}"
        )
    daily = decade_to_daily(ANURADHAPURA_PADDY_CWR[key])
    return daily, duration


def field_to_tank(daily_cwr_mm, cfg: Config):
    """
    Convert FIELD CWR (mm/day) to GROSS-AT-TANK release depth (mm/day) by dividing
    by conveyance efficiency (Option A - conveyance only).

    Returns
    -------
    gross_mm : (season_days,)  gross release depth required at the tank (mm/day)
    """
    eta_c = cfg.CONVEYANCE_EFFICIENCY
    return daily_cwr_mm / eta_c


def generate_demand(command_area_acres, cfg: Config = DEFAULT_CONFIG,
                    duration_days: int = None):
    """
    Generate the full-season daily demand matrix for all tanks from the official
    Anuradhapura paddy CWR.

    Parameters
    ----------
    command_area_acres : (N,)  command area of each tank in acres
    cfg                : Config
    duration_days      : 135 / 105 / 90 (default 135 or cfg.PADDY_DURATION_DAYS)

    Returns
    -------
    demand_m3 : (N, season_days)  daily demand in m3/day (gross at tank)
    diagnostics : dict
    """
    daily_cwr_mm, duration = get_cwr_daily(cfg, duration_days)      # field, mm/day
    gross_mm = field_to_tank(daily_cwr_mm, cfg)                     # tank, mm/day
    season_days = len(gross_mm)

    area_ha = np.asarray(command_area_acres, dtype=float) * cfg.ACRE_TO_HA  # (N,)
    demand_m3 = gross_mm[np.newaxis, :] * area_ha[:, np.newaxis] * 10       # (N, days)

    field_total = float(daily_cwr_mm.sum())
    diagnostics = {
        "season": cfg.SEASON,
        "duration_days": duration,
        "season_days": season_days,
        "field_cwr_total_mm": round(field_total, 1),
        "official_total_mm": CWR_SEASONAL_TOTAL.get((cfg.SEASON, duration)),
        "conveyance_efficiency": cfg.CONVEYANCE_EFFICIENCY,
        "gross_at_tank_total_mm": round(float(gross_mm.sum()), 1),
        "daily_cwr_mm": daily_cwr_mm,
        "gross_mm": gross_mm,
    }
    return demand_m3, diagnostics


def demand_window(demand_m3_full, t_current, T):
    """
    Extract a T-day demand window starting at day t_current for one MPC step.
    If the window runs past season end, the last day's demand is held (crop
    harvested; no further irrigation, so this is a mild over-estimate that the
    optimiser simply won't need to satisfy).

    Parameters
    ----------
    demand_m3_full : (N, season_days)
    t_current      : int
    T              : int

    Returns
    -------
    (N, T)
    """
    N, season_days = demand_m3_full.shape
    window = np.zeros((N, T))
    for d in range(T):
        idx = min(t_current + d, season_days - 1)
        window[:, d] = demand_m3_full[:, idx]
    return window