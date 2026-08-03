"""
config.py
=========
All hyperparameters, physical constants, and design switches for Module 4 in one place.

Change this file (not the algorithm files) to adjust the planning horizon, number of
scenarios, NSGA-II settings, TOPSIS weights, or the agreed hydrological assumptions.

CHANGELOG (Module 3 update response):
  + SHAPE_FACTOR              - capacity derivation, calibrated on Nachchaduwa
  + MODULE3_STORAGE_BASIS     - whether Module 3 reports area% or volume%
  + AREA_TO_VOLUME_EXPONENT   - conic reservoir approximation for the cross-check
  + CROSSCHECK_BASELINE_DAY   - which forecast day the ratio comparison anchors on
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Module 2's live GNN output - the cascade transfer/alpha matrix. Module 4 no
# longer keeps its own copy under data/; it reads Module 2's output directly,
# so a re-trained/updated alpha matrix is picked up automatically. Resolved
# from this file's own location (not cwd), so it works no matter where a
# Module 4 script is invoked from. NOT alpha_matrix_v2.csv.
_MODULE2_ALPHA_MATRIX_PATH = str(
    Path(__file__).resolve().parents[3]
    / "Module2" / "development-history" / "outputs" / "alpha_matrix.csv"
)

# Module 3's live daily forecast output folder (one forecast_<date>.csv per
# date, written by Module3/forecast-service/batch_forecast.py). Only used
# when MODULE3_LIVE_FORECAST=True - see data_loader.load_module3_risk_live.
# Resolved from this file's own location, same reasoning as the alpha path.
_MODULE3_FORECAST_OUTPUTS_DIR = str(
    Path(__file__).resolve().parents[3]
    / "Module3" / "forecast-service" / "outputs"
)


@dataclass
class Config:
    # ==============================================================
    # PLANNING HORIZON  &  SEASON
    # ==============================================================
    # STORAGE SOURCE PER MPC DAY
    #   False (simulation)  - the loop propagates its own state forward with the
    #                         mass balance; tank_storage.csv is read once at t=0.
    #   True  (deployment)  - today's storage is re-measured from file each day,
    #                         so the controller is corrected against reality and
    #                         internal state propagation is skipped.
    RELOAD_STORAGE_EACH_DAY: bool = False

    # MODULE 3 RISK/FORECAST SOURCE PER MPC DAY
    #   False (default) - module3_risk.csv is read once from DATA_DIR, exactly
    #                      as before this flag existed (bulk-CSV / full-season
    #                      runs via run_simulation.py are NOT affected by this
    #                      flag at all - they never read it).
    #   True  (deployment) - main.py's single-day path reads Module 3's live
    #                      daily output directly (Module3/forecast-service/
    #                      outputs/forecast_<RUN_DATE>.csv) instead of the
    #                      static module3_risk.csv - see
    #                      data_loader.load_module3_risk_live. Requires
    #                      SEASON_LENGTH == 1 (validated below): Module 3 only
    #                      guarantees a file for the one date you generated,
    #                      not a run of consecutive days.
    MODULE3_LIVE_FORECAST: bool = True

    # Which date's Module 3 forecast to read when MODULE3_LIVE_FORECAST=True.
    # None -> real "today" (date.today()) at the point of use. Set explicitly
    # to test/run against a specific already-generated forecast_<date>.csv
    # without waiting for the system clock.
    RUN_DATE: Optional[str] = "2026-07-29"

    MODULE3_FORECAST_OUTPUTS_DIR: str = _MODULE3_FORECAST_OUTPUTS_DIR

    T: int = 7                    # planning horizon in days (receding-horizon window)
    SEASON_LENGTH: int = 1       # full irrigation season length for the MPC loop
    SEASON: str = "maha"          # 'maha' (Oct-Mar) or 'yala' (Apr-Aug)

    # ==============================================================
    # STOCHASTIC SCENARIOS   (Section: scenario generation)
    # ==============================================================
    K: int = 20                  # number of Gamma-sampled rainfall scenarios
    # Length of the rainfall history used to fit the inflow distribution.
    # Documentation only - data_loader reads whatever columns rainfall_history.csv
    # actually has. Set it to match so the two do not silently disagree.
    #   184 = one Yala season (Mar 1 - Aug 31)
    #   552 = three Yala seasons concatenated (recommended):
    #         python fetch_rainfall.py --years 2022 2023 2024
    RAINFALL_HISTORY_DAYS: int = 552  # window length for fitting the Gamma distribution.
                                      #   NOTE: 90 (climatological) is used deliberately
                                      #   to DECOUPLE the scenario mean from Module 3's
                                      #   short-term drought signal (avoids double-counting
                                      #   under Option A). See explanation .md, "Double
                                      #   counting" section.
    # Bernoulli-Gamma (occurrence-intensity) scenario sampling. Daily dry-zone
    # rainfall is zero-inflated; a single Gamma fitted to all days preserves the
    # mean but samples almost no dry days, turning episodic rainfall into a
    # constant drip. False = original single-Gamma fit, for the ablation.
    BERNOULLI_GAMMA: bool = True

    # A day counts as WET above this daily rainfall. Reanalysis products report
    # trace drizzle on most days; at a 0 mm threshold those count as rain, p_wet
    # is inflated and the episodic structure is lost. 1 mm is the standard
    # meteorological definition of a rain day.
    WET_DAY_THRESHOLD_MM: float = 1.0

    # Physical ceiling on one day's catchment rainfall (mm). 0 disables.
    # Inflow is capped at C * A_km2 * mm * 1000, guarding against Gamma draws
    # no storm could deliver.
    MAX_DAILY_RAINFALL_MM: float = 150.0

    RUNOFF_COEFFICIENT: float = 0.16 # C: fraction of rainfall that becomes catchment
                                      #    runoff into the tank. Dry-zone lumped value.
                                      #    SUBJECT TO SENSITIVITY ANALYSIS.
    SCENARIO_SEED: int = 42       # RNG seed for reproducible scenarios

    # ==============================================================
    # NSGA-II  HYPERPARAMETERS   (Section: optimiser)
    # ==============================================================
    POP_SIZE: int = 200           # population size
    N_GEN: int = 300              # number of generations
    CROSSOVER_PROB: float = 0.9   # SBX crossover probability
    CROSSOVER_ETA: float = 15.0   # SBX distribution index
    MUTATION_PROB: float = 0.1    # polynomial mutation probability
    MUTATION_ETA: float = 20.0    # polynomial mutation distribution index
    NSGA2_SEED: int = 42          # RNG seed for reproducible optimisation

    # Seed the initial population with the no-action plan (zero release) plus
    # the reference plan. The baseline floors make the feasible set non-empty,
    # but zero release is one point in a ~217-dimensional box and a uniformly
    # random population does not find it - a correct feasible set still gave
    # 0/40 feasible individuals after 40 generations in testing. Set False to
    # reproduce the unseeded behaviour.
    SEED_NO_ACTION_PLAN: bool = True

    # Terminal-sink operating rule: decrement the sink's available water as it
    # releases across the horizon. Without this the sink can REQUEST up to T
    # times the water it holds. No water is invented (the simulator caps
    # R_eff), and feasibility and the objectives are unaffected since the sink
    # is excluded from C1-C5 - but the sink's reported release and storage
    # trajectory are wrong, and that trajectory is your cascade-health metric
    # and Module 3 cross-check series. False = original behaviour.
    SINK_RULE_DECREMENT_AVAIL: bool = True

    # Constraint-handling penalty weight (soft penalty added to objectives).
    PENALTY_WEIGHT: float = 1.0e6

    # ==============================================================
    # TOPSIS  BASE WEIGHTS   (Section: decision layer / Option A)
    # Must sum to 1.0. These are the operator's DEFAULT policy priorities,
    # used on a day with no elevated risk.
    # ==============================================================
    W_SHORTAGE_BASE: float = 0.35   # f1 - unmet demand (most critical)
    W_OVERFLOW_BASE: float = 0.25   # f2 - spill / bund safety
    W_EQUITY_BASE:   float = 0.20   # f3 - fairness between tanks
    W_LOSS_BASE:     float = 0.20   # f4 - water waste

    # ---- Option A: risk-driven weight adjustment sensitivities ----
    ALPHA_DROUGHT_SHORTAGE: float = 0.20   # drought raises shortage weight
    ALPHA_DROUGHT_EQUITY:   float = 0.10   # drought raises equity weight
    BETA_OVERFLOW:          float = 0.20   # overflow raises overflow weight
    GAMMA_OVERFLOW_LOSS:    float = 0.10   # overflow lowers loss weight
    WEIGHT_FLOOR: float = 0.05             # every weight kept >= this after adjustment

    USE_MODULE3_RISK_WEIGHTS: bool = True  # master switch for Option A.

    # ==============================================================
    # CONSTRAINT SWITCHES   (Section: constraints)
    # ==============================================================
    ENABLE_C1_STORAGE_BOUNDS: bool = True   # S_min <= S <= S_max
    ENABLE_C2_RELEASE_BOUNDS: bool = True   # 0 <= R <= R_max
    ENABLE_C3_DROUGHT_FLOOR:  bool = True   # deterministic seasonal safety floor
    ENABLE_C4_MASS_CONSERVE:  bool = True   # sum(R) <= available water
    ENABLE_C5_NON_NEGATIVITY: bool = True   # explicit non-negativity
    ENABLE_C6_MAHAWELI:       bool = False  # OFF for Mahakanumulla (no Mahaweli inflow)

    C3_SAFETY_BUFFER_FRACTION: float = 0.05  # keep S >= S_min + 5%*S_max
                                             # (= 15% of capacity when
                                             #  DEAD_STORAGE_FRACTION = 0.10)
    DELTA_M_FRACTION: float = 0.10           # C6: +/- 10% of SCHEDULED release volume

    # ==============================================================
    # C1 / C3 FLOOR FORMULATION      (see constraints.py)
    # --------------------------------------------------------------
    # The original C1 low side and C3 were ABSOLUTE hard state constraints.
    # Release is unidirectional - the optimiser can only remove water - so a
    # tank that STARTS below the floor makes the feasible set empty and NSGA-II
    # returns 0/POP feasible on every generation, whatever the budget.
    #
    # True (default): baseline-relative floor
    #       S(t) >= min( S_base(t), floor )
    # where S_base is the trajectory under the no-action plan. The controller
    # is responsible only for the storage IT removes. Zero release reproduces
    # S_base and lies inside [0, R_max], so a feasible plan always exists.
    #
    # False: original absolute floor. Keep for the ablation row
    #        ("hard C3: 0/184 feasible days") in the evaluation chapter.
    C3_BASELINE_FLOOR: bool = True
    C1_BASELINE_DEFICIT: bool = True

    # How hard a tank BELOW the floor must work to recover, in [0, 1].
    #   1.0 - must track the no-action trajectory: banks 100% of incoming water
    #         and serves no demand until it climbs back over the floor.
    #   0.0 - must merely not fall below today's storage: may release everything
    #         that arrives, and never recovers.
    #   0.5 - banks half the natural recovery, releases the other half.
    # This is a hedging-policy choice, not a numerical one. It trades reserve
    # recovery against equity (f3) and shortage (f1) for the depleted tanks.
    # The feasibility guarantee holds for any value in [0, 1].
    C3_RECOVERY_FRACTION: float = 1.0

    # Evaluate the floors on the pre-clip trajectory. Clipping at S_min invents
    # water and censors the violation at buffer*S_max per tank-day. False =
    # legacy behaviour.
    C3_FLOOR_ON_UNCLIPPED: bool = True

    # Verify numerically, once per MPC day before the search starts, that the
    # no-action plan satisfies C3. Costs one cascade simulation and fails loudly
    # instead of burning ~8 minutes on an infeasible problem.
    C3_ASSERT_ZERO_FEASIBLE: bool = True

    # Tier 2: if NSGA-II still finds nothing feasible, re-solve with C3 dropped
    # from out['G'] and flag the day as 'relaxed' rather than skipping it.
    # C1/C2/C4/C5 continue to bind. Should essentially never fire.
    C3_SOFT_FALLBACK: bool = True

    # ==============================================================
    # SIMULATOR PHYSICS              (see state_transition.py)
    # --------------------------------------------------------------
    # True (physical): E_actual = min(E, water available above dead storage).
    # A tank cannot evaporate water it does not hold. With False a nearly-empty
    # tank evaporates its full potential E and lands below S_min even at R = 0,
    # making the C1 low-side violation uncontrollable.
    STORAGE_LIMITED_EVAPORATION: bool = True

    # Sweeps reconciling each tank's availability with the EFFECTIVE upstream
    # return flow. The original code sized availability from the REQUESTED
    # upstream release, crediting tanks with water their neighbours could not
    # send. The iteration starts from the conservative no-upstream estimate and
    # increases monotonically, so stopping early under-estimates availability
    # rather than inventing water. 1 = legacy, 3 = ample.
    UPSTREAM_FIXED_POINT_ITERS: int = 3

    # ==============================================================
    # TERMINAL SINK  (major downstream reservoir handled as cascade endpoint)
    # --------------------------------------------------------------
    # In a village-tank cascade the last tank is a large terminal reservoir that
    # COLLECTS the outflow of every upstream tank (Mahakanumulla -> Nachchaduwa).
    # Its storage is a CONSEQUENCE of the upstream decisions, not an independent
    # decision, and its command-area demand is ~an order of magnitude larger than
    # any village tank, so optimising its release in the same objective would let
    # it dominate f1 and distort the whole allocation.
    #
    # When a tank is listed here it is:
    #   - STILL fully simulated (mass balance, alpha inflow, evaporation, spill),
    #   - STILL cross-checked against Module 3 (validation unaffected),
    #   - EXCLUDED from the optimisation objectives f1-f4 and from the optimiser's
    #     decision variable,
    #   - operated by a simple rule (release = min(demand, available water)),
    #   - reported, with its storage trajectory used as a cascade-health metric.
    #
    # Set to [] to optimise every tank (original 32-tank behaviour). List the
    # tank_id string(s) exactly as they appear in tank_params.csv to separate them.
    TERMINAL_SINK_TANKS: list = field(default_factory=lambda: ["Nachchaduwa_Wewa","Kudaittikattiya_Wewa",
                              "Settikulama_Wewa","Galwaduwawa_Wewa"])

    # ==============================================================
    # PHYSICAL IRRIGATION CONSUMPTION   (see state_transition.py)
    # --------------------------------------------------------------
    # True (default, physically correct):
    #     each tank's command area consumes  min(R_effective, D)  and ONLY the
    #     leftover (field drainage / return flow) is routed downstream by alpha.
    #     Irrigation water is therefore actually removed from the cascade instead
    #     of being scored as delivered and then flowing on to the next tank.
    #
    #     Note this does NOT force full demand satisfaction: consumption is capped
    #     by what was actually released (min, not D), so the optimiser remains free
    #     to leave demand unmet - that trade-off still lives in f1/f4.
    #
    # False (legacy / ablation):
    #     the whole effective release cascades downstream (alpha @ R_eff) and
    #     demand exists only as a scoring target. Use this to reproduce the older
    #     results or to quantify the effect of consumption in the evaluation
    #     chapter.
    ENABLE_CROP_CONSUMPTION: bool = True

    # ==============================================================
    # CAPACITY DERIVATION   (used by prepare_tank_params.py)
    # --------------------------------------------------------------
    # Most tanks have no measured capacity, so Full Supply Level (FSL) volume is
    # estimated from water spread area and maximum depth:
    #
    #     S_max = WSA_at_FSL (m2)  x  max_depth (m)  x  SHAPE_FACTOR
    #
    # SHAPE_FACTOR accounts for the tank being a bowl, not a box.
    # Reference points:
    #     cone idealisation        = 0.333
    #     paraboloid idealisation  = 0.500
    #     CALIBRATED on Nachchaduwa (known 45,150 acre-ft vs WSA x depth) = 0.692
    #     box (no correction)      = 1.000
    #
    # Nachchaduwa is a major reservoir in a defined valley; the village tanks of the
    # Mahakanumulla cascade are broader and shallower, so a value slightly below the
    # Nachchaduwa calibration is used as the default.
    #
    # SENSITIVITY RANGE FOR EVALUATION: 0.50 - 0.70
    # ==============================================================
    SHAPE_FACTOR: float = 0.60

    # Where a measured capacity exists (e.g. Nachchaduwa), it is used directly and
    # SHAPE_FACTOR is bypassed for that tank.
    DEAD_STORAGE_FRACTION: float = 0.10   # S_min = 10% of S_max   (ASSUMPTION)
    MAX_RELEASE_FRACTION: float = 0.15    # R_max = 15% of S_max   (ASSUMPTION)
    EVAP_MM_PER_DAY: float = 5.0          # open-water evaporation, dry zone

    # ==============================================================
    # MODULE 3 CROSS-CHECK   (see cross_validation.py)
    # --------------------------------------------------------------
    # Module 3 reports storage as a percentage, but its 100% reference is Module 1's
    # total_ha (maximum historical flood EXTENT, an AREA), whereas Module 4 works in
    # VOLUME relative to FSL. The two denominators are incompatible, so the cross-check
    # compares TRAJECTORY SHAPE (each series normalised to its own baseline day) rather
    # than absolute level. Normalising cancels the unknown denominator algebraically:
    #
    #     F(t)/F(baseline) = A(t)/A(baseline)      <- total_ha cancels
    #
    # and the area ratio is converted to a volume ratio by the conic approximation:
    #
    #     V(t)/V(baseline) = ( A(t)/A(baseline) ) ^ AREA_TO_VOLUME_EXPONENT
    # ==============================================================
    MODULE3_STORAGE_BASIS: str = "volume"    # 'area'   -> apply the exponent below
                                           # 'volume' -> compare ratios directly
                                           # Set to 'volume' if Module 3 confirms its
                                           # storage_forecast is volumetric.
    AREA_TO_VOLUME_EXPONENT: float = 1.5   # V proportional to A^1.5 (conic reservoir)
    CROSSCHECK_BASELINE_DAY: int = 1       # forecast day used as the ratio anchor
                                           # (1 = first forecast day t+1; always available)
    CROSSCHECK_DIVERGENCE_THRESHOLD: float = 0.15  # flag tanks whose mean ratio
                                                   # difference exceeds this (15%)

    # ==============================================================
    # DEMAND MODEL  (FAO-56)
    # ==============================================================
    ET0_MM_PER_DAY: float = 5.0            # reference evapotranspiration
    PERCOLATION_MM_PER_DAY: float = 3.0    # paddy percolation loss
    LAND_PREP_TOTAL_MM: float = 200.0      # one-off puddling + initial flood
    LAND_PREP_DAYS: int = 20               # spread over first N days of the season
    IRRIGATION_EFFICIENCY: float = 0.65    # eta: conveyance x application
    P_EFF_MM_PER_DAY: float = 0.0          # SCOPE LIMITATION: field rainfall NOT
                                           #   subtracted (rainfall enters as inflow Q)

    KC_STAGES_MAHA: List = field(default_factory=lambda: [
        ("initial",     30, 1.05, 1.05),
        ("development",  30, 1.05, 1.20),
        ("mid_season",   40, 1.20, 1.20),
        ("late_season",  30, 1.20, 0.70),
    ])
    KC_STAGES_YALA: List = field(default_factory=lambda: [
        ("initial",     25, 1.05, 1.05),
        ("development",  25, 1.05, 1.20),
        ("mid_season",   35, 1.20, 1.20),
        ("late_season",  25, 1.20, 0.70),
    ])

    # ==============================================================
    # GOVERNMENT CWR DEMAND  (used by demand_generator.py)
    # --------------------------------------------------------------
    # The CWR is a FIELD-level requirement that already includes land preparation
    # and the crop coefficient. It is converted to a gross-at-tank release depth
    # by dividing by CONVEYANCE_EFFICIENCY (Option A - conveyance only).
    # ==============================================================
    PADDY_DURATION_DAYS: int = 135         # 135, 105, or 90 (crop age class)

    # Tank -> field conveyance efficiency, Anuradhapura minor schemes.
    # Range across the three schemes studied: ~0.67-0.73. 0.70 taken as default.
    # This is CONVEYANCE ONLY; field-application losses are outside model scope.
    CONVEYANCE_EFFICIENCY: float = 0.70

    # ==============================================================
    # LOGGING / OBSERVABILITY
    # --------------------------------------------------------------
    # These switches never change a number, a decision, or a written file.
    # Set VERBOSE_LOGGING = False for silent batch runs.
    # ==============================================================
    VERBOSE_LOGGING: bool = True   # master switch for all structured logging
    NSGA2_LOG_EVERY: int = 10      # NSGA-II progress line every N generations
                                   # (read-only callback; cannot affect the search)
    LOG_SCENARIOS: bool = True     # Gamma-fit + scenario tensor summary
    LOG_TOPSIS: bool = True        # weights before/after Option A + top-3 table
    LOG_CONSTRAINTS: bool = True   # once-per-day feasibility snapshot
    LOG_TOP_TANKS: int = 3         # how many tanks to name in the ACT summary

    # ==============================================================
    # UNIT CONVERSIONS
    # ==============================================================
    ACRE_TO_HA: float = 0.404686
    ACRE_TO_SQKM: float = 0.00404686
    FT_TO_M: float = 0.3048
    ACRE_FT_TO_M3: float = 1233.48
    SQKM_TO_SQM: float = 1_000_000.0

    # ==============================================================
    # I/O  PATHS
    # ==============================================================
    DATA_DIR: str = "data/"
    OUTPUT_DIR: str = "outputs/"

    FILE_TANK_PARAMS: str = "tank_params.csv"
    FILE_TANK_STORAGE: str = "tank_storage.csv"
    FILE_DEMAND: str = "demand.csv"
    FILE_NETWORK_ALPHA: str = _MODULE2_ALPHA_MATRIX_PATH
    FILE_RAINFALL_HISTORY: str = "rainfall_history.csv"
    FILE_MODULE3_RISK: str = "module3_risk.csv"
    FILE_MAHAWELI: str = "mahaweli_schedule.csv"

    FILE_DECISIONS: str = "mpc_decisions.csv"
    FILE_WEIGHTS_LOG: str = "topsis_weights_log.csv"
    FILE_VALIDATION: str = "module3_crosscheck.csv"

    # ==============================================================
    # CONSOLIDATED OUTPUT MODE (additive, opt-in)
    # ==============================================================
    # Default behaviour (flag False) is completely unchanged: the day-by-day
    # simulation runner still writes one output/<date>/ folder per simulated
    # day with its own mpc_decisions.csv / topsis_weights_log.csv /
    # module3_crosscheck.csv / day_summary.json, exactly as before.
    #
    # When True, the runner ALSO appends every simulated day as ROWS onto a
    # small set of run-wide CSVs at the output root (see
    # module4/consolidated_output.py), so the whole run can be read as one
    # (or a few) day-indexed CSVs instead of walking hundreds of folders.
    # This is purely additive - nothing about the per-day folders changes.
    CONSOLIDATED_OUTPUT: bool = True
    FILE_CONSOLIDATED_DECISIONS: str = "consolidated_mpc_decisions.csv"
    FILE_CONSOLIDATED_WEIGHTS_LOG: str = "consolidated_topsis_weights_log.csv"
    FILE_CONSOLIDATED_CROSSCHECK: str = "consolidated_module3_crosscheck.csv"
    FILE_CONSOLIDATED_SUMMARY: str = "consolidated_daily_summary.csv"

    def validate(self):
        """Sanity-check the config on construction."""
        base_sum = (self.W_SHORTAGE_BASE + self.W_OVERFLOW_BASE +
                    self.W_EQUITY_BASE + self.W_LOSS_BASE)
        assert abs(base_sum - 1.0) < 1e-9, \
            f"TOPSIS base weights must sum to 1.0, got {base_sum}"
        assert self.SEASON in ("maha", "yala"), \
            f"SEASON must be 'maha' or 'yala', got {self.SEASON}"
        assert self.T >= 1, "Planning horizon T must be >= 1"
        assert self.K >= 1, "Scenario count K must be >= 1"
        assert 0.0 < self.SHAPE_FACTOR <= 1.0, \
            f"SHAPE_FACTOR must be in (0,1], got {self.SHAPE_FACTOR}"
        assert self.MODULE3_STORAGE_BASIS in ("area", "volume"), \
            f"MODULE3_STORAGE_BASIS must be 'area' or 'volume', " \
            f"got {self.MODULE3_STORAGE_BASIS}"
        assert 1 <= self.CROSSCHECK_BASELINE_DAY <= self.T, \
            f"CROSSCHECK_BASELINE_DAY must be in [1,{self.T}]"
        assert self.PADDY_DURATION_DAYS in (135, 105, 90), \
            f"PADDY_DURATION_DAYS must be 135, 105, or 90, got {self.PADDY_DURATION_DAYS}"
        assert 0.0 < self.CONVEYANCE_EFFICIENCY <= 1.0, \
            f"CONVEYANCE_EFFICIENCY must be in (0,1], got {self.CONVEYANCE_EFFICIENCY}"
        if self.MODULE3_LIVE_FORECAST:
            assert self.SEASON_LENGTH == 1, \
                "MODULE3_LIVE_FORECAST=True requires SEASON_LENGTH == 1 " \
                f"(got {self.SEASON_LENGTH}); Module 3 only guarantees a " \
                "forecast file for the single date you generated, not a " \
                "run of consecutive days."
        return self

    def kc_stages(self):
        """Return the Kc stage list for the configured season."""
        return self.KC_STAGES_MAHA if self.SEASON == "maha" else self.KC_STAGES_YALA


# A ready-to-use default instance
DEFAULT_CONFIG = Config().validate()