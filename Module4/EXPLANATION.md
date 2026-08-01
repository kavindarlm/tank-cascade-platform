# Module 4 — Multi-Objective Model Predictive Control (MO-MPC)
### Complete Implementation & Methodology Guide

**Integrated Digital Management of Interconnected Tank Cascade Systems in Sri Lanka**
Mind Mappers — Group 16 · Faculty of Information Technology · University of Moratuwa · 2026

---

## Table of Contents

1. [What Module 4 Does](#1-what-module-4-does)
2. [The Big Picture: How the Modules Fit Together](#2-the-big-picture)
3. [The Seven-Step Pipeline](#3-the-seven-step-pipeline)
4. [File-by-File Reference](#4-file-by-file-reference)
5. [The Mathematics](#5-the-mathematics)
6. [Key Design Decisions (and Why)](#6-key-design-decisions)
7. [The Data: What You Need and Its Format](#7-the-data)
8. [How to Run](#8-how-to-run)
9. [What Was Fixed vs the Original Design](#9-what-was-fixed)
10. [Assumptions and Limitations](#10-assumptions-and-limitations)
11. [Evaluation Plan](#11-evaluation-plan)

---

## 1. What Module 4 Does

Module 4 is the **decision-support brain** of the tank cascade system. Every day it
answers one operational question:

> *Given today's observed storage, the cascade network, forecast demand, and rainfall
> uncertainty, how much water should each tank release over the next 7 days so that
> shortage, overflow, inequity, and water loss are all minimised together?*

It takes the outputs of Modules 1, 2, and 3, runs a multi-objective optimisation
across many possible rainfall futures, and produces **ranked, explainable release
strategies** for water managers — implementing only today's decision, then re-planning
tomorrow with fresh data.

The framework is **Multi-Objective Model Predictive Control (MO-MPC)**:
- **Multi-Objective** — four competing goals kept separate, solved with NSGA-II, and
  reconciled afterward with TOPSIS.
- **Model Predictive Control** — plan over a 7-day horizon, act on day 0 only, re-observe,
  repeat. This receding horizon makes the controller robust to forecast error.

---

## 2. The Big Picture

```
   MODULE 1                 MODULE 2                 MODULE 3
   Satellite storage        GNN cascade graph        Forecast (LSTM/TCN)
        │                        │                        │
        │ S_current              │ alpha matrix           │ risk probs + 7-day
        │ (today's storage)      │ (who feeds whom)       │ storage forecast
        ▼                        ▼                        ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │                          MODULE 4                                 │
   │                                                                   │
   │   Open-Meteo rainfall ──► Scenario Generator (Gamma fit)          │
   │                                    │                              │
   │                                    ▼                              │
   │   Mass Balance ◄──────────── 100 inflow scenarios                 │
   │        │                                                          │
   │        ▼                                                          │
   │   4 Objectives  +  6 Constraints                                  │
   │        │                                                          │
   │        ▼                                                          │
   │   NSGA-II  ──►  Pareto front  ──►  TOPSIS (risk-weighted)         │
   │                                         │                         │
   │                                         ▼                         │
   │                              Top-3 strategies                     │
   │                                         │                         │
   │                              Apply day-0 release                  │
   └─────────────────────────────────────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                            ▼
             Sluice release              Dashboard + warnings
             (field operators)          (Module 3 risk shown here)
```

**Three distinct roles for the three upstream modules inside Module 4:**

| Module | Provides | Where it enters Module 4 |
|--------|----------|--------------------------|
| Module 1 | Today's storage `S_current` | Initial state of the mass balance, every MPC step |
| Module 2 | Cascade transfer matrix `alpha` | The `alpha @ R` term in the mass balance |
| Module 3 | Risk probabilities + 7-day forecast | **TOPSIS weights** (risk) + **cross-check** (forecast) — *not* the optimiser |

> **Important architectural point:** Module 3 does **not** feed the scenario generator.
> Rainfall scenarios come directly from historical Open-Meteo data. Module 3's role is
> (a) to nudge the decision weights toward drought/overflow protection, and (b) to serve
> as an independent forecast we cross-check our physics against. See
> [Section 6](#6-key-design-decisions) for the full reasoning.

---

## 3. The Seven-Step Pipeline

Every day of the irrigation season, the controller executes these seven steps
(implemented in `mpc_loop.py`):

### Step 0 — Configuration (`config.py`)
All hyperparameters, physical constants, and design switches live in one dataclass.
Change this file, not the algorithms.

### Step 1 — Observe (`data_loader.py`)
Load today's storage from Module 1, today's risk from Module 3, and the static inputs
(tank parameters, demand, alpha, rainfall history).

### Step 2 — Predict (`scenario_generator.py`)
Fit a Gamma distribution to each tank's recent rainfall history, convert to catchment
inflow, and sample **K = 100** plausible 7-day inflow trajectories.

### Step 3 — Simulate (`state_transition.py`)
For any candidate release plan, propagate the **mass balance** forward through all 7
days and all tanks, recording storage, overflow spill, and mass-conservation gaps.

### Step 4 — Score (`objectives.py` + `constraints.py`)
Evaluate the **four objectives** (averaged over the 100 scenarios) and the **six
constraints** (violation magnitudes) for each candidate plan.

### Step 5 — Optimise (`nsga2_optimizer.py`)
Run **NSGA-II** to find the Pareto front — the set of best-possible trade-off plans that
no other plan beats on all four objectives.

### Step 6 — Decide (`topsis.py`)
Adjust the TOPSIS weights using Module 3's risk (Option A), then rank the Pareto front
by closeness to the ideal point and return the **top 3 strategies**.

### Step 7 — Act & Cross-Check (`cross_validation.py` + `output.py`)
Compare the chosen plan's simulated trajectory against Module 3's forecast, apply **only
the day-0 release**, log everything, and slide the horizon forward one day.

---

## 4. File-by-File Reference

```
module4/
├── main.py                   Entry point — runs the full season
├── requirements.txt          Python dependencies
├── EXPLANATION.md            This document
│
├── module4/                  The package
│   ├── __init__.py
│   ├── config.py             ALL hyperparameters & switches (edit this)
│   ├── demand_generator.py   FAO-56 stage-based paddy demand
│   ├── scenario_generator.py Gamma-fit rainfall → inflow scenarios
│   ├── state_transition.py   Mass balance simulation engine
│   ├── objectives.py         The 4 objective functions f1..f4
│   ├── constraints.py        The 6 constraint checkers C1..C6
│   ├── nsga2_optimizer.py    NSGA-II via pymoo
│   ├── topsis.py             TOPSIS selection + Option A risk weights
│   ├── cross_validation.py   Module 3 forecast vs Module 4 simulation
│   ├── data_loader.py        Load & validate all inputs
│   ├── mpc_loop.py           The receding-horizon controller
│   └── output.py             Recommendations & CSV exports
│
├── data/                     Input CSVs go here (see Section 7)
├── outputs/                  Results written here
└── tests/                    Unit tests (pytest)
```

### `config.py` — the control panel
Every tunable number is here: planning horizon `T`, scenario count `K`, NSGA-II
population and generations, TOPSIS base weights, Option A sensitivities, constraint
on/off switches, FAO-56 demand parameters, and the runoff coefficient. It self-validates
on construction (weights must sum to 1, season must be valid, etc.).

### `demand_generator.py` — how much water crops need
Implements the **FAO-56 crop-water-requirement** chain. Builds the four-stage paddy
crop-coefficient curve `Kc(t)`, multiplies by reference evapotranspiration `ET0`, adds
percolation and land-preparation water, divides by irrigation efficiency, and scales by
each tank's command area to get demand `D_i(t)` in m³/day. Field rainfall is deliberately
**not** subtracted (scope limitation — rainfall enters as inflow instead).

### `scenario_generator.py` — modelling rainfall uncertainty
Takes historical daily rainfall, converts it to catchment inflow via a runoff
coefficient, fits a **Gamma distribution** per tank by moment matching, and samples K
inflow scenarios. Gamma is the standard hydrological choice for non-negative, right-skewed
rainfall. Module 3 does **not** enter here.

### `state_transition.py` — the physics engine
The **mass balance**: `S(t+1) = S(t) + Q + alpha@R − R − E`. Crucially, it records
overflow spill and below-dead-storage deficit **before** clipping storage to physical
bounds, and caps each release at the water actually available — so overflow is detectable
and no water is invented. (These are the two most important fixes; see
[Section 9](#9-what-was-fixed).)

### `objectives.py` — measuring "badness"
The four cost functions:
- `f1_shortage` — total unmet demand
- `f2_overflow` — total spill above capacity (read from the pre-clip spill series)
- `f3_equity` — dispersion of satisfaction ratios, **excluding zero-demand tanks**
- `f4_water_loss` — total over-release beyond demand

All four are averaged over the K scenarios to give the *expected* cost.

### `constraints.py` — the hard limits
Six constraint checkers, each returning a violation magnitude (0 = satisfied):
- C1 storage bounds, C2 release bounds, C3 deterministic drought floor,
  C4 mass conservation (with inflow), C5 non-negativity, C6 Mahaweli compliance.
Each is individually toggleable; C6 is **off** for Mahakanumulla (no Mahaweli inflow).

### `nsga2_optimizer.py` — the search
Wraps the simulation + objectives + constraints in a pymoo `Problem`. The constraints
are written to `out['G']` so pymoo's **constraint-domination** handles feasibility
natively. Returns the Pareto front.

### `topsis.py` — picking one plan
Normalises the objectives, applies **risk-adjusted weights** (Option A: drought raises
shortage & equity weights; overflow raises overflow weight and lowers loss weight), and
ranks by closeness to the ideal solution. Returns the top 3 with per-tank satisfaction.

### `cross_validation.py` — inter-module sanity check
Simulates the chosen plan and compares its storage trajectory (as % of capacity) against
Module 3's independent 7-day forecast. Reports the divergence. A diagnostic only — it does
not feed back into the optimiser.

### `data_loader.py` — the front door
Loads and validates every input file, converts rainfall to scenarios, computes the
network-level risk probabilities for TOPSIS, and assembles the `state` dict. **It never
fabricates data** — a missing file raises a clear error naming the file and its expected
columns.

### `mpc_loop.py` — the conductor
Runs the seven steps for every day of the season, applies only day 0, logs the applied
weights and decisions, and produces the cross-check tables.

### `output.py` — presentation
Prints operator recommendations and exports three CSVs: daily decisions, applied TOPSIS
weights, and the Module 3 cross-check.

---

## 5. The Mathematics

### 5.1 Decision variable
The release matrix, the only thing the optimiser changes:
```
R ∈ ℝ^(N×T),   R_i(t) = water released by tank i on day t  (m³)
```
Flattened to a vector of length N·T for NSGA-II, with box bounds 0 ≤ R ≤ R_max.

### 5.2 State transition (mass balance)
```
S_i(t+1) = S_i(t) + Q_i(t) + Σ_j α_ij·R_j(t) − R_i(t) − E_i(t)
```
- `Q_i(t)` — stochastic rainfall inflow (per scenario)
- `Σ_j α_ij·R_j(t)` — upstream water arriving, computed as the matrix product `α @ R_t`
- `R_i(t)` — controlled release
- `E_i(t)` — evaporation

Overflow spill `max(0, S − S_max)` and deficit `max(0, S_min − S)` are recorded before
clipping; releases are capped at available water to preserve conservation of mass.

### 5.3 Scenario generation
Convert rainfall to inflow, fit Gamma by moment matching:
```
Q_hist = C · rainfall_mm · catchment_km² · 1000
κ = μ²/σ²   (shape) ,   θ = σ²/μ   (scale)
Q^(k)_i(t) ~ Gamma(κ_i, θ_i)   for k = 1..K
```

### 5.4 The four objectives (all minimised, averaged over K scenarios)
```
f1 = Σ_i Σ_t max(0, D_i(t) − R_i(t))                     shortage
f2 = Σ_i Σ_t max(0, S_i(t) − S_max_i)                    overflow (pre-clip spill)
f3 = Σ_t Σ_{i∈active} |ρ_i(t) − ρ̄(t)|,  ρ_i = R_i/D_i    equity
f4 = Σ_i Σ_t max(0, R_i(t) − D_i(t))                     water loss
```
The optimiser sees the expectation `f̄_r = (1/K) Σ_k f_r(R, S^(k))` (sample-average
approximation of the expected cost).

### 5.5 The six constraints (violation ≥ 0, 0 = satisfied)
```
C1:  S_min ≤ S_i(t) ≤ S_max
C2:  0 ≤ R_i(t) ≤ R_max_i
C3:  S_i(t) ≥ S_min_i + 0.15·S_max_i        (deterministic seasonal floor)
C4:  Σ_i R_i(t) ≤ Σ_i (S_i + Q_i)           (mass conservation, with inflow)
C5:  R_i(t) ≥ 0 ,  S_i(t) ≥ 0
C6:  |R_m(t) − R_sched_m(t)| ≤ δ_m          (Mahaweli — OFF for Mahakanumulla)
```

### 5.6 NSGA-II
Optimises over the four objectives using Pareto dominance (plan A beats B if it is ≥ on
all objectives and > on at least one), fast non-dominated sorting, and crowding distance
for diversity. Constraints handled by constraint-domination in pymoo. SBX crossover +
polynomial mutation.

### 5.7 TOPSIS with Option A risk weights
```
Risk-adjusted weights (then floored at 0.05 and renormalised to sum to 1):
   w1 = w1_base + α_ds·P_drought      (shortage ↑ under drought)
   w3 = w3_base + α_de·P_drought      (equity   ↑ under drought)
   w2 = w2_base + β·P_overflow        (overflow ↑ under overflow risk)
   w4 = w4_base − γ·P_overflow        (loss     ↓ under overflow risk)

TOPSIS:
   Ṽ = (F / ‖F‖_col) · w                 normalise + weight
   V⁺ = min_col Ṽ ,  V⁻ = max_col Ṽ      ideal / anti-ideal (costs)
   d⁺ = ‖V − V⁺‖ ,  d⁻ = ‖V − V⁻‖
   C = d⁻ / (d⁺ + d⁻)                     closeness (higher = better)
```
Rank descending by C, return top 3.

### 5.8 Receding horizon
Solve the 7-day problem, apply only `R(·,0)`, re-observe, slide forward one day. Days
1..6 exist only to prevent myopia and are recomputed next step with better information.

---

## 6. Key Design Decisions

### 6.1 Why Module 3's probabilities drive TOPSIS weights, not scenarios (Option A)
Module 3 predicts **storage state** (a consequence of rainfall *and* releases), not
rainfall itself. Its 7-day forecast assumes some release pattern baked into its training
data — a pattern the optimiser may not choose. So it **cannot** serve as the dynamics
model or the scenario source without making the optimisation circular.

Instead, under **Option A**, Module 3's next-day risk probabilities adjust the TOPSIS
decision weights: elevated drought risk shifts the recommendation toward supply
reliability and fairness; elevated overflow risk shifts it toward spill protection. This
gives Module 3 a clean, single, well-defined point of influence.

### 6.2 Why rainfall scenarios use a 90-day window (avoiding double-counting)
Module 3's drought signal is high **because** recent rainfall has been low. If the
scenario generator also fitted its Gamma to that same recent low rainfall, the same
signal would enter the optimiser twice (once through scenario physics, once through
weights), making the controller over-react. Fitting to a **90-day climatological window**
decouples the scenario mean from the short-term drought signal, so the two information
channels stay independent. *(Alternative: fit to 30 days and explicitly document the
coupling as intentional conservatism. The config makes this a one-line switch.)*

### 6.3 Why the four objectives stay separate (Pareto, not weighted sum)
A single weighted sum requires committing to weights before seeing the trade-offs, and
cannot reach non-convex regions of the Pareto front. Keeping the objectives separate lets
NSGA-II map the *entire* trade-off surface; the value judgement (weights) is applied only
afterward in TOPSIS — transparently and adjustably.

### 6.4 Why field rainfall is excluded from demand (P_eff = 0)
Three reasons: (a) the controller's boundary ends at the tank sluice; (b) catchment
rainfall already enters as inflow `Q`, so subtracting field rainfall from demand would
double-count; (c) the resulting *gross* demand is conservative, biasing the controller
toward over-supply rather than crop failure — the right direction of error for food
security.

### 6.5 Why C3 is now a deterministic floor
The original C3 used Module 3's drought probability to raise the storage floor. Under
Option A, that probability moved to the TOPSIS weights, so C3 was reformulated as a fixed
seasonal hedging buffer (`S ≥ S_min + 0.15·S_max`). This keeps the "safety reserve"
concept without re-using Module 3's signal in two places.

### 6.6 Why C6 is off for Mahakanumulla
The Mahakanumulla cascade is rain-fed with no Mahaweli inflow, so the Mahaweli-schedule
constraint has nothing to bind. It is switched **off** in config. (If you later simulate a
Mahaweli-fed tank such as Nachchaduwa, set `ENABLE_C6_MAHAWELI = True` and provide the
schedule file.)

---

## 7. The Data

> **Nothing here is fabricated yet.** This section specifies the exact format each input
> file must have. Wiring in the real data (from Modules 1/2/3 and Open-Meteo) is the next
> phase. Where relevant, notes explain what we currently have and what still needs
> sourcing.

All files live in `data/`. All arrays use zero-indexed tanks and days. Storage/release
in m³.

### 7.1 `tank_params.csv` — static tank properties
```
tank_id, S_min, S_max, R_max, E_day0..E_day6, catchment_area_km2
```
- `S_min` dead storage (m³), `S_max` capacity (m³), `R_max` max release (m³/day)
- `E_day0..E_day6` evaporation per day (m³/day) — from meteorological data
- `catchment_area_km2` — needed to convert rainfall to inflow

**What we have:** capacity, catchment area, and (via conversion) depth for most of the 32
Mahakanumulla tanks in the merged dataset. `S_min`, `R_max`, and daily evaporation still
need sourcing from Irrigation Department records / field survey.

### 7.2 `tank_storage.csv` — today's storage (Module 1)
```
tank_id, storage_m3
```
Reloaded each MPC step. **Source:** Module 1 satellite estimation.

### 7.3 `demand.csv` — full-season daily demand
```
tank_id, day0, day1, ..., day{season-1}
```
**Source:** produced by `demand_generator.py` from the FAO-56 chain, using each tank's
command area. **What we have:** command area for 24 of 32 tanks (Sheet2) + Nachchaduwa;
8 tanks use a median fallback (flag these in analysis).

### 7.4 `network_alpha.csv` — cascade transfer matrix (Module 2)
```
      , tank_0, tank_1, ...
tank_0,  0.00 ,  0.00 , ...
tank_1,  0.75 ,  0.00 , ...
```
`alpha[i][j]` = fraction of tank j's release arriving at tank i (row = destination).
**Source:** Module 2 GNN adjacency output.

### 7.5 `rainfall_history.csv` — recent rainfall (Open-Meteo)
```
tank_id, d0, d1, ..., d{H-1}
```
Daily rainfall (mm/day) for the last `RAINFALL_HISTORY_DAYS` at each tank's coordinates.
**Source:** Open-Meteo API (same as Module 3). Latitude/longitude for all 32 tanks are in
`nodes_complete.csv`.

### 7.6 `module3_risk.csv` — forecast + risk (Module 3)
```
tank_id, drought_probability, overflow_probability, normal_probability,
         pred_t+1_storage .. pred_t+7_storage
```
- The three probabilities (for tomorrow) drive the Option A TOPSIS weights.
- The 7-day storage forecast (%) is used only for the cross-check.
**Source:** Module 3 CSV output. If absent, weights fall back to base and the cross-check
is skipped.

> ⚠️ **Module 3 caveat:** its overflow class currently has F1 = 0.000 (the test window
> contained almost no real overflow events). Until retrained, treat overflow-driven weight
> adjustments as low-confidence — the config lets you zero `BETA_OVERFLOW` if you want to
> ignore the overflow signal entirely for now.

### 7.7 `mahaweli_schedule.csv` — optional (only if C6 on)
```
reservoir_id, day0, day1, ...
```
Not needed for Mahakanumulla.

---

## 8. How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place all required CSVs in data/ (see Section 7)

# 3. Run the full season
python main.py

# 4. Run the unit tests
pytest tests/ -v
```

Outputs land in `outputs/`:
- `mpc_decisions.csv` — the day-0 release for every tank, every day
- `topsis_weights_log.csv` — the applied weights and risk probs per day
- `module3_crosscheck.csv` — simulated vs forecast storage divergence

To adjust anything (horizon, scenarios, weights, which constraints are active), edit
`module4/config.py` — never the algorithm files.

---

## 9. What Was Fixed

The original implementation had several defects that made the optimiser silently solve a
different, easier, incorrect problem. All are corrected here:

| # | Bug | Fix |
|---|-----|-----|
| 1 | **Overflow undetectable** — storage was clipped to capacity *before* f2/C1 were evaluated, so f2 ≡ 0 always | `state_transition.py` records spill *before* clipping; f2 and C1 read that spill |
| 2 | **Constraints inert** — pymoo problem declared `n_constr=0`; C3–C6 had no effect | `nsga2_optimizer.py` declares the active constraint count and writes violations to `out['G']` |
| 3 | **Mass not conserved** — clipping at S_min invented water | releases capped at available water; the shortfall is recorded |
| 4 | **δ_m wrong** — Mahaweli tolerance came from tank capacity | now based on scheduled release volume |
| 5 | **C4 omitted inflow** — used `ΣR ≤ ΣS` instead of `ΣR ≤ Σ(S+Q)` | inflow included |
| 6 | **f3 mishandled zero-demand tanks** — assigned ratio 0, creating fake inequity | zero-demand tanks excluded from mean and sum |
| 7 | **C3 dangling under Option A** — its risk input moved to TOPSIS | reformulated as a deterministic seasonal floor |

All fixes are covered by unit tests in `tests/` (18 tests, all passing).

---

## 10. Assumptions and Limitations

State these in the thesis; each is defensible and citable.

1. **P_eff = 0** — field rainfall not subtracted from demand (boundary of control +
   avoids double-counting; conservative). *Sensitivity test recommended: P_eff up to
   3 mm/day.*
2. **Runoff coefficient C = 0.25** — lumped dry-zone value (IWMI). *Subject to sensitivity
   analysis over [0.15, 0.40].*
3. **Irrigation efficiency η = 0.65** — gravity-fed cascade value. *Subject to sensitivity
   analysis over [0.55, 0.75].*
4. **Independent daily rainfall samples** — no temporal autocorrelation within a scenario
   (acceptable over 7 days; AR(1) is a future refinement).
5. **Stationary 90-day rainfall distribution** — refitted every MPC step, so regime shifts
   are picked up within a day.
6. **Constant ET0 = 5 mm/day** — first-pass; replace with a daily Anuradhapura series.
7. **8 tanks on median command area** — flag in analysis; report metrics with and without.
8. **Module 3 overflow class unvalidated** — F1 = 0.000; overflow-driven weighting is
   low-confidence until Module 3 is retrained.

---

## 11. Evaluation Plan

Three levels, mirroring the eight-week development plan:

**Level 1 — Unit verification (done):** hand-checked mass balance, objectives, and
constraints (`tests/`, 18 passing).

**Level 2 — Optimiser quality:** verify all returned solutions are feasible; track the
hypervolume indicator across generations to confirm convergence; check the Pareto front
spans a meaningful range on every objective; confirm results are stable across random
seeds and smooth under ±0.05 weight perturbation.

**Level 3 — System validation:** run a full season and compare against baselines —
(a) release-at-demand rule, (b) solve-once-static plan — reporting seasonal totals of
f1–f4. MPC should dominate the naive rule on shortage+overflow jointly and beat the static
plan increasingly as forecast noise grows. Add stress scenarios (forced drought / forced
wet spell), an equity audit (per-tank satisfaction spread), and ablations (set α=0; use
K=1; drop C3) to attribute performance to each research component. Include the sensitivity
analysis on C and η.

---

*Mind Mappers — Group 16 · University of Moratuwa · 2026*
