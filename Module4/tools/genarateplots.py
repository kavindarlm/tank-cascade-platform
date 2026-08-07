"""
generate_plots.py
==================
Generates every dissertation-ready figure directly from the CSVs already
produced by evaluate_season.py - no re-reading of output_dir, no
re-computation of anything already validated there. Works on any season
this pipeline evaluates, not just the 2025 run.

USAGE
-----
    python generate_plots.py output_25-03-08/_evaluation

Optional comparison plots (skipped with a clear message if not supplied):
    --baseline-eval-dir   a second _evaluation folder (e.g. a proportional-
                          allocation baseline run) -> baseline comparison plot
    --old-smax-eval-dir   an _evaluation folder from a run under the OLD
                          (pre-satellite) S_max values -> capacity sensitivity
    --runoff-sweep-csv    CSV with columns C, satisfaction_pct, from
                          sensitivity_runoff.py -> runoff sensitivity plot
    --calibrated-c        the calibrated runoff coefficient, for the vertical
                          marker on the runoff sensitivity plot

WHICH PLOTS THIS PRODUCES, AND FROM WHAT
------------------------------------------
 1. storage_and_satisfaction.png   season_metrics.csv + storage_trajectory.csv
 2. objective_trajectories.png     season_metrics.csv (sel_f1..sel_f4)
 3. monthly_breakdown.png          monthly_breakdown.csv (extrapolation flagged)
 4. per_tank_equity.png            per_tank_summary.csv
 5. crosscheck_agreement.png       crosscheck_daily.csv
 9. channel_loss.png               alpha_channel_loss.csv
10. rainfall_response.png          season_metrics.csv (storage delta vs release)
 6. capacity_sensitivity.png       [optional] two _evaluation dirs
 7. runoff_sensitivity.png         [optional] a sweep CSV
 8. baseline_comparison.png        [optional] two _evaluation dirs

Plot 11 (a Pareto front snapshot) is NOT produced here: no run currently
persists the full front, only the selected point, so there is nothing to
plot without adding a front dump to the pipeline first. Noted, not silently
skipped - see the end of the run for a reminder if you want it added.

All figures are written as both .png (for quick viewing) and a matplotlib
figure a dissertation-quality reader can re-export - 300 DPI, serif-free
figure text sized for a single-column dissertation page.
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ======================================================================
# Style - kept plain and print-friendly rather than screen-flashy, since
# these are destined for a dissertation page, not a slide.
# ======================================================================
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.axisbelow": True,
    "figure.constrained_layout.use": True,
})

C_STORAGE = "#2a5d8a"
C_SAT = "#c0562c"
C_HELD = "#c0562c"
C_FREE = "#4a7a8c"
C_F1 = "#a13d3d"
C_F2 = "#c0862c"
C_F3 = "#4a6fa5"
C_F4 = "#5b8c5a"
PHASE_COLORS = ["#f2d7d5", "#fdebd0", "#d5f2e3"]


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("eval_dir",
                    help="the _evaluation folder produced by evaluate_season.py")
    ap.add_argument("--out", default=None,
                    help="where to write PNGs (default: <eval_dir>/plots)")
    ap.add_argument("--baseline-eval-dir", default=None)
    ap.add_argument("--old-smax-eval-dir", default=None)
    ap.add_argument("--runoff-sweep-csv", default=None,
                    help="CSV with columns: C, satisfaction_pct")
    ap.add_argument("--calibrated-c", type=float, default=None)
    ap.add_argument("--phase-boundaries", type=int, nargs="*", default=None,
                    help="day_offset values marking phase transitions, e.g. "
                         "9 15 for a recovery/transition/stable split. If "
                         "omitted, phases are auto-detected from the first "
                         "run of satisfaction >= 60%% sustained for 5+ days.")
    ap.add_argument("--show-interpolated-gaps", action="store_true",
                    help="Draw a thin dashed grey line connecting across any "
                         "data gap (e.g. a missing-input outage), purely for "
                         "visual continuity when presenting a figure. OFF by "
                         "default: a data gap means Module 4 never produced a "
                         "decision for those days, and the default behaviour "
                         "leaves that visible as a real break rather than "
                         "implying results that were never simulated. The "
                         "gap is always shaded and labelled with its day "
                         "count regardless of this flag.")
    return ap.parse_args()


def warn(msg):
    print(f"  [!] {msg}", file=sys.stderr)


def load(eval_dir, name, required=True):
    p = os.path.join(eval_dir, name)
    if not os.path.exists(p):
        if required:
            raise SystemExit(f"required file not found: {p}\n"
                             f"(run evaluate_season.py first)")
        return None
    return pd.read_csv(p)


def to_dates(series):
    return pd.to_datetime(series)


# ======================================================================
# Phase auto-detection (for plot 1's shaded bands)
# ======================================================================

def detect_phases(season_metrics):
    """
    Finds the reserve-recovery -> transition -> stable boundary automatically:
    the recovery phase ends at the first day satisfaction reaches 60% and
    stays there (does not drop below 50%) for the next 5 consecutive days.
    Falls back to no shading if satisfaction never stabilises this way.
    """
    sat = season_metrics["sel_satisfaction_pct_mean"].to_numpy()
    n = len(sat)
    stable_start = None
    for i in range(n - 5):
        if sat[i] >= 60 and np.all(sat[i:i + 5] >= 50):
            stable_start = i
            break
    if stable_start is None:
        return None
    transition_start = max(0, stable_start - 6)
    return transition_start, stable_start


# ======================================================================
# Data-gap detection - shared by every date-axis plot
# ======================================================================

def find_date_gaps(dates):
    """
    Finds runs of missing CALENDAR days within [min(dates), max(dates)].

    A gap here means: no completed simulation exists for that date at all
    (status != "complete" in day_summary.json, so the day never appears in
    season_metrics.csv - see evaluate_season.py). This is NOT a rendering
    artefact. It reflects a real absence of Module 1/3 input for that date
    (e.g. the confirmed 10-day 2026-05-28..2026-06-06 outage), which means
    Module 4 never ran and never produced a decision for those days.

    Returns a list of (gap_start, gap_end, n_days) for each contiguous run
    of missing dates, sorted chronologically.
    """
    dates = pd.to_datetime(sorted(dates))
    full_range = pd.date_range(dates.min(), dates.max(), freq="D")
    present = set(dates)
    missing = [d for d in full_range if d not in present]
    if not missing:
        return []

    gaps = []
    run_start = missing[0]
    prev = missing[0]
    for d in missing[1:]:
        if (d - prev).days > 1:
            gaps.append((run_start, prev, (prev - run_start).days + 1))
            run_start = d
        prev = d
    gaps.append((run_start, prev, (prev - run_start).days + 1))
    return gaps


def mark_gaps(ax, gaps, y_frac=0.5, label_days_threshold=1):
    """
    Shades each gap in hatched grey and labels it with the day count, on the
    given axis. Never draws a connecting line - the visual break is the
    honest representation; this only makes it unambiguous that the break is
    a known, counted data gap rather than a plotting error.
    """
    for gap_start, gap_end, n_days in gaps:
        ax.axvspan(gap_start, gap_end + pd.Timedelta(days=1),
                  facecolor="#dddddd", edgecolor="#999999", hatch="///",
                  alpha=0.55, zorder=1)
        if n_days >= label_days_threshold:
            xm = gap_start + (gap_end - gap_start) / 2
            ax.text(xm, y_frac, f"No data\n({n_days}d)", ha="center",
                    va="center", fontsize=7, color="#555",
                    transform=ax.get_xaxis_transform(), zorder=2,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                             edgecolor="none", alpha=0.75))


def draw_interpolated_connector(ax, df, date_col, value_col, gaps):
    """
    OPTIONAL, OFF BY DEFAULT (see --show-interpolated-gaps). Draws a thin,
    unmistakably-styled dashed grey line connecting the last point before a
    gap to the first point after it, purely so the eye can follow the
    overall trend across a gap when presenting the figure. This is never
    real simulated data and is never used in any statistic - it exists only
    as a rendering aid, and is styled deliberately differently (thin, grey,
    dashed, no markers) so it cannot be mistaken for the actual trajectory.
    """
    added_legend_entry = False
    for gap_start, gap_end, n_days in gaps:
        before = df[df[date_col] < gap_start]
        after = df[df[date_col] > gap_end]
        if before.empty or after.empty:
            continue
        x = [before[date_col].iloc[-1], after[date_col].iloc[0]]
        y = [before[value_col].iloc[-1], after[value_col].iloc[0]]
        ax.plot(x, y, color="#999999", linewidth=0.9, linestyle="--",
               alpha=0.7, zorder=1,
               label=("Interpolated across data gap (not simulated)"
                      if not added_legend_entry else None))
        added_legend_entry = True


# ======================================================================
# Plot 1 - storage + satisfaction, two panels, shared date axis
# ======================================================================

def plot_storage_satisfaction(eval_dir, out_dir, phase_boundaries=None,
                              show_interpolated_gaps=False):
    sm = load(eval_dir, "season_metrics.csv")
    st = load(eval_dir, "storage_trajectory.csv")
    sm["date"] = to_dates(sm["date"])
    st["date"] = to_dates(st["date"])
    df = sm.merge(st, on="date", how="left", suffixes=("", "_traj"))

    gaps = find_date_gaps(df["date"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

    if phase_boundaries is None:
        auto = detect_phases(sm)
        boundaries = list(auto) if auto else []
    else:
        boundaries = sorted(phase_boundaries)

    if boundaries:
        edges = [0] + boundaries + [len(df)]
        labels = ["Reserve recovery", "Transition", "Stable operation"]
        labels = labels[:len(edges) - 1]
        for k in range(len(edges) - 1):
            x0 = df["date"].iloc[max(0, edges[k])]
            x1 = df["date"].iloc[min(len(df) - 1, edges[k + 1] - 1)]
            for ax in (ax1, ax2):
                ax.axvspan(x0, x1, color=PHASE_COLORS[k % len(PHASE_COLORS)],
                          alpha=0.5, zorder=0)

    village = df["village_storage_m3"] if "village_storage_m3" in df else df["total_storage_m3"]
    if show_interpolated_gaps and gaps:
        draw_interpolated_connector(ax1, df.assign(_v=village), "date", "_v", gaps)
    ax1.plot(df["date"], village / 1e6, color=C_STORAGE, linewidth=1.6, zorder=3)
    ax1.set_ylabel("Village storage (million m$^3$)")
    ax1.set_title("Village storage and demand satisfaction over the season")

    if show_interpolated_gaps and gaps:
        draw_interpolated_connector(ax2, df, "date", "sel_satisfaction_pct_mean", gaps)
    ax2.plot(df["date"], df["sel_satisfaction_pct_mean"], color=C_SAT,
             linewidth=1.4, zorder=3)
    ax2.axhline(50, color="grey", linewidth=0.7, linestyle=":")
    ax2.set_ylabel("Mean satisfaction (%)")
    ax2.set_ylim(0, 100)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    fig.autofmt_xdate()

    if gaps:
        mark_gaps(ax1, gaps, y_frac=0.5)
        mark_gaps(ax2, gaps, y_frac=0.5)

    if boundaries:
        for k, lab in enumerate(labels):
            x0 = df["date"].iloc[max(0, edges[k])]
            x1 = df["date"].iloc[min(len(df) - 1, edges[k + 1] - 1)]
            xm = x0 + (x1 - x0) / 2
            # Only place inline text if the band is wide enough to hold it
            # without overlapping its neighbour - narrow bands (e.g. a short
            # transition window) go in the legend instead. This is what was
            # producing the "Reserve TranSitecovery" overlap: a 6-day
            # transition band is far narrower than the label text at any
            # readable font size, and two centred texts that close together
            # always overlap regardless of font size chosen.
            band_days = edges[k + 1] - edges[k]
            if band_days >= 12:
                ax1.text(xm, 0.96, lab, ha="center", va="top",
                         fontsize=7.5, color="#444",
                         transform=ax1.get_xaxis_transform())

        from matplotlib.patches import Patch
        handles = [Patch(facecolor=PHASE_COLORS[k % len(PHASE_COLORS)],
                        alpha=0.5, label=lab)
                  for k, lab in enumerate(labels)]
        if gaps:
            handles.append(Patch(facecolor="#dddddd", edgecolor="#999999",
                                 hatch="///", alpha=0.55, label="No data (input gap)"))
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(handles=handles + handles2, fontsize=7, loc="lower right",
                  framealpha=0.9)
    elif gaps:
        from matplotlib.patches import Patch
        handles = [Patch(facecolor="#dddddd", edgecolor="#999999", hatch="///",
                        alpha=0.55, label="No data (input gap)")]
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(handles=handles + handles2, fontsize=7, loc="lower right",
                  framealpha=0.9)

    if gaps:
        total_missing = sum(n for _, _, n in gaps)
        fig.text(0.5, -0.02,
                 f"{len(gaps)} data gap(s) totalling {total_missing} day(s) "
                 f"excluded (no completed simulation - see days_error in "
                 f"the evaluation report)",
                 ha="center", fontsize=7, color="#666")

    path = os.path.join(out_dir, "storage_and_satisfaction.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ======================================================================
# Plot 2 - objective trajectories, 2x2
# ======================================================================

def plot_objective_trajectories(eval_dir, out_dir, show_interpolated_gaps=False):
    sm = load(eval_dir, "season_metrics.csv")
    sm["date"] = to_dates(sm["date"])
    gaps = find_date_gaps(sm["date"])

    fig, axes = plt.subplots(2, 2, figsize=(8, 5.5), sharex=True)
    specs = [
        ("sel_f1_shortage", "f1 - Shortage (m$^3$)", C_F1),
        ("sel_f2_overflow", "f2 - Overflow (m$^3$)", C_F2),
        ("sel_f3_equity", "f3 - Equity", C_F3),
        ("sel_f4_water_loss", "f4 - Water loss (m$^3$)", C_F4),
    ]
    for ax, (col, title, color) in zip(axes.flat, specs):
        if show_interpolated_gaps and gaps:
            draw_interpolated_connector(ax, sm, "date", col, gaps)
        ax.plot(sm["date"], sm[col], color=color, linewidth=0.9, zorder=3)
        if gaps:
            mark_gaps(ax, gaps, y_frac=0.5, label_days_threshold=3)
        ax.set_title(title, fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    fig.suptitle("Selected-solution objective values over the season", fontsize=11)
    if gaps:
        total_missing = sum(n for _, _, n in gaps)
        fig.text(0.5, -0.01,
                 f"{len(gaps)} data gap(s) totalling {total_missing} day(s) "
                 f"excluded (no completed simulation)",
                 ha="center", fontsize=7, color="#666")
    fig.autofmt_xdate()
    path = os.path.join(out_dir, "objective_trajectories.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ======================================================================
# Plot 3 - monthly breakdown, with extrapolation exposure flagged
# ======================================================================

def plot_monthly_breakdown(eval_dir, out_dir):
    mb = load(eval_dir, "monthly_breakdown.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.6))

    x = np.arange(len(mb))
    bars = ax1.bar(x, mb["mean_satisfaction_pct"], color=C_SAT, alpha=0.85)
    for i, pct in enumerate(mb["pct_days_extrapolated"]):
        if pct > 0:
            bars[i].set_hatch("//")
            bars[i].set_edgecolor("#7a3010")
    ax1.set_xticks(x)
    ax1.set_xticklabels(mb["month"], rotation=45, ha="right")
    ax1.set_ylabel("Mean satisfaction (%)")
    ax1.set_ylim(0, 100)
    ax1.set_title("Monthly mean satisfaction")

    ax2.bar(x, mb["release"] / 1e6, color=C_STORAGE, alpha=0.85, label="Release")
    ax2.bar(x, mb["consumed"] / 1e6, color="#8fb3cc", alpha=0.85, label="Consumed")
    ax2.set_xticks(x)
    ax2.set_xticklabels(mb["month"], rotation=45, ha="right")
    ax2.set_ylabel("Million m$^3$")
    ax2.set_title("Monthly release / consumption")
    ax2.legend(fontsize=8)

    # legend patch for the hatch meaning
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=C_SAT, alpha=0.85, label="Fully data-supported"),
              Patch(facecolor=C_SAT, alpha=0.85, hatch="//", edgecolor="#7a3010",
                    label="Contains extrapolated demand days")]
    ax1.legend(handles=handles, fontsize=7, loc="lower left")

    fig.suptitle("Monthly breakdown (hatched = month contains demand-table "
                 "extrapolated days)", fontsize=10)
    path = os.path.join(out_dir, "monthly_breakdown.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# ======================================================================
# Plot 4 - per-tank equity, ranked, coloured by C3-held status
# ======================================================================

def plot_per_tank_equity(eval_dir, out_dir):
    pt = load(eval_dir, "per_tank_summary.csv")
    pt = pt.dropna(subset=["satisfaction_pct"]).copy()
    if pt.empty:
        warn("per_tank_summary.csv has no satisfaction_pct values "
             "(demand was not reconstructed for this run) - skipping "
             "per-tank equity plot. Re-run evaluate_season.py with "
             "--demand-csv/--duration to enable this plot.")
        return None

    pt = pt[~pt["structural_terminus"].fillna(False)]
    pt = pt.sort_values("satisfaction_pct")

    fig, ax = plt.subplots(figsize=(6.5, max(4, 0.22 * len(pt))))
    colors = [C_HELD if h else C_FREE for h in pt["held_by_c3_season"].fillna(False)]
    ax.barh(pt["tank_id"], pt["satisfaction_pct"], color=colors)
    ax.set_xlabel("Season demand satisfaction (%)")
    ax.set_xlim(0, 100)
    ax.set_title("Per-tank demand satisfaction, ranked "
                 "(excludes structural network termini)")

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C_HELD, label="C3-held (majority of season)"),
                       Patch(color=C_FREE, label="Free")],
             fontsize=8, loc="lower right")

    path = os.path.join(out_dir, "per_tank_equity.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# ======================================================================
# Plot 5 - cross-check direction agreement over time
# ======================================================================

def plot_crosscheck(eval_dir, out_dir, summary_report=None):
    cc = load(eval_dir, "crosscheck_daily.csv")
    cc["date"] = to_dates(cc["date"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    ax1.plot(cc["date"], cc["direction_agreement_pct"], color="#3a7a5a",
             linewidth=1.0, alpha=0.8)
    # 15-day rolling mean to show any trend without visual noise dominating
    roll = cc["direction_agreement_pct"].rolling(15, min_periods=5).mean()
    ax1.plot(cc["date"], roll, color="#1f4d33", linewidth=2.0,
             label="15-day rolling mean")
    ax1.axhline(50, color="grey", linewidth=0.7, linestyle=":", label="Chance (50%)")
    ax1.set_ylabel("Direction agreement (%)")
    ax1.set_ylim(0, 100)
    ax1.legend(fontsize=8)
    ax1.set_title("Module 3 vs Module 4 trajectory-direction agreement")

    ax2.plot(cc["date"], cc["mean_abs_ratio_diff"], color="#8a4a2a", linewidth=1.0)
    ax2.set_ylabel("Mean |ratio diff|")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    fig.autofmt_xdate()

    if summary_report:
        ac = (summary_report.get("crosscheck") or {}).get("anchor_correction")
        if isinstance(ac, dict) and ac.get("direction_agreement_pct_CORRECTED") is not None:
            note = (f"Raw mean: {summary_report['crosscheck']['mean_direction_agreement_pct_RAW']:.1f}%  "
                   f"|  Corrected (excl. small-anchor days): "
                   f"{ac['direction_agreement_pct_CORRECTED']:.1f}%")
            fig.text(0.5, -0.02, note, ha="center", fontsize=7.5, color="#444")

    path = os.path.join(out_dir, "crosscheck_agreement.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ======================================================================
# Plot 9 - channel loss / network capture
# ======================================================================

def plot_channel_loss(eval_dir, out_dir):
    al = load(eval_dir, "alpha_channel_loss.csv")
    al = al.sort_values("network_capture_fraction")

    fig, ax = plt.subplots(figsize=(6.5, max(4, 0.2 * len(al))))
    colors = ["#999999" if t else C_STORAGE for t in al["structural_terminus"]]
    ax.barh(al["tank_id"], al["network_capture_fraction"], color=colors)
    ax.set_xlabel("Network capture fraction (share of return flow the "
                 "network delivers somewhere)")
    ax.set_xlim(0, 1.05)
    ax.axvline(1.0, color="grey", linewidth=0.7, linestyle=":")
    ax.set_title("Per-tank network capture (grey = structural network terminus,\n"
                 "excluded from the channel-loss statistic)")

    path = os.path.join(out_dir, "channel_loss.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# ======================================================================
# Plot 10 - rainfall response: storage gain vs same-day release/satisfaction
# ======================================================================

def plot_rainfall_response(eval_dir, out_dir, gain_threshold_m3=50000):
    sm = load(eval_dir, "season_metrics.csv")
    sm["date"] = to_dates(sm["date"])
    sm = sm.sort_values("date").reset_index(drop=True)
    sm["storage_gain"] = sm["total_storage_m3"].diff()

    events = sm[sm["storage_gain"] > gain_threshold_m3]
    if events.empty:
        warn(f"no days exceeded the {gain_threshold_m3:,.0f} m3 storage-gain "
             f"threshold - skipping rainfall response plot. Try a lower "
             f"--gain-threshold if the run is short or the cascade is small.")
        return None

    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    ax1.scatter(events["storage_gain"] / 1e3, events["sel_total_release"] / 1e6,
               color=C_STORAGE, alpha=0.75, s=28, label="Total release")
    ax1.set_xlabel("Storage gain that day (thousand m$^3$)")
    ax1.set_ylabel("Total release (million m$^3$)", color=C_STORAGE)
    ax1.tick_params(axis="y", labelcolor=C_STORAGE)

    ax2 = ax1.twinx()
    ax2.scatter(events["storage_gain"] / 1e3, events["sel_satisfaction_pct_mean"],
               color=C_SAT, alpha=0.6, s=22, marker="^", label="Satisfaction %")
    ax2.set_ylabel("Satisfaction (%)", color=C_SAT)
    ax2.tick_params(axis="y", labelcolor=C_SAT)
    ax2.set_ylim(0, 100)

    ax1.set_title(f"Controller response on rainfall-event days "
                 f"(storage gain > {gain_threshold_m3/1e3:.0f}k m$^3$, "
                 f"n={len(events)})")

    path = os.path.join(out_dir, "rainfall_response.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# ======================================================================
# Optional plot 6 - capacity estimation sensitivity
# ======================================================================

def plot_capacity_sensitivity(eval_dir, old_smax_eval_dir, out_dir):
    if old_smax_eval_dir is None:
        return None
    old_report = json.load(open(os.path.join(old_smax_eval_dir, "summary_report.json")))
    new_report = json.load(open(os.path.join(eval_dir, "summary_report.json")))

    old_sat = old_report["delivery"].get("season_mean_satisfaction_pct")
    new_sat = new_report["delivery"].get("season_mean_satisfaction_pct")
    old_f1 = old_report["delivery"].get("total_f1_shortage_m3")
    new_f1 = new_report["delivery"].get("total_f1_shortage_m3")
    if None in (old_sat, new_sat, old_f1, new_f1):
        warn("could not find satisfaction/shortage figures in one of the "
             "two summary_report.json files - skipping capacity "
             "sensitivity plot.")
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.6))
    ax1.bar(["Engineering\nS_max", "Satellite\nS_max"], [old_sat, new_sat],
           color=[C_FREE, C_HELD])
    ax1.set_ylabel("Mean satisfaction (%)")
    ax1.set_ylim(0, 100)
    ax1.set_title("Satisfaction")

    ax2.bar(["Engineering\nS_max", "Satellite\nS_max"],
           [old_f1 / 1e6, new_f1 / 1e6], color=[C_FREE, C_HELD])
    ax2.set_ylabel("Total shortage (million m$^3$)")
    ax2.set_title("Absolute shortage")

    fig.suptitle("Impact of capacity estimation source on outcomes", fontsize=10)
    path = os.path.join(out_dir, "capacity_sensitivity.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# ======================================================================
# Optional plot 7 - runoff coefficient sensitivity
# ======================================================================

def plot_runoff_sensitivity(runoff_sweep_csv, calibrated_c, out_dir):
    if runoff_sweep_csv is None:
        return None
    if not os.path.exists(runoff_sweep_csv):
        warn(f"--runoff-sweep-csv {runoff_sweep_csv} not found - skipping.")
        return None
    df = pd.read_csv(runoff_sweep_csv)
    if not {"C", "satisfaction_pct"}.issubset(df.columns):
        warn("runoff sweep CSV must have columns 'C' and 'satisfaction_pct' "
             "- skipping.")
        return None
    df = df.sort_values("C")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["C"], df["satisfaction_pct"], marker="o", color=C_STORAGE)
    if calibrated_c is not None:
        ax.axvline(calibrated_c, color=C_SAT, linestyle="--",
                  label=f"Calibrated C = {calibrated_c:.2f}")
        ax.legend(fontsize=8)
    ax.set_xlabel("Runoff coefficient C")
    ax.set_ylabel("Mean satisfaction (%)")
    ax.set_title("Sensitivity of season satisfaction to the runoff coefficient")

    path = os.path.join(out_dir, "runoff_sensitivity.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# ======================================================================
# Optional plot 8 - baseline comparison
# ======================================================================

def plot_baseline_comparison(eval_dir, baseline_eval_dir, out_dir):
    if baseline_eval_dir is None:
        return None
    proposed = json.load(open(os.path.join(eval_dir, "summary_report.json")))
    baseline = json.load(open(os.path.join(baseline_eval_dir, "summary_report.json")))

    metrics = [
        ("total_f1_shortage_m3", "f1 Shortage"),
        ("total_f2_overflow_m3", "f2 Overflow"),
        ("total_f4_water_loss_m3", "f4 Water loss"),
    ]
    p_vals = [proposed["delivery"].get(k, 0) / 1e6 for k, _ in metrics]
    b_vals = [baseline["delivery"].get(k, 0) / 1e6 for k, _ in metrics]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.8))
    x = np.arange(len(metrics))
    w = 0.35
    ax1.bar(x - w / 2, p_vals, w, label="MO-MPC", color=C_HELD)
    ax1.bar(x + w / 2, b_vals, w, label="Baseline", color=C_FREE)
    ax1.set_xticks(x)
    ax1.set_xticklabels([m[1] for m in metrics], rotation=20, ha="right")
    ax1.set_ylabel("Million m$^3$")
    ax1.legend(fontsize=8)
    ax1.set_title("Objective totals")

    p_sat = proposed["delivery"].get("season_mean_satisfaction_pct", 0)
    b_sat = baseline["delivery"].get("season_mean_satisfaction_pct", 0)
    ax2.bar(["MO-MPC", "Baseline"], [p_sat, b_sat], color=[C_HELD, C_FREE])
    ax2.set_ylabel("Mean satisfaction (%)")
    ax2.set_ylim(0, 100)
    ax2.set_title("Demand satisfaction")

    fig.suptitle("MO-MPC vs. proportional-allocation baseline", fontsize=10)
    path = os.path.join(out_dir, "baseline_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# ======================================================================
# Main
# ======================================================================

def main():
    args = parse_args()
    out_dir = args.out or os.path.join(args.eval_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    summary_report = None
    sr_path = os.path.join(args.eval_dir, "summary_report.json")
    if os.path.exists(sr_path):
        summary_report = json.load(open(sr_path, encoding="utf-8"))

    print(f"Generating plots from {args.eval_dir} -> {out_dir}\n")

    produced = []
    skipped = []

    def run(fn, *fargs, name=""):
        try:
            p = fn(*fargs)
            if p:
                produced.append(p)
                print(f"  [ok] {os.path.basename(p)}")
            else:
                skipped.append(name)
        except Exception as e:
            warn(f"{name} failed: {e}")
            skipped.append(name)

    run(plot_storage_satisfaction, args.eval_dir, out_dir, args.phase_boundaries,
        args.show_interpolated_gaps, name="storage_and_satisfaction")
    run(plot_objective_trajectories, args.eval_dir, out_dir,
        args.show_interpolated_gaps, name="objective_trajectories")
    run(plot_monthly_breakdown, args.eval_dir, out_dir,
        name="monthly_breakdown")
    run(plot_per_tank_equity, args.eval_dir, out_dir,
        name="per_tank_equity")
    run(plot_crosscheck, args.eval_dir, out_dir, summary_report,
        name="crosscheck_agreement")
    run(plot_channel_loss, args.eval_dir, out_dir,
        name="channel_loss")
    run(plot_rainfall_response, args.eval_dir, out_dir, 50000,
        name="rainfall_response")

    run(plot_capacity_sensitivity, args.eval_dir, args.old_smax_eval_dir, out_dir,
        name="capacity_sensitivity [optional: --old-smax-eval-dir]")
    run(plot_runoff_sensitivity, args.runoff_sweep_csv, args.calibrated_c, out_dir,
        name="runoff_sensitivity [optional: --runoff-sweep-csv]")
    run(plot_baseline_comparison, args.eval_dir, args.baseline_eval_dir, out_dir,
        name="baseline_comparison [optional: --baseline-eval-dir]")

    print(f"\n{len(produced)} plot(s) written to {out_dir}/")
    if skipped:
        print(f"{len(skipped)} skipped: {', '.join(skipped)}")
    print("\nNOTE: a Pareto-front snapshot plot was not generated. No run "
         "currently persists the full front (only the selected point), so "
         "there is nothing to plot without adding a front dump to the "
         "pipeline. Add one if this figure is wanted for Chapter 2's "
         "multi-objective-vs-scalarised argument.")


if __name__ == "__main__":
    main()