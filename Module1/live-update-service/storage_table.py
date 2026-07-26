"""
storage_table.py — the shared table (CSV in git) + serving contract.

Responsibilities:
  * load / save the table
  * append newly-observed rows (with a last_updated timestamp)
  * retroactive outlier repair on the LIVE tail only (never touches
    validated_2021_2025 rows)
  * serving: get_tank_volume() returns the OBSERVED row, or the raw
    before/after neighbours — NO interpolation (that is Module 3's job)

Module boundary discipline (unchanged): this module only ever writes a
row for a date on which a real satellite acquisition existed. Missing
dates stay missing. Estimation of unobserved dates belongs to Module 3.
"""
import datetime as dt
import pandas as pd
import config as C


def _now_iso():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_table():
    df = pd.read_csv(C.TABLE_CSV)
    df["date"] = pd.to_datetime(df["date"])
    return df


def save_table(df):
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out = out[C.ALL_COLUMNS]
    out.to_csv(C.TABLE_CSV, index=False)


def last_observed_date(df):
    """Newest date already in the table — the point to resume from."""
    return pd.to_datetime(df["date"]).max()


def existing_dates(df):
    return set(pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d"))


def append_rows(df, new_rows):
    """new_rows: list of dicts with BASE_COLUMNS keys. Tags them 'live'."""
    if not new_rows:
        return df, 0
    stamp = _now_iso()
    for r in new_rows:
        r["record_status"] = "live"
        r["last_updated"] = stamp
    add = pd.DataFrame(new_rows)
    add["date"] = pd.to_datetime(add["date"])
    for col in C.ALL_COLUMNS:
        if col not in add.columns:
            add[col] = "" if col in C.EXTRA_COLUMNS else None
    combined = pd.concat([df, add[C.ALL_COLUMNS]], ignore_index=True)
    combined = combined.sort_values(["date", "tank_id"]).reset_index(drop=True)
    return combined, len(add)


def retroactive_clean(df):
    """
    Apply the SAME single-date sandwich rule as 1_Outlier_Cleaning...py,
    but ONLY to 'live' rows that now have both a previous and a next
    neighbour. This is why a brand-new row is never cleaned on the day it
    lands (it has no 'after' yet) — it gets repaired later, once a newer
    observation gives it a right-hand neighbour. Validated rows are frozen.
    """
    df = df.sort_values(["pond_name", "date"]).reset_index(drop=True)
    changed = 0
    for tank, g in df.groupby("pond_name", sort=False):
        idx = g.index.tolist()
        fill = g["fill_pct"].values
        status = g["record_status"].values
        for k in range(1, len(idx) - 1):
            if status[k] != "live":
                continue
            if (fill[k - 1] - fill[k]) > 30 and (fill[k + 1] - fill[k]) > 30:
                # linear interpolate water_ha & fill_pct from the two neighbours
                w = (g["water_ha"].values[k - 1] + g["water_ha"].values[k + 1]) / 2
                f = (fill[k - 1] + fill[k + 1]) / 2
                row = df.loc[idx[k]]
                df.loc[idx[k], "water_ha"] = w
                df.loc[idx[k], "fill_pct"] = f
                df.loc[idx[k], "volume_mcm"] = C.final_volume(tank, w, row["total_ha"])
                df.loc[idx[k], "last_updated"] = _now_iso()
                changed += 1
    return df.sort_values(["date", "tank_id"]).reset_index(drop=True), changed


# ── Serving contract for Module 2 / Module 3 ──────────────────────
def get_tank_volume(pond_name, date):
    """
    Return storage info for a tank on a date, WITHOUT interpolation.

    Returns a dict:
      {'status': 'observed', 'row': {...}}                      exact date exists
      {'status': 'bracketed', 'before': {...}, 'after': {...}}  neighbours only
      {'status': 'no_data', ...}                                nothing usable
    Module 3 decides how (or whether) to interpolate between before/after.
    """
    df = load_table()
    target = pd.to_datetime(date)
    t = df[df["pond_name"] == pond_name].sort_values("date")
    if t.empty:
        return {"status": "no_data", "reason": f"unknown tank {pond_name!r}"}

    exact = t[t["date"] == target]
    if not exact.empty:
        return {"status": "observed", "row": _clean(exact.iloc[0])}

    before = t[t["date"] < target].tail(1)
    after = t[t["date"] > target].head(1)
    return {
        "status": "bracketed",
        "before": _clean(before.iloc[0]) if not before.empty else None,
        "after": _clean(after.iloc[0]) if not after.empty else None,
    }


def _clean(row):
    d = row[C.ALL_COLUMNS].to_dict()
    d["date"] = pd.to_datetime(d["date"]).strftime("%Y-%m-%d")
    return d
