"""
update.py — the single update routine. Backfill and the daily job both
call update_table(); backfill just runs it once now over a wide gap, the
scheduler runs it every day over a narrow one. One code path, no drift.

Flow:
  1. find S1 descending acquisition dates between the last recorded date
     and `up_to` that are NOT already in the table
  2. for each, run the consolidated detection on GEE and append the rows
  3. retroactively repair any live outliers that now have both neighbours
"""
import datetime as dt
import config as C
import gee_detect as G
import storage_table as T


def update_table(up_to=None, verbose=True):
    if up_to is None:
        up_to = dt.date.today().isoformat()

    df = T.load_table()
    have = T.existing_dates(df)
    resume = (T.last_observed_date(df) + dt.timedelta(days=1)).date().isoformat()

    if verbose:
        print(f"Last recorded date : {T.last_observed_date(df).date()}")
        print(f"Scanning GEE for S1 dates in [{resume} .. {up_to}] ...")

    # +1 day on the end so an acquisition dated `up_to` is included
    end_excl = (dt.date.fromisoformat(up_to) + dt.timedelta(days=1)).isoformat()
    s1_dates = G.s1_acquisition_dates(resume, end_excl)
    new_dates = [d for d in s1_dates if d not in have]

    if verbose:
        print(f"New S1 acquisition dates found: {len(new_dates)} -> {new_dates}")

    total_added = 0
    for d in new_dates:
        rows = G.detect_for_date(d)
        df, n = T.append_rows(df, rows)
        total_added += n
        if verbose:
            print(f"  {d}: appended {n} tank rows")

    if new_dates:
        df, repaired = T.retroactive_clean(df)
        if verbose and repaired:
            print(f"Retroactively repaired {repaired} live outlier row(s)")
        T.save_table(df)

    if verbose:
        print(f"Done. {total_added} rows added across {len(new_dates)} new date(s).")
    return total_added, new_dates
