"""
run_daily.py — the scheduled entry point. Checks whether GEE has any new
S1 acquisition since the last recorded date and, if so, appends it.
Most days there is nothing new (Sentinel revisit is ~6-12 days) and the
table is left unchanged — that is correct, not an error.

Local:      python run_daily.py
CI/cron:    GEE_SA_EMAIL and GEE_KEY_FILE env vars -> service-account auth
"""
import os
import gee_detect as G
import update

if __name__ == "__main__":
    sa  = os.environ.get("GEE_SA_EMAIL")
    key = os.environ.get("GEE_KEY_FILE")
    G.initialize_ee(service_account=sa, key_file=key) if (sa and key) else G.initialize_ee()
    added, dates = update.update_table()
    print(f"::daily-summary:: added={added} dates={dates}")
