"""
backfill.py — ONE-TIME. Fills the gap between the last date in the seeded
table (2025-12-20) and today. After this succeeds once, the daily job
keeps the table current. Run:  python backfill.py
"""
import ee, update
import gee_detect as G

if __name__ == "__main__":
    G.initialize_ee()                 # interactive auth is fine for a manual one-off
    update.update_table()
