"""
serve.py — thin CLI so Module 2/3 can query without importing internals.

  python serve.py "Nachchaduwa Wewa" 2026-02-14

Programmatic use is simpler still:
  from storage_table import get_tank_volume
  get_tank_volume("Nachchaduwa Wewa", "2026-02-14")
Returns the observed row, or the raw before/after neighbours. No
interpolation — Module 3 owns that.
"""
import sys, json
from storage_table import get_tank_volume

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('usage: python serve.py "<pond_name>" <YYYY-MM-DD>'); sys.exit(1)
    print(json.dumps(get_tank_volume(sys.argv[1], sys.argv[2]), indent=2))
