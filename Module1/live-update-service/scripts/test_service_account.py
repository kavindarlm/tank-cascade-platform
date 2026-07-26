"""
One-off diagnostic: confirms the GEE service account can actually read
nachchaduwa-32-final -- run this locally BEFORE wiring the key into
GitHub Actions secrets. Far easier to debug a permission error here than
inside a CI log.

Usage:
    python scripts/test_service_account.py <path-to-downloaded-key.json>

Expects the service account email to match config.GEE_PROJECT's naming
(module1-daily-update@gee-water-monitoring.iam.gserviceaccount.com) --
edit SA_EMAIL below if you named it differently.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ee
import config as C

SA_EMAIL = "module1-daily-update-990@gee-water-monitoring.iam.gserviceaccount.com"

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python scripts/test_service_account.py <path-to-key.json>")
        sys.exit(1)
    key_file = sys.argv[1]

    print(f"Authenticating as: {SA_EMAIL}")
    creds = ee.ServiceAccountCredentials(SA_EMAIL, key_file)
    ee.Initialize(creds, project=C.GEE_PROJECT)

    fc = ee.FeatureCollection(C.TANK_ASSET)
    n = fc.size().getInfo()
    print(f"Asset  : {C.TANK_ASSET}")
    print(f"Tanks found: {n}")
    if n == 32:
        print("PASS -- service account can read the full 32-tank asset.")
    else:
        print(f"WARNING -- expected 32 tanks, got {n}. Check the asset path/permissions.")