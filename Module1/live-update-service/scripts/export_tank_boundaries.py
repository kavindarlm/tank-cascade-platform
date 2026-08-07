#!/usr/bin/env python3
"""One-off export: convert the source tank-boundary shapefile into a static
GeoJSON the frontend can fetch directly (no GEE credentials needed at
request time - this is the same digitized boundary data the GEE asset
`projects/gee-water-monitoring/assets/nachchaduwa-32-final` was built from,
see ../development-history/Tank Boundary/nachchaduwa-32-final/).

Each feature's tank_id is joined from data/tank_meta.csv by pond_name
(whitespace-normalized, since both the shapefile and tank_meta.csv contain
inconsistent double-spacing in a few names - e.g. "Settikulama  Wewa").

Usage:
    python scripts/export_tank_boundaries.py
"""
import csv
import json
from pathlib import Path

import shapefile

BASE_DIR = Path(__file__).resolve().parent.parent
SHP_PATH = (
    BASE_DIR.parent / "development-history" / "Tank Boundary"
    / "nachchaduwa-32-final" / "nachchaduwa-32-final.shp"
)
TANK_META_PATH = BASE_DIR / "data" / "tank_meta.csv"
OUT_PATH = BASE_DIR / "data" / "tank_boundaries.geojson"


def _norm(name):
    return " ".join((name or "").split())


def _load_tank_ids():
    by_name = {}
    with TANK_META_PATH.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            by_name[_norm(row["pond_name"])] = (row["tank_id"], row["total_ha"])
    return by_name


def _ring_from_points(points):
    # Shapefile points are already closed rings (first == last); GeoJSON
    # expects the same. Drop Z (POLYGONZ) - the map only needs X/Y.
    return [[round(x, 7), round(y, 7)] for x, y in points]


def main():
    tank_ids = _load_tank_ids()
    sf = shapefile.Reader(str(SHP_PATH))

    features = []
    unmatched = []
    for sr in sf.iterShapeRecords():
        rec = sr.record.as_dict()
        shp = sr.shape
        pond_name = _norm(rec.get("pond_name"))
        match = tank_ids.get(pond_name)
        if match is None:
            unmatched.append(pond_name)
            continue
        tank_id, total_ha = match

        # Single ring per tank (verified: no multi-part shapes in this file).
        ring = _ring_from_points(shp.points)
        xmin, ymin, xmax, ymax = shp.bbox
        centroid = [round((xmin + xmax) / 2, 7), round((ymin + ymax) / 2, 7)]

        features.append({
            "type": "Feature",
            "properties": {
                "tank_id": tank_id,
                "pond_name": pond_name,
                "total_ha": float(total_ha),
                "centroid": centroid,
                "bbox": [xmin, ymin, xmax, ymax],
            },
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        })

    if unmatched:
        raise SystemExit(f"No tank_id match in tank_meta.csv for: {unmatched}")

    features.sort(key=lambda f: int(f["properties"]["tank_id"]))
    geojson = {"type": "FeatureCollection", "features": features}

    OUT_PATH.write_text(json.dumps(geojson, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(features)} tank boundaries to {OUT_PATH}")


if __name__ == "__main__":
    main()
