"""
gee_detect.py — the detection stage, ported to the earthengine-api.

IMPORTANT: this does NOT process pixels locally. Every reduceRegion /
mosaic / threshold below is executed on Google's GEE cloud servers (the
"GEE web"). This machine only sends the request and receives the numbers.
That is what lets the daily job — and any teammate — run it without a
heavy local geoprocessing stack.

Logic is a faithful port of:
  * 1_SMART_SENSOR_6DAY_TIME_SERIES...  (S1 descending, VV<-16, focal_mean 30m)
  * 2_FIX_S2_Tile_Mosaicking...         (same-date S2 mosaic + valid_fraction>0.5)
  * 3_Rebuild_Consolidated...           (small=S1 only; large=S2 within +/-3 days else S1 fallback)

Same-date S1 descending frames are mosaicked before per-tank reduction, so
each (tank, date) yields exactly one row and the whole study area is
covered — the same one-row-per-tank-per-date shape as the official CSV.
"""
import pandas as pd
import ee
import config as C

_TANKS = None          # ee.FeatureCollection of tanks (cached)
_STUDY = None          # study-area bounds (cached)
_META = None           # DataFrame: tank_id, pond_name, total_ha


# ── Auth ──────────────────────────────────────────────────────────
def initialize_ee(service_account=None, key_file=None):
    """
    Interactive (local dev):   initialize_ee()
    Unattended (CI/scheduler): initialize_ee(sa_email, 'key.json')
    """
    if service_account and key_file:
        creds = ee.ServiceAccountCredentials(service_account, key_file)
        ee.Initialize(creds, project=C.GEE_PROJECT)
    else:
        ee.Initialize(project=C.GEE_PROJECT)
    global _TANKS, _STUDY, _META
    _TANKS = ee.FeatureCollection(C.TANK_ASSET)
    _STUDY = _TANKS.geometry().bounds()
    _META = pd.read_csv(C.TANK_META)


def _speckle(img):
    return (img.focal_mean(C.SPECKLE_RADIUS, "circle", "meters")
               .clip(_STUDY)
               .copyProperties(img, ["system:time_start"]))


def _s1_desc(start, end):
    return (ee.ImageCollection(C.S1_COLLECTION)
            .filterBounds(_STUDY)
            .filterDate(start, end)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.eq("orbitProperties_pass", C.S1_ORBIT))
            .select("VV")
            .map(_speckle))


def s1_acquisition_dates(start, end):
    """The date grid: distinct S1 descending acquisition dates in [start, end)."""
    col = _s1_desc(start, end)
    dates = col.map(lambda i: ee.Feature(None, {"d": i.date().format("YYYY-MM-dd")}))
    lst = dates.aggregate_array("d").distinct().getInfo()
    return sorted(set(lst or []))


def _water_ha_s1(day):
    """Same-date descending mosaic -> per-tank water_ha (dict pond_name->ha)."""
    nxt = ee.Date(day).advance(1, "day")
    mosaic = _s1_desc(day, nxt).mosaic()
    mask = mosaic.lt(C.S1_VV_THRESHOLD).rename("water")
    fc = _TANKS.map(lambda t: t.set(
        "water_ha",
        mask.multiply(ee.Image.pixelArea()).reduceRegion(
            reducer=ee.Reducer.sum(), geometry=t.geometry(),
            scale=C.SCALE_M, maxPixels=C.MAXPIXELS, bestEffort=True
        ).get("water")))
    rows = fc.getInfo()["features"]
    out = {}
    for f in rows:
        p = f["properties"]
        w = p.get("water_ha")
        out[p[C.NAME_COL]] = (w / 1e4) if w is not None else 0.0
    return out


def _best_s2_for_large(day):
    """
    For each large tank, the nearest valid S2 observation within +/-3 days.
    Returns dict pond_name -> (s2_date_str, water_ha, day_gap) or absent.
    Mirrors the tile-mosaic fix: same-date mosaic + valid_fraction>0.5.
    """
    d0 = ee.Date(day)
    start = d0.advance(-C.S2_WINDOW_DAYS, "day")
    end = d0.advance(C.S2_WINDOW_DAYS + 1, "day")

    def mask_clouds(img):
        qa = img.select("QA60")
        m = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
        return (img.updateMask(m).select(["B3", "B8", "B11"]).divide(10000)
                   .clip(_STUDY)
                   .copyProperties(img, ["system:time_start"]))

    s2 = (ee.ImageCollection(C.S2_COLLECTION)
          .filterBounds(_STUDY).filterDate(start, end)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", C.S2_CLOUD_MAX))
          .map(mask_clouds)
          .map(lambda i: i.set("date_str", i.date().format("YYYY-MM-dd"))))

    date_strs = s2.aggregate_array("date_str").distinct().getInfo() or []
    large = _META[_META["pond_name"].isin(C.LARGE_TANKS)]
    result = {}
    for ds in sorted(date_strs):
        day_imgs = s2.filter(ee.Filter.eq("date_str", ds))
        mosaic = day_imgs.mosaic()
        mndwi = mosaic.normalizedDifference(C.S2_MNDWI_BANDS)
        water = mndwi.gt(0).rename("water")
        gap = abs((pd.to_datetime(ds) - pd.to_datetime(day)).days)
        tanks_fc = _TANKS.filter(ee.Filter.inList(C.NAME_COL, C.LARGE_TANKS))

        def per_tank(t):
            geom = t.geometry()
            valid = mosaic.select("B3").reduceRegion(
                reducer=ee.Reducer.count(), geometry=geom,
                scale=C.SCALE_M, maxPixels=C.MAXPIXELS, bestEffort=True).get("B3")
            w = water.multiply(ee.Image.pixelArea()).reduceRegion(
                reducer=ee.Reducer.sum(), geometry=geom,
                scale=C.SCALE_M, maxPixels=C.MAXPIXELS, bestEffort=True).get("water")
            return t.set("valid", valid).set("w", w)

        feats = tanks_fc.map(per_tank).getInfo()["features"]
        for f in feats:
            p = f["properties"]
            name = p[C.NAME_COL]
            total_ha = float(large[large["pond_name"] == name]["total_ha"].iloc[0])
            expected = total_ha * 100.0
            valid = p.get("valid") or 0
            if expected <= 0 or (valid / expected) <= C.S2_VALID_FRAC:
                continue                      # drop low-validity obs (no false zeros)
            water_ha = (p.get("w") or 0.0) / 1e4
            prev = result.get(name)
            if prev is None or gap < prev[2]:
                result[name] = (ds, water_ha, gap)
    return result


def detect_for_date(day):
    """
    Full consolidated detection for one S1 date -> list of 32 row dicts
    (BASE_COLUMNS). Applies the sensor-priority rule and frozen volume.
    """
    s1 = _water_ha_s1(day)                      # all 32 tanks, S1
    s2 = _best_s2_for_large(day)                # large tanks, best S2 in window
    ts = pd.to_datetime(day)
    rows = []
    for _, m in _META.iterrows():
        name = m["pond_name"]
        total_ha = float(m["total_ha"])
        if name in C.LARGE_TANKS and name in s2:
            s2_date, water_ha, gap = s2[name]
            sensor, s2d, day_gap = "S2_MNDWI", s2_date, int(gap)
        else:
            water_ha = s1.get(name, 0.0)
            sensor = "S1_SAR_fallback" if name in C.LARGE_TANKS else "S1_SAR"
            s2d, day_gap = None, 0
        fill_pct = (water_ha / total_ha * 100) if total_ha > 0 else 0.0
        rows.append({
            "tank_id": int(m["tank_id"]),
            "pond_name": name,
            "date": day,
            "year": ts.year,
            "month": ts.month,
            "water_ha": water_ha,
            "total_ha": total_ha,
            "fill_pct": fill_pct,
            "sensor_used": sensor,
            "s2_date": s2d,
            "day_gap": day_gap,
            "volume_mcm": C.final_volume(name, water_ha, total_ha),
            "volume_confidence": C.volume_confidence(name),
        })
    return rows
