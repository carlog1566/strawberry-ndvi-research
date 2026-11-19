#!/usr/bin/env python3
"""Per-pixel, future-date NDVI linear-regression baseline.

Input: manifest CSV with columns `date,with_soil` (single-band NDVI GeoTIFFs).
The optional `without_soil` column is ignored intentionally: the recommended
approach uses WITH-SOIL raster values, selecting vegetated pixels using only
NDVI measured at the prediction's latest INPUT date.

Commands:
  python models/LinearReg_Baseline.py train --manifest ndvi_manifest.csv
  python models/LinearReg_Baseline.py predict --manifest ndvi_manifest.csv \
      --date 2026-04-15 --out predictions/ndvi_2026-04-15.tif
"""

import argparse
import csv
import math
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

NODATA = 0
_WARNED_NODATA = set()
FEATURE_NAMES = [
    "ndvi_latest", "ndvi_previous", "trend_ndvi_per_day",
    "days_from_previous_to_latest", "days_until_target",
    "sin_target_day_of_year", "cos_target_day_of_year",
    "ndvi_latest_x_days_until_target", "trend_x_days_until_target",
]


def read_manifest(path):
    """Return observations sorted chronologically; paths relative to manifest."""
    path = Path(path).expanduser().resolve()
    observations = []
    with path.open(newline="", encoding="utf-8-sig") as fp:
        reader = csv.DictReader(fp)
        if not {"date", "with_soil"}.issubset(reader.fieldnames or []):
            raise ValueError("Manifest needs CSV columns: date,with_soil")
        for row in reader:
            day = date.fromisoformat(row["date"].strip())
            raw = Path(row["with_soil"].strip()).expanduser()
            full = raw if raw.is_absolute() else path.parent / raw
            if not full.is_file():
                raise FileNotFoundError(f"Missing NDVI file for {day}: {full}")
            observations.append((day, full.resolve()))
    observations.sort(key=lambda item: item[0])
    if len(observations) < 3:
        raise ValueError("At least 3 dated observations are needed to train.")
    if len({day for day, _ in observations}) != len(observations):
        raise ValueError("Duplicate capture dates in manifest; use one mosaic per date.")
    return observations


def build_grid(reference_path, resolution, max_pixels):
    """Create one north-up grid in the reference NDVI raster's CRS."""
    with rasterio.open(reference_path) as src:
        if src.count != 1:
            raise ValueError(f"Expected one NDVI band in {reference_path}, got {src.count}.")
        if src.crs is None or not src.crs.is_projected:
            raise ValueError("Reference NDVI must have a projected CRS in meters (e.g., EPSG:32611).")
        if src.crs.linear_units.lower() not in ("metre", "meter", "metres", "meters", "m"):
            raise ValueError(f"Reference CRS units should be meters, got {src.crs.linear_units!r}.")
        bounds = src.bounds
        width = math.ceil((bounds.right - bounds.left) / resolution)
        height = math.ceil((bounds.top - bounds.bottom) / resolution)
        if width * height > max_pixels:
            raise ValueError(
                f"Output grid has {width * height:,} pixels; raise --resolution "
                "(e.g. 1.0) or --max-pixels if you have enough RAM."
            )
        return {
            "crs": src.crs,
            "transform": from_origin(bounds.left, bounds.top, resolution, resolution),
            "width": width,
            "height": height,
            "resolution": resolution,
        }


def aligned_ndvi(path, grid):
    """Read one raster on the common grid, handling actual NoData as NaN."""
    with rasterio.open(path) as src:
        if src.count != 1:
            raise ValueError(f"Expected single-band NDVI GeoTIFF: {path}")
        if src.crs is None:
            raise ValueError(f"GeoTIFF has no CRS: {path}")
        if path not in _WARNED_NODATA:
            if src.nodata == 0:
                print(f"WARNING: {Path(path).name} declares NoData=0. "
                      "Real NDVI=0 pixels cannot be distinguished from missing pixels; "
                      "re-export with a distinct NoData value, such as -9999.")
            elif src.nodata is None:
                print(f"WARNING: {Path(path).name} has no declared NoData. "
                      "Verify that outside-of-field background is not stored as valid NDVI=0.")
            _WARNED_NODATA.add(path)
        # Average aggregation reduces the size of 0.016 m imagery for a baseline.
        # It respects the source's declared NoData value; 0 is NOT auto-masked.
        with WarpedVRT(
            src,
            crs=grid["crs"],
            transform=grid["transform"],
            width=grid["width"],
            height=grid["height"],
            resampling=Resampling.average,
            nodata=NODATA,
        ) as vrt:
            values = vrt.read(1, masked=True).filled(np.nan).astype(np.float32)
    # NDVI legitimately ranges from -1 to +1. Drop out-of-range artifact values.
    values[~np.isfinite(values) | (values < -1.01) | (values > 1.01)] = np.nan
    return values


def valid_input_pixels(previous, latest, min_vegetation):
    eligible = np.isfinite(previous) & np.isfinite(latest)
    # Apply ONLY to the latest available input image, never to the future label.
    if min_vegetation is not None:
        eligible &= latest > min_vegetation
    return eligible


def feature_matrix(previous, latest, older_date, latest_date, target_date, mask):
    """Generate features using two past images and a known desired future date."""
    history_days = (latest_date - older_date).days
    lead_days = (target_date - latest_date).days
    if history_days <= 0 or lead_days <= 0:
        raise ValueError("Require older_date < latest_date < target_date.")
    old = previous[mask].astype(np.float64)
    cur = latest[mask].astype(np.float64)
    trend = (cur - old) / history_days
    theta = 2.0 * np.pi * (target_date.timetuple().tm_yday - 1) / 365.25
    size = cur.size
    return np.column_stack([
        cur,
        old,
        trend,
        np.full(size, history_days),
        np.full(size, lead_days),
        np.full(size, np.sin(theta)),
        np.full(size, np.cos(theta)),
        cur * lead_days,
        trend * lead_days,
    ])


def spatial_region_masks(height, width):
    """North-up imagery: bottom 60% train, middle 20% val, top 20% test."""
    r20 = math.floor(height * 0.2)
    r40 = math.floor(height * 0.4)
    if r20 < 1 or r40 <= r20 or r40 >= height:
        raise ValueError("Reference grid is too short for a 60/20/20 spatial split.")
    rows = np.arange(height)[:, None]
    return {
        "train_south_60": np.broadcast_to(rows >= r40, (height, width)),
        "val_middle_20": np.broadcast_to((rows >= r20) & (rows < r40), (height, width)),
        "test_north_20": np.broadcast_to(rows < r20, (height, width)),
    }


def collect_samples(observations, grid, config):
    rng = np.random.default_rng(config["seed"])
    regions = spatial_region_masks(grid["height"], grid["width"])
    pieces = {key: {"X": [], "y": []} for key in regions}
    eligible_triplets = 0

    for i in range(2, len(observations)):
        older, latest, future = observations[i - 2:i + 1]
        history_days = (latest[0] - older[0]).days
        lead_days = (future[0] - latest[0]).days
        if history_days > config["max_history_days"] or lead_days > config["max_horizon_days"]:
            print(f"SKIP  {future[0]} (history={history_days}d, forecast={lead_days}d)")
            continue

        previous = aligned_ndvi(older[1], grid)
        current = aligned_ndvi(latest[1], grid)
        truth = aligned_ndvi(future[1], grid)
        eligible = valid_input_pixels(previous, current, config["min_vegetation"])
        eligible &= np.isfinite(truth)  # only label availability, NOT vegetation threshold
        eligible_triplets += 1
        counts = []

        for region, region_mask in regions.items():
            coordinates = np.flatnonzero(eligible & region_mask)
            if len(coordinates) == 0:
                counts.append(f"{region}:0")
                continue
            sample_n = min(config["samples_per_target_region"], len(coordinates))
            chosen = rng.choice(coordinates, size=sample_n, replace=False)
            selected = np.zeros(eligible.size, dtype=bool)
            selected[chosen] = True
            selected = selected.reshape(eligible.shape)
            X = feature_matrix(previous, current, older[0], latest[0], future[0], selected)
            y = truth[selected].astype(np.float64)
            pieces[region]["X"].append(X)
            pieces[region]["y"].append(y)
            counts.append(f"{region}:{sample_n:,}")

        print(f"USE   {older[0]} -> {latest[0]} -> {future[0]} | " + " ".join(counts))

    if eligible_triplets == 0:
        raise ValueError("No usable date triplets. Increase the --max-*-days limits.")
    datasets = {}
    for region, data in pieces.items():
        if not data["X"]:
            raise ValueError(
                f"No samples in {region}. Check raster coverage, NoData, and "
                "--min-vegetation / --include-soil."
            )
        datasets[region] = (np.concatenate(data["X"]), np.concatenate(data["y"]))
    return datasets


def show_metrics(name, model, X, y):
    predictions = np.clip(model.predict(X), -1, 1)
    persistence = X[:, 0]  # predicts NDVI will stay at its most recent observed value
    print(f"\n{name}: {len(y):,} pixel/date examples")
    for label, estimate in [("Linear regression", predictions), ("Persistence", persistence)]:
        rmse = np.sqrt(mean_squared_error(y, estimate))
        mae = mean_absolute_error(y, estimate)
        r2 = r2_score(y, estimate)
        print(f"  {label:17s} RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")


def train(args):
    observations = read_manifest(args.manifest)
    ref = Path(args.reference).resolve() if args.reference else observations[0][1]
    grid = build_grid(ref, args.resolution, args.max_pixels)
    config = {
        "min_vegetation": None if args.include_soil else args.min_vegetation,
        "max_history_days": args.max_history_days,
        "max_horizon_days": args.max_horizon_days,
        "samples_per_target_region": args.samples_per_target_region,
        "seed": args.seed,
    }
    print(f"Grid: {grid['width']} x {grid['height']}, {args.resolution} meters/pixel, {grid['crs']}")
    print(f"Training mode: {'all valid land pixels' if args.include_soil else f'input vegetation > {args.min_vegetation}'}")
    datasets = collect_samples(observations, grid, config)
    Xtrain, ytrain = datasets["train_south_60"]
    # Ordinary least squares on standardized engineered features (no regularization).
    model = make_pipeline(StandardScaler(), LinearRegression())
    model.fit(Xtrain, ytrain)
    print(f"\nTrained ordinary least-squares model on {len(ytrain):,} examples.")
    for region in ("train_south_60", "val_middle_20", "test_north_20"):
        show_metrics(region, model, *datasets[region])
    out = Path(args.model_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "grid": grid, "config": config,
                 "features": FEATURE_NAMES}, out)
    print(f"\nSaved model to: {out.resolve()}")
    print("Caution: the spatial test shares target dates with training; also test future held-out dates before deployment.")


def predict(args):
    saved = joblib.load(args.model)  # Load only your own trusted model files.
    observations = read_manifest(args.manifest)
    target = date.fromisoformat(args.date)
    past = [obs for obs in observations if obs[0] < target]
    if len(past) < 2:
        raise ValueError("The requested future date needs at least 2 older NDVI captures.")
    older, latest = past[-2:]
    gap = (latest[0] - older[0]).days
    lead = (target - latest[0]).days
    cfg = saved["config"]
    if gap > cfg["max_history_days"] or lead > cfg["max_horizon_days"]:
        raise ValueError(
            f"Prediction has history={gap}d and lead={lead}d, outside training "
            f"limits ({cfg['max_history_days']}d, {cfg['max_horizon_days']}d). "
            "Choose a closer prediction date or retrain with wider limits "
            "and sufficient similar training examples."
        )
    grid = saved["grid"]
    previous = aligned_ndvi(older[1], grid)
    current = aligned_ndvi(latest[1], grid)
    valid = valid_input_pixels(previous, current, cfg["min_vegetation"])
    if not valid.any():
        raise ValueError("No valid pixels in latest two input rasters on the reference grid.")
    X = feature_matrix(previous, current, older[0], latest[0], target, valid)
    prediction = np.full(current.shape, NODATA, dtype=np.float32)
    prediction[valid] = np.clip(saved["model"].predict(X), -1, 1).astype(np.float32)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out, "w", driver="GTiff", height=grid["height"], width=grid["width"],
        count=1, dtype="float32", crs=grid["crs"], transform=grid["transform"],
        nodata=NODATA, compress="deflate", predictor=3,
    ) as dst:
        dst.write(prediction, 1)
        dst.set_band_description(1, f"Predicted NDVI for {target}")
    print(f"Predicted NDVI for {target} using {older[0]} and {latest[0]}")
    print(f"Predicted {valid.sum():,} pixels; NoData={NODATA}; output={out.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    train_cmd = commands.add_parser("train", help="Train and evaluate a baseline")
    train_cmd.add_argument("--manifest", required=True, help="CSV: date,with_soil")
    train_cmd.add_argument("--reference", default=None, help="Raster defining the study-area extent")
    train_cmd.add_argument("--resolution", type=float, default=0.5, help="Model grid meters/pixel (default 0.5)")
    train_cmd.add_argument("--max-pixels", type=int, default=3_000_000)
    train_cmd.add_argument("--max-history-days", type=int, default=120)
    train_cmd.add_argument("--max-horizon-days", type=int, default=120)
    train_cmd.add_argument("--samples-per-target-region", type=int, default=15_000)
    train_cmd.add_argument("--min-vegetation", type=float, default=0.2)
    train_cmd.add_argument("--include-soil", action="store_true", help="Train on all valid pixels, including soil")
    train_cmd.add_argument("--seed", type=int, default=42)
    train_cmd.add_argument("--model-out", default="ndvi_linear_model.joblib")
    train_cmd.set_defaults(func=train)

    predict_cmd = commands.add_parser("predict", help="Write a future-date predicted NDVI GeoTIFF")
    predict_cmd.add_argument("--manifest", required=True)
    predict_cmd.add_argument("--model", default="ndvi_linear_model.joblib")
    predict_cmd.add_argument("--date", required=True, help="YYYY-MM-DD")
    predict_cmd.add_argument("--out", required=True, help="Output GeoTIFF path")
    predict_cmd.set_defaults(func=predict)

    args = parser.parse_args()
    if getattr(args, "resolution", 1) <= 0:
        parser.error("--resolution must be positive")
    if getattr(args, "samples_per_target_region", 1) <= 0:
        parser.error("--samples-per-target-region must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
