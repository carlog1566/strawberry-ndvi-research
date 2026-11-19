"""Windowed, disk-backed alignment; no temporal interpolation or target masking."""
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.windows import Window
from rasterio.warp import transform_bounds
from affine import Affine


def manifest(path, data_root=None, require_seasons=False):
    path = Path(path).resolve()
    frame = pd.read_csv(path)
    if 'season_id' in frame:
        if 'season' in frame and not frame['season_id'].equals(frame['season']):
            raise ValueError('Conflicting season and season_id columns')
        frame['season'] = frame['season_id']
    if 'path' not in frame and 'with_soil' in frame:
        frame = frame.rename(columns={'with_soil': 'path'})
    if not {'date', 'path'} <= set(frame):
        raise ValueError('Manifest requires date,season,path (season defaults to calendar year)')
    frame['date'] = pd.to_datetime(frame.date, format='%Y-%m-%d', errors='raise')
    if 'season' not in frame:
        if require_seasons:
            raise ValueError('Explicit season column required: label actual planting/harvest cycles')
        warnings.warn('No season column: using separate calendar years. Review agronomic boundaries.')
        frame['season'] = frame.date.dt.year.astype(str)
    if frame[['date', 'season', 'path']].isna().any().any():
        raise ValueError('Missing manifest date, season, or path')
    frame['season'] = frame.season.astype(str)
    if frame.duplicated(['season', 'date']).any():
        raise ValueError('Duplicate acquisition date within a season')
    root = Path(data_root).resolve() if data_root else path.parent
    for col in ['path'] + [c for c in frame if c.endswith('_path')]:
        frame[col] = [str((root / str(p)).resolve()) for p in frame[col]]
        for p in frame[col]:
            if not Path(p).is_file():
                raise FileNotFoundError(p)
    return frame.sort_values(['season', 'date']).reset_index(drop=True)


def windows(height, width, size):
    for row in range(0, height, size):
        for col in range(0, width, size):
            yield Window(col, row, min(size, width-col), min(size, height-row))


def clean(values, ndvi=True):
    values = values.astype('float32')
    values[~np.isfinite(values)] = np.nan
    if ndvi:
        values[(values < -1) | (values > 1)] = np.nan
    return values


def preprocess(manifest_path, out, cfg, data_root=None, reference=None):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Cache already exists; use a new directory to avoid stale data')
    out.mkdir(parents=True, exist_ok=True)
    frame = manifest(manifest_path, data_root, cfg.get('require_seasons', False))
    ref = reference or frame.path.iloc[0]
    with rasterio.open(ref) as src:
        if src.crs is None or abs(src.transform.determinant) == 0:
            raise ValueError('Reference requires a CRS and nonsingular transform')
        grid = dict(crs=src.crs.to_string(), transform=list(src.transform)[:6],
                    height=src.height, width=src.width)
        transform = src.transform
    # Union snapped to the reference lattice retains partially covered edges.
    if cfg.get('grid_extent', 'reference') == 'union':
        if transform.b or transform.d or transform.a <= 0 or transform.e >= 0:
            raise ValueError('Union grid requires a north-up reference')
        bounds = []
        for path in frame.path:
            with rasterio.open(path) as src:
                if src.crs is None:
                    raise ValueError(f'Missing CRS: {path}')
                bounds.append(transform_bounds(src.crs, grid['crs'], *src.bounds))
        inverse = ~transform
        corners = [inverse * (x, y) for b in bounds for x, y in [(b[0], b[3]), (b[2], b[1])]]
        c0, r0 = [math.floor(min(p[k] for p in corners)+1e-6) for k in (0, 1)]
        c1, r1 = [math.ceil(max(p[k] for p in corners)-1e-6) for k in (0, 1)]
        transform = transform * Affine.translation(c0, r0)
        grid.update(transform=list(transform)[:6], width=c1-c0, height=r1-r0)
    features = cfg['extra_features']
    if any(f'{name}_path' not in frame for name in features):
        raise ValueError('Each optional feature requires a NAME_path manifest column')
    shape = (len(frame), grid['height'], grid['width'], 1 + len(features))
    import shutil
    required = int(np.prod(shape))*4
    if required > shutil.disk_usage(out).free:
        raise OSError(f'Cache needs {required / 2**30:.1f} GiB; insufficient disk space')
    cache = np.lib.format.open_memmap(out / 'values.npy', mode='w+', dtype='float32', shape=shape)
    report = []
    for i, obs in frame.iterrows():
        for channel, col in enumerate(['path'] + [f'{name}_path' for name in features]):
            with rasterio.open(obs[col]) as src:
                if src.crs is None or src.count != 1:
                    raise ValueError(f'Expected single-band georeferenced raster: {obs[col]}')
                if src.nodata == 0:
                    warnings.warn(f'{obs[col]} declares NoData=0; genuine zero cannot be recovered')
                aligned = (src.crs.to_string() == grid['crs'] and src.transform == transform
                           and (src.height, src.width) == (grid['height'], grid['width']))
                report.append(dict(path=obs[col], aligned=aligned, crs=src.crs.to_string(),
                                   transform=list(src.transform)[:6], resolution=list(src.res),
                                   nodata=str(src.nodata)))
                # Nearest preserves measured pixel values and vegetation boundaries.
                with WarpedVRT(src, crs=grid['crs'], transform=transform,
                               height=grid['height'], width=grid['width'],
                               dtype='float32', nodata=float('nan'),
                               resampling=Resampling.nearest) as vrt:
                    for win in windows(grid['height'], grid['width'], cfg['tile_size']):
                        arr = vrt.read(1, window=win, masked=True).filled(np.nan)
                        r, c = int(win.row_off), int(win.col_off)
                        arr = clean(arr, channel == 0)
                        if channel == 0:
                            arr[arr <= cfg['threshold']] = np.nan
                        cache[i, r:r+int(win.height), c:c+int(win.width), channel] = arr
        cache.flush()
    metadata = dict(grid=grid, observations=[dict(date=r.date.strftime('%Y-%m-%d'),
                    season=r.season, path=r.path) for r in frame.itertuples()],
                    extra_features=features, alignment_report=report,
                    reference=str(Path(ref).resolve()), config=cfg,
                    explicit_seasons=bool({'season', 'season_id'} & set(pd.read_csv(manifest_path).columns)),
                    season_status=frame.get('season_status', pd.Series(['unspecified']*len(frame))).tolist(),
                    estimated_bytes=required)
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    return metadata


def open_cache(root):
    root = Path(root)
    return json.loads((root / 'metadata.json').read_text()), np.load(root / 'values.npy', mmap_mode='r')
