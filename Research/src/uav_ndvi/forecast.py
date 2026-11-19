"""Export tiled forecast rasters, coverage reports and previews."""
from contextlib import ExitStack
from pathlib import Path
import json

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from affine import Affine
import torch

from .models.linear_regression import features
from .dataset import history_indices, eligible, make_item, collate
from .preprocessing import open_cache, windows
from .train import load_model
from .utils import device_batch

NODATA = -9999.


def preview(path, title, error=False):
    # A bounded thumbnail; full resolution remains in GeoTIFF.
    with rasterio.open(path) as src:
        scale = min(1., 1200/max(src.height, src.width))
        image = src.read(1, out_shape=(max(1, int(src.height*scale)), max(1, int(src.width*scale))), masked=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap('magma' if error else 'RdYlGn').copy()
    cmap.set_bad('#929292')
    im = ax.imshow(image, cmap=cmap, vmin=0., vmax=.5 if error else 1.)
    ax.set_title(title)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, label='NDVI error' if error else 'NDVI')
    fig.savefig(Path(path).with_suffix('.png'), bbox_inches='tight')
    plt.close(fig)


def forecast(cache, run, date, season, out, model_name='transformer', mask_path=None, classification=False):
    meta, values = open_cache(cache)
    model, saved = load_model(run)
    cfg = saved['config']
    if meta['grid'] != saved['metadata']['grid'] or meta['extra_features'] != cfg['extra_features']:
        raise ValueError('Inference cache must use training reference grid and feature order')
    ids = history_indices(meta, str(season), date, cfg)
    if len(ids) < cfg['min_history']:
        raise ValueError('Not enough acquisitions strictly before target within the selected season')
    if (np.datetime64(date)-np.datetime64(meta['observations'][ids[-1]]['date'])).astype(int) > cfg['max_horizon_days']:
        raise ValueError('Forecast exceeds configured horizon')
    truth_ids = [i for i, o in enumerate(meta['observations']) if o['date'] == date and o['season'] == str(season)]
    linear = joblib.load(Path(run) / 'linear.joblib') if model_name == 'linear' else None
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Forecast already exists; use a fresh output directory')
    out.mkdir(parents=True, exist_ok=True)
    grid = meta['grid']
    profile = dict(driver='GTiff', height=grid['height'], width=grid['width'], count=1,
        dtype='float32', crs=grid['crs'], transform=Affine(*grid['transform']), nodata=NODATA,
        compress='deflate', tiled=True, blockxsize=256, blockysize=256, BIGTIFF='IF_SAFER')
    names = ['predicted', 'latest_history', 'history_count', 'forecast_coverage', 'coverage_reason', 'horizon_days']
    if model_name == 'transformer':
        names += [f'q{q}' for q in cfg['quantiles'] if q != .5]
    if truth_ids:
        names += ['actual', 'error', 'absolute_error', 'target_validity']
    if classification:
        names += ['health']
    counts = 0
    reasons = {str(i): 0 for i in range(5)}
    valid_targets, covered_targets = 0, 0
    with ExitStack() as stack:
        external = stack.enter_context(rasterio.open(mask_path)) if mask_path else None
        if external and (external.crs.to_string() != grid['crs'] or external.transform != profile['transform']
                         or (external.height, external.width) != (grid['height'], grid['width'])):
            raise ValueError('Prediction mask must already match the reference grid exactly')
        outputs = {name: stack.enter_context(rasterio.open(out / f'{name}.tif', 'w', **profile)) for name in names}
        for dst in outputs.values():
            dst.update_tags(target_date=date, season=str(season), model=model_name,
                            prediction_mask='external AND history eligibility' if external else 'latest valid historical NDVI > threshold AND minimum history')
        if classification:
            outputs['health'].update_tags(classes='0=Unfavorable,1=Marginal,2=Healthy')
        outputs['coverage_reason'].update_tags(classes='0=no observed history,1=insufficient history,2=horizon exceeded,3=external mask excluded,4=forecastable')
        for win in windows(grid['height'], grid['width'], cfg['tile_size']):
            r, c, h, w = map(int, [win.row_off, win.col_off, win.height, win.width])
            raw = np.asarray(values[ids, r:r+h, c:c+w, :])
            good = eligible(raw, cfg)
            observed = np.isfinite(raw[..., 0])
            history_count = observed.sum(axis=0)
            latest = np.full((h, w), NODATA, dtype='float32')
            for j in range(len(ids)):
                latest[observed[j]] = raw[j, ..., 0][observed[j]]
            reason = np.where(history_count == 0, 0, 1).astype('float32')
            reason[good] = 4
            # Check horizon from the last valid observation of EACH pixel.
            latest_days = np.full((h, w), -np.inf)
            target_day = np.datetime64(date, 'D').astype(int)
            for j, i in enumerate(ids):
                observed_day = np.datetime64(meta['observations'][i]['date'], 'D').astype(int)
                latest_days[np.isfinite(raw[j, ..., 0])] = observed_day
            horizon = target_day-latest_days
            reason[good & (horizon > cfg['max_horizon_days'])] = 2
            good &= horizon <= cfg['max_horizon_days']
            if external:
                mask = external.read(1, window=win, masked=True)
                allowed = (~np.ma.getmaskarray(mask)) & np.isfinite(mask.data) & (mask.data > 0)
                reason[good & ~allowed] = 3
                good &= allowed
            rr, cc = np.nonzero(good)
            maps = {name: np.full((h, w), NODATA, dtype='float32') for name in names}
            maps.update(latest_history=latest, history_count=history_count.astype('float32'),
                        forecast_coverage=good.astype('float32'), coverage_reason=reason,
                        horizon_days=np.where(np.isfinite(horizon), horizon, NODATA).astype('float32'))
            for code in reasons:
                reasons[code] += int((reason == int(code)).sum())
            for start in range(0, len(rr), cfg['batch_size']):
                rows, cols = rr[start:start+cfg['batch_size']], cc[start:start+cfg['batch_size']]
                items = [make_item(meta, raw[:, a, b, :], ids, date) for a, b in zip(rows, cols)]
                if model_name == 'transformer':
                    with torch.no_grad():
                        pred = model(device_batch(collate(items), 'cpu')).numpy()
                    point = pred[:, cfg['quantiles'].index(.5)]
                    for j, q in enumerate(cfg['quantiles']):
                        if q != .5:
                            maps[f'q{q}'][rows, cols] = pred[:, j]
                elif model_name == 'linear':
                    point = linear.predict(np.stack([features(item) for item in items]))
                else:
                    point = latest[rows, cols]
                maps['predicted'][rows, cols] = point
                if classification:
                    maps['health'][rows, cols] = np.where(point >= .6, 2, np.where(point >= .4, 1, 0))
            counts += len(rr)
            if truth_ids:
                truth = values[truth_ids[0], r:r+h, c:c+w, 0]
                valid = np.isfinite(truth) & (truth > cfg['threshold'])
                maps['target_validity'] = valid.astype('float32')
                valid_targets += int(valid.sum())
                maps['actual'][valid] = truth[valid]
                valid &= good
                covered_targets += int(valid.sum())
                maps['error'][valid] = maps['predicted'][valid]-truth[valid]
                maps['absolute_error'][valid] = abs(maps['error'][valid])
            for name, dst in outputs.items():
                dst.write(maps[name], 1, window=win)
    for name in ['predicted', 'latest_history', 'forecast_coverage'] + (['actual', 'absolute_error', 'target_validity'] if truth_ids else []):
        preview(out / f'{name}.tif', f'{name.title()}: {date} (gray = NoData)', name == 'absolute_error')
    (out / 'coverage.json').write_text(json.dumps(dict(forecast_pixels=counts,
        valid_target_pixels=valid_targets, forecast_valid_target_pixels=covered_targets,
        forecast_percent=100*covered_targets/valid_targets if valid_targets else None,
        reason_counts=reasons, reasons={'0': 'no historical coverage', '1': 'insufficient usable history',
        '2': 'horizon exceeded', '3': 'external mask', '4': 'forecastable'},
        target_date=date, season_id=season, model=model_name,
        map_scope='All eligible pixels; experiment metrics use held-out samples only'), indent=2))
    if counts == 0:
        raise ValueError('No eligible prediction pixels; outputs contain only NoData')
    print(f'Exported {counts:,} pixel forecasts to {out.resolve()}')
    return counts
