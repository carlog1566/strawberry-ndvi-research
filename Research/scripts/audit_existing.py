"""Read-only audit of preserved research outputs; writes only a fresh report."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from uav_ndvi.dataset import history_indices, eligible
from uav_ndvi.preprocessing import open_cache, windows
from uav_ndvi.evaluate import metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='runs/strawberry')
    parser.add_argument('--forecast', default='predictions/june30/predicted.tif')
    parser.add_argument('--cache', default='cache/uav')
    parser.add_argument('--out', default='docs/existing_audit.json')
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists():
        raise FileExistsError('Choose a fresh report path')
    run = Path(args.run)
    p = pd.read_csv(run / 'test_predictions.csv')
    report = {'distribution': p[['actual', 'linear', 'transformer']].describe().to_dict(),
              'metrics': {name: metrics(p.actual, p[name]) for name in ['linear', 'transformer']},
              'confusion': {name: pd.read_csv(run / f'test_{name}_confusion.csv', index_col=0).to_dict()
                            for name in ['linear', 'transformer']}}
    meta, values = open_cache(args.cache)
    cfg = json.loads((run / 'run.json').read_text())['config']
    persistence = np.full(len(p), np.nan)
    for date, group in p.groupby('date'):
        candidates = [o for o in meta['observations'] if o['date'] == date]
        if len(candidates) != 1:
            raise ValueError('Ambiguous date in preserved results')
        ids = history_indices(meta, candidates[0]['season'], date, cfg)
        rr, cc = group.row.to_numpy(), group.column.to_numpy()
        latest = np.full(len(group), np.nan)
        for i in ids:
            raw = values[i, rr, cc, 0]
            latest[np.isfinite(raw)] = raw[np.isfinite(raw)]
        persistence[group.index.to_numpy()] = latest
    if not np.isfinite(persistence).all():
        raise ValueError('Missing persistence inputs for saved evaluation samples')
    report['metrics']['persistence'] = metrics(p.actual, persistence)
    counts = dict(total=0, predicted=0, nodata=0, no_history=0, insufficient_history=0,
                  horizon_exceeded=0, unexplained_missing=0)
    with rasterio.open(args.forecast) as src:
        tags = src.tags()
        ids = history_indices(meta, tags['season'], tags['target_date'], cfg)
        day = np.datetime64(tags['target_date'], 'D').astype(int)
        if src.shape != (meta['grid']['height'], meta['grid']['width']):
            raise ValueError('Forecast does not match cache grid')
        for win in windows(src.height, src.width, 256):
            r, c, h, w = map(int, [win.row_off, win.col_off, win.height, win.width])
            hist = np.asarray(values[ids, r:r+h, c:c+w, :])
            n = np.isfinite(hist[..., 0]).sum(0)
            good = eligible(hist, cfg)
            latest = np.full((h, w), -np.inf)
            for j, i in enumerate(ids):
                latest[np.isfinite(hist[j, ..., 0])] = np.datetime64(meta['observations'][i]['date'], 'D').astype(int)
            missing = np.ma.getmaskarray(src.read(1, window=win, masked=True))
            nohistory = missing & (n == 0)
            insufficient = missing & (n > 0) & ~good
            horizon = missing & good & (day-latest > cfg['max_horizon_days'])
            counts['total'] += h*w
            counts['predicted'] += int((~missing).sum())
            counts['nodata'] += int(missing.sum())
            counts['no_history'] += int(nohistory.sum())
            counts['insufficient_history'] += int(insufficient.sum())
            counts['horizon_exceeded'] += int(horizon.sum())
            counts['unexplained_missing'] += int((missing & ~(nohistory | insufficient | horizon)).sum())
        report['forecast'] = dict(tags=tags, counts=counts, resolution=src.res,
                                  nodata=src.nodata, shape=src.shape)
    report['limits'] = 'Audit of old cache, which clips to its reference footprint and uses unverified calendar-year seasons.'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report['forecast'], indent=2))


if __name__ == '__main__':
    main()
