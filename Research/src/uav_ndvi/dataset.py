import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocessing import open_cache, windows


def history_indices(meta, season, target_date, cfg):
    target = np.datetime64(target_date, 'D')
    # Validation measurements are excluded from ALL train/validation histories.
    # 2025 rolling origins can use earlier observed test measurements, frozen model.
    heldout = set(cfg.get('excluded_dates', []))
    if cfg.get('experiment') == 'cross_year_2025':
        heldout |= set(cfg.get('validation_dates', []))
    ids = [i for i, o in enumerate(meta['observations']) if o['season'] == season
           and o['date'] not in heldout
           and 0 < (target - np.datetime64(o['date'], 'D')).astype(int) <= cfg['max_history_days']]
    return sorted(ids, key=lambda i: meta['observations'][i]['date'])[-cfg['max_history']:]


def target_partition(obs, cfg):
    if obs['date'] in cfg.get('excluded_dates', []):
        return -1
    year = int(obs['date'][:4])
    if year == cfg['test_year']:
        return 2
    if obs['date'] in cfg['validation_dates']:
        return 1
    if year in cfg['train_years']:
        # Even explicitly chosen noncontiguous validation dates cannot make
        # later 2024 targets part of fitting an earlier validation forecast.
        if year == 2024 and cfg['validation_dates'] and obs['date'] >= min(cfg['validation_dates']):
            return -1
        return 0
    return -1


def eligible(history, cfg):
    finite = np.isfinite(history[..., 0])
    count = finite.sum(axis=0)
    latest = np.full(count.shape, np.nan, dtype='float32')
    for i in range(len(history)):
        latest = np.where(finite[i], history[i, ..., 0], latest)
    return (count >= cfg['min_history']) & (latest > cfg['threshold'])


def spatial_partition(rows, cols, grid, cfg):
    coord, length = (rows, grid['height']) if cfg['split_axis'] == 'row' else (cols, grid['width'])
    a, b = int(length * .7), int(length * .85)
    if a < 1 or b <= a or b >= length:
        raise ValueError('Grid too small for spatial partitions')
    result = np.where(coord < a, 0, np.where(coord < b, 1, 2))
    gap = cfg['buffer_pixels']
    if gap:
        result = np.where((abs(coord-a) < gap) | (abs(coord-b) < gap), -1, result)
    return result


def build_examples(cache, out, cfg):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Example directory is not empty; choose a fresh directory')
    meta, values = open_cache(cache)
    cfg = dict(cfg)
    if cfg.get('require_seasons') and not meta.get('explicit_seasons'):
        raise ValueError('Rebuild cache with explicit growing-season labels')
    if cfg.get('experiment') == 'cross_year_2025':
        # Deterministic chronological tail, selected from dates only, never NDVI.
        if not cfg['validation_dates']:
            dates = sorted(o['date'] for o in meta['observations'] if o['date'].startswith('2024-') and o['date'] not in cfg.get('excluded_dates', []))
            if not dates:
                raise ValueError('Cross-year experiment requires 2024 validation dates')
            cfg['validation_dates'] = dates[-max(1, int(np.ceil(len(dates)*.15))):]
        available = {o['date'] for o in meta['observations']}
        if not set(cfg['validation_dates']) <= available:
            raise ValueError('Validation dates absent from cache')
    if meta['extra_features'] != cfg['extra_features']:
        raise ValueError('Configured feature order differs from preprocessed cache')
    rng = np.random.default_rng(cfg['seed'])
    # Uniform random-priority reservoir across ALL eligible pixel/date examples.
    samples = [np.empty((0, 3), dtype='int64') for _ in range(3)]
    priorities = [np.empty(0) for _ in range(3)]
    total = [0, 0, 0]
    coverage = []
    for target, obs in enumerate(meta['observations']):
        ids = history_indices(meta, obs['season'], obs['date'], cfg)
        valid_counts, eligible_counts = [0, 0, 0], [0, 0, 0]
        for win in windows(meta['grid']['height'], meta['grid']['width'], cfg['tile_size']):
            r, c, h, w = map(int, [win.row_off, win.col_off, win.height, win.width])
            hist = np.asarray(values[ids, r:r+h, c:c+w, :])
            good = eligible(hist, cfg)
            latest_day = np.full((h, w), -np.inf)
            for j, i in enumerate(ids):
                day = np.datetime64(meta['observations'][i]['date'], 'D').astype(int)
                latest_day[np.isfinite(hist[j, ..., 0])] = day
            target_day = np.datetime64(obs['date'], 'D').astype(int)
            good &= target_day-latest_day <= cfg['max_horizon_days']
            truth = values[target, r:r+h, c:c+w, 0]
            valid_truth = np.isfinite(truth) & (truth > cfg['threshold'])
            good &= valid_truth
            vr, vc = np.nonzero(valid_truth)
            vp = spatial_partition(vr+r, vc+c, meta['grid'], cfg) if cfg.get('experiment') != 'cross_year_2025' else np.full(len(vr), target_partition(obs, cfg))
            for split in range(3):
                valid_counts[split] += int((vp == split).sum())
            rr, cc = np.nonzero(good)
            rr, cc = rr+r, cc+c
            partitions = spatial_partition(rr, cc, meta['grid'], cfg) if cfg.get('experiment') != 'cross_year_2025' else np.full(len(rr), target_partition(obs, cfg))
            for split in range(3):
                selected = partitions == split
                n = int(selected.sum())
                total[split] += n
                eligible_counts[split] += n
                new = np.column_stack([np.full(n, target), rr[selected], cc[selected]])
                samples[split] = np.concatenate([samples[split], new])
                priorities[split] = np.concatenate([priorities[split], rng.random(n)])
                cap = cfg['samples_per_split']
                if len(samples[split]) > cap:
                    keep = np.argpartition(priorities[split], cap-1)[:cap]
                    samples[split], priorities[split] = samples[split][keep], priorities[split][keep]
        for split, name in enumerate(['train', 'validation', 'test']):
            coverage.append(dict(date=obs['date'], split=name, valid_target_pixels=valid_counts[split],
                                 forecastable_pixels=eligible_counts[split]))
    out = Path(out)
    if (out / 'examples.json').exists():
        raise FileExistsError('Example set already exists; use a fresh directory')
    out.mkdir(parents=True, exist_ok=True)
    for name, sample in zip(['train', 'validation', 'test'], samples):
        if len(sample) == 0:
            raise ValueError(f'No eligible {name} samples; check footprint, seasons, history, and split')
    for name, sample in zip(['train', 'validation', 'test'], samples):
        np.save(out / f'{name}.npy', sample)
    (out / 'examples.json').write_text(json.dumps(dict(cache=str(Path(cache).resolve()), config=cfg,
        eligible_counts=total, sampled_counts=[len(s) for s in samples], coverage=coverage, metadata=meta), indent=2))


def make_item(meta, raw, ids, target_date, target=np.nan):
    """No target value enters history or future features; zero is only a masked placeholder."""
    dates = np.array([meta['observations'][i]['date'] for i in ids], dtype='datetime64[D]')
    days = (dates-dates[0]).astype('float32')
    target_day = float((np.datetime64(target_date, 'D')-dates[0]).astype(int))
    doy = (dates - dates.astype('datetime64[Y]')).astype('float32')
    t = np.datetime64(target_date, 'D')
    target_doy = float((t-t.astype('datetime64[Y]')).astype(int))
    temporal = np.column_stack([np.sin(doy*2*np.pi/365.25), np.cos(doy*2*np.pi/365.25)])
    missing = ~np.isfinite(raw)
    history = np.column_stack([np.nan_to_num(raw, nan=0.), missing.astype('float32'), temporal]).astype('float32')
    future = np.array([target_day/365.25, np.sin(target_doy*2*np.pi/365.25),
                       np.cos(target_doy*2*np.pi/365.25)], dtype='float32')
    return dict(history=torch.from_numpy(history), days=torch.from_numpy(days),
                missing=torch.from_numpy(missing[:, 0]), future=torch.from_numpy(future),
                target_day=torch.tensor(target_day, dtype=torch.float32),
                target=torch.tensor(target, dtype=torch.float32))


class PixelDataset(Dataset):
    def __init__(self, examples, split):
        self.root = Path(examples)
        self.info = json.loads((self.root / 'examples.json').read_text())
        self.meta = self.info['metadata']
        self.cfg = self.info['config']
        self.index = np.load(self.root / f'{split}.npy')
        self.values = None

    def __len__(self):
        return len(self.index)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['values'] = None
        return state

    def __getitem__(self, i):
        if self.values is None:
            _, self.values = open_cache(self.info['cache'])
        target, r, c = self.index[i]
        obs = self.meta['observations'][target]
        ids = history_indices(self.meta, obs['season'], obs['date'], self.cfg)
        raw = np.array(self.values[ids, r, c, :], copy=True)
        item = make_item(self.meta, raw, ids, obs['date'], self.values[target, r, c, 0])
        item['index'] = torch.tensor(i)
        return item


def collate(items):
    n = max(len(i['days']) for i in items)
    batch = {}
    for key in ['history', 'days', 'missing']:
        arrays = []
        for item in items:
            a = item[key]
            shape = (n-len(a),) + a.shape[1:]
            arrays.append(torch.cat([a, torch.full(shape, key == 'missing', dtype=a.dtype)]))
        batch[key] = torch.stack(arrays)
    batch['pad'] = torch.tensor([[j >= len(i['days']) for j in range(n)] for i in items])
    for key in ['future', 'target_day', 'target']:
        batch[key] = torch.stack([i[key] for i in items])
    return batch
