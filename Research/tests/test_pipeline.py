import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
import torch

from uav_ndvi.config import load_config
from uav_ndvi.preprocess import preprocess, open_cache
from uav_ndvi.dataset import build_examples, PixelDataset, collate, history_indices, make_item
from uav_ndvi.model import AgriMatNetUAV, ElapsedDayEncoding
from uav_ndvi.training import train
from uav_ndvi.evaluation import evaluate
from uav_ndvi.inference import forecast


def synthetic(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(5)
    records = []
    for season in ['2024', '2025']:
        for j, suffix in enumerate(['03-01', '03-04', '04-13', '05-02', '05-21']):
            data = (.42+.035*j + rng.normal(0, .01, (24, 12))).astype('float32')
            data[:, :2] = -9999
            data[5, 4] = np.nan if j == 1 else data[5, 4]
            data[6, 4] = 0  # real zero, not NoData
            data[7, 4] = -9999
            path = root / f'{season}-{suffix}.tif'
            # One observation shifted by one pixel: must be warped back.
            transform = from_origin(500000 + (.016 if j == 1 else 0), 3760000, .016, .016)
            with rasterio.open(path, 'w', driver='GTiff', width=12, height=24,
                               count=1, dtype='float32', nodata=-9999,
                               crs='EPSG:32611', transform=transform) as dst:
                dst.write(data, 1)
            records.append(dict(date=f'{season}-{suffix}', season=season, path=path.name))
    manifest = root / 'manifest.csv'
    pd.DataFrame(records).to_csv(manifest, index=False)
    return manifest


@pytest.fixture
def prepared(tmp_path):
    cfg = load_config()
    cfg.update(d_model=8, num_heads=2, num_layers=1, dim_feedforward=16,
               epochs=2, patience=2, samples_per_split=30, batch_size=16,
               tile_size=8, dropout=0., grid_extent='reference')
    torch.set_num_threads(1)
    preprocess(synthetic(tmp_path / 'data'), tmp_path / 'cache', cfg)
    build_examples(tmp_path / 'cache', tmp_path / 'examples', cfg)
    return tmp_path, cfg


def test_alignment_missing_values(prepared):
    root, _ = prepared
    meta, values = open_cache(root / 'cache')
    assert not meta['alignment_report'][1]['aligned']
    assert values.shape == (10, 24, 12, 1)
    assert np.isnan(values[0, 0, 0, 0])
    assert np.isnan(values[0, 6, 4, 0])  # configured soil exclusion, never zero-filled NDVI
    assert np.isnan(values[0, 7, 4, 0])
    assert np.isnan(values[1, 5, 5, 0])  # shifted NaN
    assert values[0, 8, 4, 0] == pytest.approx(values[1, 8, 5, 0], abs=.08)


def test_sequences_and_no_future_leakage(prepared):
    root, cfg = prepared
    ds = PixelDataset(root / 'examples', 'train')
    assert len(set(len(ds[i]['days']) for i in range(len(ds)))) > 1
    for i in range(len(ds)):
        target, r, c = ds.index[i]
        obs = ds.meta['observations'][target]
        ids = history_indices(ds.meta, obs['season'], obs['date'], cfg)
        assert all(ds.meta['observations'][j]['date'] < obs['date'] for j in ids)
        assert all(ds.meta['observations'][j]['season'] == obs['season'] for j in ids)
        item = ds[i]
        assert (item['days'] < item['target_day']).all()
        raw = np.array(ds.values[ids, r, c, :])
        changed = make_item(ds.meta, raw, ids, obs['date'], target=-100.)
        assert torch.equal(item['history'], changed['history'])
        assert torch.equal(item['future'], changed['future'])
    # Acquisition spacing must preserve 3-day versus 40-day intervals.
    meta = ds.meta
    ids = history_indices(meta, '2024', '2024-05-02', cfg)
    raw = np.array(ds.values[ids, 10, 6, :])
    item = make_item(meta, raw, ids, '2024-05-02')
    assert item['days'].tolist() == [0., 3., 43.]
    raw[1, 0] = np.nan
    missing = make_item(meta, raw, ids, '2024-05-02')
    assert missing['missing'].tolist() == [False, True, False]
    assert missing['history'][1, 0] == 0


def test_spatial_isolation(prepared):
    root, _ = prepared
    sets = []
    for split in ['train', 'validation', 'test']:
        ds = PixelDataset(root / 'examples', split)
        sets.append(set(map(tuple, ds.index[:, 1:])))
    assert not sets[0] & sets[1]
    assert not sets[0] & sets[2]
    assert not sets[1] & sets[2]


def test_forward_masks_and_time(prepared):
    root, cfg = prepared
    ds = PixelDataset(root / 'examples', 'train')
    model = AgriMatNetUAV(cfg).eval()
    batch = collate([ds[0], ds[1], ds[2]])
    with torch.no_grad():
        result = model(batch)
    assert result.shape == (3, 3)
    assert torch.isfinite(result).all()
    pe = ElapsedDayEncoding(8, 0.)
    assert not torch.allclose(pe(torch.zeros(1, 2, 8), torch.tensor([[0., 3.]])),
                              pe(torch.zeros(1, 2, 8), torch.tensor([[0., 40.]])))
    # Ignored padding/missing token values must not change forecasts.
    ordered = sorted([ds[i] for i in range(len(ds))], key=lambda x: len(x['days']))
    masked = collate([ordered[0], ordered[-1]])
    with torch.no_grad():
        before = model(masked)
        masked['history'][masked['pad']] = 1000.
        after = model(masked)
    assert torch.allclose(before, after, atol=1e-6)


def test_dataset_future_mutation(prepared):
    root, _ = prepared
    ds = PixelDataset(root / 'examples', 'train')
    before = ds[0]
    target, r, c = ds.index[0]
    mutable = np.load(root / 'cache/values.npy', mmap_mode='r+')
    mutable[target:, r, c, :] = .999
    mutable.flush()
    after = ds[0]
    assert torch.equal(before['history'], after['history'])
    assert torch.equal(before['future'], after['future'])
    assert before['target'] != after['target']


def test_optional_features_and_external_mask(prepared):
    root, cfg = prepared
    frame = pd.read_csv(root / 'data/manifest.csv')
    frame['rededge_path'] = frame.path
    frame.to_csv(root / 'data/extra.csv', index=False)
    cfg = dict(cfg, extra_features=['rededge'])
    preprocess(root / 'data/extra.csv', root / 'extra_cache', cfg)
    build_examples(root / 'extra_cache', root / 'extra_examples', cfg)
    ds = PixelDataset(root / 'extra_examples', 'train')
    assert ds[0]['history'].shape[-1] == 6
    model = AgriMatNetUAV(cfg, n_features=2).eval()
    assert torch.isfinite(model(collate([ds[0], ds[1]]))).all()
    from uav_ndvi.baseline import matrix
    assert np.isfinite(matrix(ds)).all()


def test_end_to_end_and_export(prepared):
    root, cfg = prepared
    records = train(root / 'examples', root / 'run', cfg)
    assert len(records) == 2
    results = evaluate(root / 'run')
    assert len(results) == 3 and all(np.isfinite(r['rmse']) for r in results)
    for model in ['transformer', 'linear', 'persistence']:
        out = root / model
        count = forecast(root / 'cache', root / 'run', '2025-05-21', '2025', out, model, classification=True)
        assert count > 0
        with rasterio.open(out / 'predicted.tif') as dst, rasterio.open(root / 'data/2024-03-01.tif') as src:
            assert dst.transform == src.transform and dst.crs == src.crs
            assert dst.shape == src.shape
            arr = dst.read(1, masked=True)
            assert arr.mask[:, :2].all()
            assert np.isfinite(arr.compressed()).all()
        assert (out / 'error.tif').exists()
        assert (out / 'health.tif').exists()
    assert (root / 'run/evaluation_test/test_comparison.csv').exists()
    predictions = pd.read_csv(root / 'run/evaluation_test/test_predictions.csv')
    assert (predictions.horizon_days > 0).all()
    assert set(predictions.columns) >= {'persistence', 'linear', 'transformer', 'x', 'y'}
    # Future truth can affect error maps, never the requested prediction.
    mutable = np.load(root / 'cache/values.npy', mmap_mode='r+')
    mutable[9, :, :, :] = .99
    mutable.flush()
    forecast(root / 'cache', root / 'run', '2025-05-21', '2025', root / 'changed')
    with rasterio.open(root / 'changed/predicted.tif') as a, rasterio.open(root / 'transformer/predicted.tif') as b:
        np.testing.assert_array_equal(a.read(1), b.read(1))
    with rasterio.open(root / 'data/2024-03-01.tif') as src:
        profile = src.profile
    mask = np.zeros((24, 12), dtype='float32')
    mask[:, 6:] = 1
    with rasterio.open(root / 'mask.tif', 'w', **profile) as dst:
        dst.write(mask, 1)
    forecast(root / 'cache', root / 'run', '2025-05-21', '2025', root / 'masked', mask_path=root / 'mask.tif')
    with rasterio.open(root / 'masked/predicted.tif') as src:
        assert src.read(1, masked=True).mask[:, :6].all()
    # Future date with no observed truth still exports; season cannot borrow other years.
    forecast(root / 'cache', root / 'run', '2025-06-01', '2025', root / 'future')
    assert not (root / 'future/actual.tif').exists()
    with pytest.raises(ValueError, match='Not enough'):
        forecast(root / 'cache', root / 'run', '2025-03-02', '2025', root / 'insufficient')


if __name__ == '__main__':
    import sys
    print(synthetic(sys.argv[1]))
