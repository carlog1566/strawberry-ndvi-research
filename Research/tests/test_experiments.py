import json

import numpy as np
import pandas as pd
import pytest
import torch
import rasterio

from tests.test_pipeline import synthetic, prepared
from uav_ndvi.config import load_config
from uav_ndvi.preprocess import preprocess, open_cache, manifest
from uav_ndvi.dataset import build_examples, PixelDataset, history_indices
from uav_ndvi.training import train, load_model
from uav_ndvi.evaluation import evaluate
from uav_ndvi.inference import forecast


def test_union_retains_partial_edges(tmp_path):
    cfg = load_config()
    preprocess(synthetic(tmp_path / 'data'), tmp_path / 'cache', cfg)
    meta, values = open_cache(tmp_path / 'cache')
    assert meta['grid']['width'] == 13
    assert np.isnan(values[0, :, 12, 0]).all()
    assert np.isfinite(values[1, :, 12, 0]).all()
    assert np.isnan(values[1, :, 0, 0]).all()


def test_explicit_season_id(tmp_path):
    path = synthetic(tmp_path / 'data')
    frame = pd.read_csv(path)
    frame = frame.rename(columns={'season': 'season_id'})
    frame.to_csv(path, index=False)
    assert manifest(path, require_seasons=True).season.iloc[0] == '2024'
    frame.drop(columns='season_id').to_csv(path, index=False)
    with pytest.raises(ValueError, match='Explicit season'):
        manifest(path, require_seasons=True)


def test_cross_year_frozen_rolling_origin(prepared):
    root, cfg = prepared
    cfg = dict(cfg, experiment='cross_year_2025', validation_dates=['2024-05-21'])
    build_examples(root / 'cache', root / 'cross_examples', cfg)
    for split, year in [('train', '2024'), ('validation', '2024'), ('test', '2025')]:
        ds = PixelDataset(root / 'cross_examples', split)
        for i, _, _ in ds.index:
            obs = ds.meta['observations'][i]
            assert obs['date'].startswith(year)
            if split == 'validation':
                assert obs['date'] == '2024-05-21'
            if split == 'train':
                assert obs['date'] != '2024-05-21'
            ids = history_indices(ds.meta, obs['season'], obs['date'], ds.cfg)
            assert all(ds.meta['observations'][j]['date'] < obs['date'] for j in ids)
            assert all(ds.meta['observations'][j]['date'] != '2024-05-21' for j in ids)
    train(root / 'cross_examples', root / 'cross_run', cfg)
    before = (root / 'cross_run/transformer.pt').read_bytes()
    evaluate(root / 'cross_run')
    for date in ['2025-04-13', '2025-05-21']:
        forecast(root / 'cache', root / 'cross_run', date, '2025', root / date, 'persistence')
    assert before == (root / 'cross_run/transformer.pt').read_bytes()
    with rasterio.open(root / '2025-05-21/forecast_coverage.tif') as src:
        coverage = src.read(1)
    with rasterio.open(root / '2025-05-21/coverage_reason.tif') as src:
        assert np.array_equal(coverage == 1, src.read(1) == 4)
    with rasterio.open(root / '2025-05-21/absolute_error.tif') as src:
        assert (src.read(1, masked=True).compressed() >= 0).all()


def test_heldout_validation_never_becomes_training_history(prepared):
    root, cfg = prepared
    cfg = dict(cfg, experiment='cross_year_2025', validation_dates=['2024-05-02'])
    build_examples(root / 'cache', root / 'cross', cfg)
    ds = PixelDataset(root / 'cross', 'train')
    before = [ds[i]['history'].clone() for i in range(len(ds))]
    values = np.load(root / 'cache/values.npy', mmap_mode='r+')
    values[3] = .999
    values[5:] = .888  # every 2025 observation
    values.flush()
    assert all(torch.equal(item, ds[i]['history']) for i, item in enumerate(before))


def test_no_crossing_unverified_boundaries():
    cfg = load_config()
    meta = {'observations': [dict(date='2024-12-06', season='provisional_late24'),
                             dict(date='2025-02-27', season='provisional_25')]}
    assert history_indices(meta, 'provisional_25', '2025-04-01', cfg) == [1]
    # If confirmed as one crop cycle, an excluded date still cannot enter history.
    meta['observations'][0]['season'] = 'provisional_25'
    cfg.update(excluded_dates=['2024-12-06'], experiment='cross_year_2025')
    assert history_indices(meta, 'provisional_25', '2025-04-01', cfg) == [1]


def test_chronological_validation_and_invalid_training_years(tmp_path):
    from uav_ndvi.dataset import target_partition
    cfg = dict(load_config(), experiment='cross_year_2025', validation_dates=['2024-06-27'])
    assert target_partition({'date': '2024-05-30'}, cfg) == 0
    assert target_partition({'date': '2024-06-27'}, cfg) == 1
    assert target_partition({'date': '2024-08-15'}, cfg) == -1
    assert target_partition({'date': '2025-04-01'}, cfg) == 2
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps(dict(experiment='cross_year_2025', train_years=[2026])))
    with pytest.raises(ValueError, match='2022-2024'):
        load_config(path)


def test_old_checkpoint_still_loads():
    from pathlib import Path
    run = Path('runs/strawberry')
    if not (run / 'transformer.pt').exists():
        pytest.skip('Existing research checkpoint unavailable')
    model, saved = load_model(run)
    assert not model.residual_prediction


def test_model_import_and_pickle_compatibility(monkeypatch):
    import pickle
    from uav_ndvi import model as legacy_transformer, baseline as legacy_linear
    from uav_ndvi.models import transformer, linear_regression

    for name in ['AgriMatNetUAV', 'ElapsedDayEncoding', 'quantile_loss']:
        assert getattr(legacy_transformer, name) is getattr(transformer, name)
    for name in ['features', 'matrix', 'fit']:
        assert getattr(legacy_linear, name) is getattr(linear_regression, name)

    # Simulate serialized globals written with the former implementation paths.
    # These references must still resolve to the current implementations.
    symbols = [transformer.AgriMatNetUAV, transformer.ElapsedDayEncoding,
               transformer.quantile_loss, linear_regression.features,
               linear_regression.matrix, linear_regression.fit]
    with monkeypatch.context() as old_paths:
        for symbol in symbols:
            module = 'uav_ndvi.model' if symbol in symbols[:3] else 'uav_ndvi.baseline'
            old_paths.setattr(symbol, '__module__', module)
        serialized = pickle.dumps(symbols)
    restored = pickle.loads(serialized)
    assert all(old is new for old, new in zip(restored, symbols))


def test_pipeline_import_and_pickle_compatibility(monkeypatch):
    import importlib
    import pickle

    modules = [
        ('training', 'train', ['train', 'load_model', 'loader', 'seed_all', 'device_batch']),
        ('evaluation', 'evaluate', ['evaluate', 'metrics', 'health', 'plot_history']),
        ('inference', 'forecast', ['forecast', 'preview', 'NODATA']),
        ('preprocess', 'preprocessing', ['preprocess', 'manifest', 'open_cache', 'windows', 'clean']),
    ]
    symbols = []
    with monkeypatch.context() as old_paths:
        for old_name, new_name, names in modules:
            old = importlib.import_module(f'uav_ndvi.{old_name}')
            new = importlib.import_module(f'uav_ndvi.{new_name}')
            for name in names:
                symbol = getattr(new, name)
                assert getattr(old, name) is symbol
                if callable(symbol):
                    old_paths.setattr(symbol, '__module__', old.__name__)
                    symbols.append(symbol)
        serialized = pickle.dumps(symbols)
    assert all(old is new for old, new in zip(pickle.loads(serialized), symbols))

    from uav_ndvi import utils, train
    assert utils.seed_all is train.seed_all
    assert utils.device_batch is train.device_batch


def test_legacy_launcher_train_and_predict(tmp_path):
    import importlib.util
    from pathlib import Path
    import subprocess
    import sys
    from uav_ndvi import legacy_linear_regression as legacy

    project = Path(__file__).resolve().parents[1]
    launcher = project / 'models/LinearReg_Baseline.py'
    spec = importlib.util.spec_from_file_location('legacy_launcher', launcher)
    compatibility = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compatibility)
    assert compatibility.train is legacy.train
    assert compatibility.predict is legacy.predict
    assert compatibility.FEATURE_NAMES is legacy.FEATURE_NAMES

    path = synthetic(tmp_path / 'inputs')
    frame = pd.read_csv(path).rename(columns={'path': 'with_soil'})
    frame.to_csv(path, index=False)
    model = tmp_path / 'model.joblib'
    # Execute the historical script from outside the repository, without installation.
    trained = subprocess.run(
        [sys.executable, '-B', str(launcher), 'train', '--manifest', str(path),
         '--resolution', '0.016', '--max-pixels', '1000',
         '--samples-per-target-region', '10', '--model-out', str(model)],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert trained.returncode == 0, trained.stdout + trained.stderr
    assert model.is_file()
    outputs = []
    for name, entrypoint, workdir in [
        ('launcher', [str(launcher)], tmp_path),
        ('module', ['-m', 'uav_ndvi.legacy_linear_regression'], project),
    ]:
        output = tmp_path / f'{name}.tif'
        predicted = subprocess.run(
            [sys.executable, '-B', *entrypoint, 'predict', '--manifest', str(path),
             '--model', str(model), '--date', '2025-05-22', '--out', str(output)],
            cwd=workdir, capture_output=True, text=True,
        )
        assert predicted.returncode == 0, predicted.stdout + predicted.stderr
        with rasterio.open(output) as raster:
            values = raster.read(1, masked=True)
            assert values.count() > 0
            assert np.isfinite(values.compressed()).all()
            outputs.append((raster.profile, values))
    assert outputs[0][0] == outputs[1][0]
    assert np.array_equal(outputs[0][1].mask, outputs[1][1].mask)
    assert np.array_equal(outputs[0][1].data, outputs[1][1].data)
