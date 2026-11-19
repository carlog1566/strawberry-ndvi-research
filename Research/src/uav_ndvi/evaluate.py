"""Evaluate saved models on shared samples and export diagnostic reports."""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import classification_report

from .models.linear_regression import matrix
from .dataset import PixelDataset
from .train import load_model, loader
from .utils import device_batch


def health(values):
    return np.where(values >= .6, 2, np.where(values >= .4, 1, 0))


def metrics(y, p):
    return dict(n=len(y), rmse=float(np.sqrt(mean_squared_error(y, p))),
        mae=float(mean_absolute_error(y, p)), r2=float(r2_score(y, p)) if len(y) > 1 and np.var(y) > 0 else None,
        macro_f1=float(f1_score(health(y), health(p), labels=[0, 1, 2], average='macro', zero_division=0)),
        accuracy=float(accuracy_score(health(y), health(p))),
        actual_mean=float(np.mean(y)), predicted_mean=float(np.mean(p)),
        actual_std=float(np.std(y)), predicted_std=float(np.std(p)),
        bias=float(np.mean(p-y)), outside_ndvi_range=float(np.mean((p < -1) | (p > 1))))


def plot_history(run):
    run = Path(run)
    history = pd.read_csv(run / 'history.csv')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name in ['train', 'validation']:
        axes[0].plot(history.epoch, history[f'{name}_loss'], label=name)
        axes[1].plot(history.epoch, history[f'{name}_rmse'], label=name)
    for ax, title in zip(axes, ['Training objective', 'Point RMSE']):
        ax.set(xlabel='Epoch', ylabel=title)
        ax.legend()
    fig.tight_layout()
    fig.savefig(run / 'training.png')
    plt.close(fig)


def evaluate(run, split='test', out=None):
    run = Path(run)
    info = json.loads((run / 'run.json').read_text())
    dataset = PixelDataset(info['examples'], split)
    output = Path(out) if out else run / f'evaluation_{split}'
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Evaluation exists; select a fresh --out directory')
    output.mkdir(parents=True, exist_ok=True)
    model, saved = load_model(run)
    if saved['metadata'] != dataset.meta:
        raise ValueError('Evaluation cache differs from training cache')
    cfg = saved['config']
    with torch.no_grad():
        quantiles = np.concatenate([model(device_batch(b, 'cpu')).numpy() for b in loader(dataset, cfg)])
    median = quantiles[:, cfg['quantiles'].index(.5)]
    linear = joblib.load(run / 'linear.joblib').predict(matrix(dataset))
    y = np.array([float(dataset[i]['target']) for i in range(len(dataset))])
    dates = np.array([dataset.meta['observations'][i]['date'] for i in dataset.index[:, 0]])
    persistence, horizons = [], []
    for i in range(len(dataset)):
        item = dataset[i]
        valid = ~item['missing'].numpy()
        persistence.append(float(item['history'][valid, 0][-1]))
        horizons.append(int(float(item['target_day']) - float(item['days'][valid][-1])))
    persistence, horizons = np.array(persistence), np.array(horizons)
    records, per_date, per_horizon = [], [], []
    for name, pred in [('persistence', persistence), ('linear', linear), ('transformer', median)]:
        records.append(dict(model=name, **metrics(y, pred)))
        for date in sorted(set(dates)):
            select = dates == date
            per_date.append(dict(model=name, date=date, **metrics(y[select], pred[select])))
        for horizon in sorted(set(horizons)):
            select = horizons == horizon
            per_horizon.append(dict(model=name, horizon_days=int(horizon), **metrics(y[select], pred[select])))
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.hexbin(y, pred, gridsize=40, mincnt=1)
        ax.plot([-1, 1], [-1, 1], '--', color='red')
        ax.set(xlabel='Actual NDVI', ylabel='Predicted NDVI', title=f'{name}: {split}')
        fig.savefig(output / f'{split}_{name}_scatter.png', bbox_inches='tight')
        plt.close(fig)
        pd.DataFrame(confusion_matrix(health(y), health(pred), labels=[0, 1, 2]),
                     index=['unfavorable', 'marginal', 'healthy'],
                     columns=['unfavorable', 'marginal', 'healthy']).to_csv(output / f'{split}_{name}_confusion.csv')
        report = classification_report(health(y), health(pred), labels=[0, 1, 2],
            target_names=['unfavorable', 'marginal', 'healthy'], output_dict=True, zero_division=0)
        pd.DataFrame(report).T.to_csv(output / f'{split}_{name}_classes.csv')
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        bins = np.linspace(-1, 1, 81)
        axes[0].hist(y, bins=bins, alpha=.5, label='actual')
        axes[0].hist(pred, bins=bins, alpha=.5, label='predicted')
        axes[0].set(xlabel='NDVI', ylabel='Sample count')
        axes[0].legend()
        cm = confusion_matrix(health(y), health(pred), labels=[0, 1, 2])
        axes[1].imshow(cm, cmap='Blues')
        for a in range(3):
            for b in range(3):
                axes[1].text(b, a, str(cm[a, b]), ha='center', va='center')
        axes[1].set(xticks=range(3), yticks=range(3),
            xticklabels=['unfavorable', 'marginal', 'healthy'],
            yticklabels=['unfavorable', 'marginal', 'healthy'], xlabel='Predicted', ylabel='Actual')
        fig.tight_layout()
        fig.savefig(output / f'{split}_{name}_diagnostics.png')
        plt.close(fig)
    lower, upper = quantiles[:, 0], quantiles[:, -1]
    interval = dict(lower_quantile=cfg['quantiles'][0], upper_quantile=cfg['quantiles'][-1],
        nominal_coverage=cfg['quantiles'][-1]-cfg['quantiles'][0],
        coverage=float(np.mean((y >= lower) & (y <= upper))),
        mean_width=float(np.mean(upper-lower)),
        crossing_rate=float(np.mean(np.any(np.diff(quantiles, axis=1) < 0, axis=1))))
    # Raw quantiles are retained: crossings are reported, never silently sorted.
    (output / f'{split}_interval.json').write_text(json.dumps(interval, indent=2))
    pd.DataFrame(records).to_csv(output / f'{split}_comparison.csv', index=False)
    pd.DataFrame(per_date).to_csv(output / f'{split}_per_date.csv', index=False)
    pd.DataFrame(per_horizon).to_csv(output / f'{split}_per_horizon.csv', index=False)
    coverage = pd.DataFrame(dataset.info.get('coverage', []))
    if not coverage.empty:
        coverage = coverage[coverage.split == split].copy()
        coverage['forecast_percent'] = 100*coverage.forecastable_pixels / coverage.valid_target_pixels.replace(0, np.nan)
        coverage['evaluated_samples'] = coverage.date.map(pd.Series(dates).value_counts()).fillna(0).astype(int)
        coverage.to_csv(output / f'{split}_coverage.csv', index=False)
        denominator = int(coverage.valid_target_pixels.sum())
        (output / f'{split}_coverage.json').write_text(json.dumps(dict(
            valid_target_pixels=denominator, forecastable_pixels=int(coverage.forecastable_pixels.sum()),
            forecast_percent=100*float(coverage.forecastable_pixels.sum())/denominator if denominator else None,
            evaluated_samples=len(dataset), sampling='uniform pixel-date reservoir; coverage measured before sampling'), indent=2))
    for table, x, suffix in [(per_date, 'date', 'date'), (per_horizon, 'horizon_days', 'horizon')]:
        fig, ax = plt.subplots(figsize=(10, 4))
        for name, group in pd.DataFrame(table).groupby('model'):
            ax.plot(group[x], group.rmse, marker='.', label=name)
        ax.set(xlabel=x, ylabel='RMSE')
        ax.legend()
        if suffix == 'date':
            ax.tick_params(axis='x', rotation=90)
        fig.tight_layout()
        fig.savefig(output / f'{split}_rmse_by_{suffix}.png')
        plt.close(fig)
    predictions = pd.DataFrame(dict(date=dates, row=dataset.index[:, 1], column=dataset.index[:, 2],
                                   horizon_days=horizons, actual=y, persistence=persistence, linear=linear, transformer=median))
    from affine import Affine
    transform = Affine(*dataset.meta['grid']['transform'])
    xy = [transform*(int(c)+.5, int(r)+.5) for _, r, c in dataset.index]
    predictions['x'], predictions['y'] = np.array(xy).T
    for j, q in enumerate(cfg['quantiles']):
        predictions[f'q{q}'] = quantiles[:, j]
    predictions.to_csv(output / f'{split}_predictions.csv', index=False)
    (output / 'provenance.json').write_text(json.dumps(dict(run=str(run.resolve()),
        checkpoint_epoch=saved['epoch'], config=saved['config'], examples=info['examples'],
        season_status=dataset.meta.get('season_status'),
        interpretation='Unseen calendar year; unseen crop cycle requires verified season labels'
            if cfg.get('experiment') == 'cross_year_2025' else 'Geographic holdout on a fixed common grid'), indent=2))
    print(pd.DataFrame(records).to_string(index=False))
    return records
