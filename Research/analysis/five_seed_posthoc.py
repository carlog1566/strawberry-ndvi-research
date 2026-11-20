"""Read-only post-hoc ranking metrics and historical attention for saved NDVI runs.

Run from Research:
python -B -m analysis.five_seed_posthoc --out analysis_results/five_seed_posthoc_2026-11-19

No training, source edits, output replacement or checkpoint writes are performed.
"""
import argparse
import copy
import hashlib
import html
import json
import os
from pathlib import Path
import platform
import time

# Keep even plotting's generated font/config cache inside the new analysis area.
os.environ['MPLCONFIGDIR'] = str(Path(__file__).resolve().parent / '.matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)
import torch


MODELS = ('linear', 'transformer')
CUTOFFS = (0.4, 0.6)
KEYS = ['date', 'row', 'column']
ATOL = 2e-6
RTOL = 1e-5


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def protected_inventory(root):
    """Hash small/critical inputs; also record every protected file's size and mtime."""
    root = Path(root)
    inventory = {}
    for path in root.rglob('*'):
        relative = path.relative_to(root)
        if not path.is_file() or '.git' in relative.parts:
            continue
        if relative.parts[0] in ('analysis', 'analysis_results'):
            continue
        stat = path.stat()
        entry = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        if stat.st_size <= 2_000_000 or path.suffix in ('.pt', '.joblib'):
            entry['sha256'] = digest(path)
        inventory[relative.as_posix()] = entry
    return inventory


def assert_inventory_unchanged(before, after):
    changed = sorted(path for path in before.keys() | after.keys()
                     if before.get(path) != after.get(path))
    if changed:
        raise RuntimeError('Protected files changed during analysis: ' + ', '.join(changed[:20]))


def ranking_metrics(actual, score, cutoff):
    """Cut off truth only: predicted NDVI stays continuous and is never clipped."""
    actual, score = np.asarray(actual, dtype=float), np.asarray(score, dtype=float)
    if actual.ndim != 1 or score.shape != actual.shape:
        raise ValueError('Truth and scores must be equal-length vectors')
    if not len(actual) or not np.isfinite(actual).all() or not np.isfinite(score).all():
        return dict(status='unavailable: empty or nonfinite inputs', n=len(actual),
                    positives=None, negatives=None, prevalence=None, roc_auc=None,
                    average_precision=None), None
    target = (actual >= cutoff).astype(int)
    positives = int(target.sum())
    result = dict(status='available', n=len(actual), positives=positives,
                  negatives=int(len(actual)-positives), prevalence=float(target.mean()),
                  roc_auc=None, average_precision=None)
    if positives in (0, len(actual)):
        result['status'] = 'unavailable: both target classes are required'
        return result, None
    result['roc_auc'] = float(roc_auc_score(target, score))
    result['average_precision'] = float(average_precision_score(target, score))
    fpr, tpr, roc_threshold = roc_curve(target, score, drop_intermediate=False)
    precision, recall, pr_threshold = precision_recall_curve(target, score)
    curves = dict(fpr=fpr, tpr=tpr, roc_threshold=roc_threshold,
                  precision=precision, recall=recall, pr_threshold=pr_threshold)
    return result, curves


def state_digest(model):
    result = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        result.update(name.encode())
        result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


class HistoricalAttentionObserver:
    """Observe unmodified Q/K/V; recompute per-head weights in separate eval copies.

    Original need_weights=False calls, parameters and forward methods are unchanged.
    Hooks can disable a fused encoder fast path; callers must check output agreement.
    """
    def __init__(self, model):
        self.model = model
        self.modules = [layer.self_attn for layer in model.history_encoder.layers]
        self.observers = [copy.deepcopy(module).eval().requires_grad_(False)
                          for module in self.modules]
        self.handles = []
        self.weights = {}
        self.expected_outputs = {}
        self.max_attention_output_error = 0.0

    def _before(self, index, args, kwargs):
        if len(args) < 3:
            raise RuntimeError('Historical self-attention Q/K/V inputs were unavailable')
        query = args[0].detach()
        key = query if args[1] is args[0] else args[1].detach()
        value = key if args[2] is args[1] else args[2].detach()
        with torch.no_grad():
            output, weights = self.observers[index](
                query, key, value, key_padding_mask=kwargs.get('key_padding_mask'),
                attn_mask=kwargs.get('attn_mask'), is_causal=kwargs.get('is_causal', False),
                need_weights=True, average_attn_weights=False)
        if weights is None or weights.ndim != 4:
            raise RuntimeError('Per-head historical attention was not returned')
        self.weights[index] = weights.detach().cpu().numpy()
        self.expected_outputs[index] = output.detach()

    def _after(self, index, output):
        actual = output[0].detach()
        expected = self.expected_outputs[index]
        self.max_attention_output_error = max(
            self.max_attention_output_error, float((actual-expected).abs().max()))
        torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)

    def __enter__(self):
        for index, module in enumerate(self.modules):
            self.handles.append(module.register_forward_pre_hook(
                lambda mod, args, kwargs, i=index: self._before(i, args, kwargs),
                with_kwargs=True))
            self.handles.append(module.register_forward_hook(
                lambda mod, args, output, i=index: self._after(i, output)))
        return self

    def clear(self):
        self.weights.clear()
        self.expected_outputs.clear()

    def arrays(self):
        if set(self.weights) != set(range(len(self.modules))):
            raise RuntimeError('Some historical layers bypassed attention hooks; no inference changes attempted')
        return np.stack([self.weights[i] for i in range(len(self.modules))])

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class AttentionAccumulator:
    def __init__(self, layers, heads, observations):
        self.layers, self.heads, self.observations = layers, heads, observations
        self.pair_sum = np.zeros((layers, heads, observations, observations), dtype=np.float64)
        self.pair_count = np.zeros((observations, observations), dtype=np.int64)
        self.received_sum = np.zeros((layers, heads, observations), dtype=np.float64)
        self.relative_sum = np.zeros_like(self.received_sum)
        self.key_count = np.zeros(observations, dtype=np.int64)
        self.samples = 0

    def add(self, weights, observation_ids, valid):
        positions = np.flatnonzero(valid)
        ids = np.asarray(observation_ids, dtype=int)[positions]
        if len(ids) < 1 or len(ids) != len(np.unique(ids)):
            raise ValueError('Attention example needs unique valid acquisition IDs')
        selected = weights[:, :, positions, :][:, :, :, positions]
        if not np.isfinite(selected).all():
            raise ValueError('Attention contains nonfinite valid entries')
        np.testing.assert_allclose(selected.sum(axis=-1), 1, atol=ATOL, rtol=RTOL)
        invalid = np.flatnonzero(~np.asarray(valid, dtype=bool))
        if len(invalid):
            np.testing.assert_allclose(weights[:, :, positions, :][:, :, :, invalid], 0, atol=ATOL)
        self.key_count[ids] += 1
        self.pair_count[np.ix_(ids, ids)] += 1
        for layer in range(self.layers):
            self.pair_sum[layer][np.ix_(np.arange(self.heads), ids, ids)] += selected[layer]
            received = selected[layer].mean(axis=1)  # Average over valid query tokens.
            self.received_sum[layer][:, ids] += received
            self.relative_sum[layer][:, ids] += received * len(ids)
        self.samples += 1
        return selected, ids

    def date_frame(self, metadata, seed):
        rows = []
        for obs_id in np.flatnonzero(self.key_count):
            obs = metadata['observations'][obs_id]
            for layer in range(self.layers):
                for head in range(self.heads):
                    n = int(self.key_count[obs_id])
                    rows.append(dict(seed=seed, layer=layer+1, head=head+1,
                                     acquisition_id=int(obs_id), date=obs['date'], season=obs['season'],
                                     valid_key_samples=n,
                                     mean_received_attention=self.received_sum[layer, head, obs_id]/n,
                                     uniform_relative_attention=self.relative_sum[layer, head, obs_id]/n))
        return pd.DataFrame(rows)

    def pair_frame(self, metadata, seed):
        rows = []
        for query, key in zip(*np.nonzero(self.pair_count)):
            for layer in range(self.layers):
                for head in range(self.heads):
                    rows.append(dict(seed=seed, layer=layer+1, head=head+1,
                                     query_acquisition_id=int(query), key_acquisition_id=int(key),
                                     query_date=metadata['observations'][query]['date'],
                                     key_date=metadata['observations'][key]['date'],
                                     valid_pair_samples=int(self.pair_count[query, key]),
                                     mean_attention=self.pair_sum[layer, head, query, key]/self.pair_count[query, key]))
        return pd.DataFrame(rows)


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def curve_plot(curves_by_task, path, title):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for row, cutoff in enumerate(CUTOFFS):
        for model, color in [('linear', 'tab:orange'), ('transformer', 'tab:blue')]:
            item = curves_by_task.get((cutoff, model))
            if item is None:
                continue
            metric, curve = item
            axes[row, 0].plot(curve['fpr'], curve['tpr'], color=color,
                              label=f"{model}: AUC {metric['roc_auc']:.4f}")
            axes[row, 1].step(curve['recall'], curve['precision'], where='post', color=color,
                              label=f"{model}: AP {metric['average_precision']:.4f}")
            axes[row, 1].axhline(metric['prevalence'], color='grey', linestyle=':', linewidth=.7)
        axes[row, 0].plot([0, 1], [0, 1], '--', color='grey', linewidth=.8)
        axes[row, 0].set(xlabel='False positive rate', ylabel='True positive rate',
                         title=f'ROC: actual NDVI >= {cutoff}', xlim=(0, 1), ylim=(0, 1.02))
        axes[row, 1].set(xlabel='Recall', ylabel='Precision',
                         title=f'PR: actual NDVI >= {cutoff} (dotted = prevalence)',
                         xlim=(0, 1), ylim=(0, 1.02))
        for ax in axes[row]:
            if ax.lines:
                ax.legend(fontsize=8)
    fig.suptitle(title)
    save_figure(fig, path)


def attention_plots(accumulator, dates, metadata, out, seed):
    active = np.flatnonzero(accumulator.key_count)
    labels = [metadata['observations'][i]['date'] for i in active]
    received = (accumulator.received_sum[:, :, active] /
                accumulator.key_count[active][None, None, :])
    relative = (accumulator.relative_sum[:, :, active] /
                accumulator.key_count[active][None, None, :])
    fig, axes = plt.subplots(2, 1, figsize=(16, 8))
    axes[0].bar(range(len(active)), received.mean(axis=(0, 1)))
    axes[0].set(ylabel='Mean received attention mass', title='Averaged over valid queries, heads and layers')
    axes[1].bar(range(len(active)), relative.mean(axis=(0, 1)), color='tab:green')
    axes[1].axhline(1, color='black', linestyle='--', linewidth=.8)
    axes[1].set(ylabel='Uniform-relative attention', title='1 = uniform attention at each example history length')
    for ax in axes:
        ax.set_xticks(range(len(active)), labels, rotation=90, fontsize=7)
    fig.suptitle(f'Seed {seed}: historical key-date attention (descriptive, not causal importance)')
    save_figure(fig, out / 'received_attention_by_date.png')
    fig, ax = plt.subplots(figsize=(16, 11))
    values = relative.reshape(accumulator.layers*accumulator.heads, len(active))
    image = ax.imshow(values, aspect='auto', cmap='RdBu_r',
                      norm=TwoSlopeNorm(vmin=0, vcenter=1, vmax=max(2, float(values.max()))))
    ax.set_xticks(range(len(active)), labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(values)), [f'L{l+1} H{h+1}' for l in range(accumulator.layers)
                                     for h in range(accumulator.heads)], fontsize=7)
    ax.set_title(f'Seed {seed}: per-layer/head uniform-relative received attention')
    fig.colorbar(image, ax=ax, label='Mean attention / per-example uniform baseline')
    save_figure(fig, out / 'layer_head_date_attention.png')
    count = accumulator.pair_count[np.ix_(active, active)]
    total = accumulator.pair_sum[-1].mean(axis=0)[np.ix_(active, active)]
    values = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
    fig, ax = plt.subplots(figsize=(14, 12))
    image = ax.imshow(values, cmap='viridis', vmin=0, aspect='equal')
    ax.set_xticks(range(len(active)), labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(active)), labels, fontsize=7)
    ax.set(xlabel='Attended historical key date', ylabel='Historical query date',
           title=f'Seed {seed}: last-layer head mean; conditional on each pair being valid')
    fig.colorbar(image, ax=ax, label='Mean attention weight (blank = no valid pair)')
    save_figure(fig, out / 'last_layer_date_pairs.png')


def example_plot(weights, dates, path, title):
    layers = len(weights)
    columns = min(2, layers)
    fig, axes = plt.subplots(int(np.ceil(layers/columns)), columns,
                             figsize=(12, 5*int(np.ceil(layers/columns))), squeeze=False)
    for layer, ax in enumerate(axes.flat):
        if layer >= layers:
            ax.set_visible(False)
            continue
        image = ax.imshow(weights[layer].mean(axis=0), vmin=0, vmax=1, cmap='viridis')
        ax.set_xticks(range(len(dates)), dates, rotation=90, fontsize=7)
        ax.set_yticks(range(len(dates)), dates, fontsize=7)
        ax.set(xlabel='Attended historical key date', ylabel='Historical query date',
               title=f'Layer {layer+1}, mean of {weights.shape[1]} heads')
        fig.colorbar(image, ax=ax, fraction=.046, pad=.04)
    fig.suptitle(title)
    save_figure(fig, path)


def analyze_attention(run, predictions, out, seed, limit=None, batch_size=128):
    from uav_ndvi.dataset import PixelDataset, collate, history_indices
    from uav_ndvi.train import load_model

    out.mkdir(parents=True, exist_ok=False)
    model, saved = load_model(run)
    info = json.loads((run / 'run.json').read_text())
    dataset = PixelDataset(info['examples'], 'test')
    if saved['metadata'] != dataset.meta:
        raise ValueError('Checkpoint metadata and saved test dataset differ')
    expected_keys = pd.DataFrame(dict(
        date=[dataset.meta['observations'][int(i)]['date'] for i in dataset.index[:, 0]],
        row=dataset.index[:, 1], column=dataset.index[:, 2]))
    aligned = predictions.set_index(KEYS).reindex(pd.MultiIndex.from_frame(expected_keys))
    if len(predictions) != len(dataset) or aligned['actual'].isna().any():
        raise ValueError('Saved prediction coordinates do not match the test dataset')
    stop = len(dataset) if limit is None else min(limit, len(dataset))
    if stop < 1:
        raise ValueError('No test examples selected for attention')
    # Select examples by dates only, before inspecting any attention or model scores.
    selected = {}
    for index, (target, row, column) in enumerate(dataset.index[:stop]):
        obs = dataset.meta['observations'][int(target)]
        old = selected.get(obs['season'])
        if old is None or obs['date'] > old[0]:
            selected[obs['season']] = (obs['date'], index)
    illustration_ids = {item[1] for item in selected.values()}
    histories = {}
    for target in np.unique(dataset.index[:stop, 0]):
        obs = dataset.meta['observations'][int(target)]
        histories[int(target)] = history_indices(dataset.meta, obs['season'], obs['date'], dataset.cfg)
    layers = len(model.history_encoder.layers)
    heads = model.history_encoder.layers[0].self_attn.num_heads
    accumulator = AttentionAccumulator(layers, heads, len(dataset.meta['observations']))
    initial_state = state_digest(model)
    checkpoint_hash = digest(run / 'transformer.pt')
    max_forward_error = max_saved_error = 0.0
    example_records = []
    with HistoricalAttentionObserver(model) as observer:
        for start in range(0, stop, batch_size):
            indices = range(start, min(stop, start+batch_size))
            items = [dataset[i] for i in indices]
            batch = collate(items)
            # Normal inference has no hooks on the original model during this check.
            # Temporarily remove only our handles, then install them again afterward.
            observer.__exit__()
            with torch.no_grad():
                ordinary = model(batch)
            observer.__enter__()
            observer.clear()
            with torch.no_grad():
                instrumented = model(batch)
            torch.testing.assert_close(ordinary, instrumented, atol=ATOL, rtol=RTOL)
            max_forward_error = max(max_forward_error, float((ordinary-instrumented).abs().max()))
            saved_values = aligned.iloc[start:start+len(items)]
            available_quantiles = [f'q{q}' for q in model.quantiles]
            if all(column in saved_values for column in available_quantiles):
                expected = saved_values[available_quantiles].to_numpy(dtype=np.float32)
            else:
                expected = saved_values[['transformer']].to_numpy(dtype=np.float32)
            measured = ordinary.numpy() if expected.shape[1] > 1 else ordinary[:, [model.quantiles.index(.5)]].numpy()
            np.testing.assert_allclose(measured, expected, atol=ATOL, rtol=RTOL)
            max_saved_error = max(max_saved_error, float(np.max(np.abs(measured-expected))))
            weights = observer.arrays()
            valid = (~(batch['pad'] | batch['missing'])).numpy()
            for offset, index in enumerate(indices):
                actual = float(items[offset]['target'])
                if not np.isclose(actual, float(saved_values.iloc[offset]['actual']), atol=1e-8):
                    raise ValueError('Stored test targets differ from the loaded dataset')
                target, row, column = map(int, dataset.index[index])
                ids = histories[target]
                padded_ids = np.full(valid.shape[1], -1, dtype=int)
                padded_ids[:len(ids)] = ids
                selected_weights, used_ids = accumulator.add(weights[:, offset], padded_ids, valid[offset])
                if index in illustration_ids:
                    dates = [dataset.meta['observations'][i]['date'] for i in used_ids]
                    obs = dataset.meta['observations'][target]
                    stem = f'example_{index}'
                    np.savez_compressed(out / f'{stem}_per_head.npz', attention=selected_weights,
                                        historical_dates=np.array(dates), acquisition_ids=used_ids,
                                        historical_ndvi=items[offset]['history'][valid[offset][:len(ids)], 0].numpy())
                    example_plot(selected_weights, dates, out / f'{stem}_layer_means.png',
                                 f'Seed {seed}, test sample {index}, target {obs["date"]}: {obs["season"]}')
                    example_records.append(dict(sample_index=index, row=row, column=column,
                                                target_date=obs['date'], season=obs['season'],
                                                historical_dates=dates, file=f'{stem}_per_head.npz'))
            if start == 0 or start+batch_size >= stop or (start//batch_size+1) % 25 == 0:
                print(f'Seed {seed}: historical attention {min(start+batch_size, stop)}/{stop}', flush=True)
        max_attention_output_error = observer.max_attention_output_error
    if state_digest(model) != initial_state or digest(run / 'transformer.pt') != checkpoint_hash:
        raise RuntimeError('Model state or checkpoint changed during attention observation')
    dates = accumulator.date_frame(dataset.meta, seed)
    dates.to_csv(out / 'historical_date_attention.csv', index=False)
    accumulator.pair_frame(dataset.meta, seed).to_csv(out / 'historical_date_pairs.csv', index=False)
    support = pd.DataFrame([dict(acquisition_id=i, date=obs['date'], season=obs['season'],
                                 valid_key_samples=int(accumulator.key_count[i]),
                                 attention_available=bool(accumulator.key_count[i]))
                            for i, obs in enumerate(dataset.meta['observations'])])
    support.to_csv(out / 'acquisition_support.csv', index=False)
    write_json(out / 'illustrative_examples.json', example_records)
    attention_plots(accumulator, dates, dataset.meta, out, seed)
    result = dict(status='available', seed=seed, selected_epoch=int(saved['epoch']),
                  samples_analyzed=accumulator.samples, full_test_samples=len(dataset), layers=layers, heads=heads,
                  observed_historical_dates=int(np.count_nonzero(accumulator.key_count)),
                  max_instrumented_prediction_abs_error=max_forward_error,
                  max_prediction_vs_saved_csv_abs_error=max_saved_error,
                  max_observer_vs_original_attention_output_abs_error=max_attention_output_error,
                  model_state_sha256=initial_state, checkpoint_sha256=checkpoint_hash,
                  model_state_unchanged=True, tolerance=dict(atol=ATOL, rtol=RTOL),
                  selection='All saved test samples unless explicit attention-limit; illustrative examples '
                            'are first saved samples at the latest target date in each season/block.')
    write_json(out / 'attention_verification.json', result)
    return dates, result


def ranking_summary(records):
    frame = pd.DataFrame(records)
    rows = []
    for (cutoff, model), group in frame.groupby(['cutoff', 'model']):
        valid = group[group.status == 'available']
        row = dict(cutoff=float(cutoff), positive_definition=f'actual NDVI >= {cutoff}',
                   model=model, requested_seeds=len(group), available_seeds=len(valid))
        for metric in ['roc_auc', 'average_precision']:
            scores = pd.to_numeric(valid[metric])
            row[metric+'_mean'] = float(scores.mean()) if len(scores) else np.nan
            row[metric+'_std'] = float(scores.std(ddof=1)) if len(scores) > 1 else np.nan
        for field in ['n', 'positives', 'negatives', 'prevalence']:
            row[field] = group.iloc[0][field]
        rows.append(row)
    return frame, pd.DataFrame(rows)


def ranking_summary_plot(curves, summary, out):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for row, cutoff in enumerate(CUTOFFS):
        for model, color in [('linear', 'tab:orange'), ('transformer', 'tab:blue')]:
            group = summary[(summary.cutoff == cutoff) & (summary.model == model)]
            if group.empty or group.iloc[0].available_seeds == 0:
                continue
            metric = group.iloc[0]
            items = [value for (seed, cut, name), value in curves.items() if cut == cutoff and name == model]
            for index, curve in enumerate(items):
                if model == 'linear' and index:
                    continue
                label_roc = f"{model} AUC {metric.roc_auc_mean:.4f} +/- {metric.roc_auc_std:.4f}" if index == 0 else None
                label_pr = f"{model} AP {metric.average_precision_mean:.4f} +/- {metric.average_precision_std:.4f}" if index == 0 else None
                alpha = 1 if model == 'linear' else .5
                axes[row, 0].plot(curve['fpr'], curve['tpr'], color=color, alpha=alpha, label=label_roc)
                axes[row, 1].step(curve['recall'], curve['precision'], where='post', color=color,
                                  alpha=alpha, label=label_pr)
        prevalence = summary[summary.cutoff == cutoff].iloc[0]['prevalence']
        axes[row, 0].plot([0, 1], [0, 1], '--', color='grey', linewidth=.8)
        if pd.notna(prevalence):
            axes[row, 1].axhline(prevalence, linestyle=':', color='grey',
                                  label=f'No-skill prevalence {prevalence:.5f}')
        axes[row, 0].set(xlabel='False positive rate', ylabel='True positive rate',
                         title=f'ROC, actual NDVI >= {cutoff}', xlim=(0, 1), ylim=(0, 1.02))
        axes[row, 1].set(xlabel='Recall', ylabel='Precision',
                         title=f'Precision-recall, actual NDVI >= {cutoff}', xlim=(0, 1), ylim=(0, 1.02))
        for ax in axes[row]:
            handles, labels = ax.get_legend_handles_labels()
            if labels:
                ax.legend(handles, labels, fontsize=8)
    fig.suptitle('Five saved seeds; continuous NDVI scores; lines are individual runs')
    save_figure(fig, out / 'roc_pr_five_seed_summary.png')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, metric, title in zip(axes, ['roc_auc', 'average_precision'], ['ROC-AUC', 'Average precision']):
        for model, offset, color in [('linear', -.12, 'tab:orange'), ('transformer', .12, 'tab:blue')]:
            data = summary[summary.model == model].sort_values('cutoff')
            axis.errorbar(np.arange(len(data))+offset, data[metric+'_mean'],
                          yerr=data[metric+'_std'].fillna(0), fmt='o', capsize=4, color=color, label=model)
        axis.set(xticks=[0, 1], xticklabels=['NDVI >= 0.4', 'NDVI >= 0.6'],
                 ylabel=title, title=title+' mean and seed SD', ylim=(0, 1))
        axis.legend()
    fig.suptitle('Error bars describe training-seed variability, not independent fields')
    save_figure(fig, out / 'ranking_metrics_seed_mean_sd.png')
    healthy_precision_detail(curves, summary, out)


def healthy_precision_detail(curves, summary, out):
    fig, ax = plt.subplots(figsize=(10, 5))
    for model, color in [('linear', 'tab:orange'), ('transformer', 'tab:blue')]:
        group = summary[(summary.cutoff == .6) & (summary.model == model)]
        if group.empty or not group.iloc[0].available_seeds:
            continue
        metric = group.iloc[0]
        items = [value for (seed, cutoff, name), value in curves.items() if cutoff == .6 and name == model]
        for index, curve in enumerate(items):
            if model == 'linear' and index:
                continue
            label = f'{model}: mean AP {metric.average_precision_mean:.5f}' if index == 0 else None
            ax.step(curve['recall'], curve['precision'], where='post', color=color,
                    alpha=.5 if model == 'transformer' else 1, label=label)
    group = summary[summary.cutoff == .6]
    if not group.empty and pd.notna(group.iloc[0].prevalence):
        prevalence = group.iloc[0].prevalence
        ax.axhline(prevalence, color='grey', linestyle=':', label=f'Prevalence {prevalence:.5f}')
    ax.set(xlabel='Recall', ylabel='Precision (axis zoomed)', xlim=(0, 1), ylim=(0, .05),
           title='Healthy targets: NDVI >= 0.6; precision zoomed to 0-0.05')
    ax.legend()
    fig.suptitle('Higher precision outside this zoom is not shown; full curves are in the main summary')
    save_figure(fig, out / 'healthy_precision_recall_zoom.png')


def summarize_attention(frames, out):
    if not frames:
        return pd.DataFrame()
    detailed = pd.concat(frames, ignore_index=True)
    detailed.to_csv(out / 'attention_all_seeds_layer_head_dates.csv', index=False)
    per_seed = detailed.groupby(['seed', 'acquisition_id', 'date', 'season'], as_index=False).agg(
        mean_received_attention=('mean_received_attention', 'mean'),
        uniform_relative_attention=('uniform_relative_attention', 'mean'),
        valid_key_samples=('valid_key_samples', 'first'))
    per_seed.to_csv(out / 'attention_per_seed_dates.csv', index=False)
    summary = per_seed.groupby(['acquisition_id', 'date', 'season'], as_index=False).agg(
        mean_received_attention=('mean_received_attention', 'mean'),
        received_attention_seed_std=('mean_received_attention', 'std'),
        mean_uniform_relative_attention=('uniform_relative_attention', 'mean'),
        uniform_relative_seed_std=('uniform_relative_attention', 'std'),
        available_seeds=('seed', 'nunique'), min_valid_key_samples=('valid_key_samples', 'min'),
        max_valid_key_samples=('valid_key_samples', 'max'))
    summary.to_csv(out / 'attention_date_seed_summary.csv', index=False)
    fig, axes = plt.subplots(2, 1, figsize=(16, 8))
    for ax, metric, error, title in [
        (axes[0], 'mean_received_attention', 'received_attention_seed_std', 'Received attention mass'),
        (axes[1], 'mean_uniform_relative_attention', 'uniform_relative_seed_std', 'Uniform-relative received attention')]:
        ax.errorbar(np.arange(len(summary)), summary[metric], yerr=summary[error].fillna(0), fmt='o', capsize=2)
        ax.set_xticks(range(len(summary)), summary.date, rotation=90, fontsize=7)
        ax.set(ylabel=title, title=title+' (head/layer means, then seed mean +/- SD)')
    axes[1].axhline(1, color='grey', linestyle='--', linewidth=.8)
    fig.suptitle('Historical self-attention conditioned on valid acquisition availability; not causal importance')
    save_figure(fig, out / 'attention_five_seed_date_summary.png')
    return summary


def write_report(out, metrics, summary, availability, attention_summary, integrity):
    images = ['roc_pr_five_seed_summary.png', 'ranking_metrics_seed_mean_sd.png',
              'healthy_precision_recall_zoom.png', 'attention_five_seed_date_summary.png']
    content = ['<!doctype html><html><head><meta charset="utf-8"><title>NDVI post-hoc analysis</title>',
               '<style>body{font:16px system-ui;max-width:1200px;margin:35px auto;padding:0 20px;line-height:1.5}'
               'table{border-collapse:collapse;font-size:13px}th,td{border:1px solid #ccc;padding:6px}'
               'img{max-width:100%}code{background:#eee}h2{margin-top:35px}</style></head><body>',
               '<h1>Five-seed NDVI post-hoc analysis</h1>',
               '<p>Existing predictions, checkpoints and saved test samples only. No retraining. '
               'Original experimental outputs and model/preprocessing source are preserved.</p>',
               '<h2>Definitions</h2><ul><li>Positive targets are actual NDVI &ge; 0.4 (marginal-or-healthy) '
               'or &ge; 0.6 (healthy).</li><li>Scores are raw continuous point predictions. No clipping, '
               'hard prediction thresholding or probability calibration was applied.</li>'
               '<li>ROC-AUC measures ranking; AP uses scikit-learn average precision, not trapezoidal PR area.</li>'
               '<li>The PR no-skill reference is positive prevalence. A model can rank healthy targets even '
               'if none of its scores reach 0.6.</li></ul>',
               '<h2>Seed summary</h2>', summary.to_html(index=False, float_format=lambda x: f'{x:.6f}'),
               '<p>SD is sample SD across the available seeds on the same fixed test set. Linear regression '
               'is deterministic. These are not independent fields/datasets; no additional significance tests '
               'or independent-pixel confidence intervals were performed.</p>',
               '<h2>Per-seed results</h2>', metrics.to_html(index=False, float_format=lambda x: f'{x:.6f}'),
               '<h2>Historical attention method and limitations</h2><p>Temporary non-returning hooks observe '
               'each historical self-attention module\'s actual Q/K/V and masks. Separate copies with identical '
               'weights recompute per-head softmax attention in evaluation mode. The original forward methods, '
               'parameters and need_weights=False calls remain unchanged. Hooks can change fused-kernel dispatch; '
               'ordinary, instrumented and saved predictions are checked with documented float32 tolerances.</p>',
               '<p>Only valid historical query/key positions are aggregated. Received mass is averaged over '
               'valid query tokens. Uniform-relative attention divides by 1 / valid-history-length within each '
               'example before averaging, reducing the mechanical effect of different history lengths. '
               'Acquisition support counts are saved. Dates never used as valid history are unavailable, not '
               'zero-importance dates. Pairwise means are conditional on each pair co-occurring and need not '
               'sum to one across a global date row.</p>',
               '<p>This is historical-to-historical self-attention, not target-to-history cross-attention. '
               'It is not a causal explanation or a complete measure of prediction importance: value projections, '
               'feedforward blocks, residuals, masked pooling, the future branch and persistence anchor also '
               'affect output. No intervention/ablation or new performance claim was made.</p>',
               '<p>Illustrations are selected deterministically by dates: the first saved test example at '
               'the latest target date in each represented season/block. Full per-head arrays are in NPZ files; '
               'the illustrative plots average heads within each layer.</p>']
    if not attention_summary.empty:
        content += ['<h2>Dates with largest mean uniform-relative received attention</h2>',
                    '<p>Descriptive conditional ranking, not proof of most influential flights.</p>',
                    attention_summary.sort_values('mean_uniform_relative_attention', ascending=False).head(10).to_html(
                        index=False, float_format=lambda x: f'{x:.6f}')]
    content += ['<h2>Availability and prediction-agreement checks</h2>', pd.DataFrame(availability).to_html(index=False),
                '<h2>Protected input audit</h2><pre>'+html.escape(json.dumps(integrity, indent=2))+'</pre>']
    for name in images:
        if (out / name).is_file():
            content.append(f'<h2>{html.escape(name)}</h2><img src="{name}" alt="{name}">')
    content += ['<h2>Per-seed attention figures</h2>']
    for path in sorted(out.glob('seed_*/attention/received_attention_by_date.png')):
        relative = path.relative_to(out).as_posix()
        content.append(f'<h3>{html.escape(path.parent.parent.name)}</h3><img src="{relative}">')
        for example in sorted(path.parent.glob('example_*_layer_means.png')):
            relative = example.relative_to(out).as_posix()
            content.append(f'<a href="{relative}">Illustration: {html.escape(example.name)}</a><br>')
    content.append('<h2>Sources</h2><ul><li><a href="https://scikit-learn.org/stable/modules/generated/'
                   'sklearn.metrics.average_precision_score.html">scikit-learn AP definition</a></li>'
                   '<li><a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html">'
                   'PyTorch per-head attention</a></li></ul></body></html>')
    (out / 'report.html').write_text('\n'.join(content), encoding='utf-8')
    lines = ['FIVE-SEED POST-HOC RESULTS FOR PRESENTATION',
             'No retraining. Existing continuous point predictions and historical self-attention only.',
             'Positive target: actual NDVI >= 0.4 (marginal-or-healthy) or >= 0.6 (healthy).',
             'These are two binary rankings, not a three-class ROC or calibrated class probabilities.', '']
    for row in summary.itertuples():
        lines.append(f'NDVI >= {row.cutoff}, {row.model}: ROC-AUC {row.roc_auc_mean:.6f} '
                     f'+/- {row.roc_auc_std:.6f}; AP {row.average_precision_mean:.6f} '
                     f'+/- {row.average_precision_std:.6f}; prevalence {row.prevalence:.6f}; '
                     f'{row.available_seeds}/{row.requested_seeds} seeds.')
    lines += ['', 'The +/- values are sample SD across seeds on this fixed dataset, not confidence intervals '
              'across independent fields. No new significance tests were performed. The earlier RMSE p-value '
              'does not test differences in AUC or AP.',
              'AP is sklearn average precision, not trapezoidal PR area. PR no-skill baseline is prevalence.',
              'A score can rank healthy samples without any predicted NDVI reaching 0.6. '
              'Ranking metrics do not reverse the previously observed poor fixed-threshold classification.',
              'Historical attention covers valid keys/queries only, with support counts and a uniform-history '
              'baseline correction. It is not causal attribution or target-to-history cross-attention.',
              'Full per-head illustrative NPZ attention has axes [layer, head, query_date, key_date].',
              'Original file audit unchanged: '+str(integrity['unchanged']),
              'Attention extraction status by seed:']
    lines += [f"Seed {item['seed']}: {item['attention_status']}; samples {item.get('attention_samples', 'unavailable')}"
              for item in availability]
    (out / 'presentation_summary.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def run_analysis(project, runs_file, out, batch_size=128, cpu_threads=4, attention_limit=None):
    project, out = Path(project).resolve(), Path(out).resolve()
    allowed = project / 'analysis_results'
    if not out.is_relative_to(allowed) or out == allowed:
        raise ValueError('Output must be a fresh child of the separate analysis_results directory')
    if out.exists():
        raise FileExistsError('Choose a new output directory; existing results are never overwritten')
    if batch_size < 1 or cpu_threads < 1 or (attention_limit is not None and attention_limit < 1):
        raise ValueError('Batch size, CPU threads and any attention limit must be positive')
    before = protected_inventory(project)
    out.mkdir(parents=True)
    write_json(out / 'protected_inputs_before.json', before)
    torch.set_num_threads(cpu_threads)
    runs = [Path(path).resolve() for path in json.loads(Path(runs_file).read_text())]
    if len(runs) != 5:
        raise ValueError('This analysis expects the five saved runs, not additional or repeated evaluations')
    records, availability, attention_frames, all_curves = [], [], [], {}
    reference_samples = reference_config = None
    seen_seeds = set()
    for run in runs:
        info = json.loads((run / 'run.json').read_text())
        seed = int(info['config']['seed'])
        if seed in seen_seeds:
            raise ValueError('Repeated seeds do not represent five independent initialization runs')
        seen_seeds.add(seed)
        cfg = {key: value for key, value in info['config'].items() if key != 'seed'}
        if reference_config is None:
            reference_config = cfg
        elif reference_config != cfg:
            raise ValueError('The saved runs have different non-seed configurations')
        destination = out / f'seed_{seed}'
        destination.mkdir()
        status = dict(seed=seed, ranking_status='unavailable', attention_status='unavailable')
        path = run / 'evaluation_test/test_predictions.csv'
        if not path.is_file():
            status['reason'] = 'Saved test prediction CSV missing; no predictions were regenerated'
            availability.append(status)
            continue
        predictions = pd.read_csv(path)
        required = KEYS + ['actual', *MODELS]
        if not set(required).issubset(predictions.columns) or predictions.duplicated(KEYS).any():
            status['reason'] = 'Missing required prediction fields or duplicate test coordinates'
            availability.append(status)
            continue
        samples = predictions.sort_values(KEYS)[KEYS + ['actual']].reset_index(drop=True)
        if reference_samples is None:
            reference_samples = samples
        elif not samples.equals(reference_samples):
            raise ValueError('Test samples/targets differ across seeds; no data were changed to align them')
        curves_by_task = {}
        seed_records = []
        for cutoff in CUTOFFS:
            for model in MODELS:
                metric, curve = ranking_metrics(predictions.actual, predictions[model], cutoff)
                row = dict(seed=seed, model=model, cutoff=cutoff,
                           positive_definition=f'actual NDVI >= {cutoff}', **metric)
                records.append(row)
                seed_records.append(row)
                if curve is not None:
                    curves_by_task[(cutoff, model)] = (metric, curve)
                    all_curves[(seed, cutoff, model)] = curve
                    tag = f'{model}_ndvi_ge_{str(cutoff).replace(".", "_")}'
                    pd.DataFrame(dict(false_positive_rate=curve['fpr'], true_positive_rate=curve['tpr'],
                                      score_threshold=curve['roc_threshold'])).to_csv(destination / f'{tag}_roc.csv', index=False)
                    pd.DataFrame(dict(recall=curve['recall'], precision=curve['precision'],
                                      score_threshold=np.r_[curve['pr_threshold'], np.nan])).to_csv(
                                          destination / f'{tag}_precision_recall.csv', index=False)
        pd.DataFrame(seed_records).to_csv(destination / 'ranking_metrics.csv', index=False)
        curve_plot(curves_by_task, destination / 'roc_pr.png', f'Seed {seed}: saved continuous NDVI predictions')
        status['ranking_status'] = ('available' if len(curves_by_task) == len(CUTOFFS)*len(MODELS)
                                    else 'partially available: see ranking_metrics.csv' if curves_by_task
                                    else 'unavailable: see ranking_metrics.csv')
        try:
            dates, verification = analyze_attention(run, predictions, destination / 'attention', seed,
                                                     limit=attention_limit, batch_size=batch_size)
            attention_frames.append(dates)
            status.update(attention_status='available', attention_samples=verification['samples_analyzed'],
                          max_forward_abs_error=verification['max_instrumented_prediction_abs_error'],
                          max_saved_prediction_abs_error=verification['max_prediction_vs_saved_csv_abs_error'])
        except Exception as error:
            status['attention_status'] = 'unavailable: ' + type(error).__name__ + ': ' + str(error)
            write_json(destination / 'attention_unavailable.json', dict(reason=status['attention_status']))
        availability.append(status)
        write_json(out / 'availability.json', availability)
        print(f'Seed {seed}: ranking {status["ranking_status"]}; attention {status["attention_status"]}', flush=True)
    if not records:
        write_json(out / 'availability.json', availability)
        raise RuntimeError('No saved predictions were available; see availability.json')
    metrics, summary = ranking_summary(records)
    metrics.to_csv(out / 'ranking_metrics_all_seeds.csv', index=False)
    summary.to_csv(out / 'ranking_metrics_summary.csv', index=False)
    ranking_summary_plot(all_curves, summary, out)
    attention_summary = summarize_attention(attention_frames, out)
    after = protected_inventory(project)
    changes = sorted(path for path in before.keys() | after.keys() if before.get(path) != after.get(path))
    integrity = dict(unchanged=not changes, changed_files=changes, protected_files=len(before),
                     hash_policy='SHA256 for files <=2 MB and all checkpoints/joblib; size+mtime_ns for every protected file',
                     no_training=True, existing_results_overwritten=False,
                     existing_model_source_modified=False, hooks_removed_after_use=True)
    write_json(out / 'input_integrity.json', integrity)
    write_json(out / 'analysis_protocol.json', dict(
        runs=[str(path) for path in runs], seeds=sorted(seen_seeds), cutoffs=CUTOFFS,
        models=MODELS, score='raw saved continuous point NDVI', positive='actual >= cutoff',
        average_precision='sklearn average_precision_score; not trapezoidal PR area',
        cohort='Same saved spatial test examples; no new data split or preprocessing',
        attention_limit=attention_limit, attention_batch_size=batch_size, cpu_threads=cpu_threads,
        software=dict(python=platform.python_version(), torch=torch.__version__, sklearn=sklearn.__version__),
        attention_method='Observational Q/K/V hooks plus independent equal-weight attention copies; '
                         'no model forward replacement or trained-parameter changes',
        interpretations=['Binary ranking, not multiclass AUC or calibrated probability evaluation',
                         'Seed SD conditional on fixed test data; no independent-pixel inference',
                         'Historical self-attention is descriptive and not causal attribution']))
    write_report(out, metrics, summary, availability, attention_summary, integrity)
    assert_inventory_unchanged(before, after)
    print(summary.to_string(index=False), flush=True)
    print(f'Protected originals unchanged. Report: {out / "report.html"}', flush=True)
    return summary, availability


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--runs-file', default='runs/model_comparison/five_seed_2026-11-18/completed_runs.json')
    parser.add_argument('--out', required=True)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--attention-limit', type=int, help='Optional explicit prefix limit; omit for all test samples')
    args = parser.parse_args()
    run_analysis(args.project_root, args.runs_file, args.out, args.batch_size,
                 args.cpu_threads, args.attention_limit)


if __name__ == '__main__':
    main()
