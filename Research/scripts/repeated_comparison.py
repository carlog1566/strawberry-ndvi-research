"""Repeat fixed-split training and compare paired run-level RMSE scores."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def paired_statistics(transformer, linear, alpha=0.05):
    transformer = np.asarray(transformer, dtype=float)
    linear = np.asarray(linear, dtype=float)
    if transformer.ndim != 1 or transformer.shape != linear.shape or len(linear) < 2:
        raise ValueError('Need at least two paired run scores of equal length')
    if not np.isfinite(transformer).all() or not np.isfinite(linear).all():
        raise ValueError('Run scores must be finite')
    if not 0 < alpha < 1:
        raise ValueError('alpha must be between zero and one')
    difference = transformer - linear
    result = dict(
        n_pairs=len(linear), primary_metric='rmse', alpha=alpha,
        difference_definition='transformer RMSE minus linear RMSE; negative favors transformer',
        transformer_mean=float(transformer.mean()), transformer_std=float(transformer.std(ddof=1)),
        linear_mean=float(linear.mean()), linear_std=float(linear.std(ddof=1)),
        mean_difference=float(difference.mean()), difference_std=float(difference.std(ddof=1)),
        linear_scores_identical=bool(np.all(linear == linear[0])),
        alternative='two-sided', df=len(linear)-1,
        inference_scope='Training-seed variability conditional on this fixed training, validation and test dataset. '
                        'Not uncertainty across new fields, years or independent datasets. '
                        'Assumes independent seed realizations and approximately normal run-level differences. '
                        'The test set was evaluated previously; treat this as an exploratory comparison.',
    )
    if np.ptp(difference) == 0:
        result.update(test_status='undefined: paired differences have zero variance',
                      t_statistic=None, p_value=None, confidence_interval=None, reject_equal_means=None)
    else:
        test = stats.ttest_rel(transformer, linear, alternative='two-sided')
        interval = test.confidence_interval(confidence_level=1-alpha)
        result.update(test_status='computed', t_statistic=float(test.statistic),
                      p_value=float(test.pvalue), confidence_interval=[float(interval.low), float(interval.high)],
                      reject_equal_means=bool(test.pvalue < alpha))
    return result


def collect_scores(runs):
    records, seen_seeds = [], set()
    reference_config = reference_samples = reference_examples = None
    for path in runs:
        path = Path(path)
        info = json.loads((path / 'run.json').read_text())
        cfg = info['config'].copy()
        seed = cfg.pop('seed')
        examples = str(Path(info['examples']).resolve())
        if reference_examples is None:
            reference_examples = examples
        elif examples != reference_examples:
            raise ValueError('Training/validation example directories differ; use a fixed saved split')
        if seed in seen_seeds:
            raise ValueError(f'Duplicate seed {seed}: do not count repeated identical seeds as independent runs')
        seen_seeds.add(seed)
        if reference_config is None:
            reference_config = cfg
        elif cfg != reference_config:
            raise ValueError('Model settings differ beyond seed; this runner compares fixed protocols')
        predictions = pd.read_csv(path / 'evaluation_test/test_predictions.csv')
        keys = ['date', 'row', 'column']
        if predictions.duplicated(keys).any():
            raise ValueError('Duplicate test sample coordinates/date')
        predictions = predictions.sort_values(keys).reset_index(drop=True)
        samples = predictions[keys + ['actual']]
        if reference_samples is None:
            reference_samples = samples
        elif not samples.equals(reference_samples):
            raise ValueError('Each pair and repetition must use exactly the same test samples and targets')
        actual = predictions.actual.to_numpy()
        values = {name: float(np.sqrt(np.mean((predictions[name].to_numpy()-actual)**2)))
                  for name in ['transformer', 'linear']}
        if len(predictions) == 0 or not np.isfinite(list(values.values())).all():
            raise ValueError('Empty or nonfinite test predictions')
        records.append(dict(seed=seed, run=str(path.resolve()), n_test=len(predictions),
                            transformer_rmse=values['transformer'], linear_rmse=values['linear'],
                            difference=values['transformer']-values['linear']))
    return pd.DataFrame(records)


def summarize(runs, out, alpha=0.05):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if any((out / name).exists() for name in ['paired_scores.csv', 'paired_test.json', 'comparison.md']):
        raise FileExistsError('Comparison reports already exist; use a fresh summary directory')
    scores = collect_scores(runs)
    result = paired_statistics(scores.transformer_rmse, scores.linear_rmse, alpha)
    scores.to_csv(out / 'paired_scores.csv', index=False)
    (out / 'paired_test.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    text = '# Repeated fixed-split comparison\n\n'
    text += f"Paired runs: {result['n_pairs']}. Each run uses the same test samples.\n\n"
    text += '| Model | Mean test RMSE | Sample standard deviation |\n| --- | ---: | ---: |\n'
    for name in ['transformer', 'linear']:
        text += f"| {name} | {result[name+'_mean']:.8f} | {result[name+'_std']:.8f} |\n"
    text += f"\nMean paired difference (transformer minus linear): {result['mean_difference']:.8f}.\n\n"
    if result['test_status'] == 'computed':
        low, high = result['confidence_interval']
        text += (f"Two-sided paired t-test: t({result['df']}) = {result['t_statistic']:.6f}, "
                 f"p = {result['p_value']:.8g}. {(1-alpha)*100:g}% confidence interval "
                 f"for the mean difference: [{low:.8f}, {high:.8f}].\n\n")
    else:
        text += result['test_status'] + '.\n\n'
    if result['reject_equal_means'] is False:
        text += ('At the specified alpha, this test did not detect a mean RMSE difference. '
                 'This does not establish equivalence.\n\n')
    elif result['reject_equal_means'] is True:
        favored = 'transformer' if result['mean_difference'] < 0 else 'linear regression'
        text += (f'The test detects a mean RMSE difference favoring {favored}, conditional '
                 'on this fixed dataset and the stated assumptions.\n\n')
    text += ('A negative difference favors the transformer. Linear regression is deterministic on fixed '
             'samples, so identical linear scores are expected and are not independent datasets. '
             'The t-test uses one difference per training seed, not one per pixel.\n\n'
             + result['inference_scope'] + '\n\n'
             'With few repeats, the normal-difference assumption is difficult to assess. '
             'A small p-value here does not establish superiority on unseen dates or fields.\n\n'
             '[SciPy paired t-test documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html)\n')
    (out / 'comparison.md').write_text(text)
    print(json.dumps(result, indent=2), flush=True)
    return result


def train_seed(examples, run, cfg, cpu_threads, log_path):
    import torch
    from uav_ndvi.train import train
    from uav_ndvi.evaluate import evaluate

    torch.set_num_threads(cpu_threads)
    with Path(log_path).open('w', encoding='utf-8') as log, redirect_stdout(log), redirect_stderr(log):
        train(examples, run, cfg)
        evaluate(run, split='validation')
        evaluate(run, split='test')
    return str(Path(run).resolve())


def run_repetitions(args):
    from uav_ndvi.config import load_config

    if len(args.seeds) < 2 or len(args.seeds) != len(set(args.seeds)):
        raise ValueError('Provide at least two distinct seeds')
    if any(seed < 0 or seed >= 2**32 for seed in args.seeds):
        raise ValueError('Seeds must be between zero and 2**32 - 1')
    if not 0 < args.alpha < 1:
        raise ValueError('alpha must be between zero and one')
    if args.cpu_threads < 1:
        raise ValueError('cpu-threads must be positive')
    if args.jobs < 1:
        raise ValueError('jobs must be positive')
    out = Path(args.out)
    if out.exists():
        raise FileExistsError('Choose a new repetition output directory')
    examples = Path(args.examples).resolve()
    cfg = load_config(args.config)
    reference = Path(args.reference_run).resolve() if args.reference_run else None
    reference_seed = None
    if reference:
        saved = json.loads((reference / 'run.json').read_text())
        reference_seed = saved['config']['seed']
        if reference_seed not in args.seeds or Path(saved['examples']).resolve() != examples:
            raise ValueError('Reference run must use these examples and a requested seed')
        if {k: v for k, v in saved['config'].items() if k != 'seed'} != {k: v for k, v in cfg.items() if k != 'seed'}:
            raise ValueError('Reference run settings differ from requested protocol')
        if not (reference / 'evaluation_test/test_predictions.csv').is_file():
            raise ValueError('Reference run needs completed test predictions')
    out.mkdir(parents=True)
    (out / 'logs').mkdir()
    fingerprints = {name: hashlib.sha256((examples / name).read_bytes()).hexdigest()
                    for name in ['train.npy', 'validation.npy', 'test.npy', 'examples.json']}
    protocol = dict(mode='fixed-split initialization-seed repetitions', seeds=args.seeds,
                    examples=str(examples), config=cfg, example_sha256=fingerprints,
                    reference_run=str(reference) if reference else None, cpu_threads=args.cpu_threads, jobs=args.jobs,
                    metric='rmse', alternative='two-sided', alpha=args.alpha,
                    existing_test_set_already_inspected=True)
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    completed = {reference_seed: reference} if reference else {}
    if reference:
        print(f'Using completed reference seed {reference_seed}: {reference}', flush=True)
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        pending = {}
        for seed in args.seeds:
            if seed == reference_seed:
                continue
            run = out / f'seed_{seed}'
            future = pool.submit(train_seed, str(examples), str(run), dict(cfg, seed=seed),
                                 args.cpu_threads, str(out / 'logs' / f'seed_{seed}.log'))
            pending[future] = (seed, run)
        while pending:
            done, _ = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            for future in done:
                seed, run = pending.pop(future)
                completed[seed] = Path(future.result())
                print(f'Seed {seed} complete ({len(completed)}/{len(args.seeds)})', flush=True)
                (out / 'completed_runs.json').write_text(json.dumps(
                    [str(completed[seed].resolve()) for seed in args.seeds if seed in completed], indent=2))
            for seed, run in pending.values():
                history = run / 'history.csv'
                if history.exists():
                    try:
                        table = pd.read_csv(history)
                        print(f'Seed {seed}: epoch {len(table)}, best validation RMSE '
                              f'{table.validation_rmse.min():.6f}', flush=True)
                    except pd.errors.EmptyDataError:
                        pass  # The worker may be rewriting history at this instant.
    if any(hashlib.sha256((examples / name).read_bytes()).hexdigest() != digest
           for name, digest in fingerprints.items()):
        raise ValueError('Saved examples changed during the repeated experiment')
    runs = [completed[seed] for seed in args.seeds]
    summarize(runs, out, args.alpha)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run', help='Train paired models for distinct seeds on a fixed saved split')
    run.add_argument('--examples', required=True)
    run.add_argument('--config', required=True)
    run.add_argument('--seeds', nargs='+', type=int, required=True)
    run.add_argument('--reference-run')
    run.add_argument('--cpu-threads', type=int, default=4)
    run.add_argument('--jobs', type=int, default=1, help='Number of separate training processes')
    run.add_argument('--out', required=True)
    run.add_argument('--alpha', type=float, default=0.05)
    summary = commands.add_parser('summarize', help='Analyze completed runs; no retraining')
    summary.add_argument('--runs', nargs='+', required=True)
    summary.add_argument('--out', required=True)
    summary.add_argument('--alpha', type=float, default=0.05)
    args = parser.parse_args()
    if args.command == 'run':
        run_repetitions(args)
    else:
        summarize(args.runs, args.out, args.alpha)


if __name__ == '__main__':
    main()
