import json

import numpy as np
import pandas as pd
import pytest

from scripts.repeated_comparison import collect_scores, paired_statistics, summarize


def test_paired_statistic_uses_run_differences():
    transformer = np.array([0.09, 0.091, 0.089, 0.092, 0.088])
    linear = np.full(5, 0.1)
    difference = transformer-linear
    result = paired_statistics(transformer, linear)
    expected_t = difference.mean()/(difference.std(ddof=1)/np.sqrt(5))
    assert result['n_pairs'] == 5
    assert result['df'] == 4
    assert result['t_statistic'] == pytest.approx(expected_t)
    assert result['mean_difference'] == pytest.approx(-0.01)
    assert result['confidence_interval'][1] < 0
    assert result['p_value'] < 0.01
    assert result['linear_scores_identical']


def test_zero_variance_does_not_fabricate_a_p_value():
    result = paired_statistics([0.1, 0.1, 0.1], [0.2, 0.2, 0.2])
    assert result['p_value'] is None
    assert result['reject_equal_means'] is None


@pytest.mark.parametrize('transformer,linear', [([0.1], [0.2]), ([0.1, np.nan], [0.2, 0.3]),
                                               ([0.1, 0.2], [0.3, 0.4, 0.5])])
def test_invalid_pairs_rejected(transformer, linear):
    with pytest.raises(ValueError):
        paired_statistics(transformer, linear)


def make_run(root, seed, offset=0):
    run = root / f'run_{seed}_{offset}'
    (run / 'evaluation_test').mkdir(parents=True)
    (run / 'run.json').write_text(json.dumps({'config': {'seed': seed, 'epochs': 100},
                                            'examples': str(root / 'shared_examples')}))
    actual = np.array([0.4, 0.5, 0.6])
    pd.DataFrame({'date': ['2025-04-14']*3, 'row': [1, 2, 3], 'column': [0, 0, 0],
                  'actual': actual+offset, 'linear': actual+0.1,
                  'transformer': actual+0.05+seed*0.001}).to_csv(
                      run / 'evaluation_test/test_predictions.csv', index=False)
    return run


def test_duplicate_seeds_are_not_independent_repeats(tmp_path):
    first = make_run(tmp_path, 42)
    with pytest.raises(ValueError, match='Duplicate seed'):
        collect_scores([first, first])


def test_mismatched_test_targets_rejected(tmp_path):
    with pytest.raises(ValueError, match='same test samples'):
        collect_scores([make_run(tmp_path, 1), make_run(tmp_path, 2, offset=0.01)])


def test_summary_outputs_pairs_not_pooled_pixels(tmp_path):
    runs = [make_run(tmp_path, seed) for seed in [1, 2, 3]]
    out = tmp_path / 'summary'
    result = summarize(runs, out)
    assert result['n_pairs'] == 3
    assert len(pd.read_csv(out / 'paired_scores.csv')) == 3
    assert json.loads((out / 'paired_test.json').read_text())['df'] == 2
    assert 'conditional on this fixed' in (out / 'comparison.md').read_text()
    with pytest.raises(FileExistsError):
        summarize(runs, out)
