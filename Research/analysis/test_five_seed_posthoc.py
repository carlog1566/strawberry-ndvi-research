"""Isolated tests for added analysis; no fitting or training is performed."""
from pathlib import Path

import numpy as np
import pytest
import torch

from analysis.five_seed_posthoc import (
    AttentionAccumulator, HistoricalAttentionObserver, assert_inventory_unchanged,
    protected_inventory, ranking_metrics, state_digest,
)
from uav_ndvi.dataset import collate, make_item
from uav_ndvi.models.transformer import AgriMatNetUAV


def test_healthy_ranking_uses_continuous_scores_not_hard_labels():
    # All predictions are below 0.6, yet they rank the healthy samples perfectly.
    actual = [.2, .4, .5, .6, .8]
    scores = [.21, .3, .35, .42, .45]
    result, curves = ranking_metrics(actual, scores, .6)
    assert result['positives'] == 2
    assert result['prevalence'] == pytest.approx(.4)
    assert result['roc_auc'] == 1
    assert result['average_precision'] == 1
    assert len(curves['precision']) == len(curves['pr_threshold']) + 1


def test_boundary_is_greater_than_or_equal_and_no_clipping():
    result, curves = ranking_metrics([.2, .4, .5, .6], [-.8, .1, .2, 1.2], .4)
    assert result['positives'] == 3
    assert result['roc_auc'] == 1
    assert 1.2 in curves['roc_threshold']
    assert -.8 in curves['roc_threshold']


def test_constant_scores_have_chance_auc_and_prevalence_ap():
    result, _ = ranking_metrics([.2, .2, .2, .7], [.3]*4, .6)
    assert result['roc_auc'] == .5
    assert result['average_precision'] == .25


@pytest.mark.parametrize('actual,cutoff', [([.2, .3], .6), ([.6, .7], .6)])
def test_single_class_is_explicitly_unavailable(actual, cutoff):
    result, curves = ranking_metrics(actual, [.25, .35], cutoff)
    assert result['status'].startswith('unavailable')
    assert result['roc_auc'] is None and result['average_precision'] is None
    assert curves is None


def test_nonfinite_input_is_not_silently_dropped():
    result, curves = ranking_metrics([.2, .7], [.3, np.nan], .6)
    assert result['n'] == 2
    assert result['status'].startswith('unavailable')
    assert curves is None


def test_attention_observation_preserves_forward_state_and_masks():
    torch.manual_seed(123)
    torch.set_num_threads(1)
    cfg = dict(d_model=8, num_heads=2, num_layers=2, dim_feedforward=16,
               dropout=.1, quantiles=[.1, .5, .9], residual_prediction=False)
    model = AgriMatNetUAV(cfg).eval()
    metadata = dict(observations=[dict(date=date, season='2025')
                                 for date in ['2025-03-01', '2025-03-04', '2025-04-13']])
    items = [make_item(metadata, np.array([[.3], [np.nan], [.5]], dtype='float32'),
                       [0, 1, 2], '2025-05-02'),
             make_item(metadata, np.array([[.4], [.45]], dtype='float32'),
                       [0, 1], '2025-05-02')]
    batch = collate(items)
    initial = state_digest(model)
    forward = model.forward.__func__
    with torch.no_grad():
        ordinary = model(batch)
    with HistoricalAttentionObserver(model) as observer:
        with torch.no_grad():
            observed = model(batch)
        weights = observer.arrays()
        assert weights.shape == (2, 2, 2, 3, 3)  # layer,batch,head,query,key
        assert np.all(weights[:, 0, :, :, 1] == 0)  # Missing observation.
        assert np.all(weights[:, 1, :, :, 2] == 0)  # Padding.
        torch.testing.assert_close(observed, ordinary, atol=2e-6, rtol=1e-5)
    assert state_digest(model) == initial
    assert model.forward.__func__ is forward
    assert all(not layer.self_attn._forward_pre_hooks and not layer.self_attn._forward_hooks
               for layer in model.history_encoder.layers)


def test_date_aggregation_corrects_uniform_history_length_and_support():
    accumulator = AttentionAccumulator(1, 2, 3)
    first = np.zeros((1, 2, 3, 3))
    first[:, :, :2, :2] = .5
    accumulator.add(first, [0, 1, -1], np.array([True, True, False]))
    accumulator.add(np.full((1, 2, 3, 3), 1/3), [0, 1, 2], np.array([True]*3))
    np.testing.assert_equal(accumulator.key_count, [2, 2, 1])
    np.testing.assert_equal(accumulator.pair_count, [[2, 2, 1], [2, 2, 1], [1, 1, 1]])
    metadata = dict(observations=[dict(date=f'2025-03-0{i+1}', season='a') for i in range(3)])
    frame = accumulator.date_frame(metadata, 42)
    np.testing.assert_allclose(frame.uniform_relative_attention, 1)
    assert accumulator.samples == 2


def test_input_audit_detects_changes_in_temporary_fixture_only(tmp_path):
    original = tmp_path / 'config.json'
    original.write_text('{"unchanged": true}')
    before = protected_inventory(tmp_path)
    (tmp_path / 'analysis_results').mkdir()
    (tmp_path / 'analysis_results/result.json').write_text('{}')
    assert_inventory_unchanged(before, protected_inventory(tmp_path))
    original.write_text('{"unchanged": false}')
    with pytest.raises(RuntimeError, match='config.json'):
        assert_inventory_unchanged(before, protected_inventory(tmp_path))
