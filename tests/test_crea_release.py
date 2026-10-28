"""Focused CREA tests. Author: Jia Rong."""
import numpy as np

from rpv_experiments.causal_aggregation import causal_hedge_mix, robust_error_scale
from rpv_experiments.config import load_config, units_for_stage


def test_configuration():
    cfg, _ = load_config('configs/experiment_final_v1.yaml')
    assert len(units_for_stage(cfg, 'formal')) == 5
    assert cfg['seeds']['formal'] == [42, 3407, 101, 2026, 777]


def test_future_targets_do_not_change_earlier_forecasts():
    y = np.zeros((5, 2, 3), dtype=np.float32)
    predictions = [np.zeros_like(y), np.ones_like(y)]
    changed = y.copy()
    changed[-1] = 1000
    args = dict(origins=np.arange(10, 15), eta_by_horizon=[.1, .3],
                error_scale=np.ones((2, 3)))
    first, weights = causal_hedge_mix(y, predictions, **args)
    second, other = causal_hedge_mix(changed, predictions, **args)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(weights, other)
    np.testing.assert_allclose(weights.sum(axis=1), 1)


def test_four_policies_induce_effective_convex_coefficient():
    rng = np.random.default_rng(42)
    a, r = rng.normal(size=(2, 8, 3, 4))
    y = rng.normal(size=a.shape)
    alpha = np.array([.1, .3, .7])[None, :, None]
    gate = rng.uniform(size=a.shape)
    s, l = (1-alpha)*a+alpha*r, (1-gate)*a+gate*r
    pred, w = causal_hedge_mix(y, [a, r, s, l], np.arange(8)+12,
                              [.01, .03, .1], error_scale=np.ones((3, 4)))
    effective = w[:, 1]+alpha*w[:, 2]+gate*w[:, 3]
    assert effective.min() >= 0 and effective.max() <= 1
    np.testing.assert_allclose(pred, (1-effective)*a+effective*r, atol=2e-7)


def test_robust_scale_is_finite_for_constant_targets():
    scale = robust_error_scale(np.ones((10, 3, 4)))
    assert scale.shape == (3, 4)
    assert np.isfinite(scale).all() and np.all(scale > 0)


def test_relation_shift_correction_pairs_the_same_initialization():
    from scripts.msn_shift_control import analyze

    rows = []
    for dataset in ['pems04', 'pems07', 'metr_la']:
        for seed in range(5):
            natural_drop = 0.001 * seed
            for fraction in [0., .25, .5, .75, 1.]:
                rows.append(dict(dataset=dataset, seed=seed,
                    requested_endpoint_change_fraction=fraction,
                    prior_expert_weight_drop=natural_drop + fraction * .02))
    result = analyze(list(reversed(rows)))
    assert len(result['rows']) == 12
    for row in result['rows']:
        np.testing.assert_allclose(row['mean_extra_drop_pp'],
                                   2 * row['requested_fraction'], atol=1e-12)
        np.testing.assert_allclose(row['sd_extra_drop_pp'], 0, atol=1e-12)
        assert row['positive_percent'] == 100
