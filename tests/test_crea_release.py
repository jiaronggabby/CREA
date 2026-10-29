"""Focused CREA tests. Author: Jia Rong."""
from pathlib import Path
import re

import numpy as np
import pytest

from rpv_experiments.causal_aggregation import causal_hedge_mix, robust_error_scale
from rpv_experiments.config import load_config, units_for_stage


CREA_COMMANDS = (
    "plan", "preflight", "prepare-data", "prepare-controls", "run",
    "final-stats", "final-ablations", "final-standard-reporting",
    "final-shift", "final-efficiency",
)


@pytest.fixture
def crea_cli(monkeypatch):
    # Support both this source checkout and the exported public tree.
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "publication" / "crea"))
    import crea

    return crea


@pytest.mark.parametrize("command", CREA_COMMANDS)
def test_public_cli_allows_documented_commands(crea_cli, command):
    args = crea_cli.build_parser().parse_args([command])
    assert args.command == command
    assert args.config == "configs/experiment_final_v1.yaml"


@pytest.mark.parametrize("command", [
    "summarize", "analyze", "decision", "mechanism-decision", "lock", "unknown",
])
def test_public_cli_rejects_legacy_commands_before_dispatch(
    crea_cli, monkeypatch, capsys, command,
):
    import rpv_experiments.cli as shared_cli

    monkeypatch.setattr(shared_cli, "main", lambda: pytest.fail("unexpected dispatch"))
    monkeypatch.setattr("sys.argv", ["crea", command])
    with pytest.raises(SystemExit) as error:
        crea_cli.main()
    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert "usage: crea" in stderr
    assert f"invalid choice: '{command}'" in stderr


@pytest.mark.parametrize("config_options, expected", [
    ([], "configs/experiment_final_v1.yaml"),
    (["--config", "custom config.yaml"], "custom config.yaml"),
    (["--config=custom.yaml"], "custom.yaml"),
    (["--config", "analyze"], "analyze"),
    (["--conf", "custom.yaml"], "custom.yaml"),
])
def test_public_cli_preserves_config_and_run_options(
    crea_cli, monkeypatch, config_options, expected,
):
    import sys
    import rpv_experiments.cli as shared_cli

    parsed = []
    monkeypatch.setattr(
        shared_cli, "main",
        lambda: parsed.append(shared_cli.build_parser().parse_args()),
    )
    argv = ["crea", *config_options, "run", "--stage", "formal",
            "--datasets", "pems04", "--experiments", "E2", "--seed", "42",
            "--resume", "--limit", "1", "--shard-index", "1", "--num-shards", "2"]
    monkeypatch.setattr(sys, "argv", argv)
    crea_cli.main()
    assert sys.argv is argv
    assert len(parsed) == 1
    args = parsed[0]
    assert args.config == expected
    assert args.command == "run" and args.stage == "formal"
    assert args.datasets == "pems04" and args.experiments == "E2"
    assert args.seed == 42 and args.resume and args.limit == 1
    assert args.shard_index == 1 and args.num_shards == 2


@pytest.mark.parametrize("options", [["--help"], ["run", "--help"]])
def test_public_cli_help_is_scoped(crea_cli, monkeypatch, capsys, options):
    import rpv_experiments.cli as shared_cli

    monkeypatch.setattr(shared_cli, "main", lambda: pytest.fail("unexpected dispatch"))
    monkeypatch.setattr("sys.argv", ["crea", *options])
    with pytest.raises(SystemExit) as error:
        crea_cli.main()
    assert error.value.code == 0
    stdout = capsys.readouterr().out
    assert "usage: crea" in stdout
    assert "rpv-experiments" not in stdout
    for command in ("summarize", "analyze", "decision", "mechanism-decision", "lock"):
        assert re.search(rf"\b{re.escape(command)}\b", stdout) is None
    if options == ["--help"]:
        for command in CREA_COMMANDS:
            assert command in stdout
        assert "--config" in stdout
    else:
        assert "--resume" in stdout and "--seed" in stdout


def test_public_cli_does_not_restrict_shared_parser(crea_cli):
    from rpv_experiments.cli import build_parser

    crea_cli.build_parser()
    assert build_parser().parse_args(["analyze"]).command == "analyze"


def test_public_cli_restores_argv_when_dispatch_fails(crea_cli, monkeypatch):
    import sys
    import rpv_experiments.cli as shared_cli

    def fail():
        raise RuntimeError("dispatcher failure")

    argv = ["crea", "plan"]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(shared_cli, "main", fail)
    with pytest.raises(RuntimeError, match="dispatcher failure"):
        crea_cli.main()
    assert sys.argv is argv


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


def test_raw_mape_targets_use_data_root_not_report_location(tmp_path, monkeypatch):
    import rpv_experiments.final_v1_stats as stats

    processed = tmp_path / "mounted-data" / "processed"
    dataset = processed / "pems04"
    dataset.mkdir(parents=True)
    raw = np.arange(12, dtype=np.float64).reshape(6, 2)
    np.savez(dataset / "series.npz", raw_values=raw,
             observed_mask=np.ones_like(raw, dtype=bool))
    run = tmp_path / "separate-reports" / "run"
    predictions = run / "predictions" / "test"
    predictions.mkdir(parents=True)
    np.save(predictions / "origin_indices.npy", np.array([3, 4]))
    target = raw[np.array([3, 4])[:, None] + np.arange(2)[None, :]]
    monkeypatch.setattr(stats, "PROCESSED_ROOT", processed)
    actual = stats._exact_raw_target(
        tmp_path / "unrelated-report-location", run,
        {"dataset": "pems04", "split_boundaries": {"forecaster_train": [0, 3]}},
        target.astype(np.float32),
    )
    np.testing.assert_array_equal(actual, target)


def test_test_stream_starts_from_last_validation_origin_without_tail_flush():
    validation = np.zeros((5, 2, 1), dtype=np.float32)
    experts = [np.zeros_like(validation), np.ones_like(validation)]
    scale = np.ones((2, 1))
    _, weights = causal_hedge_mix(validation, experts, np.arange(10, 15),
                                  [.1, .3], error_scale=scale)
    changed_tail = validation.copy()
    changed_tail[-1] = 100
    # This feedback is not yet eligible at the last validation origin.
    _, changed_weights = causal_hedge_mix(changed_tail, experts, np.arange(10, 15),
                                          [.1, .3], error_scale=scale)
    np.testing.assert_array_equal(weights[-1], changed_weights[-1])
    test_targets = np.zeros((3, 2, 1), dtype=np.float32)
    test_experts = [np.zeros_like(test_targets), np.ones_like(test_targets)]
    _, test_weights = causal_hedge_mix(test_targets, test_experts,
        np.arange(30, 33), [.1, .3], error_scale=scale, initial_weights=weights[-1])
    np.testing.assert_allclose(test_weights[0], weights[-1], atol=1e-7)
    # Test-origin feedback subsequently changes trust, separately for each lead.
    assert test_weights[1, 1, 0, 0] < test_weights[0, 1, 0, 0]
    np.testing.assert_array_equal(test_weights[1, :, 1], test_weights[0, :, 1])


def test_initialization_interval_uses_complete_paired_summaries():
    from scripts.msn_initialization_summary import initialization_interval

    result = initialization_interval([1., 2., 3., 4., 5.])
    assert result['mean'] == 3.
    np.testing.assert_allclose(result['sd'], np.sqrt(2.5))
    half = 2.7764451051977934 * np.sqrt(2.5 / 5)
    np.testing.assert_allclose(result['ci95'], [3-half, 3+half])


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
