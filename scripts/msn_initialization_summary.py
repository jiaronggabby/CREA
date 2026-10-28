"""Summarize CREA's saved metrics by complete random initialization.

Author: Jia Rong. No prediction arrays, model fitting, or network access.
"""
import csv
import json
from math import sqrt
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "results/final_v1_current/analysis"
SEEDS = [42, 3407, 101, 2026, 777]
REFERENCE_SEEDS = SEEDS[:3]
DATASETS = ["pems03", "pems04", "pems07", "pems08", "metr_la"]
HORIZONS = [3, 6, 12]
# Two-sided 95% Student-t critical value, four degrees of freedom.
T95_DF4 = 2.7764451051977934


def read_rows(name):
    with (ANALYSIS / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def initialization_interval(values):
    if len(values) != 5:
        raise ValueError("Five independent initialization summaries are required")
    average, sd = mean(values), stdev(values)
    half_width = T95_DF4 * sd / sqrt(5)
    return dict(mean=average, sd=sd, ci95=[average-half_width, average+half_width],
                per_initialization=values)


def analyze(rows, ablations):
    expected = {(d, s, h) for d in DATASETS for s in SEEDS for h in HORIZONS}
    index = {(r["dataset"], int(r["seed"]), int(r["horizon"])): r for r in rows}
    if len(rows) != len(expected) or set(index) != expected:
        raise ValueError("Incomplete or duplicated dataset/seed/horizon metrics")
    gains = {}
    for baseline in ["static_mix", "loss_gate"]:
        values = [mean(100 * (1-float(index[d, s, h]["mae_causal_hedge"])
                               / float(index[d, s, h]["mae_"+baseline]))
                       for d in DATASETS for h in HORIZONS) for s in SEEDS]
        gains[baseline] = initialization_interval(values)
    architecture = {}
    for d in DATASETS:
        values = [mean(float(index[d, s, h]["mae_causal_hedge"])
                       for h in HORIZONS) for s in REFERENCE_SEEDS]
        architecture[d] = dict(mean=mean(values), sd=stdev(values),
                               per_initialization=values)
    variants = {(r["dataset"], int(r["seed"]), int(r["horizon"]), r["variant"]):
                float(r["mae"]) for r in ablations}
    recovery = []
    for s in SEEDS:
        degradations = {}
        for variant in ["frozen_validation_weights", "no_delayed_feedback_diagnostic"]:
            degradations[variant] = mean(
                variants[d, s, h, variant] / variants[d, s, h, "causal_hedge"] - 1
                for d in DATASETS for h in HORIZONS)
        fixed = degradations["frozen_validation_weights"]
        zero = degradations["no_delayed_feedback_diagnostic"]
        recovery.append(100 * fixed / (fixed-zero))
    return dict(seeds=SEEDS, architecture_reference_seeds=REFERENCE_SEEDS,
                unit="one initialization, jointly averaging five datasets and three horizons",
                interval="paired Student-t interval across five initialization-level gains (df=4)",
                inference_scope="initialization variation conditional on the fixed benchmark test streams",
                gains_percent=gains, architecture_crea_mae=architecture,
                feedback_recovery_percent=initialization_interval(recovery))


def main():
    result = analyze(read_rows("final_v1_per_dataset_horizon_seed.csv"),
                     read_rows("final_v1_ablation_per_dataset_horizon_seed.csv"))
    destination = ANALYSIS / "msn_initialization_summary.json"
    destination.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
