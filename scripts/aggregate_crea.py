"""Apply CREA to completed forecaster runs. Author: Jia Rong."""
import argparse
from pathlib import Path

from rpv_experiments.causal_aggregate import aggregate_unit
from rpv_experiments.config import load_config, resolved_output_root, seeds_for_stage, units_for_stage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/experiment_final_v1.yaml')
    args = parser.parse_args()
    cfg, _ = load_config(args.config)
    root = resolved_output_root(cfg)
    for unit in units_for_stage(cfg, 'formal'):
        for seed in seeds_for_stage(cfg, 'formal'):
            run = root/'runs'/unit['experiment_id']/unit['dataset']/f'seed_{seed}'
            aggregate_unit(run)
            print(run, flush=True)


if __name__ == '__main__':
    main()
