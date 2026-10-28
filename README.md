# CREA

Code for **CREA: Causal Relational Expert Aggregation for Traffic Forecasting** (IEEE MSN 2026), by Jia Rong and Guanchao Tong.

CREA combines two frozen traffic forecasters: unrestricted sensor interaction and interaction constrained by a supplied graph. Their direct predictions, static mixing, and input-conditioned mixing form four allocation policies. Observed forecast errors update policy trust separately for each sensor and horizon.

## Installation

Use Python 3.10 or newer and install the PyTorch build appropriate for your CUDA environment, then run from the repository root:

```bash
pip install -e ".[test,figures]"
pytest -q tests -p no:cacheprovider --basetemp=tmp/pytest-runtime
```

## Data

The experiments use PEMS03, PEMS04, PEMS07, PEMS08, and METR-LA. Data remain subject to their original providers' terms and are not included in this repository.

```bash
crea prepare-data --stage formal --download
crea prepare-controls --stage formal
```

The PEMS download links are configured in `rpv_data/registry.py`. For METR-LA, obtain `metr-la.h5` and `adj_mx.pkl` from the [DCRNN data release](https://github.com/liyaguang/DCRNN), place them in `data/raw/metr_la/`, and rerun data preparation. `RPV_DATA_ROOT` can point to another data directory.

Each series uses chronological intervals of 60%/10%/10%/10%/10% for forecaster training, forecaster validation, gate training, gate validation, and testing. History and maximum forecast length are both 12. Forecaster training uses at most 4,096 windows per epoch. The configuration specifies the five initialization seeds and all selection settings.

## Training and evaluation

```bash
crea plan --stage formal
crea preflight --stage formal
crea run --stage formal --resume
python scripts/aggregate_crea.py
crea final-stats
crea final-ablations
crea final-standard-reporting
crea final-shift --datasets pems04,pems07,metr_la
crea final-efficiency
python scripts/msn_initialization_summary.py
```

The training command fits the two forecasters and the input-conditioned gate. `aggregate_crea.py` selects the update rates on gate-validation predictions and starts testing from the normalized weights used at the final validation forecast origin. Pending validation losses are discarded at this boundary; subsequent updates use only test-origin feedback as its targets arrive. Completed runs are reused with `--resume`. Use `--datasets pems04 --seed 42` on the training command to run one dataset and initialization; the full result summaries require all five datasets and seeds.

Outputs are written to `reports/selective_prior_injection_final_v1/`. For a single completed run, aggregation can also be called directly:

```bash
python -m rpv_experiments.causal_aggregate reports/selective_prior_injection_final_v1/runs/E2/pems04/seed_42
```

In the stored arrays, an origin is the index of the first target, not the last observed sample. The delayed update in `causal_aggregation.py` uses that convention. The paper indexes forecasts by the last observed sample; the two conventions differ by one.

## Analyses and figures

`scripts/per_node_static_analysis.py` computes validation-selected per-node static mixtures, dense global mixtures, and the pointwise A/R oracle from frozen checkpoints. Its command-line arguments specify the source tree, completed report directory, output directory, dataset, and seed:

```bash
python scripts/per_node_static_analysis.py --help
```

The five paper figures can be redrawn from the included compact plotting data, without training or downloading prediction arrays:

```bash
python scripts/redraw_msn_figures.py
```

The plotting script uses Arial, which must be installed locally. Figure 4 averages five initializations in 180 chronological bins and 220 consecutive-sensor-index bins; the effective-relational-reliance color scale stays at [0,1]. Figure 5 is a two-policy A/R diagnostic. Its weight-drop panel subtracts the unperturbed first-to-second-half difference within each initialization; the horizontal axis reports the fraction of original non-self edges replaced. The paired correction can also be computed with `python scripts/msn_shift_control.py`.

The saved-prediction extraction tools are `scripts/msn_relational_reliance.py` and `scripts/msn_plot_data.py`. Their path arguments allow regenerated predictions and trained checkpoints to be evaluated without retraining.

`scripts/msn_initialization_summary.py` summarizes the manuscript's paired 95% Student-t intervals from five complete initialization-level gains, keeping all datasets and horizons together. It also computes the CREA row for the common three-initialization architecture table and the initialization-level feedback-recovery ratios. These are the manuscript's reported intervals; the temporal block summaries produced by `final-stats` are a separate analysis. The included CSV summaries reproduce the reported intervals and CREA row without prediction arrays or training.

## Architectural comparisons

The adapters import the original model implementations from an explicitly supplied upstream directory. These implementations are not redistributed here; retain their original licenses.

- `scripts/run_external_same_split.py`: [STID](https://github.com/zezhishao/STID) and [STAEformer](https://github.com/XDZhelheim/STAEformer).
- `scripts/run_external_graph_baseline.py`: [DCRNN](https://github.com/liyaguang/DCRNN), [Graph WaveNet](https://github.com/nnzhan/Graph-WaveNet), [MTGNN](https://github.com/nnzhan/MTGNN), [AGCRN](https://github.com/LeiBAI/AGCRN), and [MegaCRN](https://github.com/deepkashiwa20/MegaCRN).
- `scripts/run_gc_moe_closest_same_split.py`: [GC-MoE](https://github.com/Ahghaffari/gc_moe).

Run an adapter with `--help` for its data, training, and upstream-path arguments. Architectural references use at most 1,024 training and validation windows; CREA experts use 4,096 training windows per epoch and full validation. The reference table uses seeds 42, 3407, and 101 for every method, including CREA. Main allocation comparisons use all five initializations and the same frozen A/R predictions. All methods share raw series, chronological partitions, horizons, and test origins. CREA uses per-node training normalization; external adapters use global training normalization and model-specific inputs. The architectural table therefore reports results under the stated settings rather than an equal-budget ranking.

## Citation

```bibtex
@inproceedings{rong2026crea,
  title={CREA: Causal Relational Expert Aggregation for Traffic Forecasting},
  author={Rong, Jia and Tong, Guanchao},
  booktitle={IEEE International Conference on Mobility, Sensing and Networking (MSN)},
  year={2026}
}
```

## License

Original code is released under the [MIT License](LICENSE). External datasets and upstream baseline implementations retain their own terms.
