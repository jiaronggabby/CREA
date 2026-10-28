"""Verify local utility and rebuild the effective-reliance figure without training.

Author: Jia Rong
Run extraction on the experiment host; transfer only its compact summary and
block-averaged figure data. The original predictions and checkpoints are read-only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SEEDS = (42, 3407, 101, 2026, 777)
DATASETS = ("pems03", "pems04", "pems07", "pems08", "metr_la")


def block_mean(value, rows=180, cols=220):
    groups = np.array_split(value, rows, axis=0)
    temporal = np.stack([g.mean(axis=0) for g in groups])
    return np.stack([g.mean(axis=1) for g in np.array_split(temporal, cols, axis=1)], axis=1)


def ready_loss_prefix(losses, origins, horizon_index):
    cumulative = np.cumsum(losses, axis=1)
    count = np.searchsorted(origins, origins-int(horizon_index), side="left")
    ready = np.zeros_like(losses)
    valid = count > 0
    ready[:,valid,:] = cumulative[:,count[valid]-1,:]
    return ready


def weights_from_losses(initial, ready, eta):
    initial = initial / initial.sum(axis=0, keepdims=True)
    logits = np.log(np.maximum(initial, 1e-300))[:,None,:]-eta*ready
    logits -= logits.max(axis=0, keepdims=True)
    weights = np.exp(logits)
    return weights/weights.sum(axis=0, keepdims=True)


def extract(args):
    import torch
    sys.path.insert(0, str(args.source))
    from rpv_experiments.config import load_config, units_for_stage
    from rpv_experiments.data import load_bundle
    from rpv_experiments.runner import _datasets_for_unit
    from rpv_experiments.training import build_forecaster
    from rpv_experiments.models import PriorGate

    torch.set_num_threads(4)
    cfg, _ = load_config(args.source / "configs/experiment_final_v1.yaml")
    saved_summary = args.output / "relational_utility_verification.json"
    cells = json.loads(saved_summary.read_text())["cells"] if saved_summary.exists() else []
    for di, dataset in enumerate(() if cells else DATASETS, 1):
        for seed in SEEDS:
            suffix = Path(f"runs/E{di}/{dataset}/seed_{seed}")
            original = args.original / suffix
            metrics = json.loads((original / "metrics.json").read_text())
            td = args.predictions / suffix / "predictions/test"
            arrays = {k: np.load(td / f"{k}.npy", mmap_mode="r") for k in ("targets", "pred_all", "pred_prior")}
            for h in (3, 6, 12):
                y, a, r = [np.asarray(arrays[k][:, h-1, :], dtype=np.float64) for k in ("targets", "pred_all", "pred_prior")]
                ea, er = np.abs(a-y), np.abs(r-y)
                cells.append(dict(dataset=dataset, seed=seed, horizon=h,
                                  mae_a=float(ea.mean()), mae_r=float(er.mean()),
                                  local_r_win_percent=float(100*np.mean(er < ea)),
                                  ties_percent=float(100*np.mean(er == ea)),
                                  matched_label_percent=100*metrics["horizons"][str(h)]["gates"]["loss_gate"]["prevalence"]))
            print(f"Verified A/R {dataset} {seed}", flush=True)
    inferior = [r for r in cells if r["mae_r"] > r["mae_a"]]
    summary = dict(
        source_predictions=str(args.predictions), source_checkpoints=str(args.original),
        statistic="Mean of strict pointwise R-win proportions, equal weight per dataset/seed/horizon; ties are not wins.",
        inferior_settings_percent=100*len(inferior)/len(cells),
        local_r_win_all_settings_percent=float(np.mean([r["local_r_win_percent"] for r in cells])),
        local_r_win_inferior_settings_percent=float(np.mean([r["local_r_win_percent"] for r in inferior])),
        matched_label_prevalence_percent=float(np.mean([r["matched_label_percent"] for r in cells])),
        cells=cells,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "relational_utility_verification.json").write_text(json.dumps(summary, indent=2)+"\n")

    # Only history projections and the saved gate are evaluated. The base
    # forecasters' attention blocks and prediction heads are not run.
    unit = next(u for u in units_for_stage(cfg, "formal") if u["dataset"] == "pems07")
    bundle = load_bundle(args.data_root, "pems07", cfg["split"])
    datasets = _datasets_for_unit(cfg, unit, bundle)
    atlas = []
    figure_checks = []
    for seed in SEEDS:
        suffix = Path(f"runs/E3/pems07/seed_{seed}")
        original = args.original / suffix
        m = json.loads((original / "metrics.json").read_text())
        ca = json.loads((original / "causal_aggregation.json").read_text())
        model = build_forecaster(bundle, 12, 12, cfg["training"])
        model.load_state_dict(torch.load(original/"checkpoints/all_only.pt", map_location="cpu", weights_only=False)["model"])
        model.eval()
        gate = PriorGate(64, 12, dropout=0.1, branch_feature_dim=24)
        gate.load_state_dict(torch.load(original/"checkpoints/loss_gate.pt", map_location="cpu", weights_only=False)["model"])
        gate.eval()
        alpha = float(m["static_alpha_by_horizon"]["12"])
        beta = float(m["gate_probability_calibration"]["loss_gate"]["12"])
        eta = float(ca["eta_by_horizon"]["12"])
        previous = np.full((4, bundle.variables), .25)
        scale = None
        for split in ("gate_validation", "test"):
            directory = args.predictions / suffix / "predictions" / split
            a_full = np.load(directory/"pred_all.npy", mmap_mode="r")
            r_full = np.load(directory/"pred_prior.npy", mmap_mode="r")
            y_full = np.load(directory/"targets.npy", mmap_mode="r")
            origins = np.load(directory/"origin_indices.npy")
            assert np.array_equal(origins, datasets[split].starts)
            probability = np.empty((len(origins), bundle.variables), dtype=np.float32)
            with torch.no_grad():
                for start in range(0, len(origins), 16):
                    stop = min(start+16, len(origins))
                    x = torch.from_numpy(np.stack([bundle.values[s-12:s] for s in origins[start:stop]]))
                    local = model.local_features(x)
                    global_context = local.mean(dim=1, keepdim=True).expand_as(local)
                    horizon = gate.horizon_embedding.weight[11].view(1,1,-1).expand_as(local)
                    an = (np.asarray(a_full[start:stop], dtype=np.float32)-bundle.mean[None,None,:])/bundle.std[None,None,:]
                    rn = (np.asarray(r_full[start:stop], dtype=np.float32)-bundle.mean[None,None,:])/bundle.std[None,None,:]
                    delta = torch.from_numpy(rn-an).transpose(1,2)
                    branch = torch.cat([delta, delta.abs()], dim=-1)
                    features = torch.cat([local, global_context, horizon, branch], dim=-1)
                    probability[start:stop] = torch.sigmoid(gate.network(features).squeeze(-1)).numpy()
                    if start == 0:
                        np.testing.assert_allclose(probability[start:stop], torch.sigmoid(gate(local, branch))[:,11,:].numpy(), atol=2e-7, rtol=2e-6)
            # Match the saved runner's float32 policy arithmetic.
            q = beta*probability+(1-beta)*alpha
            a = np.asarray(a_full[:,11,:], dtype=np.float32)
            r = np.asarray(r_full[:,11,:], dtype=np.float32)
            y = np.asarray(y_full[:,11,:], dtype=np.float64)
            s = alpha*r+(1-alpha)*a
            l = q*r+(1-q)*a
            predictions = np.stack([a,r,s,l]).astype(np.float64)
            if scale is None:
                scale = np.maximum(np.median(np.abs(y-np.median(y,axis=0)),axis=0)+1e-3, 1e-6)
            losses = np.clip(np.abs(predictions-y[None,:,:])/scale[None,None,:],0,10)
            ready = ready_loss_prefix(losses, origins, 11)
            weights = weights_from_losses(previous, ready, eta)
            prediction = np.sum(predictions*weights,axis=0).astype(np.float32)
            previous = weights[:,-1,:].astype(np.float32).astype(np.float64)
            if split == "test":
                effective = weights[1]+alpha*weights[2]+q*weights[3]
                np.testing.assert_allclose(prediction, (1-effective)*a+effective*r, atol=1e-4, rtol=1e-6)
                checks = dict(seed=seed, mae_rebuilt=float(np.abs(prediction-y).mean()),
                              mae_reference=ca["per_horizon"]["12"]["test_mae"]["causal_hedge"],
                              loss_gate_mae_rebuilt=float(np.abs(l-y).mean()),
                              loss_gate_mae_reference=ca["per_horizon"]["12"]["test_mae"]["loss_gate"],
                              alpha_effective_mean=float(effective.mean()), direct_r_weight_mean=float(weights[1].mean()))
                assert abs(checks["mae_rebuilt"]-checks["mae_reference"]) < 3e-5, checks
                assert abs(checks["loss_gate_mae_rebuilt"]-checks["loss_gate_mae_reference"]) < 3e-5, checks
                figure_checks.append(checks)
                atlas.append(np.stack([block_mean(np.abs(a-y)-np.abs(r-y)), block_mean(effective), block_mean(np.abs(l-y)-np.abs(prediction-y))]))
        print(f"Rebuilt effective reliance PEMS07 {seed}: {checks}", flush=True)
    np.savez_compressed(args.output/"effective_reliance_atlas.npz", panels=np.mean(atlas,axis=0), nodes=bundle.variables, origins=len(origins), horizon=12)
    summary["figure_reconstruction"] = figure_checks
    (args.output / "relational_utility_verification.json").write_text(json.dumps(summary, indent=2)+"\n")


def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, Normalize
    data = np.load(args.data)
    font_manager.findfont("Arial", fallback_to_default=False)
    plt.rcParams.update({"font.family":"Arial", "font.size":8.2, "pdf.fonttype":42,
                         "mathtext.fontset":"stix", "axes.linewidth":.6})
    # Fixed single-column canvas: typography is set at the landed paper size.
    fig = plt.figure(figsize=(3.5, 1.65))
    axes = [fig.add_axes([.15+i*(.193333+.06), .32, .193333, .36]) for i in range(3)]
    panel_labels = []
    cmap = LinearSegmentedColormap.from_list("difference", ["#4C83B6", "#F7F7F5", "#D68170"])
    for i, (ax, values) in enumerate(zip(axes, data["panels"])):
        limit=max(float(np.percentile(np.abs(values),98)),1e-6)
        step = 10**np.floor(np.log10(limit))
        limit = float(np.ceil(limit/step)*step)
        norm = Normalize(0,1) if i==1 else TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit)
        im=ax.imshow(values.T,origin="lower",aspect="auto",cmap="GnBu" if i==1 else cmap,norm=norm,interpolation="nearest",extent=(0,100,0,int(data["nodes"])-1))
        ax.set_xticks([0,50,100])
        ax.set_yticks([0,441,882])
        ax.tick_params(labelsize=7.6,length=2,pad=2)
        if i:
            ax.tick_params(axis="y",labelleft=False)
        # Author-requested label row below the images, clear of both colorbar
        # values and heatmap ticks. The letters are not drawn over the data.
        panel_labels.append(ax.annotate(chr(97+i), xy=(.5,0), xycoords="axes fraction",
                    xytext=(0,-17), textcoords="offset points", ha="center",va="top",
                    fontsize=9, fontweight="bold",clip_on=False))
        cax=fig.add_axes([ax.get_position().x0,.84,ax.get_position().width,.035])
        bar=fig.colorbar(im,cax=cax,orientation="horizontal",extend="neither" if i==1 else "both")
        bar.set_ticks([0,1] if i==1 else [-limit,limit])
        bar.ax.tick_params(labelsize=7.4,length=2,pad=1)
        bar.set_label(("A−R error",r"$\alpha^{\rm eff}$","L−CREA error")[i],fontsize=8,labelpad=4)
        bar.ax.xaxis.set_label_position("top")
    axes[0].set_ylabel("Sensor",labelpad=2)
    axes[0].set_yticks([0,441,882])
    common_label = fig.text(.5,.025,"Test-stream progress (%)",ha="center",fontsize=8.2)
    fig.canvas.draw()
    boxes=[ax.get_window_extent() for ax in axes]
    widths=[b.width for b in boxes]
    assert abs((boxes[0].x0+boxes[-1].x1)/2-fig.bbox.width/2)/fig.bbox.width < .02
    assert (max(widths)-min(widths))/np.mean(widths) < .01
    renderer = fig.canvas.get_renderer()
    occupied = [common_label] + [artist for ax in fig.axes for artist in
                [*ax.get_xticklabels(), *ax.get_yticklabels(), ax.xaxis.label, ax.yaxis.label]
                if artist.get_visible() and artist.get_text()]
    for label, ax in zip(panel_labels, axes):
        bounds = label.get_window_extent(renderer)
        assert bounds.y1 < ax.get_window_extent(renderer).y0
        assert fig.bbox.contains(bounds.x0, bounds.y0) and fig.bbox.contains(bounds.x1, bounds.y1)
        assert all(not bounds.overlaps(artist.get_window_extent(renderer)) for artist in occupied)
    fig.savefig(args.figure, dpi=600, bbox_inches=None)
    plt.close(fig)


def plot_shift(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.findfont("Arial", fallback_to_default=False)
    plt.rcParams.update({"font.family":"Arial", "font.size":8.2, "pdf.fonttype":42,
                         "mathtext.fontset":"stix", "axes.linewidth":.6,
                         "axes.spines.top":False,"axes.spines.right":False})
    rows=json.loads(args.data.read_text(encoding="utf-8"))["rows"]
    fig,axes=plt.subplots(1,2,figsize=(3.5,1.65))
    for dataset,label,color,marker in [("pems04","PEMS04","#4C78A8","o"),
                                        ("pems07","PEMS07","#5AA6A6","D"),
                                        ("metr_la","METR-LA","#D98373","s")]:
        groups=[[r for r in rows if r["dataset"]==dataset and r["requested_endpoint_change_fraction"]==level]
                for level in (0,.25,.5,.75,1)]
        assert all(len(group)==5 for group in groups)
        x=[100*np.mean([r["actual_endpoint_change_fraction"] for r in g]) for g in groups]
        for ax,key in zip(axes,["prior_expert_weight_drop","prior_harm_rate_second_half"]):
            values=[100*np.array([r[key] for r in g]) for g in groups]
            ax.errorbar(x,[v.mean() for v in values],yerr=[v.std(ddof=1) for v in values],
                        color=color,marker=marker,markersize=2.8,linewidth=.85,capsize=1.5,
                        elinewidth=.55,label=label)
    axes[0].axhline(0,color="#8A95A5",linewidth=.6)
    axes[0].set_ylabel("R-weight drop (pp)",labelpad=2)
    axes[1].set_ylabel("Origins with larger\nR MAE (%)",labelpad=2)
    for i,ax in enumerate(axes):
        ax.set_xlim(-3,103)
        ax.set_xticks([0,50,100])
        ax.tick_params(labelsize=7.6,length=2,pad=2)
        ax.grid(axis="y",color="#D9DEE7",linewidth=.45)
        ax.annotate(f"({chr(97+i)})",xy=(0,1),xycoords="axes fraction",
                    xytext=(-8,4),textcoords="offset points",ha="right",va="bottom",
                    fontsize=8,fontweight="bold",clip_on=False)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.52,1.015),ncol=3,
               frameon=False,fontsize=7.5,columnspacing=.8,handlelength=1.2,handletextpad=.3)
    fig.text(.53,.025,"Realized endpoint change (%)",ha="center",fontsize=8.2)
    fig.subplots_adjust(left=.13,right=.985,bottom=.25,top=.76,wspace=.76)
    fig.savefig(args.figure,dpi=600,bbox_inches=None)
    plt.close(fig)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=["extract","plot","plot-shift"])
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--source",type=Path,default=root)
    parser.add_argument("--original",type=Path,default=root/"reports/selective_prior_injection_final_v1")
    parser.add_argument("--predictions",type=Path,default=root/"reports/per_node_static")
    parser.add_argument("--data-root",type=Path,default=root/"data/processed")
    parser.add_argument("--output",type=Path,default=root/"reports/msn_camera_ready")
    parser.add_argument("--data",type=Path)
    parser.add_argument("--figure",type=Path)
    args=parser.parse_args()
    {"extract":extract,"plot":plot,"plot-shift":plot_shift}[args.mode](args)
