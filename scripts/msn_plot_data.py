"""Recover compact, data-derived MSN figure summaries on the experiment host.

Author: Jia Rong. No training or base-forecast inference is performed.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from msn_relational_reliance import SEEDS, ready_loss_prefix, weights_from_losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    sys.path.insert(0, str(args.source))
    from rpv_experiments.config import load_config, units_for_stage
    from rpv_experiments.data import load_bundle
    from rpv_experiments.runner import _datasets_for_unit
    from rpv_experiments.training import build_forecaster
    from rpv_experiments.models import PriorGate
    torch.set_num_threads(4)
    cfg, _ = load_config(args.source/'configs/experiment_final_v1.yaml')
    unit = next(u for u in units_for_stage(cfg, 'formal') if u['dataset'] == 'pems07')
    bundle = load_bundle(args.data_root, 'pems07', cfg['split'])
    datasets = _datasets_for_unit(cfg, unit, bundle)
    collected = {h: [] for h in (3, 6, 12)}
    checks = []
    for seed in SEEDS:
        suffix = Path(f'runs/E3/pems07/seed_{seed}')
        original = args.original/suffix
        metrics = json.loads((original/'metrics.json').read_text())
        ca = json.loads((original/'causal_aggregation.json').read_text())
        model = build_forecaster(bundle, 12, 12, cfg['training'])
        model.load_state_dict(torch.load(original/'checkpoints/all_only.pt', map_location='cpu', weights_only=False)['model'])
        model.eval()
        gate = PriorGate(64, 12, dropout=.1, branch_feature_dim=24)
        gate.load_state_dict(torch.load(original/'checkpoints/loss_gate.pt', map_location='cpu', weights_only=False)['model'])
        gate.eval()
        previous = {h: np.full((4, bundle.variables), .25) for h in collected}
        scales = {}
        for split in ('gate_validation', 'test'):
            directory = args.predictions/suffix/'predictions'/split
            a = np.load(directory/'pred_all.npy', mmap_mode='r')
            r = np.load(directory/'pred_prior.npy', mmap_mode='r')
            y = np.load(directory/'targets.npy', mmap_mode='r')
            origins = np.load(directory/'origin_indices.npy')
            assert np.array_equal(origins, datasets[split].starts)
            probabilities = np.empty((len(origins), 3, bundle.variables), dtype=np.float32)
            with torch.no_grad():
                for start in range(0, len(origins), 16):
                    stop = min(start+16, len(origins))
                    x = torch.from_numpy(np.stack([bundle.values[s-12:s] for s in origins[start:stop]]))
                    local = model.local_features(x)
                    an = (np.asarray(a[start:stop], dtype=np.float32)-bundle.mean[None,None,:])/bundle.std[None,None,:]
                    rn = (np.asarray(r[start:stop], dtype=np.float32)-bundle.mean[None,None,:])/bundle.std[None,None,:]
                    delta = torch.from_numpy(rn-an).transpose(1, 2)
                    branch = torch.cat([delta, delta.abs()], dim=-1)
                    probabilities[start:stop] = torch.sigmoid(gate(local, branch))[:,[2,5,11],:].numpy()
            for hi, h in enumerate(collected):
                alpha = float(metrics['static_alpha_by_horizon'][str(h)])
                beta = float(metrics['gate_probability_calibration']['loss_gate'][str(h)])
                eta = float(ca['eta_by_horizon'][str(h)])
                ah, rh = np.asarray(a[:,h-1,:], dtype=np.float32), np.asarray(r[:,h-1,:], dtype=np.float32)
                yh = np.asarray(y[:,h-1,:], dtype=np.float64)
                q = beta*probabilities[:,hi,:]+(1-beta)*alpha
                sh, lh = alpha*rh+(1-alpha)*ah, q*rh+(1-q)*ah
                predictions = np.stack([ah,rh,sh,lh]).astype(np.float64)
                if split == 'gate_validation':
                    scales[h] = np.maximum(np.median(np.abs(yh-np.median(yh,axis=0)),axis=0)+1e-3,1e-6)
                losses = np.clip(np.abs(predictions-yh[None,:,:])/scales[h][None,None,:],0,10)
                weights = weights_from_losses(previous[h],ready_loss_prefix(losses,origins,h-1),eta)
                previous[h] = weights[:,-1,:].astype(np.float32).astype(np.float64)
                if split == 'test':
                    pred = (predictions*weights).sum(axis=0).astype(np.float32)
                    mae = float(np.abs(pred-yh).mean())
                    reference = ca['per_horizon'][str(h)]['test_mae']['causal_hedge']
                    assert abs(mae-reference) < 3e-5, (seed,h,mae,reference)
                    checks.append(dict(seed=seed,horizon=h,mae=mae,reference=reference))
                    # Match the earlier figure's eligible-history covariate.
                    advantage = np.abs(ah-np.asarray(y[:,h-1,:],dtype=np.float32))-np.abs(rh-np.asarray(y[:,h-1,:],dtype=np.float32))
                    prefix = np.concatenate([np.zeros((1,bundle.variables)),np.cumsum(advantage,axis=0,dtype=np.float64)])
                    count = np.arange(len(origins))-(h-1)
                    valid = count > 0
                    past = prefix[np.maximum(count,0)]/np.maximum(count,1)[:,None]
                    collected[h].append((past[valid].ravel(),weights[1,valid,:].ravel()))
            print(f'{seed} {split} complete',flush=True)
    arrays = {}
    for h, pairs in collected.items():
        lo = min(np.percentile(x,1) for x,w in pairs)
        hi = max(np.percentile(x,99) for x,w in pairs)
        edges = np.linspace(lo,hi,43)
        means = []
        density = np.zeros((80,35),dtype=np.float64)
        for x,w in pairs:
            index = np.clip(np.digitize(x,edges)-1,0,41)
            count = np.bincount(index,minlength=42)
            means.append(np.divide(np.bincount(index,weights=w,minlength=42),count,out=np.full(42,np.nan),where=count>0))
            # Deterministic equal-sized background samples, as in the old figure.
            take = np.linspace(0,len(x)-1,min(18000,len(x)),dtype=int)
            hist, xe, ye = np.histogram2d(x[take],w[take],bins=(80,35),range=((lo,hi),(0,1)))
            density += hist
        arrays.update({f'centers_{h}':(edges[1:]+edges[:-1])/2,
                       f'mean_{h}':np.nanmean(means,axis=0),
                       f'sd_{h}':np.nanstd(means,axis=0,ddof=1),
                       f'density_{h}':density,f'xedges_{h}':xe,f'yedges_{h}':ye})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,**arrays)
    print(json.dumps({'checks':checks,'output':str(args.output),'bytes':args.output.stat().st_size}),flush=True)


if __name__ == '__main__':
    main()
