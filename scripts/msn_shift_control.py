"""Paired zero-perturbation correction for the CREA relation-shift study.

Author: Jia Rong. Reuses completed experiments; no training or forecasting.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'results/final_v1_current/analysis'


def analyze(rows):
    output = []
    for dataset in ['pems04','pems07','metr_la']:
        zero = {r['seed']:r['prior_expert_weight_drop'] for r in rows
                if r['dataset']==dataset and r['requested_endpoint_change_fraction']==0}
        assert len(zero)==5
        for fraction in [.25,.5,.75,1.]:
            group = [r for r in rows if r['dataset']==dataset and r['requested_endpoint_change_fraction']==fraction]
            assert len(group)==5 and {r['seed'] for r in group}==set(zero)
            values = [100*(r['prior_expert_weight_drop']-zero[r['seed']]) for r in group]
            output.append(dict(dataset=dataset,requested_fraction=fraction,
                mean_extra_drop_pp=float(np.mean(values)),sd_extra_drop_pp=float(np.std(values,ddof=1)),
                positive_percent=float(100*np.mean(np.array(values)>0)),
                per_initialization=[dict(seed=r['seed'],extra_drop_pp=v) for r,v in zip(group,values)]))
    return dict(definition='100 * [(first-half minus second-half mean R weight under perturbation) - (same difference without perturbation)], paired by dataset and initialization.',
                unit='percentage points',rows=output)


if __name__ == '__main__':
    rows=json.loads((DATA/'prior_reliability_shift/final_v1_prior_reliability_shift.json').read_text())['rows']
    result=analyze(rows)
    path=DATA/'msn_shift_control.json'
    path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(path),'conditions':len(result['rows']),
                      'all_initialization_level_differences_positive':all(r['positive_percent']==100 for r in result['rows'])},indent=2))
