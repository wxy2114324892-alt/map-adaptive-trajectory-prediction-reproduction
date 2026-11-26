"""Compare learned selection with baselines by observed-history movement."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_candidate_scorer import summary


def compare_groups(selector_report, baseline_report):
    results=selector_report['scenes']; baseline=baseline_report['rows']
    if len(results)!=selector_report['validation_count'] or len(baseline)!=baseline_report['overall']['count']:
        raise ValueError('Reported counts do not match scene records')
    lookup={str(r['id']):r for r in baseline}
    seen={str(r['id']) for r in results}
    if len(seen)!=len(results) or len(lookup)!=len(baseline) or seen!=set(lookup) or not seen:
        raise ValueError('Duplicate or mismatched validation scene IDs')
    groups={}
    for row in results:
        other=lookup[str(row['id'])]
        if not np.allclose(row['constant_velocity'], [other['baseline_ade_m'],other['baseline_fde_m']], atol=1e-5, rtol=1e-5):
            raise ValueError(f"Baseline metrics disagree for scene {row['id']}")
        groups.setdefault(other['movement'], []).append(row)
    output={}
    for group,rows in sorted(groups.items()):
        for row in rows:
            for method in ('constant_velocity','heuristic','learned','oracle'):
                if len(row[method])!=2 or not np.isfinite(row[method]).all() or min(row[method])<0:
                    raise ValueError('Invalid error value')
        output[group]={'count':len(rows),'methods':summary(rows),
                       'learned_better_ADE_than_baseline':float(np.mean([r['learned'][0]<r['constant_velocity'][0] for r in rows])),
                       'learned_better_FDE_than_baseline':float(np.mean([r['learned'][1]<r['constant_velocity'][1] for r in rows]))}
    return {'validation_count':len(results),'groups':output}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selector',type=Path,default=ROOT/'outputs/learned_selector_full/metrics.json')
    p.add_argument('--baseline',type=Path,default=ROOT/'outputs/official_full/constant_velocity_comparison.json')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/learned_selector_full/movement_groups.json')
    args=p.parse_args()
    result=compare_groups(json.loads(args.selector.read_text(encoding='utf-8')),json.loads(args.baseline.read_text(encoding='utf-8')))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
