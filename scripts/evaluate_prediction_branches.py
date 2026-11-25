"""Evaluate fixed prediction branches and oracle modes on matched validation scenes."""
import argparse
import io
import json
import random
import sqlite3
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compare_constant_velocity import constant_velocity
from src.data.scenes import predict
from src.models.predictor import MapAdaptiveTrajectoryPredictor
from src.utils.mode_selection import choose_history_consistent


def trajectory_errors(trajectories, future):
    if trajectories.ndim != 3 or trajectories.shape[0] < 2 or trajectories.shape[1:] != future.shape:
        raise ValueError('Expected m+2 trajectories matching the future shape')
    if not torch.isfinite(trajectories).all() or not torch.isfinite(future).all():
        raise ValueError('Non-finite trajectory values')
    distances = torch.linalg.vector_norm(trajectories-future.unsqueeze(0), dim=-1)
    ade, fde = distances.mean(dim=-1), distances[:, -1]
    return {'scene': (ade[-2].item(), fde[-2].item()), 'motion': (ade[-1].item(), fde[-1].item()),
            'oracle': (ade.min().item(), fde.min().item())}


def summarize(rows):
    if not rows: raise ValueError('No scenes evaluated')
    result = {}
    for method in ('constant_velocity', 'scene', 'motion', 'history_match', 'oracle'):
        ade = np.asarray([r[method][0] for r in rows]); fde = np.asarray([r[method][1] for r in rows])
        if not np.isfinite(ade).all() or not np.isfinite(fde).all(): raise ValueError('Non-finite metric')
        result[method] = {'mean_ADE_m': float(ade.mean()), 'mean_FDE_m': float(fde.mean()),
                          'median_ADE_m': float(np.median(ade)), 'median_FDE_m': float(np.median(fde)),
                          'miss_rate_over_2m': float(np.mean(fde > 2))}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, default=ROOT/'data/processed/scenes.sqlite')
    p.add_argument('--checkpoint', type=Path, default=ROOT/'outputs/official_full/checkpoint.pt')
    p.add_argument('--output', type=Path, default=ROOT/'outputs/official_full/branch_comparison.json')
    p.add_argument('--count', type=int, default=2000, help='Validation sample size; 0 evaluates all')
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--device', default='cpu')
    args=p.parse_args()
    if args.count < 0: p.error('count must be non-negative')
    checkpoint=torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    model=MapAdaptiveTrajectoryPredictor(agent_input_dim=4).to(args.device)
    model.load_state_dict(checkpoint['model']); model.eval()
    with sqlite3.connect(f'file:{args.cache.resolve().as_posix()}?mode=ro', uri=True) as db:
        ids=sorted(str(r[0]) for r in db.execute("SELECT id FROM scenes WHERE split='val'"))
        if not ids or len(ids) != len(set(ids)) or set(ids) != set(map(str, checkpoint['val_ids'])):
            raise ValueError('Validation IDs differ from the checkpoint split')
        random.Random(args.seed).shuffle(ids)
        if args.count: ids=ids[:args.count]
        rows=[]
        with torch.no_grad():
            for sid in ids:
                payload=db.execute("SELECT payload FROM scenes WHERE split='val' AND id=?", (sid,)).fetchone()[0]
                scene=torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
                future=scene['future']; predictions=predict(model, scene, args.device).cpu()
                errors=trajectory_errors(predictions, future)
                baseline=constant_velocity(scene['histories'][0], scene['intervals'], len(future))
                d=torch.linalg.vector_norm(baseline-future, dim=-1)
                selected=choose_history_consistent(predictions, baseline)
                selected_error=torch.linalg.vector_norm(predictions[selected]-future, dim=-1)
                rows.append({'id': sid, 'modes': len(predictions), 'selected_mode': selected,
                             'constant_velocity': (d.mean().item(),d[-1].item()),
                             'history_match': (selected_error.mean().item(),selected_error[-1].item()), **errors})
                if len(rows)%500==0: print(f'Evaluated {len(rows)}/{len(ids)} scenes', flush=True)
    report={'count': len(rows), 'seed': args.seed, 'checkpoint_epochs': checkpoint['epoch'],
            'methods': summarize(rows), 'scenes': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='scenes'}, indent=2))


if __name__=='__main__': main()
