"""Compare saved multimodal validation errors to a constant-velocity baseline."""
import argparse
import io
import json
import sqlite3
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.argoverse_loader import history_quality
from scripts.analyze_history_quality import categories


def constant_velocity(history, intervals, horizon, step_seconds=0.1):
    """Extrapolate the velocity over the last four observed intervals, in meters."""
    if history.ndim != 2 or history.shape[-1] != 2 or len(history) < 5:
        raise ValueError('History must have at least five XY points')
    if len(intervals) != len(history)-1 or not torch.isfinite(intervals).all() or torch.any(intervals <= 0):
        raise ValueError('Expected one positive finite interval per observed step')
    if horizon < 1 or not np.isfinite(step_seconds) or step_seconds <= 0 or not torch.isfinite(history).all():
        raise ValueError('Invalid horizon, step duration or history')
    velocity = (history[-1] - history[-5]) / intervals[-4:].sum()
    times = torch.arange(1, horizon+1, dtype=history.dtype, device=history.device) * step_seconds
    return history[-1] + times[:, None] * velocity


def summarize(rows):
    if not rows: raise ValueError('No validation scenes')
    result = {'count': len(rows)}
    for key in ('baseline_ade_m', 'baseline_fde_m', 'model_minADE_m', 'model_minFDE_m'):
        values = np.asarray([r[key] for r in rows], dtype=float)
        if not np.isfinite(values).all() or (values < 0).any(): raise ValueError(f'Invalid {key}')
        result[key] = {'mean': float(values.mean()), 'median': float(np.median(values))}
    for prefix in ('baseline', 'model'):
        key = 'baseline_fde_m' if prefix == 'baseline' else 'model_minFDE_m'
        result[prefix+'_miss_rate_over_2m'] = float(np.mean([r[key] > 2 for r in rows]))
    result['model_lower_ADE_fraction'] = float(np.mean([r['model_minADE_m'] < r['baseline_ade_m'] for r in rows]))
    result['model_lower_FDE_fraction'] = float(np.mean([r['model_minFDE_m'] < r['baseline_fde_m'] for r in rows]))
    return result


def compare(db, report):
    scenes = report['scenes']
    if not scenes or len(scenes) != report['val_count']: raise ValueError('Validation count mismatch')
    metrics = {str(s['id']): s for s in scenes}
    if len(metrics) != len(scenes): raise ValueError('Duplicate validation IDs')
    ids = {str(r[0]) for r in db.execute("SELECT id FROM scenes WHERE split='val'")}
    if ids != set(metrics): raise ValueError('Cache and metric validation IDs differ')
    rows = []
    for scene_id, payload in db.execute("SELECT id,payload FROM scenes WHERE split='val'"):
        scene = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
        history, future, dt = scene['histories'][0], scene['future'], scene['intervals']
        if future.ndim != 2 or future.shape[1] != 2 or not torch.isfinite(future).all():
            raise ValueError(f'Invalid future for {scene_id}')
        prediction = constant_velocity(history, dt, len(future))
        errors = torch.linalg.vector_norm(prediction-future, dim=-1)
        metric = metrics[str(scene_id)]
        movement, _ = categories(history_quality(history, dt))
        rows.append({'id': str(scene_id), 'movement': movement, 'baseline_ade_m': errors.mean().item(),
                     'baseline_fde_m': errors[-1].item(), 'model_minADE_m': metric['minADE_m'], 'model_minFDE_m': metric['minFDE_m']})
        if len(rows) % 5000 == 0: print(f'Compared {len(rows)} scenes', flush=True)
    groups = {name: summarize([r for r in rows if r['movement'] == name]) for name in sorted({r['movement'] for r in rows})}
    return {'overall': summarize(rows), 'movement_groups': groups, 'future_step_seconds': .1,
            'velocity_history_intervals': 4, 'rows': rows}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, default=ROOT/'data/processed/scenes.sqlite')
    p.add_argument('--metrics', type=Path, default=ROOT/'outputs/official_full/metrics.json')
    p.add_argument('--output', type=Path, default=ROOT/'outputs/official_full/constant_velocity_comparison.json')
    args=p.parse_args()
    report=json.loads(args.metrics.read_text(encoding='utf-8'))
    with sqlite3.connect(f'file:{args.cache.resolve().as_posix()}?mode=ro', uri=True) as db: result=compare(db, report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k != 'rows'}, indent=2))


if __name__ == '__main__': main()
