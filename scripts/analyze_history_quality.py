"""Analyze saved validation errors by observed-history characteristics."""
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


def categories(quality):
    distance = quality['last4_displacement_m']
    movement = 'under_0.5m' if distance < .5 else ('0.5_to_2m' if distance < 2 else 'at_least_2m')
    angle = quality['short_long_angle_deg']
    direction = 'undefined' if angle is None else ('over_60deg' if angle > 60 else 'at_most_60deg')
    return movement, direction


def summarize(rows):
    if not rows:
        raise ValueError('No matching validation scenes')
    result = {}
    for category in ('movement', 'direction'):
        groups = {}
        for row in rows:
            groups.setdefault(row[category], []).append(row)
        result[category] = {}
        for name, group in sorted(groups.items()):
            ade = np.array([r['minADE_m'] for r in group]); fde = np.array([r['minFDE_m'] for r in group])
            result[category][name] = {'count': len(group), 'mean_minADE_m': float(ade.mean()),
                'median_minADE_m': float(np.median(ade)), 'mean_minFDE_m': float(fde.mean()),
                'miss_rate_over_2m': float((fde > 2).mean()),
                'mean_modes': float(np.mean([r['modes'] for r in group]))}
    return result


def analyze(db, report):
    scenes = report['scenes']
    if len(scenes) != report['val_count'] or not scenes:
        raise ValueError('Validation scene count mismatch or empty results')
    metrics = {str(s['id']): s for s in scenes}
    if len(metrics) != len(scenes):
        raise ValueError('Duplicate validation IDs in metrics')
    cached = {str(row[0]) for row in db.execute("SELECT id FROM scenes WHERE split='val'")}
    if set(metrics) != cached:
        raise ValueError(f'Validation IDs differ: {len(set(metrics)-cached)} missing, {len(cached-set(metrics))} extra in cache')
    rows = []
    for scene_id, payload in db.execute("SELECT id, payload FROM scenes WHERE split='val'"):
        scene = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
        q = history_quality(scene['histories'][0], scene['intervals'])
        movement, direction = categories(q)
        metric = metrics[str(scene_id)]
        values = (metric['minADE_m'], metric['minFDE_m'])
        if any(not np.isfinite(v) or v < 0 for v in values) or metric['modes'] < 1:
            raise ValueError(f'Invalid metrics for scene {scene_id}')
        rows.append({'movement': movement, 'direction': direction, **{k: metric[k] for k in ('minADE_m','minFDE_m','modes')}})
        if len(rows) % 5000 == 0: print(f'Analyzed {len(rows)} scenes', flush=True)
    return {'scene_count': len(rows), 'groups': summarize(rows)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, default=ROOT/'data/processed/scenes.sqlite')
    p.add_argument('--metrics', type=Path, default=ROOT/'outputs/official_full/metrics.json')
    p.add_argument('--output', type=Path, default=ROOT/'outputs/official_full/history_analysis.json')
    args = p.parse_args()
    report = json.loads(args.metrics.read_text(encoding='utf-8'))
    with sqlite3.connect(f'file:{args.cache.resolve().as_posix()}?mode=ro', uri=True) as db:
        result = analyze(db, report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
