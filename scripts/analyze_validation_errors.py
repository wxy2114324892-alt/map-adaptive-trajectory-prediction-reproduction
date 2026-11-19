"""Summarize scene-level validation errors without rerunning model inference."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def summarize(report, miss_threshold=2.0):
    """Compute percentiles and endpoint miss rate from saved per-scene errors."""
    scenes = report['scenes']
    if not scenes:
        raise ValueError('No validation scenes in metrics file')
    if len(scenes) != report['val_count']:
        raise ValueError('Scene count does not match the reported validation count')
    if not np.isfinite(miss_threshold) or miss_threshold <= 0:
        raise ValueError('Miss threshold must be positive and finite')
    result = {'count': len(scenes), 'miss_threshold_m': float(miss_threshold)}
    for key in ('minADE_m', 'minFDE_m'):
        values = np.asarray([s[key] for s in scenes], dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f'Invalid {key} values')
        result[key] = {'mean': float(np.mean(values)), 'median': float(np.median(values)),
                       'p90': float(np.percentile(values, 90)), 'p95': float(np.percentile(values, 95))}
    ends = np.asarray([s['minFDE_m'] for s in scenes])
    result['endpoint_miss_rate'] = float(np.mean(ends > miss_threshold))
    result['modes_min'] = min(s['modes'] for s in scenes)
    result['modes_max'] = max(s['modes'] for s in scenes)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', type=Path, default=ROOT/'outputs/official_full/metrics.json')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/official_full/error_analysis')
    parser.add_argument('--threshold', type=float, default=2.0)
    args = parser.parse_args()
    report = json.loads(args.metrics.read_text(encoding='utf-8'))
    stats = summarize(report, args.threshold)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'summary.json').write_text(json.dumps(stats, indent=2)+'\n', encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    for ax, key, title in zip(axes, ('minADE_m','minFDE_m'), ('Average displacement (minADE)', 'Final displacement (minFDE)')):
        values = np.sort(np.asarray([s[key] for s in report['scenes']], dtype=float))
        fraction = np.arange(1, len(values)+1)/len(values)*100
        ax.plot(values, fraction, color='#2261ae', linewidth=2)
        for pct, color in ((50, '#168367'), (90, '#d17b00')):
            x = np.percentile(values, pct)
            ax.scatter([x], [pct], color=color, zorder=3, label=f'{pct}th percentile: {x:.2f} m')
        if key == 'minFDE_m':
            ax.axvline(args.threshold, color='#ad3c3c', linestyle='--', label=f'Miss threshold: {args.threshold:g} m')
        xmax = max(np.percentile(values, 99), args.threshold if key == 'minFDE_m' else 0)
        ax.set_xlim(0, max(xmax*1.05, 1)); ax.set_ylim(0, 100)
        ax.set_title(title); ax.set_xlabel('Error (m)'); ax.set_ylabel('Validation scenes at or below error (%)')
        ax.grid(alpha=.25); ax.legend(fontsize=8, loc='lower right')
    fig.suptitle(f'Validation error distribution across {stats["count"]:,} scenes', fontsize=13)
    fig.tight_layout(); fig.savefig(args.output/'error_distribution.png', dpi=160); plt.close(fig)
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
