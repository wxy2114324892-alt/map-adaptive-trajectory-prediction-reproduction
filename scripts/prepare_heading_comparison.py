"""Create matched legacy and rotated-heading subsets from an existing legacy cache."""
import argparse
import io
import random
import sqlite3
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_dataset import connect, configure_heading_strategy
from src.data.argoverse_loader import align_to_target_heading, estimate_heading


def verify_legacy_source(db):
    """Reject a cache known to use a different coordinate strategy."""
    has_metadata = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='cache_metadata'").fetchone()
    if has_metadata:
        row = db.execute("SELECT value FROM cache_metadata WHERE key='heading_strategy'").fetchone()
        if row and row[0] != 'legacy':
            raise ValueError(f'Expected a legacy source cache, got {row[0]}')


def robust_from_legacy(scene):
    """Rotate all local tensors together; stationary global fallback is unrecoverable."""
    history = scene['histories'][0]
    _, source = estimate_heading(history, 'robust')
    if source == 'stationary_fallback':
        return None
    result = dict(scene)
    for key in ('histories', 'centerlines', 'future'):
        result[key] = align_to_target_heading(scene[key], history, 'robust')
    result['heading_strategy'] = 'robust'
    result['heading_source'] = source
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=ROOT/'data/processed/scenes.sqlite')
    p.add_argument('--directory', type=Path, default=ROOT/'data/processed/heading_comparison')
    p.add_argument('--train-count', type=int, default=2000)
    p.add_argument('--val-count', type=int, default=500)
    p.add_argument('--seed', type=int, default=2026)
    args = p.parse_args()
    if args.train_count < 1 or args.val_count < 1:
        p.error('Both scene counts must be positive')
    args.directory.mkdir(parents=True, exist_ok=True)
    targets = [args.directory/'legacy.sqlite', args.directory/'robust.sqlite']
    if any(p.exists() for p in targets):
        raise FileExistsError('Comparison cache already exists; choose a new directory')
    with sqlite3.connect(f'file:{args.source.resolve().as_posix()}?mode=ro', uri=True) as src:
        verify_legacy_source(src)
        old, new = (connect(path) for path in targets)
        configure_heading_strategy(old, 'legacy'); configure_heading_strategy(new, 'robust')
        rng = random.Random(args.seed)
        for split, wanted in (('train', args.train_count), ('val', args.val_count)):
            ids = [r[0] for r in src.execute('SELECT id FROM scenes WHERE split=? ORDER BY id', (split,))]
            rng.shuffle(ids); count = excluded = rotated = 0
            for scene_id in ids:
                payload = src.execute('SELECT payload FROM scenes WHERE split=? AND id=?', (split, scene_id)).fetchone()[0]
                scene = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
                converted = robust_from_legacy(scene)
                if converted is None:
                    excluded += 1; continue
                rotated += converted['heading_source'] != 'short'
                buffer = io.BytesIO(); torch.save(converted, buffer)
                old.execute('INSERT INTO scenes VALUES (?,?,?)', (split, scene_id, payload))
                new.execute('INSERT INTO scenes VALUES (?,?,?)', (split, scene_id, buffer.getvalue()))
                count += 1
                if count >= wanted: break
            old.commit(); new.commit()
            if count != wanted: raise RuntimeError(f'Only {count} of {wanted} {split} scenes available')
            print(f'{split}: {count} scenes, {rotated} longer-heading, {excluded} stationary fallback excluded', flush=True)
        old.close(); new.close()


if __name__ == '__main__': main()
