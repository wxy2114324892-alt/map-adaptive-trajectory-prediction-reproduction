"""Stream official Argoverse archives into a resumable on-disk scene cache."""
import argparse
import io
import sqlite3
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'third_party' / 'argoverse-api'))
from src.data.scenes import prepare_scene

URLS = {name: f'https://s3.amazonaws.com/argoverse/datasets/av1.1/tars/forecasting_{name}_v1.1.tar.gz' for name in ('train', 'val')}


def connect(path):
    db = sqlite3.connect(path, timeout=60)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS scenes (split TEXT NOT NULL, id TEXT NOT NULL, payload BLOB NOT NULL, PRIMARY KEY (split,id))')
    db.execute('CREATE TABLE IF NOT EXISTS failures (split TEXT NOT NULL, id TEXT NOT NULL, error TEXT NOT NULL, PRIMARY KEY (split,id))')
    db.commit()
    return db


def configure_heading_strategy(db, strategy):
    """Record one strategy per cache, treating pre-existing caches as legacy."""
    if strategy not in ('legacy', 'robust'):
        raise ValueError(f'Unknown heading strategy: {strategy}')
    db.execute('BEGIN IMMEDIATE')
    try:
        db.execute('CREATE TABLE IF NOT EXISTS cache_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
        setting = db.execute("SELECT value FROM cache_metadata WHERE key='heading_strategy'").fetchone()
        existing = db.execute('SELECT count(*) FROM scenes').fetchone()[0]
        existing_strategy = setting[0] if setting else ('legacy' if existing else strategy)
        if existing_strategy != strategy:
            raise ValueError(f'Cache uses {existing_strategy} heading; choose a separate cache for {strategy}.')
        db.execute("INSERT OR IGNORE INTO cache_metadata VALUES ('heading_strategy', ?)", (existing_strategy,))
        db.commit()
    except Exception:
        db.rollback()
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--split', choices=URLS, required=True)
    ap.add_argument('--cache', type=Path, default=ROOT/'data/processed/scenes.sqlite')
    ap.add_argument('--map-dir', type=Path, default=ROOT/'data/raw/argoverse_map/map_files')
    ap.add_argument('--archive', type=Path, help='Local tar.gz instead of official URL')
    ap.add_argument('--limit', type=int, default=0, help='Maximum number of cached scenes, 0 means all')
    ap.add_argument('--heading-strategy', choices=('legacy', 'robust'), default='legacy')
    args = ap.parse_args()
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    db = connect(args.cache)
    try:
        configure_heading_strategy(db, args.heading_strategy)
    except ValueError as exc:
        db.close()
        ap.error(str(exc))
    count = db.execute('SELECT count(*) FROM scenes WHERE split=?', (args.split,)).fetchone()[0]
    if args.limit and count >= args.limit:
        print(f'{args.split}: already has {count} scenes', flush=True); return
    from argoverse.map_representation.map_api import ArgoverseMap
    amap = ArgoverseMap(root=args.map_dir)
    source = args.archive.open('rb') if args.archive else urllib.request.urlopen(URLS[args.split], timeout=120)
    start = time.monotonic()
    with source, tarfile.open(fileobj=source, mode='r|gz') as archive, tempfile.TemporaryDirectory(dir=ROOT/'data/processed') as temp:
        for member in archive:
            if not member.isfile() or not member.name.lower().endswith('.csv'): continue
            scene_id = Path(member.name).stem
            if db.execute('SELECT 1 FROM scenes WHERE split=? AND id=?', (args.split, scene_id)).fetchone(): continue
            raw = archive.extractfile(member)
            if raw is None: continue
            path = Path(temp) / 'scene.csv'
            path.write_bytes(raw.read())
            try:
                scene = prepare_scene(path, amap, heading_strategy=args.heading_strategy); scene['id'] = scene_id
                payload = io.BytesIO(); torch.save(scene, payload)
                db.execute('INSERT INTO scenes VALUES (?,?,?)', (args.split, scene_id, payload.getvalue()))
                db.execute('DELETE FROM failures WHERE split=? AND id=?', (args.split, scene_id))
                count += 1
            except (ValueError, IndexError, KeyError, AssertionError) as exc:
                db.execute('INSERT OR REPLACE INTO failures VALUES (?,?,?)', (args.split, scene_id, f'{type(exc).__name__}: {exc}'))
                print(f'{args.split} {scene_id}: skipped: {exc}', flush=True)
            db.commit()
            if count % 10 == 0:
                print(f'{args.split}: {count} scenes; elapsed {time.monotonic()-start:.0f}s', flush=True)
            if args.limit and count >= args.limit: break
    db.close()
    print(f'{args.split}: cached {count} scenes', flush=True)


if __name__ == '__main__': main()
