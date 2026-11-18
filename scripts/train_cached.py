"""Train and evaluate from separately cached official dataset splits."""
import argparse
import io
import json
import random
import sqlite3
import sys
import time
from pathlib import Path
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.scenes import predict
from src.models.predictor import MapAdaptiveTrajectoryPredictor
from src.models.losses import modified_mtp_loss
from src.utils.metrics import min_ade, min_fde


def read_scene(db, split, scene_id):
    row = db.execute('SELECT payload FROM scenes WHERE split=? AND id=?', (split, scene_id)).fetchone()
    if row is None: raise ValueError(f'Missing {split} scene {scene_id}')
    return torch.load(io.BytesIO(row[0]), map_location='cpu', weights_only=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cache', type=Path, default=ROOT/'data/processed/scenes.sqlite')
    ap.add_argument('--output', type=Path, default=ROOT/'outputs/official_experiment')
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--train-limit', type=int, default=0)
    ap.add_argument('--val-limit', type=int, default=0)
    ap.add_argument('--save-every', type=int, default=1000)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--deadline', type=float, default=0, help='Unix timestamp after which to stop training and evaluate')
    args = ap.parse_args()
    db = sqlite3.connect(f'file:{args.cache.resolve().as_posix()}?mode=ro', uri=True)
    train = [x[0] for x in db.execute("SELECT id FROM scenes WHERE split='train' ORDER BY id")]
    val = [x[0] for x in db.execute("SELECT id FROM scenes WHERE split='val' ORDER BY id")]
    if args.train_limit: train = train[:args.train_limit]
    if args.val_limit: val = val[:args.val_limit]
    if not train or not val: ap.error('Cache must contain both official train and validation scenes')
    torch.manual_seed(args.seed)
    model = MapAdaptiveTrajectoryPredictor(agent_input_dim=4).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    args.output.mkdir(parents=True, exist_ok=True)
    file = args.output/'checkpoint.pt'
    epoch = offset = 0; losses = []
    if args.resume:
        state = torch.load(file, map_location=args.device, weights_only=True)
        if state['seed'] != args.seed or not set(state['train_ids']).issubset(train) or not set(state['val_ids']).issubset(val):
            raise ValueError('Checkpoint seed or cached scenes do not match')
        train, val = state['train_ids'], state['val_ids']
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        epoch, offset, losses = state['epoch'], state['offset'], state['losses']
    def save(epoch, offset):
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
                 'offset': offset, 'seed': args.seed, 'train_ids': train, 'val_ids': val, 'losses': losses}
        tmp = file.with_suffix('.tmp'); torch.save(state, tmp); tmp.replace(file)
    stopped_for_deadline = False
    for ep in range(epoch, args.epochs):
        order = train.copy(); random.Random(args.seed + ep).shuffle(order)
        model.train(); total = 0.; steps = 0
        for i in range(offset if ep == epoch else 0, len(order)):
            scene = read_scene(db, 'train', order[i]); optimizer.zero_grad()
            pred = predict(model, scene, args.device)
            loss, _ = modified_mtp_loss(pred, scene['future'].to(args.device))
            loss.backward(); optimizer.step(); total += loss.item(); steps += 1
            if args.deadline and time.time() >= args.deadline:
                save(ep, i+1)
                stopped_for_deadline = True
                print(f'Training deadline reached at epoch {ep+1}, {i+1}/{len(order)}; saving and evaluating', flush=True)
                break
            if (i+1) % args.save_every == 0:
                save(ep, i+1); print(f'epoch {ep+1}, {i+1}/{len(order)}', flush=True)
        if stopped_for_deadline:
            break
        losses.append({'epoch': ep+1, 'mean_remaining_loss': total/max(steps, 1), 'steps': steps})
        save(ep+1, 0); print(f'epoch {ep+1} complete, loss {total/max(steps,1):.4f}', flush=True)
    # Evaluation uses the saved checkpoint and only the official validation split.
    restored = MapAdaptiveTrajectoryPredictor(agent_input_dim=4).to(args.device)
    restored.load_state_dict(torch.load(file, map_location=args.device, weights_only=True)['model']); restored.eval()
    metrics = []
    with torch.no_grad():
        for scene_id in val:
            scene = read_scene(db, 'val', scene_id); pred = predict(restored, scene, args.device).cpu()
            metrics.append({'id': scene_id, 'modes': len(pred), 'minADE_m': min_ade(pred, scene['future']).item(), 'minFDE_m': min_fde(pred, scene['future']).item()})
    report = {'train_count': len(train), 'val_count': len(val), 'epochs': args.epochs, 'seed': args.seed,
              'stopped_for_deadline': stopped_for_deadline, 'completed_epochs': torch.load(file, map_location='cpu', weights_only=True)['epoch'],
              'mean_minADE_m': sum(x['minADE_m'] for x in metrics)/len(metrics), 'mean_minFDE_m': sum(x['minFDE_m'] for x in metrics)/len(metrics), 'scenes': metrics}
    (args.output/'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    scene = read_scene(db, 'val', val[0])
    with torch.no_grad(): pred = predict(restored, scene, args.device).cpu()
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, line in enumerate(scene['centerlines']):
        ax.plot(line[:, 0], line[:, 1], '--', color='gray', alpha=.5, label='Candidate centerlines' if i == 0 else None)
    for i, track in enumerate(scene['histories']):
        ax.plot(track[:, 0], track[:, 1], color='blue' if i == 0 else 'lightblue', label='Target history' if i == 0 else ('Other histories' if i == 1 else None))
    for i, trajectory in enumerate(pred):
        ax.plot(trajectory[:, 0], trajectory[:, 1], alpha=.7, label=f'Prediction {i+1}')
    future = scene['future']; ax.plot(future[:, 0], future[:, 1], color='green', linewidth=2, label='Ground truth future')
    focus = torch.cat((scene['histories'][0], future, pred.reshape(-1, 2)))
    low, high = focus.min(0).values-10, focus.max(0).values+10
    ax.set_xlim(low[0].item(), high[0].item()); ax.set_ylim(low[1].item(), high[1].item())
    ax.set_aspect('equal', adjustable='box'); ax.set(xlabel='Local X (m)', ylabel='Local Y (m)', title=f"Validation scene {val[0]}")
    ax.grid(); ax.legend(); fig.tight_layout(); fig.savefig(args.output/'prediction.png', dpi=150); plt.close(fig)
    db.close()
    print(json.dumps({k:v for k,v in report.items() if k != 'scenes'}, indent=2))


if __name__ == '__main__': main()
