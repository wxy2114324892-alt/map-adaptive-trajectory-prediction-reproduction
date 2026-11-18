"""Visualize a fixed-seed sample of predictions from the held-out validation set."""
import argparse
import csv
import json
import random
import sqlite3
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_cached import read_scene
from src.data.scenes import predict
from src.models.predictor import MapAdaptiveTrajectoryPredictor
from src.utils.metrics import min_ade, min_fde


def draw(ax, scene, predictions, item):
    # Dashed candidate lines belong to the target vehicle, not its neighbours.
    for line, owner in zip(scene['centerlines'], scene['owners']):
        if owner == 0:
            xy = line.cpu().numpy()
            ax.plot(xy[:, 0], xy[:, 1], '--', color='0.65', lw=1, alpha=.7)
    for history in scene['histories'][1:]:
        xy = history.cpu().numpy()
        ax.plot(xy[:, 0], xy[:, 1], color='lightsteelblue', lw=1, alpha=.65)
    history = scene['histories'][0].cpu().numpy()
    truth = scene['future'].cpu().numpy()
    ax.plot(history[:,0], history[:,1], color='royalblue', lw=2)
    for track in predictions.numpy():
        ax.plot(track[:,0], track[:,1], color='darkorange', lw=1.2, alpha=.7)
    ax.plot(truth[:,0], truth[:,1], color='seagreen', lw=2.4)
    ax.scatter(history[-1,0], history[-1,1], color='royalblue', marker='*', s=65, zorder=5)
    focus = np.concatenate([history, truth, predictions.numpy().reshape(-1,2)])
    lo, hi = focus.min(axis=0)-6, focus.max(axis=0)+6
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    ax.set_aspect('equal', adjustable='box'); ax.grid(alpha=.2)
    ax.set_title(f"Scene {item['id']}  |  minADE {item['minADE_m']:.2f} m  |  minFDE {item['minFDE_m']:.2f} m", fontsize=9)
    ax.set_xlabel('Local X (m)', fontsize=8); ax.set_ylabel('Local Y (m)', fontsize=8)
    ax.tick_params(labelsize=7)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--count', type=int, default=12)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--output', type=Path, default=ROOT/'outputs/official_full/validation_examples')
    args = p.parse_args()
    report = json.loads((ROOT/'outputs/official_full/metrics.json').read_text(encoding='utf-8'))
    entries = report['scenes']
    if args.count < 1 or args.count > len(entries):
        p.error('count must be between 1 and the validation-set size')
    chosen = random.Random(args.seed).sample(entries, args.count)
    checkpoint = torch.load(ROOT/'outputs/official_full/checkpoint.pt', map_location='cpu', weights_only=True)
    model = MapAdaptiveTrajectoryPredictor(agent_input_dim=4)
    model.load_state_dict(checkpoint['model']); model.eval()
    db = sqlite3.connect(f"file:{(ROOT/'data/processed/scenes.sqlite').resolve().as_posix()}?mode=ro", uri=True)
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    cols=3; rows=(len(chosen)+cols-1)//cols
    overview, axes = plt.subplots(rows, cols, figsize=(15, 2.8*rows), squeeze=False)
    legend=[Line2D([0],[0],color='royalblue',lw=2,label='Observed target history'),
            Line2D([0],[0],color='lightsteelblue',lw=2,label='Other agents: history only'),
            Line2D([0],[0],color='gray',ls='--',label='Target candidate centerlines'),
            Line2D([0],[0],color='darkorange',lw=2,label='Model predictions'),
            Line2D([0],[0],color='seagreen',lw=2,label='Actual future')]
    with torch.no_grad():
        for index, entry in enumerate(chosen):
            scene=read_scene(db,'val',entry['id'])
            prediction=predict(model,scene,'cpu').cpu()
            item={'id':entry['id'],'modes':len(prediction),
                  'minADE_m':min_ade(prediction,scene['future']).item(),
                  'minFDE_m':min_fde(prediction,scene['future']).item()}
            # Ensure figures are consistent with the previously saved validation run.
            if abs(item['minADE_m']-entry['minADE_m']) > 1e-3 or abs(item['minFDE_m']-entry['minFDE_m']) > 1e-3:
                print(f"Note: scene {item['id']} recalculated {item['minADE_m']:.4f}/{item['minFDE_m']:.4f}, original {entry['minADE_m']:.4f}/{entry['minFDE_m']:.4f}", flush=True)
            results.append(item)
            fig, ax=plt.subplots(figsize=(9,7)); draw(ax,scene,prediction,item)
            fig.legend(handles=legend,loc='upper center',ncol=2,fontsize=8)
            fig.tight_layout(rect=[0,0,1,.90]); fig.savefig(args.output/f"scene_{item['id']}.png",dpi=150); plt.close(fig)
            draw(axes[index//cols,index%cols],scene,prediction,item)
            print(f"{index+1}/{args.count}: scene {item['id']}",flush=True)
    for index in range(len(chosen), rows*cols): axes[index//cols,index%cols].axis('off')
    overview.legend(handles=legend,loc='upper center',bbox_to_anchor=(0.5,0.965),ncol=3,fontsize=10)
    overview.suptitle(f'Validation examples (random seed {args.seed}; not selected for performance)',y=.995)
    overview.tight_layout(rect=[0,0,1,.90]); overview.savefig(args.output/'overview.png',dpi=150); plt.close(overview)
    with (args.output/'selected_scenes.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['id','modes','minADE_m','minFDE_m']); writer.writeheader(); writer.writerows(results)
    (args.output/'README.txt').write_text(f"{args.count} validation scenes sampled uniformly with seed {args.seed} from {len(entries)} saved validation results.\n"
        "Orange: all model predictions. Green: recorded future (not model input). Blue: past. Gray dashed: target CCLs.\n"
        "Numbers are per-scene minimum errors across all output trajectories, not prediction probabilities.\n"
        "Each plot covers one chosen scene; inspect all reported metrics for aggregate evaluation.\n",encoding='utf-8')
    db.close()
    print(f'Saved overview and {len(results)} plots to {args.output}')


if __name__=='__main__': main()
