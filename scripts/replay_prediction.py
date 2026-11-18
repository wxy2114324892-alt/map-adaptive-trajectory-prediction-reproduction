"""Replay a recorded Argoverse scene, updating trajectory predictions over time.

Each prediction uses only the preceding 20 recorded timestamps. This is an
open-loop visualization, not vehicle control or a benchmark evaluation.
"""
import argparse
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'third_party' / 'argoverse-api'))
from src.data.argoverse_loader import load_scene_histories, align_to_target_heading
from src.data.scenes import predict, resample
from src.models.predictor import MapAdaptiveTrajectoryPredictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, default=ROOT/'data/raw/forecasting_sample/data/2645.csv')
    parser.add_argument('--map-dir', type=Path, default=ROOT/'data/raw/argoverse_map/map_files')
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'outputs/official_full/checkpoint.pt')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/official_full/dynamic_replay.gif')
    parser.add_argument('--stride', type=int, default=5, help='Observed frames per prediction update (5 = 0.5 s at 10 Hz)')
    args = parser.parse_args()
    if args.stride < 1:
        parser.error('--stride must be positive')
    from argoverse.map_representation.map_api import ArgoverseMap
    raw = pd.read_csv(args.scene)
    times = sorted(raw.TIMESTAMP.unique())
    if len(times) < 20:
        raise ValueError('Scene needs at least 20 observed timestamps')
    target = raw[raw.OBJECT_TYPE == 'AGENT'].sort_values('TIMESTAMP')
    city = target.CITY_NAME.iloc[0]
    steps = list(range(19, len(times), args.stride))
    if steps[-1] != len(times)-1:
        steps.append(len(times)-1)
    amap = ArgoverseMap(root=args.map_dir)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = MapAdaptiveTrajectoryPredictor(agent_input_dim=4)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    data = []
    with tempfile.TemporaryDirectory() as temp, torch.no_grad():
        for step in steps:
            observed_times = times[step-19:step+1]
            observed = raw[raw.TIMESTAMP.isin(observed_times)]
            csv = Path(temp)/'history.csv'
            observed.to_csv(csv, index=False)
            histories, _, _, origin = load_scene_histories(csv)
            history = histories[0]
            angle = torch.atan2(history[-1,1]-history[-5,1], history[-1,0]-history[-5,0])
            c, s = torch.cos(angle), torch.sin(angle)
            rot = torch.stack([torch.stack([c,-s]), torch.stack([s,c])])
            lines, owners, target_lines = [], [], []
            for owner, track in enumerate(histories):
                try:
                    candidates = amap.get_candidate_centerlines_for_traj((track+origin).numpy(), city)
                except (ValueError, IndexError, AssertionError):
                    candidates = []
                for candidate in candidates:
                    try:
                        line = torch.tensor(resample(candidate), dtype=torch.float32)
                    except ValueError:
                        continue
                    lines.append(align_to_target_heading((line-origin).float(), history))
                    owners.append(owner)
                    if owner == 0:
                        target_lines.append(line.numpy())
            intervals = np.diff(np.asarray(observed_times, dtype=float))
            if np.any(intervals <= 0):
                raise ValueError('Invalid timestamps')
            scene = {'histories': align_to_target_heading(histories, history),
                     'centerlines': torch.stack(lines) if lines else torch.empty(0,30,2),
                     'owners': owners, 'intervals': torch.tensor(intervals, dtype=torch.float32)}
            predictions = predict(model, scene, 'cpu')
            predictions_global = (predictions @ rot.T + origin).numpy()
            data.append({'step':step, 'origin':origin.numpy(), 'lines':target_lines,
                         'history':(histories[0]+origin).numpy(), 'pred':predictions_global})
            print(f'Prediction at recorded frame {step+1}/{len(times)}: {len(predictions)} trajectories', flush=True)
    # Coordinates are shown relative to the first prediction's location, with a fixed frame.
    base = data[0]['origin']
    actual = target[['X','Y']].to_numpy()-base
    all_points = np.concatenate([actual] + [x['pred'].reshape(-1,2)-base for x in data])
    lo, hi = all_points.min(0)-8, all_points.max(0)+8
    fig, ax = plt.subplots(figsize=(10,7))
    def render(i):
        ax.clear()
        frame = data[i]
        for j, line in enumerate(frame['lines']):
            line = line-base
            ax.plot(line[:,0], line[:,1], color='gray', linestyle='--', alpha=.5, label='Candidate lane centerlines' if j==0 else None)
        t = times[frame['step']]
        known = target[target.TIMESTAMP <= t][['X','Y']].to_numpy()-base
        ax.plot(known[:,0],known[:,1],color='royalblue',linewidth=2,label='Observed actual path')
        pos=frame['origin']-base
        ax.scatter(*pos,color='blue',s=100,zorder=5,label='Current vehicle position')
        for j, trajectory in enumerate(frame['pred']):
            trajectory=trajectory-base
            ax.plot(trajectory[:,0],trajectory[:,1],color='darkorange',alpha=.7,linewidth=1.5,label='Future predictions (updated)' if j==0 else None)
        ax.set_xlim(lo[0],hi[0]); ax.set_ylim(lo[1],hi[1]); ax.set_aspect('equal',adjustable='box')
        ax.set_title(f"Recorded scene {args.scene.stem}: update {i+1}/{len(data)}")
        ax.set_xlabel('Relative X (m)'); ax.set_ylabel('Relative Y (m)'); ax.grid(alpha=.25)
        ax.legend(loc='best',fontsize=8)
        ax.text(.02,.02,'Open-loop replay: vehicle follows recorded data, not the prediction',transform=ax.transAxes,fontsize=8)
    ani=FuncAnimation(fig,render,frames=len(data),interval=800,repeat=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    ani.save(args.output,writer=PillowWriter(fps=1.25))
    plt.close(fig)
    print(f'Saved {len(data)} prediction updates to {args.output}')


if __name__ == '__main__':
    main()
