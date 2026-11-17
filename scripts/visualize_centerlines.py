"""Extract and visualize candidate centerlines for an Argoverse 1 scene.

Run from the project root after cloning the official Argoverse API into
third_party/argoverse-api. The API is imported from source on Windows.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "argoverse-api"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=ROOT / "data/raw/forecasting_sample/data/2645.csv")
    parser.add_argument("--map-dir", type=Path, default=ROOT / "data/raw/argoverse_map/map_files")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/argoverse_2645_centerlines.png")
    args = parser.parse_args()

    from argoverse.map_representation.map_api import ArgoverseMap

    data = pd.read_csv(args.scene)
    agent = data.loc[data["OBJECT_TYPE"] == "AGENT"].sort_values("TIMESTAMP")
    if len(agent) < 20:
        raise ValueError("Target agent must have at least 20 observed points")
    history = agent[["X", "Y"]].iloc[:20].to_numpy()
    future = agent[["X", "Y"]].iloc[20:50].to_numpy()
    city = agent["CITY_NAME"].iloc[0]
    amap = ArgoverseMap(root=args.map_dir)
    centerlines = amap.get_candidate_centerlines_for_traj(history, city)
    if not centerlines:
        raise RuntimeError("No candidate centerlines found")

    origin = history[-1]
    fig, ax = plt.subplots(figsize=(9, 9))
    for index, line in enumerate(centerlines):
        line = line - origin
        ax.plot(line[:, 0], line[:, 1], "--", color="tab:green", alpha=0.65, label="Candidate centerlines" if index == 0 else None)
    ax.plot(*(history - origin).T, color="tab:blue", marker=".", label="Observed history")
    if len(future):
        ax.plot(*(future - origin).T, color="tab:orange", marker=".", label="Ground-truth future")
    ax.scatter(0, 0, marker="*", s=130, color="tab:red", label="Current target position", zorder=5)
    ax.set(xlabel="Relative X (m)", ylabel="Relative Y (m)", title=f"{city}: {len(centerlines)} candidate centerlines")
    ax.axis("equal")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print(f"City: {city}; candidate centerlines: {len(centerlines)}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

