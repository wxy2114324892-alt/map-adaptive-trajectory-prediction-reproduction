# Heading and data-quality experiment

The default `legacy` heading keeps the original four-interval estimate and remains compatible with existing caches and checkpoints. `robust` is an experimental preprocessing option: it uses the last ten intervals when the short displacement is below 0.5 m or when short and long directions differ by more than 60 degrees, provided the longer displacement is at least 0.5 m. If neither is reliable, the zero-angle fallback is used. These heuristic thresholds are not validated improvements.

Each newly prepared scene records `heading_strategy`, `heading_source`, and `history_quality` (short displacement, largest observed step speed, repeated recent positions and short/long angle). These flags are screening indicators, not proof of bad data or grounds for automatic exclusion. All heading and quality calculations use observed history only.

Prepare a separate cache for comparison; do not combine strategies in one cache:

```powershell
.\outputs\python310\python.exe scripts\prepare_dataset.py --split train --limit 100 --heading-strategy robust --cache data\processed\heading_robust_pilot.sqlite
.\outputs\python310\python.exe scripts\prepare_dataset.py --split val --limit 100 --heading-strategy robust --cache data\processed\heading_robust_pilot.sqlite
.\outputs\python310\python.exe scripts\train_cached.py --cache data\processed\heading_robust_pilot.sqlite --output outputs\heading_robust_pilot --epochs 1 --device cpu
```

Run a corresponding legacy pilot in a separate cache and output directory with matching split IDs, seed, training settings and evaluation. Prepare only small samples first. Preparation streams the dataset archive unless `--archive` supplies a local copy, and requires the Argoverse map/API setup. Changing preprocessing changes the coordinates seen by the model: retrain for a fair comparison rather than treating the old checkpoint on a new coordinate frame as a valid evaluation. Existing cached scenes lack these new metadata; regenerate them in a new cache if the metadata is needed.
