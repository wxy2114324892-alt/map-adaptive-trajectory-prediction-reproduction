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

## Observed-history characteristics and validation errors

On the existing one-epoch model, we joined the saved metrics to the cached histories by exact official validation-scene ID. All 39,468 IDs matched. These are the **legacy-heading** results; no model was retrained or evaluated with the experimental heading. Grouping uses only observed target history. Distances are in the cached target-centered frame; distance and angle are invariant under rotation.

| Last four-interval displacement | Scenes | Mean minADE | Mean minFDE | Endpoint miss rate (>2 m) | Mean modes |
| --- | ---: | ---: | ---: | ---: | ---: |
| <0.5 m | 391 | 3.17 m | 5.90 m | 81.8% | 6.33 |
| 0.5 to <2 m | 6,458 | 1.67 m | 3.36 m | 66.7% | 5.08 |
| >=2 m | 32,619 | 1.72 m | 3.34 m | 58.8% | 5.16 |

| Angle between displacement over four and ten intervals | Scenes | Mean minADE | Mean minFDE | Endpoint miss rate (>2 m) | Mean modes |
| --- | ---: | ---: | ---: | ---: | ---: |
| <=60 degrees | 39,321 | 1.71 m | 3.34 m | 60.2% | 5.16 |
| >60 degrees | 116 | 5.41 m | 10.22 m | 91.4% | 5.69 |
| Undefined (either displacement <=0.1 m) | 31 | 5.53 m | 10.50 m | 80.6% | 6.48 |

The thresholds are screening choices, not ground-truth labels. The short-displacement and direction groups overlap, and direction is not defined for tiny displacements. Low-motion scenes and large direction disagreements have higher errors in this dataset, but the groups are small and may differ in map structure, motion, mode count, and other factors. These descriptive comparisons do not establish why errors are high or whether changing heading reduces them. Best-of-mode errors favor having more available predictions, and miss rate is not a collision or safety rate.

Recompute locally from the original cache and per-scene metrics:

```powershell
.\outputs\python310\python.exe scripts\analyze_history_quality.py
```

The output is `outputs/official_full/history_analysis.json`; the code requires the cache and metrics to have exactly matching validation IDs. A fair next experiment would train separate models with matched splits, seed and settings for legacy and robust preprocessing and compare the same held-out scenes.

## Matched heading pilot

A small paired experiment used the existing legacy cache to create two subsets with identical scene IDs. Selection was random without replacement (seed 2026): 2,000 official training and 500 official validation scenes. For the robust version, histories, candidate centerlines and future labels were rotated together using the longer observed-history direction when selected by the heuristic; candidate lines themselves were not regenerated. The original legacy cache does not preserve enough information to recover the global zero-angle stationary fallback. Therefore the selection skipped that case in both versions: 10 training and one validation candidates. The longer-direction rule applied to 25 training and two validation scenes.

Both models were trained independently from scratch with seed 42, Adam learning rate 0.001, two epochs, CUDA, and otherwise identical settings. Predictions were evaluated against the corresponding coordinate frame in the same 500 held-out scenarios. These are one run, not an estimate of statistical significance.

| Heading | Validation mean minADE | Validation mean minFDE |
| --- | ---: | ---: |
| Legacy | 1.884 m | 3.789 m |
| Experimental robust | 2.031 m | 3.941 m |

The two validation scenes where the longer direction was selected had minFDE of 2.95 and 1.81 m with the legacy model, versus 1.42 and 4.15 m with the robust model, respectively. The outcomes differ and two cases are far too few to establish an effect. Results for all scenes can also differ from GPU nondeterminism and the changed training inputs. These findings do not support declaring the experimental approach better; the existing full-validation model's results are not changed by this pilot.

The paired caches are local at `data/processed/heading_comparison/` and the checkpoints and per-scene metrics are at `outputs/heading_comparison/`. To reproduce into a new directory, using the original legacy cache:

```powershell
.\outputs\python310\python.exe scripts\prepare_heading_comparison.py --directory data\processed\heading_comparison_retry --train-count 2000 --val-count 500 --seed 2026
.\outputs\python310\python.exe scripts\train_cached.py --cache data\processed\heading_comparison_retry\legacy.sqlite --output outputs\heading_comparison_retry\legacy --epochs 2 --seed 42 --device cuda
.\outputs\python310\python.exe scripts\train_cached.py --cache data\processed\heading_comparison_retry\robust.sqlite --output outputs\heading_comparison_retry\robust --epochs 2 --seed 42 --device cuda
```

Existing comparison caches are never overwritten. The source must be a legacy cache; old caches without strategy metadata are assumed to be legacy. Rotation consistency and the unrecoverable stationary case are covered by `tests/test_heading_comparison.py`.
