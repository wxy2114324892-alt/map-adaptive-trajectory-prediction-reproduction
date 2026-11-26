# Experimental learned trajectory selector

The original model returns a variable number of candidate trajectories without ranking them. This experiment freezes that model and trains a separate scorer to select a single candidate before observing the future. It does not alter the original predictor or claim to reproduce the paper's decoder.

For each candidate, the scorer sees the 20 observed target positions, its 30 predicted positions, the difference between the candidate and a constant-velocity extrapolation, and its node category (target CCL, scene reasoning, or motion maintaining). Coordinates are target-centered and scaled by 50 m. The reference velocity uses the last four observed intervals and assumes nominal 0.1 s future steps. A shared MLP assigns one score to each mode; the largest score selects the output. These scores are not calibrated probabilities.

Ground-truth futures are used to label the lowest-ADE candidate during training and to calculate validation metrics. They are not inputs to `candidate_features` or `select_candidate`. Mode sets have variable sizes; training masks padded modes. A separate subset of the official training split selects the epoch by cross-entropy; the official validation split is used for evaluation, not epoch selection. The original trajectory predictor was itself trained on the full training split, including the scorer's training and tuning scenes, so those candidate predictions are in-sample for the predictor. Out-of-fold predictions would reduce this limitation. The official validation set has been analyzed in earlier project experiments and should not be treated as an untouched final test set.

## Reproduce

The following command needs the original locally prepared Argoverse cache, original predictor checkpoint and dependencies described in the main README. It creates a separate feature cache and artifacts under `outputs/learned_selector_full/` on D: in the original setup. This is a substantial computation. Use a new output directory for a different sample size, seed or predictor checkpoint; the feature cache checks those settings and the checkpoint SHA-256. If interrupted, restarting the same command reuses successfully cached features and retrains the scorer.

```powershell
.\outputs\python310\python.exe scripts\train_candidate_scorer.py --device cuda --train-count 10000 --tune-count 2000 --val-count 0 --epochs 15 --batch-size 64 --seed 2026 --output outputs\learned_selector_full
```

`--val-count 0` evaluates all cached official validation scenes. `scorer.pt` contains scorer weights, the feature configuration, predictor checkpoint hash, training history and selected epoch. `metrics.json` contains aggregate and per-scene results; `features.sqlite` contains derived features and labels and should remain local with the underlying data. The scorer expects predictions from the matching predictor and the same coordinate and sampling conventions. Resume reuses generated features; it does not resume a partly trained scorer.

The selector is experimental and not automatically enabled in the predictor. Neither the heuristic nor learned selector uses a learned collision model or provides safety guarantees.

## Full validation result

A single run used 10,000 randomly sampled official training scenes to fit the scorer and another 2,000 disjoint training scenes to select the epoch; seed 2026. The scorer ran for 15 epochs, and epoch 15 had the lowest tuning cross-entropy. The original predictor checkpoint was frozen and had completed one training epoch. Evaluation used all 39,468 successfully cached official validation scenes; four original validation scenes had been skipped during preprocessing. All listed methods are evaluated on the same scenes.

| Method | Uses ground truth to select? | Mean ADE | Mean FDE | Median ADE | Median FDE | Endpoint error >2 m |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Constant velocity | No | 2.597 m | 5.659 m | 1.848 m | 4.138 m | 74.48% |
| History-consistency heuristic | No | 2.584 m | 5.548 m | 1.989 m | 4.263 m | 77.95% |
| Learned selector | No | 2.429 m | 5.185 m | 1.879 m | 3.988 m | 76.46% |
| Best candidate for each metric (oracle) | Yes | 1.728 m | 3.369 m | 1.347 m | 2.500 m | 60.35% |

The learned selector reduced mean ADE and FDE relative to the tested history heuristic in this run. It selected the same candidate as the heuristic in about 68.8% of scenes; its ADE was strictly lower in 17.4% of scenes and higher in the remainder. Relative to constant velocity, its mean ADE/FDE are lower, but its median ADE and 2 m miss rate are higher. Thus it does not consistently outperform that baseline across all metrics. The oracle's ADE and FDE can select different modes and are not available at inference. The result is not a safety guarantee or evidence of collision avoidance.

The scorer checkpoint was reloaded and its decisions matched the saved results on 50 independently recomputed validation scenes. This is a single run and one predictor checkpoint; different seeds, epochs, feature choices, scene correlation and a clean held-out test set have not been evaluated. No conclusion about statistical significance or real-world driving performance follows from this experiment.

## Breakdown by observed movement

We matched all 39,468 selector results to the earlier baseline analysis by validation scene ID, checking that the independently calculated baseline errors agree. Groups use target displacement over the last four observed intervals (approximately 0.4 s), not future motion:

| Observed displacement | Scenes | Constant velocity mean ADE / FDE | Learned selection mean ADE / FDE | Constant velocity / learned miss rate >2 m |
| --- | ---: | ---: | ---: | ---: |
| <0.5 m | 391 | 4.642 / 10.707 m | 4.869 / 10.288 m | 98.72% / 92.58% |
| 0.5 to <2 m | 6,458 | 2.899 / 6.776 m | 2.550 / 5.667 m | 86.98% / 79.89% |
| >=2 m | 32,619 | 2.512 / 5.377 m | 2.376 / 5.028 m | 71.72% / 75.59% |

Low observed displacement does not imply the vehicle remains stationary in the future. In the smallest group, learned selection has worse mean ADE but lower mean FDE and miss rate than the constant-velocity baseline; error remains large in both. In the largest group, the selector improves mean error while increasing miss rate. These are descriptive subgroup results from a single run, not causal explanations, and the smallest group contains only 391 scenes. The variable-mode oracle remains an after-the-fact bound, not a deployable selection.

Reproduce the breakdown from the two saved per-scene result files:

```powershell
.\outputs\python310\python.exe scripts\analyze_selector_groups.py
```

The resulting `outputs/learned_selector_full/movement_groups.json` contains the group summaries. The script rejects duplicate/mismatched IDs and baseline discrepancies before comparison.
