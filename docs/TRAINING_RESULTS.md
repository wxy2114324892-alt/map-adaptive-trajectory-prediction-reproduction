# Argoverse 1 trajectory prediction: training results

This project is an unofficial, partial implementation inspired by **Map-Adaptive Multimodal Trajectory Prediction Using Hierarchical Graph Neural Networks** (IEEE RA-L, 2023). It demonstrates an end-to-end training and evaluation pipeline; it is not a claim of exact paper reproduction.

## Experiment

- Dataset: Argoverse 1 Motion Forecasting, official training and validation splits; HD map information supplies candidate centerlines.
- Successfully processed scenes: 205,863 training and 39,468 validation; 79 training and 4 validation scenes were skipped by preprocessing checks.
- Training: one epoch, Adam, learning rate 0.001, seed 42; vehicle history and lane features feed a GRU/graph-attention predictor.
- Validation: mean minADE **1.728 m**, mean minFDE **3.369 m**, over the processed validation scenes and all modes returned by the map-adaptive predictor. Lower is better. These best-of-set metrics do not estimate mode probabilities.
- Settings differ from the paper, including the number of epochs, implementation details, loss, output-mode protocol and validation versus test split. These metrics should not be directly compared with the paper's fixed-six-mode test-set table.

## Examples

![Twelve randomly sampled validation scenes](figures/validation_overview.png)

The plots show 12 validation examples sampled using random seed 2026, not chosen by model performance. Blue is observed target history, pale blue shows other agents' observed history, dashed gray is target candidate lane centerlines, orange is model predictions, and green is the recorded future. A few per-scene plot labels vary slightly from the original aggregate evaluation because predictions were recalculated in a different execution environment; the aggregate values above come from the original validation run. Twelve examples illustrate outcomes but do not replace evaluation over the full validation split.

## Reproducibility and limitations

Data archives, processed cache, the trained checkpoint, third-party source and generated outputs are not included in this repository. Training and evaluation require separately obtained Argoverse 1 data, its map/API, compatible dependencies, and the appropriate data-use terms. Candidate centerlines are derived only from observed histories. The current Windows API setup imports its source directly because the upstream installer does not officially support Windows. The model's ADE-selected Smooth L1 objective is an approximation and is not confirmed to be an exact implementation of the cited MTP loss.

This is a trajectory prediction experiment, not a closed-loop vehicle controller. A separate optional replay visualization shows changing predictions while replaying recorded motion; it is not an experiment reported in the paper and the replay's sample is not assumed to be held out.

## Validation error distribution

The aggregate analysis below uses the 39,468 successfully processed validation scenes recorded in `outputs/official_full/metrics.json`, without rerunning inference. It reports minimum errors across the trajectory modes returned for each scene.

| Metric | Mean | Median | 90th percentile | 95th percentile |
| --- | ---: | ---: | ---: | ---: |
| minADE (m) | 1.73 | 1.35 | 3.25 | 4.20 |
| minFDE (m) | 3.37 | 2.50 | 6.86 | 9.14 |

With an endpoint threshold of 2 m, **60.35%** of these scenes have a minFDE greater than 2 m: none of that scene's predicted endpoints is within 2 m of the recorded endpoint. This is an endpoint miss rate, not a collision rate or a probability of unsafe driving. The number of predicted trajectories varies by scene (3–20 in this result), and these validation-split numbers are **not directly comparable** to the paper's fixed-six-mode test-set benchmark.

![Cumulative validation error distributions](figures/validation_error_distribution.png)

Each curve shows the percentage of scenes whose minimum error is at or below the value on the horizontal axis. The plotted horizontal range ends near the 99th percentile; extreme errors beyond it are not shown. The 90th percentile means 90% of included scenes have an error at or below that value, not that 90% of predictions are safe.

To recompute from the saved local metrics file:

```powershell
.\outputs\python310\python.exe scripts\analyze_validation_errors.py
```

This produces a JSON summary and figure under `outputs/official_full/error_analysis/`. The raw per-scene metrics are local and excluded from Git.

## Comparison with a constant-velocity baseline

We compared the saved model evaluation with a single constant-velocity trajectory on the **same 39,468 processed official validation scenes**. The baseline estimates XY velocity from the last four observed time intervals, using their actual elapsed time, then extrapolates from the final observation for 30 steps at a nominal 0.1 seconds per step. It uses no future positions, map information or other agents as inputs. Cached ground-truth future timestamps are not retained, so the nominal forecast times are an approximation for recordings with irregular intervals.

| Validation metric | Constant velocity (one trajectory) | Model (best of all returned modes) |
| --- | ---: | ---: |
| Mean ADE / minADE | 2.597 m | 1.728 m |
| Median ADE / minADE | 1.848 m | 1.347 m |
| Mean FDE / minFDE | 5.659 m | 3.369 m |
| Median FDE / minFDE | 4.138 m | 2.500 m |
| Endpoint error >2 m | 74.48% | 60.35% |

The model's best-of-set ADE was lower than the baseline in 63.16% of scenes, and its best-of-set FDE was lower in 67.28%. These are **oracle comparisons**: the model returns multiple modes (3–20 in this evaluation), and each metric selects the best mode after observing the true future. The baseline has one output, and the model does not rank its predictions by probability. Therefore these figures do not establish which model trajectory could be selected in advance, nor do they provide a controlled architecture ablation. ADE and FDE can select different model modes. As elsewhere in this report, endpoint miss rate is not a collision or safety rate.

The script also reports groups by observed displacement, enabling inspection of low-motion cases. The comparisons are descriptive; group membership does not establish the causes of errors. Four validation scenes that failed the original preprocessing are absent from both methods. Recompute with the original local cache and per-scene metrics:

```powershell
.\outputs\python310\python.exe scripts\compare_constant_velocity.py
```

The script requires exact agreement between cached and reported validation IDs, and saves aggregate and per-scene results to `outputs/official_full/constant_velocity_comparison.json`. It does not retrain or rerun the learned model.

## Selecting one trajectory before the future is known

We reloaded the original one-epoch checkpoint and reran inference for all **39,468 processed validation scenes**. The scene-reasoning and motion-maintaining rows use their fixed output nodes. For a simple history-based selector, we first extrapolate the observed velocity as above, then select the model trajectory whose first ten predicted positions have the lowest mean Euclidean distance to that reference. Selection uses only history and model outputs; the recorded future is used solely for evaluation. It produces one selected trajectory and is an experimental heuristic, not a calibrated confidence score.

| Output | Mean ADE | Mean FDE | Endpoint error >2 m |
| --- | ---: | ---: | ---: |
| Constant velocity, single trajectory | 2.597 m | 5.659 m | 74.48% |
| Model scene-reasoning branch | 6.763 m | 13.065 m | 92.45% |
| Model motion-maintaining branch | 5.418 m | 10.739 m | 87.43% |
| Model history-based selected trajectory | 2.584 m | 5.548 m | 77.95% |
| Model best of all modes, selected using future | 1.728 m | 3.369 m | 60.35% |

The fixed branches perform worse than constant velocity in this evaluation. The simple selector has slightly lower mean errors than constant velocity, but its median errors and endpoint miss rate are higher; this single run does not demonstrate a consistent advantage. The oracle row selects the best mode after seeing ground truth and is not an actionable prediction-time selection. Different ADE and FDE modes may win. The results describe this model, seed, one-epoch checkpoint and processed validation split, not all models or future datasets. They do not establish why branches perform poorly: winner-take-all training may contribute, but that requires controlled experiments. None of these metrics is a safety or collision rate.

The selector is available in `src/utils/mode_selection.py`; it expects predictions and a history-derived reference in the same coordinates and sampling schedule. It is not enabled automatically in the predictor. Cached future timestamps are unavailable, so the reference assumes 0.1-second future steps. Rerun the full comparison, after preparing the matching original cache and checkpoint:

```powershell
.\outputs\python310\python.exe scripts\evaluate_prediction_branches.py --device cuda --count 0 --seed 2026 --output outputs\official_full\branch_comparison_with_selection.json
```

Use a positive `--count` for a reproducible random subset. Recomputed inference can show minor numerical differences across environments. Four scenes excluded from the original validation preparation remain excluded.
