# Task: Implement weighted-KNN PM2.5 forecaster in a Jupyter notebook

## Context

You are implementing a *modified, weighted K-Nearest Neighbors* model for forecasting PM2.5 (KD₂,₅) air pollution 12 hours ahead, based on the methodology described in the report files in this repo (the LaTeX sources `report_data_preparation/ŽiemysMindaugasDataPreparation.tex, report_euclidean/ŽiemysWeightedEuclideanDistanceReport.tex` describe the data preparation and the predictor). **Read the methodology section of the report first** — it is the source of truth. Below is a condensed summary; if anything here conflicts with the report, the report wins.

The data is already prepared and normalized to `[-1, 1]`. You are implementing **only the predictor + evaluation** in a new Jupyter notebook. I already implemented normalize_data.ipynb for data preparation report, if need to check, but want a clean notebook. The data is in /data/3_scaled/scaled.csv. COVID years are excluded already and all prepared.

## Deliverable

A single Jupyter notebook that:

1. Loads the prepared CSV.
2. Builds input (5h) / output (12h) blocks with the required metadata.
3. Splits train (pre-2025) / test (2025).
4. Runs the weighted-KNN predictor with the white-box threshold mechanism.
5. Reports the evaluation metrics: μ₅, σ₅, μ₁₂, σ₁₂, coverage C.
6. Includes a small section showing predictions vs. truth on a few example windows (a plot or table is fine — use your judgment).

Code style: **senior, fluent, functional-leaning Python**. Prefer composition, small classes with single responsibilities, no over-engineering. NumPy-vectorized where it matters (the inner loop over historical windows must be vectorized — a Python loop over ~80k windows per test hour will be too slow). Type hints on public APIs. Concise docstrings, not essays.

## Method recap

For each test hour `t`:

- **Input window x**: last 5 hours `[t-4, t]` of features.
- **Truth y**: PM2.5 for the next 12 hours `[t+1, t+12]`.
- Find historical 5h windows `y_hist` that:
  - start at the same hour-of-day as `x`,
  - have the same day category (work / pre-holiday / day-off),
  - fall within ±20 days of the test date's day-of-year (across all training years),
  - have no missing values in either the 5h input or the following 12h.
- Compute aggregated distance:

```
D = w1·d1 + w2·d2 + w3·d3 + w4·d4 + w5·d5
```

  with five Euclidean sub-distances over the 5-hour window:

| Group | Features |
|------:|----------|
| d₁    | wind_speed, wind_sin, wind_cos |
| d₂    | temperature |
| d₃    | O₃, NO, NO₂ |
| d₄    | PM10 |
| d₅    | PM2.5 |

  Initial weights: all `0.2`. Constraint: `sum(w) = 1`.

The five sub-distances d₁..d₅ must be computed as five independent Euclidean distances and only then combined by the weighted sum. Do not fuse them into a single weighted-Euclidean over all features (i.e. do not push the weights down into the squared-difference sum). The weights apply at the aggregation step, after each d_j is fully computed, so they can be tuned later without touching the distance code.

Select all neighbors with D ≤ S (initial S = 0.5). This is a threshold selection, not top-K.
If no neighbor qualifies → return "no prediction" (white-box principle).
Otherwise, prediction is the per-hour mean of the neighbors' 12h PM2.5 futures.

- Select **all** neighbors with `D ≤ S` (initial `S = 0.5`). This is a *threshold* selection, not top-K.
- If no neighbor qualifies → return *"no prediction"* (white-box principle).
- Otherwise, prediction is the per-hour mean of the neighbors' 12h PM2.5 futures.

## Metrics

For each predicted window:

```
NAE_t  = |y_t - ŷ_t| / (y_max - y_min) * 100%
NMAE_H = mean over t=1..H of NAE_t
```

Aggregate across all predicted test windows:

- μ₅, σ₅  — mean and std of NMAE over 5h horizon
- μ₁₂, σ₁₂ — mean and std of NMAE over 12h horizon
- C = N_predicted / N_total · 100%

`y_max - y_min` is the range of PM2.5 in the **test set** (per the report).

## Existing code to adapt

There is a starter `KNNPredictor` class that uses plain Euclidean top-K — do not use it, create separate one, you can use different name, it does not match the method. Reuse the *idea* of `fit` / `predict` if it helps clarity, but the API will look different (threshold, weights, metadata-aware).

There is a `make_blocks` function that flattens features per block. **Extend it** so each block also carries the metadata needed for filtering: `date`, `start_hour`, `day_category`. Either return structured records or a small dataclass — your call. Keep it pure and testable.

You may keep features as 2D `(n_blocks, hours * features)` arrays for fast distance math, as long as you also store the column slices for each of the 5 distance groups (so `d_j` can be computed by slicing).

## Architecture sketch (suggestion, not prescription)

Reasonable decomposition:

- `Block` / `BlockSet` — input array, output array, metadata (date, start_hour, day_category), and the column slices for each distance group.
- `WeightedDistance` — given group slices and weights, computes `D(x, Y_hist)` vectorized over all candidates at once. Returns shape `(n_candidates,)`.
- `NeighborFilter` — given a query window's metadata, returns a boolean mask over the historical BlockSet (hour match, day_category match, ±20-day seasonal window). Pre-compute what you can (e.g. day-of-year arrays) at construction time.
- `WeightedKNNForecaster` — `.fit(train_blocks)`, `.predict(query_block) -> ndarray | None`. Composes the filter + distance + threshold logic.
- `Evaluator` — runs the sliding-window test loop, applies the forecaster, accumulates NAEs, returns the five metrics. Also exposes per-window results for diagnostics.

The ±20-day seasonal filter must handle year wrap-around (Dec 25 ↔ Jan 5 is 11 days apart, not 354).

## Things to get right

- **Vectorize the candidate scoring.** For each query, compute all 5 sub-distances against the filtered candidate matrix in NumPy, not in a Python loop. Filtering can use boolean masks; do the masking *before* the sqrt-and-sum to save work.
- **No leakage.** Test is 2025; training/history is everything before 2025. The historical block set should not contain 2025 rows.
- **Skip incomplete test windows.** If the 5h input or 12h truth for hour `t` has gaps, skip that `t` (it doesn't count toward `N_total` or `N_predicted` — re-read the test algorithm in the report to confirm; match the report's behavior exactly).
- **Weights and threshold are constructor parameters** of the forecaster, not hard-coded. They will be tuned later.
- **Return type for predictions** should clearly distinguish "no prediction" from a zero prediction. `None` or `np.nan`-filled array — pick one and be consistent.

## Notebook structure

Cells, roughly:

1. Imports + config (paths, feature lists, hyperparameters `weights`, `S`).
2. Load CSV → DataFrame; quick sanity print (shape, date range, columns).
3. Build train + test BlockSets via the extended `make_blocks`. Print sizes.
4. Instantiate `WeightedKNNForecaster`. Show one example prediction with the chosen neighbors' count and distance distribution.
5. Run `Evaluator` over the test set. Print μ₅, σ₅, μ₁₂, σ₁₂, C.
6. Plot a couple of predicted-vs-truth examples (one "good" coverage hour, one "no prediction" if any).

Keep markdown cells short — a sentence or two of context per code cell, not paragraphs.

## What I do *not* want

- A monolithic 500-line cell.
- A class hierarchy where one would do.
- Re-implementing data cleaning — the CSV is already clean and normalized.
- pandas-heavy inner loops. Use NumPy for the hot path.
- "Defensive" exception handling that hides bugs. Let it fail loudly during development; add narrow checks only where the report's logic requires them (e.g. "no prediction" branch).
- Unused imports, commented-out code, print-statement debugging left behind.

When in doubt about a methodology detail, re-read the relevant subsection of the report rather than guessing.
