# London Air Quality — PM2.5 Forecaster

## Overview

`knn_forecaster.ipynb` implements a **threshold-based weighted KNN forecaster** that predicts PM2.5 particulate matter concentration **12 hours ahead** using a 5-hour input window. The algorithm is a white-box model: instead of a fixed number of neighbours K, it uses a distance threshold S — returning "no prediction" rather than fabricating one when no sufficiently similar historical windows exist.

The full methodological description (in Lithuanian) is in `report_euclidean/ŽiemysWeightedEuclideanDistanceReport.tex`.

---

## Data

| File | Description |
|---|---|
| `data/3_scaled/scaled.csv` | Min-max scaled hourly observations, 2012-01-01 – 2025-12-30, 82 356 rows |

Columns used: `date`, `time` (1–24), `O3`, `NO`, `NO2`, `PM10`, `PM2.5`, `wind_speed`, `temp`, `day_category`, `wind_dir_sin`, `wind_dir_cos`.

`day_category`: `0` = workday, `1` = day-off, `2` = pre-holiday.  
All feature values are scaled to `[-1, 1]`.

---

## Algorithm — step by step

### Step 1 — Build sliding-window blocks

A **block** is a gap-free sequence of `IN_H + OUT_H = 17` consecutive hourly rows:
- `X`: rows `[i, i+5)` → shape `(5, 9)` — the 5-hour input window across all 9 features
- `Y`: rows `[i+5, i+17)` → shape `(12,)` — PM2.5 values for the 12-hour forecast horizon

Blocks with any missing hour (timestamp gap > 1 h) are silently dropped.

```
raw data (hourly):  h1  h2  h3  h4  h5 | h6  h7  ... h17
                    ←—— X (input) ————→ | ←—— Y (target) ——→
```

The dataset is split strictly by year: **2012–2024 → train** (61 335 blocks), **2025 → test** (5 630 blocks).

### Step 2 — Filter candidates (NeighborFilter)

For a query block with metadata `(hour, day_cat, doy)`, the candidate pool is restricted to train blocks that satisfy **all three** conditions simultaneously:

| Filter | Condition |
|---|---|
| Hour sync | `train.hour == query.hour` |
| Day category | `train.day_cat == query.day_cat` |
| Seasonality | `circular_distance(train.doy, query.doy) ≤ 20 days` |

The circular day-of-year distance wraps around year boundaries (e.g., Jan 1 and Dec 31 are 1 day apart).  
A typical candidate pool after filtering is ~100–200 blocks.

### Step 3 — Compute weighted Euclidean distance (WeightedDistance)

Five independent Euclidean sub-distances are computed between the query window `x` and each candidate window `y`, each covering a specific feature group over the 5 input hours:

| Sub-distance | Feature group | Dimensions |
|---|---|---|
| `d1` | Wind: `wind_speed`, `wind_dir_sin`, `wind_dir_cos` | 3 × 5 = 15 |
| `d2` | Temperature: `temp` | 1 × 5 = 5 |
| `d3` | Gas pollutants: `O3`, `NO`, `NO2` | 3 × 5 = 15 |
| `d4` | Coarse particles: `PM10` | 1 × 5 = 5 |
| `d5` | Fine particles: `PM2.5` | 1 × 5 = 5 |

Each sub-distance is the standard Euclidean norm over its group's flattened dimensions:

```
d_j(x, y) = sqrt( sum over hours and features in group j of (x - y)^2 )
```

The aggregate distance is a weighted sum:

```
D(x, y) = w1·d1 + w2·d2 + w3·d3 + w4·d4 + w5·d5
```

Weights must sum to 1. Default (equal): `w = [0.2, 0.2, 0.2, 0.2, 0.2]`.  
Vectorised implementation operates on the entire candidate pool at once: `Y` has shape `(n_cand, 5, 9)`, result `D` has shape `(n_cand,)`.

### Step 4 — Threshold selection

All candidates with `D ≤ S` become neighbours. If no candidate qualifies, the model returns `None` (no prediction).

Default threshold `S = 0.5`.  
With equal weights and features scaled to `[-1, 1]`, the theoretical maximum aggregate distance is `D_max ≈ 5.78`, so `S = 0.5` corresponds to ≈ 8.6% of the maximum — strict enough for quality neighbours, permissive enough for practical coverage.

### Step 5 — Forecast

The 12-hour PM2.5 forecast is the **unweighted mean** of all qualifying neighbours' future PM2.5 values:

```
ŷ_t = mean over qualifying neighbours k of y_t^(k),   t = 1..12
```

K varies dynamically per query (there is no fixed K).

---

## Internal data structures

```python
@dataclass
class BlockSet:
    X:        np.ndarray  # (n, 5, 9)  float32  — input features
    Y:        np.ndarray  # (n, 12)    float32  — PM2.5 targets
    dates:    np.ndarray  # (n,)       str      — date string of block start
    doys:     np.ndarray  # (n,)       int16    — day-of-year of block start
    hours:    np.ndarray  # (n,)       int8     — hour (1–24) of block start
    day_cats: np.ndarray  # (n,)       int8     — day category (0/1/2)
```

Key classes:

| Class | Responsibility |
|---|---|
| `WeightedDistance` | Vectorised D(x, Y) computation over all candidates |
| `NeighborFilter` | Boolean mask array from hour / day_cat / seasonality |
| `WeightedKNNForecaster` | Orchestrates filter → distance → threshold → mean |
| `EvalResult` | μ₅, σ₅, μ₁₂, σ₁₂, coverage, per-window records |

---

## Evaluation metrics

All metrics are computed only over test windows that received a prediction.

| Metric | Formula | Meaning |
|---|---|---|
| NAE_t | `\|y_t − ŷ_t\| / (y_max − y_min) × 100%` | Per-hour normalised absolute error |
| NMAE₅ | mean of NAE over hours 1–5 | Short-range accuracy |
| NMAE₁₂ | mean of NAE over hours 1–12 | Full-horizon accuracy |
| μ / σ | mean and std of NMAE across all predicted windows | Summary statistics |
| Coverage C | `N_pred / N_total × 100%` | Fraction of windows that got a prediction |

`y_max` and `y_min` are the PM2.5 extremes in the **test set only** (not the full dataset).

### Results on 2025 test set (equal weights, S = 0.5)

| μ₅ | σ₅ | μ₁₂ | σ₁₂ | Coverage |
|---|---|---|---|---|
| 7.08% | 4.74% | 7.70% | 4.79% | 99.66% |

### Weight sensitivity (S = 0.4)

| Config | weights [d1,d2,d3,d4,d5] | μ₅ | μ₁₂ | C |
|---|---|---|---|---|
| baseline | [0.20, 0.20, 0.20, 0.20, 0.20] | 5.61% | 6.28% | 63.2% |
| pm25_heavy | [0.10, 0.10, 0.10, 0.30, 0.40] | 5.50% | 6.34% | 90.8% |
| pm_dominant | [0.05, 0.05, 0.10, 0.30, 0.50] | 5.39% | 6.38% | 96.8% |
| particles | [0.10, 0.10, 0.10, 0.35, 0.35] | 5.57% | 6.39% | 90.7% |
| meteo_heavy | [0.30, 0.30, 0.30, 0.05, 0.05] | 6.48% | 6.86% | 38.6% |

Heavier weight on PM2.5 (d5) improves μ₅ and increases coverage; heavier meteorology weight (d1+d2) hurts both.

### Threshold vs. accuracy / coverage trade-off (equal weights)

| S | μ₁₂ | C |
|---|---|---|
| 0.30 | 5.74% | 69.9% |
| 0.35 | 6.02% | 83.1% |
| 0.40 | 6.30% | 91.7% |
| 0.45 | 6.48% | 96.1% |
| 0.50 | 6.65% | 98.3% |

Lower S → fewer but more similar neighbours → lower error, lower coverage.

---

## Discrepancies between notebook and report

| # | Report states | Notebook behaviour |
|---|---|---|
| 1 | Initial S = 0.5 | `DEFAULT_THRESHOLD = 0.5` matches, but the weight-comparison grid uses S = 0.4 without explanation |
| 2 | Weights and S will be found by a formal optimisation algorithm | Notebook only does manual grid search over hand-picked configs; no optimiser is implemented |
| 3 | Stale cell output | The example-prediction cell output prints "S=0.8" but the constant is now 0.5 — the output was captured during an earlier run with a different threshold value |

---

# Project Setup

## Prerequisites
- Python 3.13 or higher ([Download](https://python.org))
- Git

## Installation

1. Clone the repository
```bash
git clone https://github.com/zminnde/london-air-quality.git
cd london-air-quality
```

2. Create virtual environment
```bash
python3 -m venv .venv
```

3. Activate virtual environment
```bash
# Mac/Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

4. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

5. Run the project
```bash
python3 normalize_data.py
```