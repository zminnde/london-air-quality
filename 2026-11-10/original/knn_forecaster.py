from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

DATA_PATH = Path("data/3_scaled/scaled.csv")
TEST_YEAR = 2025
IN_H = 5
OUT_H = 12
SEASONAL_RADIUS = 20
DEFAULT_WEIGHTS = np.full(5, 0.2)
DEFAULT_THRESHOLD = 0.5

FEAT_COLS = ["O3", "NO", "NO2", "PM10", "PM2.5", "wind_speed", "temp", "wind_dir_sin", "wind_dir_cos"]
N_FEAT = len(FEAT_COLS)

GROUPS: list[list[int]] = [
    [FEAT_COLS.index("wind_speed"), FEAT_COLS.index("wind_dir_sin"), FEAT_COLS.index("wind_dir_cos")],
    [FEAT_COLS.index("temp")],
    [FEAT_COLS.index("O3"), FEAT_COLS.index("NO"), FEAT_COLS.index("NO2")],
    [FEAT_COLS.index("PM10")],
    [FEAT_COLS.index("PM2.5")],
]


@dataclass
class BlockSet:
    """Pre-built sliding windows over a time."""
    X: np.ndarray
    Y: np.ndarray
    dates: np.ndarray
    doys: np.ndarray
    hours: np.ndarray
    day_cats: np.ndarray

    def __len__(self) -> int:
        return len(self.X)


def make_blocks(df: pd.DataFrame) -> BlockSet:
    """Slide a (IN_H + OUT_H)-length window over df, keeping only gap-free blocks."""
    features = df[FEAT_COLS].values
    pm25 = df["PM2.5"].values
    dt_h = df["datetime"].values.astype("datetime64[h]").astype(np.int64)
    hours = df["time"].values
    day_cats = df["day_category"].values
    dates = df["date"].values
    doys = df["datetime"].dt.day_of_year.values

    N = len(df)
    total = IN_H + OUT_H

    Xs, Ys, ds, doy_arr, h_arr, cat_arr = [], [], [], [], [], []

    for i in range(N - total + 1):
        if not np.all(np.diff(dt_h[i: i + total]) == 1):
            continue
        Xs.append(features[i: i + IN_H])
        Ys.append(pm25[i + IN_H: i + total])
        ds.append(dates[i])
        doy_arr.append(doys[i])
        h_arr.append(hours[i])
        cat_arr.append(day_cats[i])

    return BlockSet(
        X=np.array(Xs, dtype=np.float32),
        Y=np.array(Ys, dtype=np.float32),
        dates=np.array(ds),
        doys=np.array(doy_arr, dtype=np.int16),
        hours=np.array(h_arr, dtype=np.int8),
        day_cats=np.array(cat_arr, dtype=np.int8),
    )


class WeightedDistance:
    """Aggregated weighted distance D = Σ w_j · d_j over five independent Euclidean sub-distances."""

    def __init__(self, groups: list[list[int]], weights: np.ndarray):
        assert abs(weights.sum() - 1.0) < 1e-6, "weights must sum to 1"
        self.groups = groups
        self.weights = weights

    def __call__(self, x: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """x: (IN_H, N_FEAT), Y: (n_cand, IN_H, N_FEAT) → D: (n_cand,)"""
        D = np.zeros(len(Y), dtype=np.float64)
        for w, idx in zip(self.weights, self.groups):
            diff = x[None, :, idx] - Y[:, :, idx]  # (n_cand, IN_H, |group|)
            d_j = np.sqrt((diff * diff).sum(axis=(1, 2)))
            D += w * d_j
        return D


class NeighborFilter:
    """Produces a boolean mask over train blocks that satisfy all filtering criteria."""

    def __init__(self, train: BlockSet, radius: int = SEASONAL_RADIUS):
        self._hours = train.hours
        self._day_cats = train.day_cats
        self._doys = train.doys.astype(np.int16)
        self._radius = radius

    def mask(self, hour: int, day_cat: int, doy: int) -> np.ndarray:
        hour_ok = self._hours == hour
        cat_ok = self._day_cats == day_cat
        diff = np.abs(self._doys - doy)
        seas_ok = np.minimum(diff, 365 - diff) <= self._radius
        return hour_ok & cat_ok & seas_ok


class WeightedKNNForecaster:
    """Threshold-based weighted KNN forecaster."""

    def __init__(
            self,
            weights: np.ndarray = DEFAULT_WEIGHTS,
            threshold: float = DEFAULT_THRESHOLD,
    ):
        self.threshold = threshold
        self._dist = WeightedDistance(GROUPS, weights)
        self._train: Optional[BlockSet] = None
        self._filter: Optional[NeighborFilter] = None

    def fit(self, train: BlockSet) -> WeightedKNNForecaster:
        self._train = train
        self._filter = NeighborFilter(train)
        return self

    def predict(self, x: np.ndarray, hour: int, day_cat: int, doy: int) -> Optional[np.ndarray]:
        """Returns (OUT_H,) forecast or None when no neighbor qualifies."""
        mask = self._filter.mask(hour, day_cat, doy)
        cand_X = self._train.X[mask]
        if len(cand_X) == 0:
            return None
        D = self._dist(x, cand_X)
        qualified = D <= self.threshold
        if not qualified.any():
            return None
        return self._train.Y[mask][qualified].mean(axis=0)


@dataclass
class EvalResult:
    mu5: float
    sigma5: float
    mu12: float
    sigma12: float
    coverage: float  # percent


def evaluate(forecaster: WeightedKNNForecaster, test: BlockSet, pm25_range: float) -> EvalResult:
    """Sliding-window evaluation over all valid test blocks."""
    nmae5, nmae12 = [], []
    n_pred = 0

    for i in range(len(test)):
        x = test.X[i]
        y_true = test.Y[i]
        pred = forecaster.predict(x, int(test.hours[i]), int(test.day_cats[i]), int(test.doys[i]))

        if pred is not None:
            n_pred += 1
            nae = np.abs(y_true - pred) / pm25_range * 100.0
            nmae5.append(nae[:5].mean())
            nmae12.append(nae.mean())

    a5 = np.array(nmae5)
    a12 = np.array(nmae12)
    return EvalResult(
        mu5=float(a5.mean()),
        sigma5=float(a5.std()),
        mu12=float(a12.mean()),
        sigma12=float(a12.std()),
        coverage=n_pred / len(test) * 100.0,
    )


def main():
    raw = pd.read_csv(DATA_PATH)
    raw["datetime"] = pd.to_datetime(raw["date"]) + pd.to_timedelta(raw["time"] - 1, unit="h")
    raw = raw.sort_values("datetime").reset_index(drop=True)

    train_df = raw[raw["datetime"].dt.year < TEST_YEAR]
    test_df = raw[raw["datetime"].dt.year == TEST_YEAR]

    train_blocks = make_blocks(train_df)
    test_blocks = make_blocks(test_df)

    print(f"Train blocks: {len(train_blocks):,}")
    print(f"Test blocks:  {len(test_blocks):,}")

    forecaster = WeightedKNNForecaster().fit(train_blocks)

    pm25_test = test_blocks.Y.ravel()
    pm25_range = float(pm25_test.max() - pm25_test.min())

    result = evaluate(forecaster, test_blocks, pm25_range)

    print(f"\n{'Metric':<12} {'Value':>10}")
    print("-" * 24)
    print(f"{'μ₅  (%)':<12} {result.mu5:>10.3f}")
    print(f"{'σ₅  (%)':<12} {result.sigma5:>10.3f}")
    print(f"{'μ₁₂ (%)':<12} {result.mu12:>10.3f}")
    print(f"{'σ₁₂ (%)':<12} {result.sigma12:>10.3f}")
    print(f"{'C   (%)':<12} {result.coverage:>10.2f}")


if __name__ == "__main__":
    main()
