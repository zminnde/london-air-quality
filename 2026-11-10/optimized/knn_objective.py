# Path is like java.nio.file.Path: an object that represents a file location.
from pathlib import Path

# numpy gives fast arrays of numbers (like double[] / double[][] with built-in maths).
import numpy as np
# pandas reads CSV files into a table (rows and named columns).
import pandas as pd

# __file__ is this script's own location; parents[2] goes up to the repository root folder.
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "3_scaled" / "scaled.csv"
# Rows from this year are the test data; all earlier years are the training data.
TEST_YEAR = 2025
# How many past hours the model looks at.
INPUT_HOURS = 5
# How many future hours the model predicts.
OUTPUT_HOURS = 12
# A training block is a candidate only if its day of year is at most this many days away.
SEASON_RADIUS_DAYS = 20
# Error returned when no window at all could be predicted (100 % is the worst possible error).
WORST_ERROR = 100.0

# The columns of the CSV that describe one hour.
FEATURE_COLUMNS = ["O3", "NO", "NO2", "PM10", "PM2.5", "wind_speed", "temp", "wind_dir_sin", "wind_dir_cos"]

# The five groups of columns; each group gets its own distance and its own weight.
GROUPS = [
    # Group 0: wind.
    ["wind_speed", "wind_dir_sin", "wind_dir_cos"],
    # Group 1: temperature.
    ["temp"],
    # Group 2: gases.
    ["O3", "NO", "NO2"],
    # Group 3: PM10.
    ["PM10"],
    # Group 4: PM2.5.
    ["PM2.5"],
]
# How many groups there are (5).
GROUP_COUNT = len(GROUPS)


# Cuts a table of hourly rows into blocks: 5 input hours followed by 12 future hours.
def make_blocks(table):
    # All feature columns as a 2D array: one row per hour, one column per feature.
    features = table[FEATURE_COLUMNS].values
    # The PM2.5 column alone, as a 1D array.
    pm25 = table["PM2.5"].values
    # Each timestamp converted to a whole number of hours, so consecutive hours differ by exactly 1.
    hour_numbers = table["datetime"].values.astype("datetime64[h]").astype(np.int64)
    # Hour of the day (1..24) of every row.
    hours = table["time"].values
    # Day category (kind of day) of every row.
    day_categories = table["day_category"].values
    # Day of the year (1..366) of every row.
    days_of_year = table["datetime"].dt.day_of_year.values

    # Length of one whole block: 5 + 12 = 17 hours.
    block_length = INPUT_HOURS + OUTPUT_HOURS

    # Empty lists (like ArrayList) that collect the pieces of every valid block.
    block_inputs = []
    block_futures = []
    block_hours = []
    block_day_categories = []
    block_days_of_year = []

    # Try every possible start row; range(n) counts 0, 1, ..., n-1.
    for start in range(len(table) - block_length + 1):
        # The hour numbers of the 17 rows of this block; a[x:y] means "from x up to, not including, y".
        block_hour_numbers = hour_numbers[start: start + block_length]
        # np.diff gives the differences between neighbours; any difference other than 1 means a missing hour.
        if not np.all(np.diff(block_hour_numbers) == 1):
            # Skip blocks that have a gap in time.
            continue
        # The first 5 hours with all features are the input of the block.
        block_inputs.append(features[start: start + INPUT_HOURS])
        # The PM2.5 values of the next 12 hours are what really happened afterwards.
        block_futures.append(pm25[start + INPUT_HOURS: start + block_length])
        # Remember the hour of day of the block's first row.
        block_hours.append(hours[start])
        # Remember the day category of the block's first row.
        block_day_categories.append(day_categories[start])
        # Remember the day of year of the block's first row.
        block_days_of_year.append(days_of_year[start])

    # Convert the lists to numpy arrays and return all five together (Python can return several values).
    return (
        # Shape: (number of blocks, 5 hours, 9 features).
        np.array(block_inputs, dtype=np.float32),
        # Shape: (number of blocks, 12 hours).
        np.array(block_futures, dtype=np.float32),
        # Shape: (number of blocks).
        np.array(block_hours, dtype=np.int16),
        # Shape: (number of blocks).
        np.array(block_day_categories, dtype=np.int16),
        # Shape: (number of blocks).
        np.array(block_days_of_year, dtype=np.int16),
    )


# Does all the slow work once: everything that does not depend on the weights or the threshold.
def prepare():
    # Read the CSV file into a table.
    table = pd.read_csv(DATA_PATH)
    # Build a full timestamp from the date and the hour (hour 1 means 00:00, so subtract 1).
    table["datetime"] = pd.to_datetime(table["date"]) + pd.to_timedelta(table["time"] - 1, unit="h")
    # Sort the rows by time and renumber them 0, 1, 2, ...
    table = table.sort_values("datetime").reset_index(drop=True)

    # Rows before the test year are the training data.
    train_table = table[table["datetime"].dt.year < TEST_YEAR]
    # Rows of the test year are the test data.
    test_table = table[table["datetime"].dt.year == TEST_YEAR]

    # Cut the training rows into blocks.
    train_inputs, train_futures, train_hours, train_day_categories, train_days_of_year = make_blocks(train_table)
    # Cut the test rows into blocks.
    test_inputs, test_futures, test_hours, test_day_categories, test_days_of_year = make_blocks(test_table)

    # Number of test blocks.
    test_count = len(test_inputs)

    # For every test block: the list of training blocks that are allowed to be its neighbours.
    candidate_lists = []
    # Go through the test blocks one by one.
    for i in range(test_count):
        # True for every training block that starts at the same hour of the day.
        same_hour = train_hours == test_hours[i]
        # True for every training block that has the same day category.
        same_day_category = train_day_categories == test_day_categories[i]
        # How many days apart the two blocks are within the year.
        day_gap = np.abs(train_days_of_year - test_days_of_year[i])
        # The year is a circle (31 Dec is next to 1 Jan), so take the shorter way round.
        same_season = np.minimum(day_gap, 365 - day_gap) <= SEASON_RADIUS_DAYS
        # Positions of the training blocks that pass all three checks (& is element-wise AND).
        candidates = np.flatnonzero(same_hour & same_day_category & same_season)
        # Store the positions for this test block.
        candidate_lists.append(candidates)

    # The longest candidate list decides how wide the tables must be.
    max_candidates = max(len(candidates) for candidates in candidate_lists)

    # Table [test block][candidate][group] of distances, pre-filled with zeros.
    distances = np.zeros((test_count, max_candidates, GROUP_COUNT), dtype=np.float32)
    # Table [test block][candidate]: True for a real candidate, False for an empty (padding) cell.
    is_candidate = np.zeros((test_count, max_candidates), dtype=bool)
    # Table [test block][candidate][future hour] of the candidates' PM2.5 futures, pre-filled with zeros.
    candidate_futures = np.zeros((test_count, max_candidates, OUTPUT_HOURS), dtype=np.float32)

    # Fill both tables, one test block at a time.
    for i in range(test_count):
        # Positions of this test block's candidates.
        candidates = candidate_lists[i]
        # How many candidates this test block has.
        candidate_count = len(candidates)
        # Difference between every candidate's input and this test input; shape (candidates, 5 hours, 9 features).
        difference = train_inputs[candidates] - test_inputs[i]
        # Compute one distance per group.
        for j in range(GROUP_COUNT):
            # Column numbers of this group's features; [f(x) for x in list] builds a new list (like stream().map()).
            columns = [FEATURE_COLUMNS.index(name) for name in GROUPS[j]]
            # Keep only this group's columns; ":" means "all" in that dimension.
            group_difference = difference[:, :, columns]
            # Euclidean distance: square everything, sum over hours and features, take the square root.
            group_distance = np.sqrt((group_difference ** 2).sum(axis=(1, 2)))
            # Write the distances into the first cells of the row; the remaining cells stay empty.
            distances[i, :candidate_count, j] = group_distance
        # Mark the first cells of the row as real candidates.
        is_candidate[i, :candidate_count] = True
        # Write the candidates' futures into the first cells of the row.
        candidate_futures[i, :candidate_count] = train_futures[candidates]

    # Range of the real PM2.5 values in the test data; used to express the error in percent.
    pm25_range = float(test_futures.max() - test_futures.min())

    # Return everything that evaluate() needs.
    return distances, is_candidate, candidate_futures, test_futures, pm25_range


# The objective function: fast, called once for every point the optimizer wants to try.
def evaluate(distances, is_candidate, candidate_futures, true_futures, pm25_range, weights, threshold):
    # Number of test blocks.
    test_count = len(true_futures)

    # Combined distance = sum of weight[j] * distance[j]; "@" multiplies the last dimension by the weights.
    total_distance = distances @ weights
    # True where a real candidate is close enough to be a neighbour; shape (test blocks, candidates).
    is_neighbour = (total_distance <= threshold) & is_candidate
    # Number of neighbours of every test block (True counts as 1; axis=1 sums along the candidates).
    neighbour_count = is_neighbour.sum(axis=1)
    # True for the test blocks that have at least one neighbour.
    has_prediction = neighbour_count > 0
    # Second objective: how many test blocks have no neighbour, so no prediction is made.
    refused_count = test_count - int(has_prediction.sum())

    # If nothing could be predicted there is no error to average.
    if refused_count == test_count:
        # Report the worst result.
        return WORST_ERROR, test_count

    # For every test block t and hour h: add up the futures of its neighbours (sum over candidates c).
    future_sums = np.einsum("tc,tch->th", is_neighbour, candidate_futures)
    # The neighbour counts of the predicted blocks as a column, so each row is divided by its own count.
    counts_column = neighbour_count[has_prediction].reshape(-1, 1)
    # Prediction = average future of the neighbours; a[mask] keeps only the rows where mask is True.
    predictions = future_sums[has_prediction] / counts_column
    # Absolute error of every predicted hour, as a percentage of the PM2.5 range.
    errors = np.abs(true_futures[has_prediction] - predictions) / pm25_range * 100.0
    # First objective: the mean error over all predicted windows and all 12 hours.
    mean_error = float(errors.mean())

    # Return both objectives.
    return mean_error, refused_count
