#!/bin/bash
#
# Builds the C++ optimizer against the KNN worker and runs one optimisation.
#
# Usage:
#   ./run_on_mpi_machine.sh <path to the "parallel" repository> [MPI ranks] [evaluations]
# Example:
#   ./run_on_mpi_machine.sh ~/parallel 4 10000
#
# Optional: PYTHON=/path/to/python chooses the interpreter (it needs numpy and pandas).
#
# State when this was written (2026-11-10, on a Mac without MPI):
#   - the Python side was run and gives NMAE12 6.566 % / 861 refused for equal weights 0.2, S 0.5
#   - the C++ changes (PythonFunction.h, TestParallelHybridAlgorithm.cpp) were never compiled or run
#   - data/3_scaled/scaled.csv is ignored by git, so it must be copied to this machine by hand
#
# Decision vector: 5 weights (wind, temperature, gases, PM10, PM2.5), each in [0, 1], and S in [0, 14].
# Objectives: mean NMAE12 in percent, and the number of refused test windows (out of 5630).

# Stop at the first command that fails.
set -e

# Arguments, with defaults for the optional ones.
PARALLEL_DIR="$1"
RANKS="${2:-4}"
EVALUATIONS="${3:-10000}"

# The folder this script is in, and the repository root two levels above it.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Files and programs used below.
WORKER="$SCRIPT_DIR/worker.py"
DATA="$REPO_DIR/data/3_scaled/scaled.csv"
RESULT="$SCRIPT_DIR/pareto_result.txt"
PYTHON_BIN="$(command -v "${PYTHON:-python3}")"

# Step 1: check that everything needed is present.
if [ -z "$PARALLEL_DIR" ] || [ ! -f "$PARALLEL_DIR/src/PythonFunction.h" ]; then
    echo "First argument must be the path to the 'parallel' repository."
    exit 1
fi
if [ ! -f "$DATA" ]; then
    echo "Missing $DATA (ignored by git: copy it from the other machine)."
    exit 1
fi
command -v mpicxx > /dev/null || { echo "mpicxx not found."; exit 1; }
command -v mpirun > /dev/null || { echo "mpirun not found."; exit 1; }
"$PYTHON_BIN" -c "import numpy, pandas" || { echo "$PYTHON_BIN needs numpy and pandas."; exit 1; }

# Step 2: check the Python side alone. Expected first row: equal ... 6.566 861 84.71
"$PYTHON_BIN" "$SCRIPT_DIR/run_examples.py"

# Step 3: check the worker protocol alone. Expected output line: 6.566439 861
echo "0.2 0.2 0.2 0.2 0.2 0.5" | "$PYTHON_BIN" -u "$WORKER"

# Step 4: build the optimizer with the interpreter, the worker path and the evaluation count compiled in.
cd "$PARALLEL_DIR/build"
make clean || true
make FLAGS="-O2 -g -Wall -DPYTHON_N_MAX=$EVALUATIONS -DPYTHON_INTERPRETER='\"$PYTHON_BIN\"' -DPYTHON_WORKER_SCRIPT='\"$WORKER\"'"

# Step 5: run the optimisation; every MPI rank starts its own Python worker (about 75 MB each).
mpirun -np "$RANKS" ./optimize > "$RESULT"

# Step 6: show the end of the output (evaluation count and run time).
tail -n 8 "$RESULT"
echo "Full output, with the Pareto points and their weights and S: $RESULT"
