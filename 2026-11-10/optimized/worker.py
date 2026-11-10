# sys gives access to standard input, standard output and the exit code.
import sys

# numpy gives fast arrays of numbers.
import numpy as np

# Our own file knn_objective.py, which sits next to this script.
import knn_objective

# The C++ optimizer sends this many numbers per line: 5 weights and 1 threshold.
VALUE_COUNT = 6


# The worker: the C++ optimizer starts it once and then asks for one evaluation per line.
def main():
    # Do the slow preparation once, before the first request.
    distances, is_candidate, candidate_futures, true_futures, pm25_range = knn_objective.prepare()

    # "for line in sys.stdin" reads line by line and ends when the C++ side closes the pipe.
    for line in sys.stdin:
        # Split the line on spaces into separate text pieces.
        fields = line.split()

        # A line with a wrong number of values means the two programs disagree on the protocol.
        if len(fields) != VALUE_COUNT:
            # Write the problem to the error output (standard output is reserved for results).
            print(f"worker: expected {VALUE_COUNT} values, got {len(fields)}: {line!r}", file=sys.stderr, flush=True)
            # A non-zero exit code tells the C++ side that the worker failed.
            return 1

        # The first 5 values are the weights; convert each text piece to a number.
        weights = np.array([float(field) for field in fields[:5]])
        # The last value is the threshold S.
        threshold = float(fields[5])

        # Compute the two objectives for this point.
        mean_error, refused_count = knn_objective.evaluate(
            distances, is_candidate, candidate_futures, true_futures, pm25_range, weights, threshold
        )

        # Send both results back as one line; !r prints the number with full precision.
        sys.stdout.write(f"{mean_error!r} {refused_count}\n")
        # Push the line out immediately, otherwise the C++ side would wait forever.
        sys.stdout.flush()

    # Exit code 0 means everything went fine.
    return 0


# This block runs only when the file is started directly (like public static void main).
if __name__ == "__main__":
    # End the program with the exit code returned by main().
    sys.exit(main())
