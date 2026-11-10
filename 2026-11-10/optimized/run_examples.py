# Path represents a file location.
from pathlib import Path

# numpy gives fast arrays of numbers.
import numpy as np

# Our own file knn_objective.py, which sits next to this script.
import knn_objective

# The results file is written next to this script.
RESULT_PATH = Path(__file__).resolve().parent / "results.txt"
# The same threshold S is used for both examples.
THRESHOLD = 0.5

# The examples to run: a name and five weights in the order wind, temperature, gases, PM10, PM2.5.
EXAMPLES = [
    # All five groups equally important.
    ("equal", [0.20, 0.20, 0.20, 0.20, 0.20]),
    # A guess: the PM2.5 history matters most, temperature least.
    ("guess", [0.15, 0.05, 0.15, 0.20, 0.45]),
]


# Runs both examples and writes a short table.
def main():
    # Do the slow preparation once.
    distances, is_candidate, candidate_futures, true_futures, pm25_range = knn_objective.prepare()
    # Number of test blocks.
    test_count = len(true_futures)

    # The lines of the table; start with the header.
    lines = ["name   wind  temp  gases PM10  PM2.5  S     NMAE12 %  refused  coverage %"]

    # Go through the examples; each one is unpacked into its name and its weights.
    for name, weights in EXAMPLES:
        # Compute the two objectives for these weights.
        mean_error, refused_count = knn_objective.evaluate(
            distances, is_candidate, candidate_futures, true_futures, pm25_range, np.array(weights), THRESHOLD
        )
        # Coverage: the percentage of test blocks that did get a prediction.
        coverage = (test_count - refused_count) / test_count * 100.0
        # The five weights as text, each with two decimals, joined by two spaces.
        weights_text = "  ".join(f"{weight:.2f}" for weight in weights)
        # One table row; ":<6" pads to 6 characters, ":.3f" prints 3 decimals.
        lines.append(f"{name:<6} {weights_text}   {THRESHOLD:<5} {mean_error:<9.3f} {refused_count:<8} {coverage:.2f}")

    # Join the lines with line breaks into one text.
    text = "\n".join(lines) + "\n"
    # Write the text to the results file.
    RESULT_PATH.write_text(text)
    # Also show it on the screen.
    print(text)


# This block runs only when the file is started directly (like public static void main).
if __name__ == "__main__":
    # Start the program.
    main()
