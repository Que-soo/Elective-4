"""Run the whole ReviewLens analysis in order (do this BEFORE the presentation;
train_grid.py takes a while).

    python run_all.py                       raw file at data/jollibee_reviews_raw.csv
    python run_all.py path/to/raw.csv
"""
import subprocess
import sys

raw = sys.argv[1:]  # optional raw-file path, passed to preprocess.py
steps = [
    ["preprocess.py", *raw],
    ["eda.py"],
    ["train_grid.py"],
    ["evaluate.py"],
    ["extract_complaints.py"],
    ["validate_themes.py", "examples"],
]
for step in steps:
    print(f"\n>>> python {' '.join(step)}", flush=True)
    if subprocess.run([sys.executable, *step]).returncode != 0:
        sys.exit(f"Stopped: {step[0]} failed.")
print("\nAll steps finished. Results are in output/, output/eda/ and output/eval/.")
