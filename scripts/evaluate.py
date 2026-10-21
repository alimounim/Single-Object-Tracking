"""Score saved tracker results against the ground truth and print a ranked leaderboard.

Reads the box and time files written by scripts/run_benchmark.py, scores every sequence
with sot/metrics.py, averages the sequences (each one counts equally, the OTB / VisDrone
convention), and writes runs/<split>/leaderboard.csv sorted by AUC.

Usage (from the repo root, with the venv active):
    python scripts/evaluate.py --split val                      # every tracker found in runs/val/
    python scripts/evaluate.py --split val --per-seq            # also AUC per sequence
"""

import argparse                 # reads command-line options
import sys                      # gives access to sys.path, the list of folders Python imports from
from pathlib import Path

import numpy as np              # stacking and averaging the curves
import pandas as pd             # tables: build, sort, print and save as CSV

# Make `sot` importable from scripts/ (see show_sequence.py for the full explanation).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sot.dataset import ROOT, load_split, read_numbers  # noqa: E402
from sot.metrics import centre_error, iou, precision_curve, success_curve  # noqa: E402


def score_tracker(run_dir: Path, seqs) -> tuple[dict, dict]:
    """Score one tracker's saved results on every sequence.

    Returns (summary, per_seq): summary holds AUC, P@20 and FPS over the whole split;
    per_seq maps each sequence name to its own AUC.
    """
    succ, prec, per_seq = [], [], {}      # one curve per sequence, collected for averaging
    frames, seconds = 0, 0.0              # totals for the FPS
    for seq in seqs:
        pred = read_numbers(run_dir / f"{seq.name}.txt")     # (N, 4) predicted boxes
        if len(pred) != len(seq.gt):                         # a half-written or wrong file
            raise ValueError(f"{run_dir.name}/{seq.name}: {len(pred)} boxes, expected {len(seq.gt)}")
        s = success_curve(iou(pred, seq.gt))                 # 21 values for this sequence
        succ.append(s)
        prec.append(precision_curve(centre_error(pred, seq.gt)))  # 51 values for this sequence
        per_seq[seq.name] = s.mean()                         # this sequence's own AUC
        frames += len(seq)
        seconds += float((run_dir / f"{seq.name}_time.txt").read_text())
    # np.mean(..., axis=0) averages the list of curves point by point: every sequence counts equally.
    succ_curve = np.mean(succ, axis=0)
    prec_curve = np.mean(prec, axis=0)
    summary = {
        "tracker": run_dir.name,          # the folder name is the tracker name
        "AUC": succ_curve.mean(),         # area under the averaged success curve (max 0.952)
        "P@20": prec_curve[20],           # fraction of frames within 20 px, averaged over sequences
        "FPS": frames / seconds,          # tracker speed over the whole split
    }
    return summary, per_seq


def main() -> None:
    """Score every tracker folder in runs/<split>/, print the ranking and save it as CSV."""
    # --- Command-line options -------------------------------------------------------
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="val")
    # store_true: the flag takes no value; writing --per-seq sets it to True.
    parser.add_argument("--per-seq", action="store_true", help="also print the AUC of every sequence")
    args = parser.parse_args()

    seqs = load_split(args.split)
    split_dir = ROOT / "runs" / args.split
    # Every sub-folder of runs/<split>/ is one tracker's results.
    run_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir())

    rows, per_seq = [], {}
    for run_dir in run_dirs:
        summary, seq_auc = score_tracker(run_dir, seqs)
        rows.append(summary)
        per_seq[run_dir.name] = seq_auc   # becomes one column of the per-sequence table

    # --- Leaderboard -----------------------------------------------------------------
    # A DataFrame is a table: each dict in rows becomes one row, its keys become the columns.
    board = pd.DataFrame(rows).sort_values("AUC", ascending=False)   # best tracker first
    board.to_csv(split_dir / "leaderboard.csv", index=False, float_format="%.4f")
    # float_format rounds the printed numbers; index=False hides pandas' row numbers.
    print(board.to_string(index=False, float_format="%.3f"))
    print(f"saved {split_dir / 'leaderboard.csv'}")

    # --- Optional: AUC of every sequence, one column per tracker -------------------------
    if args.per_seq:
        # A dict of dicts becomes a table: outer keys = columns (trackers), inner keys = rows (sequences).
        table = pd.DataFrame(per_seq)
        table.index = [f"{i:2d} {name}" for i, name in enumerate(table.index)]  # add the --seq index
        print()
        print(table.to_string(float_format="%.3f"))


# Run main() only when the file is executed directly, not when imported.
if __name__ == "__main__":
    main()
