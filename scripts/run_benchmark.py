"""Run trackers over every sequence of a split and save their boxes and timings to runs/.

For each tracker and sequence this writes two files:
    runs/<split>/<tracker>/<sequence>.txt        one "x,y,w,h" line per frame (frame 0 = init box)
    runs/<split>/<tracker>/<sequence>_time.txt   seconds spent inside the tracker (init + updates)

Nothing is drawn, so it runs at full speed. Sequences whose box file already exists are
skipped, so an interrupted run can be restarted and continues where it stopped.
Score the results afterwards with scripts/evaluate.py.

Usage (from the repo root, with the venv active):
    python scripts/run_benchmark.py --split val --trackers csrt kcf mil
"""

import argparse                 # reads command-line options
import sys                      # gives access to sys.path, the list of folders Python imports from
import time                     # perf_counter(): a precise clock for timing the tracker
from pathlib import Path

import cv2                      # OpenCV: read the frames
import numpy as np              # stack boxes into an array and save them as text
from tqdm import tqdm           # progress bar for long loops

# Make `sot` importable from scripts/ (see show_sequence.py for the full explanation).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sot.dataset import ROOT, load_split  # noqa: E402
from sot.trackers import TRACKERS, make_tracker  # noqa: E402


def track_sequence(name: str, seq) -> tuple[np.ndarray, float]:
    """Run tracker `name` on one sequence (OPE: init once on frame 0, never re-initialised).

    Returns the predicted boxes as an (N, 4) int array, row 0 being the init box, and the
    seconds spent inside init() and update(). Reading images is not timed, so the speed is
    the tracker's own, not the disk's.
    """
    tracker = make_tracker(name)                # a fresh tracker for every sequence
    first = cv2.imread(str(seq.frames[0]))

    seconds = 0.0                               # running total of tracker time only
    t0 = time.perf_counter()
    tracker.init(first, seq.init_box)           # learn the target from frame 0
    seconds += time.perf_counter() - t0

    boxes = [seq.init_box]                      # frame 0's prediction is the given init box (standard OPE)
    for path in seq.frames[1:]:                 # every later frame
        img = cv2.imread(str(path))             # not timed: disk speed is not the tracker's fault
        t0 = time.perf_counter()
        ok, box = tracker.update(img)
        seconds += time.perf_counter() - t0
        # When the tracker reports failure, keep its previous box instead of a meaningless one.
        # It is still scored on that frame, so being lost still costs it (it is not skipped).
        boxes.append(tuple(box) if ok else boxes[-1])
    return np.array(boxes, dtype=int), seconds


def main() -> None:
    """Run every requested tracker on every sequence of the split, saving results as it goes."""
    # --- Command-line options -------------------------------------------------------
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="val")
    # nargs="+" accepts one or more names; choices rejects names that are not in TRACKERS.
    parser.add_argument("--trackers", nargs="+", default=["csrt"], choices=list(TRACKERS))
    args = parser.parse_args()

    seqs = load_split(args.split)
    for name in args.trackers:
        out = ROOT / "runs" / args.split / name          # e.g. <root>/runs/val/csrt
        out.mkdir(parents=True, exist_ok=True)           # create it (and runs/val/) if missing
        # tqdm wraps the list and draws a progress bar; desc is the label shown in front of it.
        for seq in tqdm(seqs, desc=name):
            box_file = out / f"{seq.name}.txt"
            if box_file.exists():                        # finished in an earlier run: skip it
                continue
            boxes, seconds = track_sequence(name, seq)
            # Save the time first and the boxes last: the box file is the "done" marker, so a
            # crash between the two writes never leaves a sequence that looks finished.
            (out / f"{seq.name}_time.txt").write_text(f"{seconds:.4f}\n")
            # One "x,y,w,h" line per frame, the same format as the annotation files, so
            # read_numbers() from sot/dataset.py can load it back.
            np.savetxt(box_file, boxes, fmt="%d", delimiter=",")
        print(f"{name}: results in {out}")


# Run main() only when the file is executed directly, not when imported.
if __name__ == "__main__":
    main()
