"""Print basic statistics of a VisDrone-SOT split: sequence length, target size, motion, attributes.

These numbers describe what a tracker is up against before we run any tracker:
how long it must survive, how few pixels the target has, how far it jumps between frames,
and which challenges (occlusion, camera motion, ...) are common.

Usage (from the repo root, with the venv active):
    python scripts/dataset_stats.py --split val
    python scripts/dataset_stats.py --split test-dev
"""

import argparse                 # reads command-line options like --split val
import sys                      # gives access to sys.path, the list of folders Python imports from
from pathlib import Path

import cv2                      # OpenCV: used here only to read one frame per sequence for its width
import numpy as np              # array maths: medians, percentiles, distances

# Python only imports from the script's own folder (scripts/) and installed packages, so
# `import sot` would fail. Adding the repo root (parents[1]) to the front of sys.path fixes that.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sot.dataset import ATTRIBUTES, load_split  # noqa: E402  (import must come after the sys.path line)


def main() -> None:
    """Load a split and print six groups of statistics about its sequences and boxes."""
    # --- Command-line options -------------------------------------------------------
    parser = argparse.ArgumentParser(description=__doc__)   # __doc__ = this file's docstring, shown by --help
    parser.add_argument("--split", default="val")           # which split to describe
    args = parser.parse_args()

    # --- Load ------------------------------------------------------------------------
    seqs = load_split(args.split)                     # list of Sequence objects
    gt = np.concatenate([s.gt for s in seqs])         # every box of every sequence, stacked: (total_frames, 4)
    w, h = gt[:, 2], gt[:, 3]                         # column 2 = widths, column 3 = heights

    # 1. Sequence length: how many frames the tracker must survive without losing the target.
    lengths = np.array([len(s) for s in seqs])        # len(s) works because Sequence defines __len__
    print(f"{len(seqs)} sequences, {lengths.sum()} frames, length min/median/max: "
          f"{lengths.min()} / {int(np.median(lengths))} / {lengths.max()}")

    # 2. Empty boxes: a width or height of 0 (or less) would mean "target not visible".
    # The | is element-wise OR, so `empty` is a True/False array with one entry per box.
    empty = (w <= 0) | (h <= 0)
    print(f"empty boxes: {empty.sum()}")              # summing True/False counts the Trues

    # 3. Target size in pixels: sqrt(w * h) is the side of a square with the same area,
    # one number that captures "how big" a box is regardless of its shape.
    size = np.sqrt(w[~empty] * h[~empty])             # ~ is element-wise NOT: keep only non-empty boxes
    # size < 20 is a True/False array; its .mean() is the fraction of Trues (True = 1, False = 0).
    print(f"target size (px): min {size.min():.1f} / median {np.median(size):.1f} / max {size.max():.1f}, "
          f"{(size < 20).mean() * 100:.0f}% under 20 px")

    # 4. Target size relative to the frame: frame sizes differ between sequences, so each
    # sequence is measured against its own width. One frame per sequence is enough, since
    # the resolution does not change inside a video.
    rel = []                                              # one value per sequence
    for s in seqs:
        frame_w = cv2.imread(str(s.frames[0])).shape[1]   # shape = (height, width, 3), so [1] is the width
        s_size = np.sqrt(s.gt[:, 2] * s.gt[:, 3])         # this sequence's box sizes
        rel.append(np.median(s_size) / frame_w * 100)     # typical size as a % of frame width
    print(f"target size (% of frame width): min {min(rel):.1f} / median {np.median(rel):.1f} / max {max(rel):.1f}")

    # 5. Motion: how far the box centre moves from one frame to the next. Trackers search only
    # near the last position, so big jumps are where they get lost. Computed per sequence so the
    # last frame of one video is never subtracted from the first frame of the next.
    steps = []                                            # one array per sequence
    for s in seqs:
        c = s.gt[:, :2] + s.gt[:, 2:] / 2                 # centre = (x + w/2, y + h/2) for every frame
        d = np.linalg.norm(np.diff(c, axis=0), axis=1)    # diff: (dx, dy) per frame; norm: sqrt(dx² + dy²)
        steps.append(d)
    steps = np.concatenate(steps)                         # join all sequences into one array
    # The 95th percentile = the step 95% of frames stay under: the "hard frames" without one outlier.
    print(f"motion per frame (px): median {np.median(steps):.1f} / 95th percentile {np.percentile(steps, 95):.1f} "
          f"/ max {steps.max():.1f}")

    # 6. Attributes: how many sequences have each challenge flag switched on.
    for a in ATTRIBUTES:
        n = sum(s.attributes.get(a, False) for s in seqs)  # .get(a, False): treat a missing flag as off
        print(f"  {a:24s} {n:3d} / {len(seqs)}")          # :24s pads the name to 24 chars so columns line up


# Run main() only when the file is executed directly, not when imported.
if __name__ == "__main__":
    main()
