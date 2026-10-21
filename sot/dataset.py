"""Load VisDrone2019-SOT sequences: frame paths, ground-truth boxes and attribute flags.

This is the only file that knows how the dataset is laid out on disk. Every other part of
the project (trackers, metrics, plots) calls `load_split()` and works with `Sequence`
objects, so if the folder layout ever changes, only this file needs fixing.

On disk, each split folder looks like this (one entry per sequence, same name in all three):

    VisDrone2019-SOT-val/
        sequences/uav0000024_00000_s/img0000001.jpg, img0000002.jpg, ...   <- the video frames
        annotations/uav0000024_00000_s.txt        <- one "x,y,w,h" line per frame
        attributes/uav0000024_00000_s_attr.txt    <- 12 flags (0/1) describing the challenges
"""

import re                                   # regular expressions: used to split lines on commas OR spaces
from dataclasses import dataclass, field    # @dataclass writes __init__ etc. for us; field() sets defaults
from pathlib import Path                    # Path objects: OS-independent file paths (/ joins parts)

import numpy as np                          # arrays for the boxes, so maths on all frames at once is easy

# Repository root. __file__ is this file's path; resolve() makes it absolute;
# parents[0] is sot/, parents[1] is the repo root. Building paths from here means the code
# works no matter which folder you run it from.
ROOT = Path(__file__).resolve().parents[1]
# Where the extracted dataset lives: <root>/data/VisDrone-SOT
DATA = ROOT / "data" / "VisDrone-SOT"
# Short split name (what you type) -> the folder name the zip extracts to.
# test-challenge is left out on purpose: it has no annotations, so it cannot be scored.
SPLITS = {
    "train": "VisDrone2019-SOT-train",
    "val": "VisDrone2019-SOT-val",
    "test-dev": "VisDrone2019-SOT-test-dev",
}
# Names for the 12 attribute flags, in the exact order they appear in each *_attr.txt file.
# The file only holds numbers like "1,1,1,0,...", so this list is what gives them meaning.
ATTRIBUTES = [
    "aspect_ratio_change", "background_clutter", "camera_motion", "fast_motion",
    "full_occlusion", "illumination_variation", "low_resolution", "out_of_view",
    "partial_occlusion", "similar_object", "scale_variation", "viewpoint_change",
]


def read_numbers(path: Path) -> np.ndarray:
    """Read a text file of numbers into a 2-D array, one row per non-empty line.

    VisDrone files are not consistent about separators (some use commas, some spaces or
    tabs), so any mix of them is accepted. Example: "608,404,27,84" -> [608., 404., 27., 84.]
    """
    rows = [
        re.split(r"[,\s]+", line.strip())   # split on one or more commas/whitespace -> list of strings
        for line in path.read_text().splitlines()  # read the whole file, then break it into lines
        if line.strip()                      # skip blank lines (e.g. a trailing newline at the end)
    ]
    return np.array(rows, dtype=np.float64)  # convert the list of string lists into a float array


@dataclass
class Sequence:
    """One tracking video plus its labels.

    The tracking task: the tracker is shown the target's box on frame 0 only (`init_box`),
    then must predict the box on every later frame. `gt` holds the true boxes for all frames,
    which are used only for scoring, never given to the tracker.
    """

    name: str                   # sequence id, e.g. "uav0000024_00000_s"
    frames: list[Path]          # image paths, in temporal order (frames[i] goes with gt[i])
    gt: np.ndarray              # (N, 4) ground-truth boxes as x, y, w, h (top-left corner, in pixels)
    attributes: dict[str, bool] = field(default_factory=dict)  # e.g. {"fast_motion": True, ...};
                                # default_factory gives each Sequence its own empty dict

    def __len__(self) -> int:
        """Number of frames, so `len(seq)` works."""
        return len(self.frames)

    @property                   # @property lets you write seq.init_box (no brackets) like an attribute
    def init_box(self) -> tuple[int, int, int, int]:
        """The only box a tracker is allowed to see: frame 0's ground truth, as whole pixels.

        OpenCV trackers want an (x, y, w, h) tuple of ints, so the floats are rounded.
        """
        return tuple(int(round(v)) for v in self.gt[0])


def load_sequence(split_dir: Path, name: str) -> Sequence:
    """Load one sequence by name from an extracted split folder.

    Raises ValueError if the number of frames and boxes differ, since then frame i and
    box i would not belong together and every score would be wrong.
    """
    # All .jpg files in the sequence folder, sorted so img0000001 comes before img0000002.
    frames = sorted((split_dir / "sequences" / name).glob("*.jpg"))
    # The ground-truth boxes: one row per frame.
    gt = read_numbers(split_dir / "annotations" / f"{name}.txt")
    # Safety check: frames and boxes must pair up one-to-one.
    if len(frames) != len(gt):
        raise ValueError(f"{name}: {len(frames)} frames but {len(gt)} boxes")

    attributes = {}             # stays empty if a sequence has no attribute file
    attr_file = split_dir / "attributes" / f"{name}_attr.txt"
    if attr_file.exists():
        flags = read_numbers(attr_file).ravel()     # ravel() flattens the (1, 12) array to 12 values
        # Pair each name with its flag: zip(["aspect_ratio_change", ...], [1, 1, ...]) -> {name: True, ...}
        attributes = {a: bool(f) for a, f in zip(ATTRIBUTES, flags)}
    return Sequence(name, frames, gt, attributes)


def load_split(split: str) -> list[Sequence]:
    """Load every sequence of a split ("train", "val" or "test-dev"), sorted by name.

    This is the main entry point, e.g. `for seq in load_split("val"): ...`
    Sorting by name keeps the order the same on every run and every computer.
    """
    split_dir = DATA / SPLITS[split]    # e.g. <root>/data/VisDrone-SOT/VisDrone2019-SOT-val
    # Every sub-folder of sequences/ is one sequence; its folder name is the sequence name.
    names = sorted(p.name for p in (split_dir / "sequences").iterdir() if p.is_dir())
    return [load_sequence(split_dir, n) for n in names]
