"""Play a sequence with its ground-truth box drawn. Keys: space = pause, q = quit.

A sanity check before any tracking: if the green box stays on the target, the frames and
annotations are loaded correctly and line up frame by frame.

Usage (from the repo root, with the venv active):
    python scripts/show_sequence.py --split val --seq 0
"""

import argparse                 # reads command-line options like --split val
import sys                      # gives access to sys.path, the list of folders Python imports from
from pathlib import Path

import cv2                      # OpenCV: read images, draw boxes, show windows

# Python only imports from the script's own folder (scripts/) and installed packages, so
# `import sot` would fail. Adding the repo root (parents[1]) to the front of sys.path fixes that.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sot.dataset import load_split  # noqa: E402  (import must come after the sys.path line)


def main() -> None:
    """Parse the options, load the chosen sequence and play it frame by frame."""
    # --- Command-line options -------------------------------------------------------
    parser = argparse.ArgumentParser(description=__doc__)   # __doc__ = this file's docstring, shown by --help
    parser.add_argument("--split", default="val")           # which split folder to read from
    parser.add_argument("--seq", type=int, default=0, help="index of the sequence in the split")
    args = parser.parse_args()                               # args.split and args.seq now hold the values

    # --- Load ------------------------------------------------------------------------
    seq = load_split(args.split)[args.seq]  # load every sequence in the split, keep the one at index --seq
    # Print the name, length and only the attributes that are switched on.
    print(f"{seq.name}: {len(seq)} frames, attributes: {[a for a, on in seq.attributes.items() if on]}")

    # --- Playback loop ---------------------------------------------------------------
    paused = False
    # zip pairs each frame path with its box; enumerate adds the frame number i.
    # (x, y, w, h) unpacks the 4 numbers of the box straight into variables.
    for i, (path, (x, y, w, h)) in enumerate(zip(seq.frames, seq.gt)):
        img = cv2.imread(str(path))         # load the JPG as a (height, width, 3) BGR array
        # Draw the box: OpenCV wants the top-left and bottom-right corners as ints.
        # (0, 255, 0) is green in BGR order; 2 is the line thickness in pixels.
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        # Write the frame number in the top-left corner, at position (10, 30).
        cv2.putText(img, f"{i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow(seq.name, cv2.resize(img, None, fx=0.6, fy=0.6))  # frames are large; shrink to 60% to fit
        # waitKey(ms) shows the window and waits for a key: 30 ms between frames is about 33 fps;
        # 0 waits forever, which is how pause works. & 0xFF keeps only the key code's last byte.
        key = cv2.waitKey(0 if paused else 30) & 0xFF
        if key == ord("q"):                 # ord("q") is the key code of the letter q
            break
        if key == ord(" "):                 # space toggles pause on/off
            paused = not paused
    cv2.destroyAllWindows()                 # close the window when the video ends or you quit


# Run main() only when the file is executed directly (python scripts/show_sequence.py),
# not when another file imports it.
if __name__ == "__main__":
    main()
