"""Run one OpenCV tracker on one sequence and show its box next to the ground truth.

This is the first real tracking loop. Every tracker in the project follows the same
two-call contract:

    tracker.init(first_frame, init_box)     # once: "this is the target"
    ok, box = tracker.update(frame)         # every later frame: "where is it now?"

The tracker never sees the ground truth after frame 0; `gt` is used here only to draw the
green reference box, so you can judge the red tracker box by eye.

Green = ground truth, red = tracker. Keys: space = pause, q = quit.
At the end the last frame stays on screen for 10 s (any key closes it sooner).

Usage (from the repo root, with the venv active):
    python scripts/track_one.py --split val --seq 0
    python scripts/track_one.py --seq 1 --tracker nano   # any name in sot.trackers.TRACKERS
"""

import argparse                 # reads command-line options like --seq 7
import sys                      # gives access to sys.path, the list of folders Python imports from
import time                     # perf_counter(): a precise clock for measuring speed
from pathlib import Path

import cv2                      # OpenCV: read images, draw, show windows

# Python only imports from the script's own folder (scripts/) and installed packages, so
# `import sot` would fail. Adding the repo root (parents[1]) to the front of sys.path fixes that.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sot.dataset import load_split  # noqa: E402  (import must come after the sys.path line)
from sot.trackers import TRACKERS, make_tracker  # noqa: E402  (builds any tracker by name)


def main() -> None:
    """Initialise the chosen tracker on frame 0, update it on every later frame, and draw both boxes."""
    # --- Command-line options -------------------------------------------------------
    parser = argparse.ArgumentParser(description=__doc__)   # __doc__ = this file's docstring, shown by --help
    parser.add_argument("--split", default="val")           # which split to read from
    parser.add_argument("--seq", type=int, default=0, help="index of the sequence in the split")
    # choices= makes argparse reject unknown names and list the valid ones in --help.
    parser.add_argument("--tracker", default="csrt", choices=list(TRACKERS))
    args = parser.parse_args()                               # args.split, args.seq, args.tracker now hold the values

    # --- Load ------------------------------------------------------------------------
    seq = load_split(args.split)[args.seq]  # load every sequence in the split, keep the one at index --seq
    # Print the name, length and only the attributes that are switched on.
    print(f"{seq.name}: {len(seq)} frames, attributes: {[a for a, on in seq.attributes.items() if on]}")

    # --- Initialise: the tracker sees frame 0 and the init box, nothing else ---------
    tracker = make_tracker(args.tracker)    # a fresh tracker of the chosen kind (deep ones load their ONNX weights here)
    first = cv2.imread(str(seq.frames[0]))  # frame 0 as a (height, width, 3) BGR array
    # init() learns what the target looks like from the pixels inside init_box (x, y, w, h ints).
    tracker.init(first, seq.init_box)

    # --- Tracking loop ---------------------------------------------------------------
    paused = False
    lost = 0                                   # frames where the tracker itself said ok = False
    start = time.perf_counter()                # start the clock, to measure speed (frames per second)
    # Start at frame 1: frame 0 was used for init. frames[1:] and gt[1:] both skip frame 0 so
    # frame i stays paired with box i; enumerate(..., start=1) makes i the real frame number.
    # (gx, gy, gw, gh) unpacks the ground-truth box; the g stands for "ground truth".
    for i, (path, (gx, gy, gw, gh)) in enumerate(zip(seq.frames[1:], seq.gt[1:]), start=1):
        img = cv2.imread(str(path))
        # update() searches near the last position and returns (ok, box): ok = False means the
        # tracker thinks it lost the target. It can also be wrong while saying ok = True.
        ok, (x, y, w, h) = tracker.update(img)  # the tracker's guess for this frame, already ints
        if not ok:
            lost += 1

        # Green = ground truth, red = tracker (BGR colour order). Corners: top-left, bottom-right.
        cv2.rectangle(img, (int(gx), int(gy)), (int(gx + gw), int(gy + gh)), (0, 255, 0), 2)
        if ok:                                  # only draw the tracker box when it claims to have the target
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # Frame number and status in red in the top-left corner.
        cv2.putText(img, f"{i}  {'ok' if ok else 'LOST'}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow(seq.name, cv2.resize(img, None, fx=0.6, fy=0.6))  # frames are large; shrink to 60% to fit

        # waitKey(1) waits only 1 ms, so playback runs as fast as the tracker allows;
        # 0 waits forever, which is how pause works. & 0xFF keeps only the key code's last byte.
        key = cv2.waitKey(0 if paused else 1) & 0xFF
        if key == ord("q"):                     # q quits early
            break
        if key == ord(" "):                     # space toggles pause on/off
            paused = not paused

    # --- Summary ---------------------------------------------------------------------
    # i is the last frame number reached. The time also includes reading, drawing and showing
    # images (and any paused time), so this is the speed of the whole loop, a bit below the tracker's own.
    fps = i / (time.perf_counter() - start)
    print(f"{args.tracker}: tracked {i} frames at {fps:.1f} fps, tracker reported lost on {lost} frames")

    # Keep the last frame on screen for 10 s so you can note down the result before the
    # window closes. waitKey takes milliseconds; pressing any key ends the wait early.
    # This comes after the fps line, so the pause does not lower the measured speed.
    cv2.waitKey(10_000)                         # 10_000 = 10000; the _ is only for readability
    cv2.destroyAllWindows()                     # close the window when the video ends or you quit


# Run main() only when the file is executed directly, not when imported.
if __name__ == "__main__":
    main()
