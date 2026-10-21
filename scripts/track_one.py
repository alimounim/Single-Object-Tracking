"""Run one tracker on one sequence, show its box next to the ground truth, and judge it live.

This is the first real tracking loop. Every tracker in the project follows the same
two-call contract:

    tracker.init(first_frame, init_box)     # once: "this is the target"
    ok, box = tracker.update(frame)         # every later frame: "where is it now?"

The tracker never sees the ground truth after frame 0. `gt` is used only by the *judge*:
on every frame we compare the tracker's box with the GT box (IoU and centre error), exactly
as scripts/evaluate.py does for the leaderboard. That separates two different things:

    ok      = the tracker's own opinion ("I still have it"), often wrong
    HIT/MISS = the judge's verdict (IoU > 0.5 against the ground truth)

Green box = ground truth, red box = tracker (thin when the tracker says ok = False and we
are holding its last box). Top-left: frame number + tracker status, then the judge's verdict.
Keys: space = pause, q = quit. At the end the last frame stays on screen for 10 s (any key
closes it sooner) and the same scores evaluate.py would give are printed.

Usage (from the repo root, with the venv active):
    python scripts/track_one.py --split val --seq 0
    python scripts/track_one.py --seq 1 --tracker nano   # any name in sot.trackers.TRACKERS
"""

import argparse                 # reads command-line options like --seq 7
import sys                      # gives access to sys.path, the list of folders Python imports from
import time                     # perf_counter(): a precise clock for measuring speed
from pathlib import Path

import cv2                      # OpenCV: read images, draw, show windows
import numpy as np              # small arrays for the metric functions

# Python only imports from the script's own folder (scripts/) and installed packages, so
# `import sot` would fail. Adding the repo root (parents[1]) to the front of sys.path fixes that.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sot.dataset import load_split  # noqa: E402  (import must come after the sys.path line)
from sot.metrics import centre_error, iou, success_curve  # noqa: E402  (the same judge evaluate.py uses)
from sot.trackers import TRACKERS, make_tracker  # noqa: E402  (builds any tracker by name)

GREEN = (0, 255, 0)             # OpenCV colours are BGR (blue, green, red), not RGB
RED = (0, 0, 255)
HIT_IOU = 0.5                   # a frame counts as a hit when IoU > 0.5, the usual cut-off in tracking papers


def main() -> None:
    """Initialise the chosen tracker on frame 0, update it on every later frame, draw and judge each frame."""
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
    lost = 0                    # frames where the tracker itself said ok = False
    last_box = seq.init_box     # the box we score; kept when the tracker says ok = False (same rule as run_benchmark.py)
    ious, errs = [], []         # per-frame judge scores, collected for the summary
    fooled = 0                  # frames where the tracker said ok but its box did not overlap the target at all
    first_miss = None           # first frame with zero overlap = the moment the target was really lost
    start = time.perf_counter()                # start the clock, to measure speed (frames per second)
    # Start at frame 1: frame 0 was used for init. frames[1:] and gt[1:] both skip frame 0 so
    # frame i stays paired with box i; enumerate(..., start=1) makes i the real frame number.
    # (gx, gy, gw, gh) unpacks the ground-truth box; the g stands for "ground truth".
    for i, (path, (gx, gy, gw, gh)) in enumerate(zip(seq.frames[1:], seq.gt[1:]), start=1):
        img = cv2.imread(str(path))
        # update() searches near the last position and returns (ok, box): ok = False means the
        # tracker thinks it lost the target. It can also be wrong while saying ok = True.
        ok, (x, y, w, h) = tracker.update(img)  # the tracker's guess for this frame, already ints
        if ok:
            last_box = (x, y, w, h)             # trust the new box only when the tracker returned one
        else:
            lost += 1                           # keep last_box: the box returned with ok = False is meaningless

        # --- Judge: compare against the ground truth. Only the evaluation sees this, never the tracker ---
        # The metric functions take (N, 4) arrays, so wrap each single box in [...] to make one row.
        pred = np.array([last_box], dtype=float)
        truth = np.array([[gx, gy, gw, gh]], dtype=float)
        frame_iou = iou(pred, truth)[0]          # overlap, 0 (miss) to 1 (perfect); [0] takes the one value out
        frame_err = centre_error(pred, truth)[0] # pixel distance between the two box centres
        ious.append(frame_iou)
        errs.append(frame_err)
        if frame_iou == 0:
            if ok:
                fooled += 1                      # confident and wrong: the tracker says ok, the judge says miss
            if first_miss is None:
                first_miss = i                   # remember only the first such frame
        hit = frame_iou > HIT_IOU

        # --- Draw ---------------------------------------------------------------------
        # Corners: top-left (x, y) and bottom-right (x + w, y + h). The last argument is line thickness.
        cv2.rectangle(img, (int(gx), int(gy)), (int(gx + gw), int(gy + gh)), GREEN, 2)
        bx, by, bw, bh = last_box                # unpack the box being scored
        # Thick red when the tracker says ok; thin red when it said lost and we are holding the old box.
        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), RED, 2 if ok else 1)
        # Line 1: frame number and the tracker's own status (its opinion).
        cv2.putText(img, f"{i}  tracker: {'ok' if ok else 'LOST'}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2)
        # Line 2: the judge's verdict, green HIT / red MISS, with IoU and centre error.
        cv2.putText(img, f"IoU {frame_iou:.2f}  err {frame_err:.0f}px  {'HIT' if hit else 'MISS'}",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, GREEN if hit else RED, 2)
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

    # The judge's scores for this one sequence: the same numbers evaluate.py would give
    # (if you quit early with q, they cover only the frames you watched).
    ious, errs = np.array(ious), np.array(errs)          # lists -> arrays so we can do maths on all frames at once
    print(f"judge: AUC {success_curve(ious).mean():.3f}, "          # mean of the success curve = Success AUC
          f"success@{HIT_IOU} {(ious > HIT_IOU).mean():.0%}, "      # share of frames that were a HIT
          f"P@20 {(errs <= 20).mean():.0%}")                        # share of frames with centre within 20 px
    print(f"really lost at frame {first_miss}; said ok while missing on {fooled} frames")  # None = never lost

    # Keep the last frame on screen for 10 s so you can note down the result before the
    # window closes. waitKey takes milliseconds; pressing any key ends the wait early.
    # This comes after the prints, so the pause does not lower the measured speed.
    cv2.waitKey(10_000)                         # 10_000 = 10000; the _ is only for readability
    cv2.destroyAllWindows()                     # close the window when the video ends or you quit


# Run main() only when the file is executed directly, not when imported.
if __name__ == "__main__":
    main()
