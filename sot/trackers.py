"""Create trackers by name, so scripts can choose them on the command line.

    tracker = make_tracker("csrt")      # a fresh, un-initialised tracker
    tracker.init(frame0, init_box)
    ok, box = tracker.update(frame)

Every tracker returned here follows the same init/update contract, so the benchmark
runner never needs to know which tracker it is running. Adding a tracker = adding one
line to TRACKERS.
"""

import cv2

# Name -> the function that builds a new tracker. The functions are stored WITHOUT calling
# them (no brackets), so each call to make_tracker() gets a brand-new tracker with no memory
# of a previous sequence.
TRACKERS = {
    "csrt": cv2.TrackerCSRT.create,     # correlation filter with spatial reliability; accurate, ~60 fps
    "kcf": cv2.TrackerKCF.create,       # kernelised correlation filter; very fast, weaker on scale change
    "mil": cv2.TrackerMIL.create,       # multiple-instance learning classifier; older, slow
}


def make_tracker(name: str):
    """Build a new tracker by name (a key of TRACKERS). Raises KeyError for an unknown name."""
    return TRACKERS[name]()             # look up the builder function, then call it with ()
