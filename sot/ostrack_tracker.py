"""Wrap the official OSTrack (one-stream ViT tracker) in the same init/update contract as OpenCV.

    tracker = OSTrackTracker()                  # loads the network onto the GPU
    tracker.init(frame0_bgr, (x, y, w, h))
    ok, (x, y, w, h) = tracker.update(frame_bgr)

OSTrack's own API differs from OpenCV's in three ways, and this class hides all three:
    1. method names: initialize() / track() instead of init() / update()
    2. colour order: it expects RGB images; OpenCV reads BGR
    3. return type: track() returns {"target_bbox": [x, y, w, h]} floats, not (ok, box)

The OSTrack repo is not a pip package: it lives next to ours (../OSTrack) and is imported by
adding that folder to sys.path. The checkpoint lives in its output/ folder (plan step 9a).
"""

import sys
from pathlib import Path

import cv2

# GitHub/Single-Object-Tracking/sot/ostrack_tracker.py -> parents[2] = GitHub/, then /OSTrack.
OSTRACK_DIR = Path(__file__).resolve().parents[2] / "OSTrack"
CONFIG = "vitb_256_mae_ce_32x4_ep300"          # ViT-Base, 256 px search, candidate elimination, 300 epochs
# The official pretrained weights; the fine-tuned ones (step 10) will be passed in as `checkpoint`.
PRETRAINED = OSTRACK_DIR / "output/checkpoints/train/ostrack" / CONFIG / "OSTrack_ep0300.pth.tar"


class OSTrackTracker:
    """OSTrack behind the OpenCV-style init/update contract. Needs a CUDA GPU."""

    def __init__(self, checkpoint: Path = PRETRAINED, config: str = CONFIG):
        """Build the network from `config` and load the weights in `checkpoint` onto the GPU."""
        if not Path(checkpoint).exists():
            raise FileNotFoundError(f"Missing checkpoint {checkpoint}; see step 9a in the plan.")
        if str(OSTRACK_DIR) not in sys.path:     # only add it once, even if we build many trackers
            sys.path.insert(0, str(OSTRACK_DIR)) # now `import lib...` finds OSTrack's own package

        # Imported here, not at the top: these modules only exist once OSTRACK_DIR is on sys.path.
        from lib.config.ostrack.config import cfg, update_config_from_file
        from lib.test.tracker.ostrack import OSTrack
        from lib.test.utils import TrackerParams

        # cfg starts with OSTrack's defaults; the experiment .yaml overrides them (model size, crop sizes...).
        update_config_from_file(str(OSTRACK_DIR / "experiments/ostrack" / f"{config}.yaml"))

        # TrackerParams is a plain settings bag. We fill it ourselves instead of calling OSTrack's
        # parameters(), which needs a machine-specific local.py file we don't have.
        params = TrackerParams()
        params.cfg = cfg
        params.template_factor = cfg.TEST.TEMPLATE_FACTOR  # template crop = 2x the target size
        params.template_size = cfg.TEST.TEMPLATE_SIZE      # ...resized to 128 x 128 px
        params.search_factor = cfg.TEST.SEARCH_FACTOR      # search crop = 4x the target size around the last box
        params.search_size = cfg.TEST.SEARCH_SIZE          # ...resized to 256 x 256 px
        params.checkpoint = str(checkpoint)
        params.save_all_boxes = False                      # we only need the single best box
        params.debug = 0                                   # 0 = no debug windows or visdom

        # Builds the ViT, loads the weights, moves it to the GPU and switches to eval mode.
        self._tracker = OSTrack(params, "visdrone")        # 2nd argument (dataset name) is unused by OSTrack

    def init(self, frame, box) -> None:
        """Crop the template around `box` (x, y, w, h) on the first frame (BGR, as read by OpenCV)."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)       # OSTrack was trained on RGB images
        self._tracker.initialize(rgb, {"init_bbox": [float(v) for v in box]})  # it wants a list of floats

    def update(self, frame):
        """Track into the next frame. Returns (ok, (x, y, w, h)) like OpenCV's trackers."""
        out = self._tracker.track(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        x, y, w, h = out["target_bbox"]                    # floats, in full-frame pixel coordinates
        # OSTrack always returns a box (no lost flag), so ok is always True: like NanoTrack, its
        # opinion carries no information and the judge decides. round() -> ints, as OpenCV returns.
        return True, (round(x), round(y), round(w), round(h))
