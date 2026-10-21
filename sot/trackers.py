"""Create trackers by name, so scripts can choose them on the command line.

    tracker = make_tracker("csrt")      # a fresh, un-initialised tracker
    tracker.init(frame0, init_box)
    ok, box = tracker.update(frame)

Every tracker returned here follows the same init/update contract, so the benchmark
runner never needs to know which tracker it is running. A classical tracker needs one line
in TRACKERS; a deep tracker also needs a small builder function (below) that points it at
its ONNX weights in models/.
"""

from pathlib import Path   # standard library: OS-independent file paths

import cv2                 # third-party: OpenCV, which provides every tracker

# Folder holding the pretrained ONNX weights (downloaded by hand, not in git).
# __file__ is sot/trackers.py, so .parent.parent is the repo root.
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def _model(filename: str) -> str:
    """Return the full path of an ONNX file in models/, failing early with a clear message.

    OpenCV's own error for a missing model is cryptic, so we check first.
    """
    path = MODELS_DIR / filename                     # / joins paths on any OS
    if not path.exists():
        raise FileNotFoundError(f"Missing model {path}; see step 8a in the plan.")
    return str(path)                                 # OpenCV wants a plain str, not a Path


# --- Deep trackers: pretrained ONNX weights ---

def _dasiamrpn():
    """DaSiamRPN: Siamese region-proposal tracker (3 ONNX files: backbone + 2 correlation kernels)."""
    params = cv2.TrackerDaSiamRPN_Params()           # empty settings object
    params.model = _model("dasiamrpn_model.onnx")    # shared feature network
    params.kernel_r1 = _model("dasiamrpn_kernel_r1.onnx")      # regression branch -> box offsets
    params.kernel_cls1 = _model("dasiamrpn_kernel_cls1.onnx")  # classification branch -> target vs background
    return cv2.TrackerDaSiamRPN.create(params)       # build the tracker with these weights


def _nano():
    """NanoTrack v2: tiny Siamese tracker for edge devices (backbone + head)."""
    params = cv2.TrackerNano_Params()                # empty settings object
    params.backbone = _model("nanotrack_backbone_sim.onnx")   # extracts features of template and search crops
    params.neckhead = _model("nanotrack_head_sim.onnx")       # correlates them and predicts the box
    return cv2.TrackerNano.create(params)            # build the tracker with these weights


def _vit():
    """VitTrack: lightweight transformer tracker (one ONNX file, attention over template + search)."""
    params = cv2.TrackerVit_Params()                 # empty settings object
    params.net = _model("object_tracking_vittrack_2023sep.onnx")  # single ViT network: template + search in, box + score out
    return cv2.TrackerVit.create(params)             # build the tracker with these weights


# --- Registry ---

# Name -> the function that builds a new tracker. The functions are stored WITHOUT calling
# them (no brackets), so each call to make_tracker() gets a brand-new tracker with no memory
# of a previous sequence.
TRACKERS = {
    "csrt": cv2.TrackerCSRT.create,     # correlation filter with spatial reliability; accurate, ~60 fps
    "kcf": cv2.TrackerKCF.create,       # kernelised correlation filter; very fast, weaker on scale change
    "mil": cv2.TrackerMIL.create,       # multiple-instance learning classifier; older, slow
    "dasiamrpn": _dasiamrpn,            # Siamese + region proposals; 3 ONNX files, ~160 MB
    "nano": _nano,                      # Siamese, ~1.8 MB, very fast
    "vit": _vit,                        # transformer (attention over template + search tokens)
}


def make_tracker(name: str):
    """Build a new tracker by name (a key of TRACKERS). Raises KeyError for an unknown name."""
    return TRACKERS[name]()             # look up the builder function, then call it with ()
