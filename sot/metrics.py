"""Tracking metrics for One-Pass Evaluation (OPE): IoU, centre error, success and precision.

All functions work on whole sequences at once: boxes are (N, 4) NumPy arrays of x, y, w, h
(top-left corner, pixels), one row per frame, so no Python loop over frames is needed.

    ious = iou(pred, gt)                       # (N,) overlap per frame, 0..1
    auc = success_curve(ious).mean()           # main metric: area under the success curve
    p20 = precision_curve(centre_error(pred, gt))[20]   # fraction of frames within 20 px
"""

import numpy as np

# Success-curve thresholds: 0.00, 0.05, ..., 1.00 (21 values), the standard OTB/VisDrone setting.
IOU_THRESHOLDS = np.linspace(0, 1, 21)
# Precision-curve thresholds: 0, 1, ..., 50 pixels; index 20 is the reported "P@20".
PIXEL_THRESHOLDS = np.arange(0, 51)


def iou(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Intersection over Union per frame: overlap area / combined area, from 0 (apart) to 1 (identical).

    pred, gt: (N, 4) arrays of x, y, w, h. Returns an (N,) array.
    """
    # Corners of the overlap rectangle: the larger of the two left/top edges and the smaller
    # of the two right/bottom edges. Columns: 0 = x, 1 = y, 2 = w, 3 = h.
    x1 = np.maximum(pred[:, 0], gt[:, 0])                            # overlap left edge
    y1 = np.maximum(pred[:, 1], gt[:, 1])                            # overlap top edge
    x2 = np.minimum(pred[:, 0] + pred[:, 2], gt[:, 0] + gt[:, 2])    # overlap right edge
    y2 = np.minimum(pred[:, 1] + pred[:, 3], gt[:, 1] + gt[:, 3])    # overlap bottom edge
    # If the boxes do not touch, x2 - x1 (or y2 - y1) is negative; clip turns that into 0 overlap.
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    # Union = both areas added, minus the overlap (otherwise it would be counted twice).
    union = pred[:, 2] * pred[:, 3] + gt[:, 2] * gt[:, 3] - inter
    # Divide only where union > 0; elsewhere (two empty boxes) the result stays 0 instead of NaN.
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def centre_error(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Distance in pixels between the centres of the predicted and true boxes, per frame."""
    # Centre = top-left corner + half the size: columns 0-1 are (x, y), columns 2-3 are (w, h).
    pc = pred[:, :2] + pred[:, 2:] / 2
    gc = gt[:, :2] + gt[:, 2:] / 2
    return np.linalg.norm(pc - gc, axis=1)     # sqrt(dx² + dy²) for each row


def success_curve(ious: np.ndarray) -> np.ndarray:
    """Fraction of frames whose IoU is above each threshold in IOU_THRESHOLDS (21 values).

    Its mean is the AUC, the main ranking metric.
    """
    # ious[None, :] is a row (1, N), IOU_THRESHOLDS[:, None] a column (21, 1); comparing them
    # "broadcasts" to a (21, N) True/False table, and .mean(axis=1) gives one fraction per threshold.
    return (ious[None, :] > IOU_THRESHOLDS[:, None]).mean(axis=1)


def precision_curve(errors: np.ndarray) -> np.ndarray:
    """Fraction of frames whose centre error is at most each threshold in PIXEL_THRESHOLDS (51 values).

    Element [20] is the reported precision at 20 pixels (P@20).
    """
    # Same broadcasting trick as success_curve: a (51, N) table, averaged along the frames.
    return (errors[None, :] <= PIXEL_THRESHOLDS[:, None]).mean(axis=1)
