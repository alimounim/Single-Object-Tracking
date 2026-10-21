# Single-Object Tracking on VisDrone2019-SOT

**[API reference](https://alimounim.github.io/Single-Object-Tracking/)**: every module, class and function, rebuilt from the docstrings on each push.

Given a target's box in the first frame of a drone video, a tracker predicts the box in every later frame. This project compares OpenCV trackers (MIL, KCF, CSRT, then deep ones) on VisDrone2019-SOT, scored by Success AUC and Precision@20 px (One-Pass Evaluation).

## Results on val (11 sequences)

| Tracker | AUC | P@20 | FPS |
| --- | --- | --- | --- |
| CSRT | 0.521 | 0.871 | 72 |
| MIL | 0.311 | 0.581 | 36 |
| KCF | 0.209 | 0.358 | 437 |

## Layout

| Path | Contents |
| --- | --- |
| `sot/` | Package: `dataset.py` (loading), `trackers.py` (tracker factory), `metrics.py` (IoU, AUC, P@20) |
| `scripts/` | Programs to run: viewer, dataset stats, single-sequence tracking, benchmark runner, evaluator |
| `data/` | VisDrone2019-SOT, extracted (not in git) |
| `runs/` | Saved tracker outputs and leaderboards (not in git) |

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/run_benchmark.py --split val --trackers csrt kcf mil
python scripts/evaluate.py --split val --per-seq
```
