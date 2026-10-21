"""Download and extract VisDrone2019-SOT splits into data/VisDrone-SOT.

The dataset is hosted on Google Drive, which does not offer plain download links for big
files, so the `gdown` package handles it. For each split this script downloads the zip,
extracts it, deletes the zip, and leaves a marker file so a re-run skips finished splits.

Usage (from the repo root, with the venv active):
    python scripts/download_visdrone_sot.py                 # every split
    python scripts/download_visdrone_sot.py --splits val    # just val
"""

import argparse                 # reads command-line options like --splits val
import zipfile                  # opens and extracts .zip archives (built into Python)
from pathlib import Path

import gdown                    # downloads files from Google Drive by file id

# Short split name -> Google Drive file id, from https://github.com/VisDrone/VisDrone-Dataset (Task 3).
# The id is the long string in a Drive share link (drive.google.com/file/d/<id>/view).
# The train set is split into two archives on Drive; both extract into the same train folder.
SPLITS = {
    "val": "18SNAOlCJtApnG2m45ud-1e_OtGYill0D",             # 1.29 GB
    "test-dev": "1xCiHjU4JlR9QsYtiHYy2UUd3m6NthoBC",        # 11.27 GB
    "train-part1": "1a-AmQjYATzj8seXLlEm9Sx8aQmClrJka",     # 7.78 GB
    "train-part2": "16YPyhNDQrTgW8I2HaH_HNEO-KTRA-xso",     # 12.59 GB
}
# Repository root (this file lives in <root>/scripts/, so parents[1] is <root>).
ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Download, extract and clean up each requested split, skipping ones already done."""
    # --- Command-line options -------------------------------------------------------
    parser = argparse.ArgumentParser(description=__doc__)
    # Where to put the data; defaults to <root>/data/VisDrone-SOT, where sot/dataset.py looks.
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "VisDrone-SOT")
    # Which splits to fetch: nargs="+" accepts one or more names; choices rejects typos.
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)     # create the folder (and parents); fine if it exists
    for split in args.splits:
        # An empty hidden file like ".val.done" records that this split finished in an earlier run.
        done_marker = args.out / f".{split}.done"
        # Skip splits finished in an earlier run, so re-running the script is safe.
        if done_marker.exists():
            print(f"{split}: already done, skipping")
            continue
        zip_path = args.out / f"{split}.zip"
        # resume=True continues a half-finished download instead of starting over.
        gdown.download(id=SPLITS[split], output=str(zip_path), quiet=False, resume=True)
        print(f"{split}: extracting")
        # `with` closes the zip file automatically when the block ends, even on an error.
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(args.out)                 # unpack every file into the output folder
        zip_path.unlink()  # free the disk space once extracted
        done_marker.touch()                         # create the marker only after everything succeeded
        print(f"{split}: done")


# Run main() only when the file is executed directly, not when imported.
if __name__ == "__main__":
    main()
