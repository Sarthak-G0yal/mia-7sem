"""Download / prepare the datasets used by both parts.

Mini-project (segmentation) -> data/seg/<name>/{images,masks[,masks_expert2]}:

    uv run python scripts/download_data.py duke_dme     # Duke DME fluid segmentation (Chiu et al., BOE 2015), ~178 MB
    uv run python scripts/download_data.py synthetic    # OCT-like synthetic B-scans with fluid pockets (no download)
    uv run python scripts/download_data.py retouch      # prints manual steps (registration required)
    uv run python scripts/download_data.py aroi         # prints manual steps (request form required)

Project (NA-DyCNN classification) -> data/ucsd, data/oct_c8, data/neh:

    uv run python scripts/download_data.py ucsd         # Kermany 2018 via the Kaggle CLI (~5 GB)
    uv run python scripts/download_data.py oct_c8       # OCT-C8 via the Kaggle CLI (~2 GB)
    uv run python scripts/download_data.py neh          # prints manual steps (Mendeley Data)

The Kaggle downloads need `pip install kaggle` (or `uv pip install kaggle`) and an API token in
~/.kaggle/kaggle.json. Copying an existing machine's data/ (or just its cache/, which is all training
needs after `main.py prepare`) with rsync works as well.

Segmentation output layout used by fbseg and notebooks/feedback_segmentation.ipynb:

    data/seg/<name>/images/<subject>_<frame>.png
    data/seg/<name>/masks/<subject>_<frame>.png           0 = background, 255 = fluid (defect)
    data/seg/<name>/masks_expert2/<subject>_<frame>.png   (Duke only) second grader, for an inter-observer ceiling
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA_ROOT = ROOT / "data" / "seg"
DUKE_URL = "https://people.duke.edu/~sf59/Datasets/2015_BOE_Chiu2.zip"


def download(url: str, destination: Path, retries: int = 20) -> None:
    """Resumable download (the Duke server is slow and drops connections)."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    if destination.exists():
        print(f"already downloaded: {destination}")
        return
    for attempt in range(1, retries + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                if offset and response.status != 206:
                    offset = 0  # server ignored the range request; start over
                total = offset + int(response.headers.get("Content-Length", 0))
                with open(partial, "ab" if offset else "wb") as handle:
                    done, last = offset, time.time()
                    while chunk := response.read(1 << 20):
                        handle.write(chunk)
                        done += len(chunk)
                        if time.time() - last > 2:
                            print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)
                            last = time.time()
            if total and partial.stat().st_size < total:
                raise urllib.error.URLError("connection closed early")
            partial.rename(destination)
            print(f"\r  downloaded {destination.stat().st_size / 1e6:.1f} MB to {destination}")
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            print(f"\n  attempt {attempt}/{retries} interrupted ({error}); resuming...")
            time.sleep(min(30, 2 * attempt))
    raise RuntimeError(f"could not download {url}")


def _load_mat(path: Path) -> dict[str, np.ndarray]:
    try:
        from scipy.io import loadmat
        return {k: v for k, v in loadmat(path).items() if not k.startswith("__")}
    except NotImplementedError:  # MATLAB v7.3 (HDF5) files
        import h5py
        with h5py.File(path, "r") as handle:
            return {k: np.array(handle[k]).T for k in handle.keys()}


def _save(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def prepare_duke(keep_zip: bool = True) -> None:
    """Duke DME: 10 subjects x 61 B-scans (496x768); 11 B-scans per subject have fluid outlined by 2 graders."""
    downloads = DATA_ROOT / "_downloads"
    archive = downloads / "2015_BOE_Chiu2.zip"
    download(DUKE_URL, archive)
    extract_dir = downloads / "duke_dme"
    if not any(extract_dir.rglob("*.mat")):
        print("extracting...")
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(extract_dir)
    out_dir = DATA_ROOT / "duke_dme"
    total = with_fluid = 0
    for mat_path in sorted(extract_dir.rglob("Subject_*.mat")):
        subject = f"s{int(mat_path.stem.split('_')[-1]):02d}"
        mat = _load_mat(mat_path)
        images, fluid1, fluid2 = mat["images"], mat["manualFluid1"], mat.get("manualFluid2")
        for frame in range(images.shape[2]):
            annotated = ~np.all(np.isnan(fluid1[:, :, frame]))
            if not annotated:
                continue
            name = f"{subject}_{frame:03d}.png"
            image = images[:, :, frame].astype(np.float32)
            image = image if image.max() > 1.5 else image * 255.0
            _save(np.clip(image, 0, 255).astype(np.uint8), out_dir / "images" / name)
            mask1 = np.nan_to_num(fluid1[:, :, frame]) > 0
            _save((mask1 * 255).astype(np.uint8), out_dir / "masks" / name)
            if fluid2 is not None and not np.all(np.isnan(fluid2[:, :, frame])):
                _save(((np.nan_to_num(fluid2[:, :, frame]) > 0) * 255).astype(np.uint8),
                      out_dir / "masks_expert2" / name)
            total += 1
            with_fluid += int(mask1.any())
        print(f"  {mat_path.name}: images {images.shape}, annotated frames so far {total}")
    print(f"duke_dme ready at {out_dir}: {total} annotated B-scans ({with_fluid} contain fluid)")
    if not keep_zip:
        archive.unlink()


def prepare_synthetic() -> None:
    from fbseg.data import make_synthetic
    make_synthetic(DATA_ROOT / "synthetic")
    print(f"synthetic ready at {DATA_ROOT / 'synthetic'}")


KAGGLE = {
    "ucsd": ("paultimothymooney/kermany2018", ROOT / "data" / "ucsd"),
    "oct_c8": ("obulisainaren/retinal-oct-c8", ROOT / "data" / "oct_c8"),
}


def kaggle_download(name: str) -> None:
    """Download + unzip a Kaggle dataset; nadycnn.data.resolve_root then finds the root by its sub-folders."""
    import shutil
    import subprocess
    slug, destination = KAGGLE[name]
    if shutil.which("kaggle") is None:
        raise SystemExit("kaggle CLI not found: `uv pip install kaggle` and put your API token in ~/.kaggle/kaggle.json")
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(["kaggle", "datasets", "download", "-d", slug, "-p", str(destination), "--unzip"], check=True)
    print(f"{name} downloaded to {destination}; next: uv run python main.py prepare")


MANUAL = {
    "neh": """NEH (Noor Eye Hospital, 2021; CNV / DRUSEN / NORMAL volumes) is on Mendeley Data:
  https://data.mendeley.com/datasets/8kt969dhx6/2  ->  "Download All"
Unzip it so that data/neh/**/NEH_UT_2021RetinalOCTDataset/{CNV,DRUSEN,NORMAL}/<patient>/... exists
(the folder is found automatically under data/neh/), then: uv run python main.py prepare""",
    "retouch": """RETOUCH (MICCAI 2017; Cirrus, Spectralis, Topcon; IRF/SRF/PED masks) needs registration:
  https://retouch.grand-challenge.org/  ->  download the training set (.mhd volumes + reference .mhd masks)
Convert each B-scan to data/seg/retouch/images/<case>_<slice>.png and a binary mask (any fluid label > 0)
to data/seg/retouch/masks/<case>_<slice>.png (same name); the vendor/case prefix keeps subjects together.""",
    "aroi": """AROI (Spectralis, 1136 annotated B-scans, layers + fluid) is released on request:
  https://ipg.fer.hr/ipg/resources/oct_image_database
Put B-scans in data/seg/aroi/images/<patient>_<slice>.png and fluid masks (PED/SRF/IRF > 0)
in data/seg/aroi/masks/<patient>_<slice>.png.""",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=["duke_dme", "synthetic", "retouch", "aroi", "ucsd", "oct_c8", "neh"])
    parser.add_argument("--delete-zip", action="store_true", help="remove the archive after conversion")
    args = parser.parse_args()
    if args.dataset == "duke_dme":
        prepare_duke(keep_zip=not args.delete_zip)
    elif args.dataset == "synthetic":
        prepare_synthetic()
    elif args.dataset in KAGGLE:
        kaggle_download(args.dataset)
    else:
        print(MANUAL[args.dataset])


if __name__ == "__main__":
    main()
