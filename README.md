# NA-DyCNN — Noise-Aware Dynamic Convolution for Retinal OCT

PyTorch reproduction of Chutani & Yalavarthy, *Noise-aware dynamic convolution for improved generalizability of retinal disease diagnosis using OCT images*, J. Biomed. Opt. 31(8), 086006 (2026) — see `paper.pdf`.

Two parts:

- **Project** — reproduction of the paper: `nadycnn/`, `main.py`, `scripts/project_*.py`
- **Mini-project** — feedback-controlled smoothing for OCT defect segmentation built on top of it:
  `fbseg/`, `notebooks/feedback_segmentation.ipynb`, `scripts/miniproject_*.py`

## Layout

```text
main.py                       project CLI: prepare / train / experiment / tune / evaluate / report / profile
nadycnn/                      PROJECT package
├── data.py                   dataset discovery + QC, 256x256 uint8 cache, GPU augmentation, speckle noise (Eq. 1)
├── models.py                 per-image Z-norm (Eq. 2), DynamicConv2d (Eq. 3), Basic/Dynamic CNN, MobileNetV3-Small, ResNet18
├── training.py               standard / noise-augmented / noise-aware objectives (Eqs. 4-6), AdamW + warm-up cosine
├── metrics.py                F2, sens., spec., κ, ECE, Err.Unc (Eq. 9), Risk@80%, Mondrian CP (Eqs. 10-11), bootstrap
├── evaluation.py             zero-shot cross-dataset evaluation, Tables 2-8, routing probe (Sec. 3.7)
└── experiment.py             run / re-evaluate / report orchestration (used by main.py and scripts/)
fbseg/                        MINI-PROJECT package
├── data.py · models.py · training.py · metrics.py · evaluation.py · plots.py
└── experiment.py             run / evaluate_saved / report (also `python -m fbseg run|evaluate|report`)
notebooks/
└── feedback_segmentation.ipynb   mini-project: data -> train -> evaluate saved models -> results -> figures
scripts/                      plain-Python entry points for a headless workstation server
├── download_data.py          datasets of both parts (Duke DME, synthetic, UCSD/OCT-C8 via Kaggle, NEH manual)
├── project_train.py          project end to end: prepare cache -> train all methods/seeds -> evaluate -> report
├── project_evaluate.py       project: evaluate saved checkpoints only (--repredict reloads best.pt)
├── miniproject_train.py      mini-project end to end (script version of the notebook) + saved figures
├── miniproject_evaluate.py   mini-project: evaluate saved checkpoints only
├── collect_results.py        zip reports, metrics and figures (no checkpoints) into results_bundle/
└── run_server.sh             unattended runner with logs: miniproject | project | all | evaluate
solution.ipynb                earlier exploratory notebook (not paper-faithful; kept for reference)
data/ cache/ runs/ runs_seg/ logs/ results_bundle/   git-ignored data and outputs
```

## Datasets (paper Table 1)

| ID | Dataset | Path | After QC | Protocol |
|---|---|---|---|---|
| D1 | UCSD (Kermany 2018) | `data/ucsd/OCT2017 /` | 74,954 | pooled, de-duplicated, 70/15/15 **patient-level** split |
| D2 | OCT-C8 | `data/oct_c8/RetinalOCT_Dataset/RetinalOCT_Dataset/` | 10,940 | CNV/DME/DRUSEN/NORMAL only, original splits |
| D3 | NEH | `data/neh/NEH_UT_2021RetinalOCTDataset/NEH_UT_2021RetinalOCTDataset/` | 12,641 | test only; frames whose label matches the volume label |

QC keeps only 512×496, 768×496 and 512×512 B-scans. Class order is `("CNV", "DME", "DRUSEN", "NORMAL")`.
For D3 (no DME) the DME logit is discarded and the remaining three are renormalized.

## Method

- **Preprocessing:** 8-bit grayscale (16-bit scaled linearly) → square zero-pad to max(W, H) → Lanczos 256×256 → per-image Z-norm (inside the model).
- **Models:** Basic CNN (4 conv blocks 16-32-64-128, GAP, 2-layer MLP; 0.11 M params) and Dynamic CNN (same, every conv → K = 4 expert dynamic conv with a GAP→1×1→ReLU→1×1→softmax router; 0.40 M params). MobileNetV3-Small (1.52 M) and ResNet18 (11.17 M) trained from scratch.
- **NA-DyCNN objective:** `L = ℓ(f(Z(x)), y) + β · (1/N) Σₙ ℓ(f(Z(x⁽ⁿ⁾)), y)`, with `x⁽ⁿ⁾ = clip(x ⊙ (1 + η), 0, 1)`, `η ~ N(0, σ²)` injected before Z-norm. (σ, β, N) = (0.06, 0.12, 4) for D1 and (0.10, 0.10, 4) for D2. Inference is a single clean pass.
- **Training:** AdamW (wd 1e-4), batch 64, linear warm-up 1e-6 → 1e-3 over 5 epochs then cosine to 1e-6, label smoothing, inverse-frequency class weights, early stopping on source-val balanced accuracy (patience 10, max 100 epochs), mild flip/rotation/translation/scale + brightness/contrast jitter.
- **Evaluation:** strict zero-shot transfer D1→D2, D1→D3, D2→D1, D2→D3; Mondrian CP at ε = 0.05 either zero-shot (source-val calibration) or site-specific (10 % of target patients, ≥ 5 per class).

## Usage

```bash
uv run python main.py prepare                                  # build cache/ (≈3 min)
uv run python main.py train --method nadycnn --source d2 --seeds 0 1 2
uv run python main.py experiment --sources d1 d2 --seeds 0 1 2  # every run behind Tables 3-8
uv run python main.py tune --source d2                         # σ/β/N grid of Fig. 4 (source val only)
uv run python main.py report                                   # aggregate into runs/report.md
uv run python main.py profile                                  # Table 2 (params, MACs, latency)
```

Methods: `basic`, `dynamic`, `dynamic_aug` (Table 8 ablation variant 3), `nadycnn`, `mobilenetv3`, `resnet18`.
Use `--max-train-samples` / `--epochs` to cut compute on small GPUs; finished runs are skipped unless `--overwrite`.

## Follow-up idea: feedback-controlled smoothing for defect segmentation (`fbseg/`)

A U-Net segments fluid (the defect) in OCT B-scans that are smoothed with a per-pixel strength map λ. λ is
updated in a 3-step loop from the segmenter's hyperspherical-prototype confidence and boundary score, so the
model can smooth noisy flat tissue and leave boundaries and small defects sharp.

```bash
uv run python scripts/download_data.py duke_dme     # Duke DME fluid masks (2 graders) -> data/seg/duke_dme
uv run python scripts/download_data.py synthetic    # offline fallback -> data/seg/synthetic
jupyter lab notebooks/feedback_segmentation.ipynb   # train (small budget) + evaluate + figures
uv run python scripts/miniproject_train.py          # same thing headless; --folds 0 for a ~10 min look
uv run python scripts/miniproject_evaluate.py       # evaluate saved checkpoints only
```

| file | content |
|---|---|
| `fbseg/data.py` | folder dataset, subject-level 5-fold CV, paired augmentation, speckle, synthetic generator |
| `fbseg/models.py` | Noise2Void denoiser + scale-space smoother, U-Net, hyperspherical / linear heads, λ controller, 8 methods |
| `fbseg/training.py` | N2V pre-training, Dice + CE + boundary + λ-prior losses, early stopping on val Dice |
| `fbseg/metrics.py` | Dice, IoU, boundary F1, HD95, small/large-defect recall, false-positive components |
| `fbseg/evaluation.py` | robustness at extra speckle σ ∈ {0, .1, .2, .3}, λ by region, Dice per iteration, inter-expert ceiling, report |

Results land in `runs_seg/<dataset>/report.md`, `summary.json` and `figures/`.

## Running on a workstation server

1. **Copy the code and the environment.** Everything except git-ignored folders, then `uv sync` (or
   `pip install -e .` / the packages in `pyproject.toml` into any env and use `PYTHON=python` below).
2. **Data.** Either copy from this machine (fastest; the cache is all the project's training needs):
   ```bash
   rsync -a cache/ server:mia/cache/                        # project: preprocessed D1/D2/D3 (~7 GB)
   rsync -a data/seg/duke_dme/ server:mia/data/seg/duke_dme/  # mini-project
   ```
   or download on the server: `scripts/download_data.py duke_dme | ucsd | oct_c8 | neh`, then
   `uv run python main.py prepare`. Dataset roots are found automatically under `data/ucsd`, `data/oct_c8`, `data/neh`.
3. **Run** (results are skipped when already present, so any command can be restarted):
   ```bash
   nohup bash scripts/run_server.sh miniproject > logs/server.out 2>&1 &   # ~1 h on a small GPU
   GPU=0 nohup bash scripts/run_server.sh project > logs/server.out 2>&1 & # full paper protocol: many hours
   bash scripts/run_server.sh project --sources d2 --seeds 0               # smaller first pass
   bash scripts/run_server.sh evaluate                                     # score saved checkpoints, no training
   ```
   Each script also runs on its own: `uv run python scripts/project_train.py --help`.

   **Choosing the device:** every entry point takes `--gpu <index>` (the number `nvidia-smi` shows) or
   `--gpu cpu`: `run_server.sh`, the `scripts/*.py`, `main.py` and `python -m fbseg`; in the notebook set
   `GPU = 1` in the first cell. Without it, GPU 0 is used. Two parts can share the two GPUs at once:
   ```bash
   nohup bash scripts/run_server.sh project --gpu 0 > logs/project.out 2>&1 &
   nohup bash scripts/run_server.sh miniproject --gpu 1 > logs/miniproject.out 2>&1 &
   ```
4. **Results to send back:** `results_bundle/results_<timestamp>.zip` (written at the end of every
   `run_server.sh` call, or by `scripts/collect_results.py`): `runs/report.md`, `runs/profile.md`,
   `runs_seg/<dataset>/{report.md, summary.json, figures/}`, per-run metrics and the logs.

## Deviations / unspecified details

- Unspecified in the paper and chosen here: router hidden width `max(C_in/4, 4)`, MLP hidden 64 with dropout 0.2, label smoothing 0.1, augmentation ranges (±10°, ±5 % shift, 0.9–1.1 scale/brightness/contrast), noise-augmentation probability 0.5 for ablation variant 3.
- The paper's NEH cohort (~10.9 k) is "patient-balanced" by an unstated rule; all 12,641 label-matched frames are used here (NORMAL count matches the paper's Fig. 5 exactly).
- OCT-C8 has no patient IDs, so each image is treated as its own patient for calibration splits and bootstrap.
- The routing probe (Sec. 3.7) balances D2/D3 by subsampling the larger set.
- bf16 autocast is used on CUDA for speed.
