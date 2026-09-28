# [AGENT_CONTEXT] NA-DyCNN Retinal OCT Pipeline

## 1. System Objective

This repository contains a PyTorch-based diagnostic pipeline for retinal OCT B-scans. The system mitigates speckle noise via an adaptive edge-preserving contrast enhancer, extracts features using a lightweight Noise-Aware Dynamic CNN (NA-DyCNN), and calibrates predictive uncertainty using Mondrian Conformal Prediction (MCP).

## 2. Directory Structure & Pathing

```text
.
├── AGENT.md                   # Agent execution instructions & architecture specs
├── main.py                    # Main PyTorch training & execution script
├── paper.pdf                  # NA-DyCNN reference architecture paper
├── pyproject.toml             # uv project configuration & dependencies
├── README.md                  # This file
├── solution.ipynb             # Interactive Jupyter Notebook pipeline
├── uv.lock                    # Dependency lockfile
└── data/                      # Local dataset storage (Real OCT scans)
    ├── ucsd/                  # Kermany 2018 (Primary Benchmark)
    │   └── OCT2017/OCT2017/
    │       ├── train/         # Subdirs: [CNV, DME, DRUSEN, NORMAL]
    │       ├── val/           # Subdirs: [CNV, DME, DRUSEN, NORMAL]
    │       └── test/          # Subdirs: [CNV, DME, DRUSEN, NORMAL]
    ├── oct_c8/                # OCT-C8 Dataset (Cross-vendor validation)
    │   └── RetinalOCT_Dataset/RetinalOCT_Dataset/
    │       ├── train/
    │       ├── val/
    │       └── test/
    └── neh/                   # NEH UT 2021 Dataset (OOD Evaluation)
        ├── metadata_train.csv
        ├── metadata_val.csv
        ├── metadata_test.csv
        └── NEH_UT_2021RetinalOCTDataset/NEH_UT_2021RetinalOCTDataset/
            ├── CNV/
            ├── DRUSEN/
            └── NORMAL/

```

## 3. Dataset Ingestion Protocol

When writing data loaders or executing training, strictly adhere to the following mapping and tensor constraints.

* **Class Mapping:** `{"NORMAL": 0, "CNV": 1, "DME": 2, "DRUSEN": 3, "AMD": 3}`
* **Input Dimension Target:** $256 \times 256$ pixels.
* **Color Space:** Grayscale (`L` mode in PIL), converted to `[1, H, W]` float32 tensors.
* **Active Dataset Dataloader:** `RealOCTDataset(Dataset)` inside `main.py` or `solution.ipynb` must point to the `data/` subdirectories defined in the tree above.

## 4. Architecture Specifications

### 4.1. Adaptive Preprocessing Module

Before feeding data to the network, standard OCT scans must pass through the edge-preserving enhancer:

1. **Padding & Resize:** Square pad to $\max(H, W)$, resize to $256 \times 256$ (Lanczos).
2. **Edge Masking:** Compute Sobel magnitude $E(x) \in [0, 1]$.
3. **CLAHE Enhancement:** Compute CLAHE-enhanced image $x_{\text{CLAHE}}$.
4. **Convex Blending:** $x_{\text{enhanced}} = E(x) \cdot x + (1 - E(x)) \cdot x_{\text{CLAHE}}$.
5. **Z-Score Normalization:** $\mathcal{Z}(x) = \frac{x - \mu}{\sigma + 1e-7}$.

### 4.2. NA-DyCNN Backbone

* **Parameter Constraint:** Total trainable parameters $\le 500,000$.
* **Router Configuration:** $K=4$ convolutional expert kernels dynamically blended per B-scan.
* **Routing Function:** $\alpha = \text{Softmax}(\text{Conv}_{1\times1}(\text{ReLU}(\text{Conv}_{1\times1}(\text{GAP}(x)))))$.
* **Effective Weight:** $\mathbf{W}_{\text{eff}} = \sum_{k=1}^{4} \alpha_k \mathbf{W}_k$.

### 4.3. Dual-Branch Optimization

The loss function requires standard clean images and stochastically noisy augmentations per forward pass:


$$x^{(n)}_{\text{noisy}} = \text{clip}\left(x_{\text{clean}} \odot (1 + \eta), 0, 1\right), \quad \eta \sim \mathcal{N}(0, \sigma^2=0.10)$$

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{ce}}(x_{\text{clean}}, y) + \beta \frac{1}{N} \sum_{i=1}^{N} \mathcal{L}_{\text{ce}}(x_{\text{noisy}}^{(i)}, y)$$

* **Hyperparameters:** $\beta = 0.10$, $N = 4$, optimizer = AdamW.

### 4.4. Mondrian Conformal Prediction (MCP)

Used for test-time uncertainty quantification to guarantee coverage ($\ge 90\%$) across shifted scanner domains:

1. Compute non-conformity scores on the calibration set: $s_i = 1 - \hat{p}(y_i \mid x_i)$.
2. Calculate class-specific quantiles $q_{1-\alpha}^{(c)}$ where $\alpha = 0.10$.
3. Generate prediction sets: $C(x_{\text{test}}) = \{c : \hat{p}(c \mid x_{\text{test}}) \ge 1 - q_{1-\alpha}^{(c)}\}$.

## 5. Execution Environment Commands

**Environment Initialization (`uv` required):**

```bash
uv venv .venv
source .venv/bin/activate
# GPU Install
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install opencv-python numpy matplotlib scikit-learn jupyter jupyterlab ipykernel

```

**Kernel Registration:**

```bash
python -m ipykernel install --user --name=na-dycnn-env --display-name "Python (NA-DyCNN uv)"

```

**Execution Vectors:**

* Notebook Mode: `jupyter lab solution.ipynb` (Select `NA-DyCNN uv` kernel).
* Headless Training: `uv run python main.py`
