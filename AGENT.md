# AGENT Instructions: Execution, Environment Setup & Tooling Guidelines

## 📌 Agent Role & Task Overview

Your task is to execute, maintain, and expand the project described in `README.md`. You are expected to act as an autonomous Deep Learning / Computer Vision Software Engineer.

Follow the instructions below for environment management, dependency handling, execution workflows, and code generation standards.

---

## 🛠️ Environment & Tooling Specifications

### 1. Package & Project Management (`uv`)
- **Primary Tool:** Use [`uv`](https://github.com/astral-sh/uv) for all Python environment management, virtual environment creation, package installation, and script execution.
- **DO NOT** use `pip` or `conda` directly if `uv` is available.

#### Recommended Commands:
```bash
# Initialize project / virtual environment
uv venv .venv

# Activate virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Add core dependencies
# uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # Adjust CUDA version as needed
# uv pip install opencv-python numpy matplotlib scikit-learn jupyter jupyterlab
#ADDED THE DEPENDENCIES
```

### 2. Jupyter Notebook Execution
- Ensure the Jupyter environment is registered within the `uv` virtual environment kernel:
```bash
uv pip install ipykernel
python -m ipykernel install --user --name=na-dycnn-env --display-name "Python (NA-DyCNN uv)"
```
- When creating or running `.ipynb` files, ensure code cells are strictly ordered and reproducible (`random_seed` set for PyTorch, NumPy, and Python's `random`).

---

## 🎯 Step-by-Step Action Plan for the Agent

1. **Read Specifications:** Thoroughly review `README.md` to understand the mathematical formulas, model parameters, pre-processing steps, and visualization requirements.
2. **Setup Environment:** Execute the `uv` setup commands specified above.
3. **Build Core Modules:**
   - Implement the `Adaptive Contrast & Edge-Preserving Enhancer` in a modular, reusable format.
   - Implement the `DynamicConv2d` PyTorch layer ($K=4$ expert weights + lightweight router).
   - Build the `NA-DyCNN` classifier backbone ($\le 0.5\text{M}$ parameters).
   - Implement the Dual-Branch Noise-Aware loss ($\mathcal{L}_{\text{std}} + \beta \mathcal{L}_{\text{aug}}$).
   - Implement Mondrian Conformal Prediction (MCP) for clinical uncertainty calibration.
4. **Generate Notebook (`solution.ipynb` or `main.ipynb`):**
   - Translate all implemented logic into cleanly documented cells.
   - Ensure synthetic B-scan data generation allows the notebook to run end-to-end without external downloads.
   - Run all cells to render inline plots for dynamic routing coefficients, feature maps, training metrics, and conformal prediction coverage.

---

## ⚠️ Coding Standards & Guidelines

- **Self-Contained Executability:** The notebook/scripts must be runnable out-of-the-box. Synthetic data must be provided as a fallback if real OCT datasets are unavailable.
- **Type Annotations & Clean Code:** Write clean PyTorch code using proper typing hints (`torch.Tensor`, `nn.Module`, etc.).
- **Parameter Check:** Always log or `print()` the model parameter count to confirm it complies with the paper's target ($\approx 0.4\text{M}$ params).
- **Device Agnostic:** Ensure all tensors and modules automatically move to `cuda`, `mps`, or `cpu`:
  ```python
  device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
  ```
