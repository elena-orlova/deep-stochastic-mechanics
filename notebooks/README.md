# DSM Experimental Notebooks

[![Python](https://img.shields.io/badge/Python-3.3+-green)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13.1-orange)](https://pytorch.org/)

Interactive notebooks demonstrating Deep Stochastic Mechanics experiments from our [ICML 2024 paper](https://proceedings.mlr.press/v235/orlova24a.html).

These notebooks prioritize readability over performance and don't use `torch.jit` optimization. For faster use, see the optimized implementations in `interacting-system/` and `non-interacting-system/` folders.

## Available Experiments

### Singular Initial Conditions
- `1d_singular_DSM.ipynb` - DSM with singular initial conditions

### 1D Harmonic Oscillator Comparisons
| Phase Type | DSM | PINN |
|------------|-----|------|
| Zero phase | `1d_oscillator_zero_phase_DSM.ipynb` | `1d_oscillator_zero_phase_PINN.ipynb` |
| Non-zero phase | `1d_oscillator_non_zero_phase_DSM.ipynb` | `1d_oscillator_non_zero_phase_PINN.ipynb` |

### Higher Dimensions
- `3d_oscillator_zero_phase_DSM.ipynb` - 3D harmonic oscillator with DSM

### Interacting Systems
See `../interacting-system/` folder for interacting particle experiments.

## 🛠️ Requirements

```bash
torch==1.13.1 tqdm==4.64.1 scipy==1.10.1 numpy==1.24.2
```

## 🚀 Quick Start

1. Install dependencies: `pip install torch==1.13.1 tqdm==4.64.1 scipy==1.10.1 numpy==1.24.2`
2. Launch Jupyter: `jupyter notebook`
3. Start with `1d_oscillator_zero_phase_DSM.ipynb` for a basic introduction


