# Interacting Bosons in Harmonic Oscillator

[![Python](https://img.shields.io/badge/Python-3.3+-green)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13.1-orange)](https://pytorch.org/)
[![Optimized](https://img.shields.io/badge/torch.jit-Optimized-blue)](https://pytorch.org/docs/stable/jit.html)
[![NetKet](https://img.shields.io/badge/NetKet-t--VMC-purple)](https://netket.readthedocs.io/)

Deep Stochastic Mechanics for *d* one-dimensional interacting bosons, with comparisons to PINN and t-VMC methods.

## 🎯 Physical System

### Non-interacting Case (Reference)
**Harmonic Oscillator Potential:**
$$V(x) = \frac{1}{2}m\omega^2 x^2$$

**Analytical Ground State:**
$$\Psi_0(x) = \left(\frac{\omega}{\pi\hbar}\right)^{d/4} e^{-\frac{\omega x^2}{2\hbar}}$$

It is a stationary system with constant mean position and variance.

### Interacting Case
**Full Potential with Interaction:**
$$V(x) = \frac{1}{2}m\omega^2 x^2 + \frac{g}{2\sqrt{2\pi s_2}} \exp\left(-\frac{(x_1 - x_2)^2}{2s_2}\right)$$

**Key Parameters:**
- **$g$**: Interaction strength (controls dynamics)
- **$s_2$**: Interaction range parameter
- **Initial condition**: Non-interacting ground state $\Psi_0(x)$

*System behavior*: Non-stationary dynamics due to particle interactions.

## Available Methods

### 1. Deep Stochastic Mechanics (DSM)
```bash
bash run_dsm.sh
```

**Features:**
- `torch.jit` optimized training
- Automatic model saving and visualization
- Density plot generation
- Statistical analysis with numerical comparisons

**Hyperparameter Examples:**
```bash
# Quick test
python train-DSM.py -n_epochs 100

```

### 2. Physics-Informed Neural Networks (PINN)
```bash
python interacting_PINN.py
```

### 3. Time-dependent Variational Monte Carlo (t-VMC)
```bash
python tvmc_jastrow_basis.py
```

## ⚙Setup Requirements

### Standard Dependencies
- Python 3.3+, PyTorch 1.13.1, NumPy, SciPy

### Method-Specific Setup

#### For Numerical Comparisons (qmsolve)
⚠️ **Important**: Modify qmsolve constants before use:

1. Locate: `<python_env>/lib/python3.8/site-packages/qmsolve/util/constants.py`
2. Replace with values from `constants.py` in this repository
3. Works for $d=2$ by default

#### For t-VMC (NetKet)
Install NetKet following their [official instructions](https://netket.readthedocs.io/en/latest/docs/install.html).

**Key t-VMC Parameters:**
- `N`: Number of interacting bosons
- `n_max`: Number of basis functions  
- `n_samples`: MC samples per optimization step
- `dt`: Time step size for optimization
- `tstops`: Time points for saving predictions

## Research Applications

This implementation enables study of:
- **Many-body quantum dynamics** with controllable interactions
- **Method comparisons** (DSM vs. PINN vs. t-VMC)
- **Scaling behavior** with particle number and interaction strength

---

*This implementation showcases DSM's effectiveness for interacting quantum many-body systems, providing a comprehensive comparison framework with established methods.*
