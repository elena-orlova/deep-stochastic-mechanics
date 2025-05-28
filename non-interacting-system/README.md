# Non-interacting Bosons in Harmonic Oscillator

[![Python](https://img.shields.io/badge/Python-3.3+-green)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13.1-orange)](https://pytorch.org/)
[![Optimized](https://img.shields.io/badge/torch.jit-Optimized-blue)](https://pytorch.org/docs/stable/jit.html)

Deep Stochastic Mechanics implementation for *d* one-dimensional non-interacting bosons in a harmonic oscillator potential.

> This folder provides optimized DSM training with `torch.jit`. For PINN comparisons, see the `../notebooks/` folder.

## Physical System

**Harmonic Oscillator Setup:**
- Domain: $x \in \mathbb{R}^1$
- Potential: $V(x) = \frac{1}{2}m\omega^2(x - 0.1)^2$ 
- Parameters: $m = 1$, $\omega = 1$
- Time evolution: $t \in [0, 1]$

**Initial Wave Function:**
$$\psi(x, 0) \propto e^{-x^2/(4\sigma^2)}$$

where $X(0) \sim \mathcal{N}(0, \sigma^2)$ with $\sigma^2 = 0.1$.

## Configuration Options

### Zero Initial Phase (Default)
- $u_0(x) = -\frac{\hbar x}{2m\sigma^2}$
- $v_0(x) \equiv 0$
- $X(0)$ comes from $X(0) \sim \mathcal{N}(0, \sigma^2),$ where $\sigma^2 = 0.1$. 

### Non-zero Initial Phase
- Initial phase: $S_0(x) = cx$ (e.g., $c = -5$)
- $v_0(x) \equiv -\frac{c\hbar}{m}$
- Corresponds to initial particle momentum

Set initial phase coefficient with `-init_phase -5` command line argument.

## Training & Execution

### Quick Start
```bash
bash run_dsm.sh
```

### Main Training Script: `train-DSM.py`
**Automatically handles:**
- DSM model training with `torch.jit` optimization
- Model checkpointing and loss visualization
- Sample generation with trained neural networks
- Density plot creation
- Numerical solution comparison

### Hyperparameter Tuning
Experiment with different settings:
```bash
# Quick test run
python train-DSM.py -n_epochs 10

# Adjust learning rate
python train-DSM.py -lr 0.001
```

This file runs DSM training, saves trained models, losses plot, samples with trained NNs and makes density plots after training; it also runs the numerical solution for the specified problem, and saves density and statistics plots.

---

*This implementation demonstrates DSM's effectiveness for quantum harmonic oscillator problems with excellent computational efficiency.*

