# Deep Stochastic Mechanics (DSM)

[![Paper](https://img.shields.io/badge/Paper-ICML%202024-red)](https://proceedings.mlr.press/v235/orlova24a.html)
[![PDF](https://img.shields.io/badge/PDF-Available-blue)](https://raw.githubusercontent.com/mlresearch/v235/main/assets/orlova24a/orlova24a.pdf)
[![Python](https://img.shields.io/badge/Python-3.3+-green)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13.1-orange)](https://pytorch.org/)

A novel deep learning approach for numerical simulation of the time-dependent Schrödinger equation, inspired by stochastic interpretation of quantum mechanics and generative diffusion models.

## 📄 Publication

**[Deep Stochastic Mechanics](https://proceedings.mlr.press/v235/orlova24a.html)**  
*Proceedings of the 41st International Conference on Machine Learning (ICML 2024)*  
Elena Orlova, Aleksei Ustimenko, Ruoxi Jiang, Peter Y. Lu, Rebecca Willett

## Overview

Traditional quantum simulation methods scale exponentially with dimension, making high-dimensional problems intractable. DSM addresses this fundamental limitation by:


- **Adaptive Dimensionality**: Leveraging the latent low-dimensional structure of wave functions
- **Markovian Sampling**: Using diffusion-based sampling to avoid MCMC samoling for every time step
- **Novel Stochastic Framework**: Introducing new equations for stochastic quantum mechanics with quadratic computational complexity
- **Deep Learning Integration**: Combining insights from generative diffusion models with quantum simulation

It leads to significant computational advantages for high-dimensional quantum systems.


## 🗂️ Repository Structure

```
📁 interacting-system/     # Interacting bosons in harmonic oscillator
📁 non-interacting-system/ # Non-interacting bosons in harmonic oscillator  
📁 notebooks/              # Experimental notebooks and demonstrations
```

### Code Organization

#### `interacting-system/`
Implementation for **interacting bosons in a harmonic oscillator** with `torch.jit` optimization for enhanced performance.

#### `non-interacting-system/`
Implementation for **non-interacting bosons in a harmonic oscillator** with `torch.jit` optimization.

#### `notebooks/`
Jupyter notebooks containing experimental code and demonstrations. Note: These notebooks use the standard PyTorch implementation without `torch.jit` optimization for better readability and experimentation.

## 🛠️ Prerequisites

### System Requirements
- **Python**: 3.3 or higher

### Core Dependencies
```bash
torch==1.13.1
tqdm==4.64.1
scipy==1.10.1
numpy==1.24.2
```

You can istall it via
```bash
# Clone the repository
git clone <repository-url>
cd deep-stochastic-mechanics

# Install dependencies
pip install torch==1.13.1 tqdm==4.64.1 scipy==1.10.1 numpy==1.24.2
```

## 🔬 Methodology

DSM reformulates quantum dynamics through stochastic differential equations, allowing neural networks to learn from particle trajectories rather than gridded wave functions. This approach:

- Samples trajectories from Markovian diffusion processes
- Learns dynamics via deep neural networks
- Scales favorably by adapting to intrinsic dimensionality
- Bridges generative AI and quantum simulation

## 📊 Performance

Numerical experiments demonstrate significant speedup over existing deep learning quantum methods while maintaining accuracy across different system configurations and dimensions.


## 📚 Citation

```bibtex
@InProceedings{pmlr-v235-orlova24a,
  title = 	 {Deep Stochastic Mechanics},
  author =       {Orlova, Elena and Ustimenko, Aleksei and Jiang, Ruoxi and Lu, Peter Y. and Willett, Rebecca},
  booktitle = 	 {Proceedings of the 41st International Conference on Machine Learning},
  pages = 	 {38779--38814},
  year = 	 {2024},
  editor = 	 {Salakhutdinov, Ruslan and Kolter, Zico and Heller, Katherine and Weller, Adrian and Oliver, Nuria and Scarlett, Jonathan and Berkenkamp, Felix},
  volume = 	 {235},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {21--27 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://raw.githubusercontent.com/mlresearch/v235/main/assets/orlova24a/orlova24a.pdf},
  url = 	 {https://proceedings.mlr.press/v235/orlova24a.html}
}
```
---

*DSM represents a significant step forward in bridging deep learning and quantum simulation, offering new possibilities for tackling previously intractable quantum many-body problems.*

