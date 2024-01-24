# Deep Stochastic Mechanics (DSM)

DSM [paper](https://arxiv.org/abs/2305.19685) code.

## Code

 - ```interacting-system``` folder is for interacting bosons in a harmonic oscillator (cthe ode uses torch.jit).
 - ```non-interacting-system``` folder is for non-interacting bosons in a harmonic oscillator (cthe ode uses torch.jit).
 - ```notebooks``` folder have notebooks for some experiments. Note that this code version does not use torch.jit.

## Prerequisites

See more details in every folder, but in general:

- Python 3.3+
- torch==1.13.1
- tqdm==4.64.1
- scipy==1.10.1
- numpy==1.24.2


