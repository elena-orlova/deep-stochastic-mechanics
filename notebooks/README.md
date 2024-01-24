# Some notebooks 

There are provided some notebooks with initial code and experiments from the paper. Note this code might be slow as it's not optimized and does not use torch.jit. Better to look at other folders.

## Code

There are the following notebooks: 
- An experiment with a singular initial condition: ```1d_singular_DSM.ipynb``` 
- A harmonic oscillator in 1d with zero initial phase, DSM and PINN: ```1d_oscillator_zero_phase_DSM.ipynb```, ```1d_oscillator_zero_phase_PINN.ipynb```
- A harmonic oscillator in 1d with non-zero initial phase, DSM and PINN: ```1d_oscillator_non_zero_phase_DSM.ipynb```, ```1d_oscillator_non_zero_phase_PINN.ipynb```
- A harmonic oscillator in 3d: ```3d_oscillator_zero_phase_DSM.ipynb```
- Interacting particles experiments are available at the corresponding folder.

## Prerequisites

- Python 3.3+
- torch==1.13.1
- tqdm==4.64.1
- scipy==1.10.1
- numpy==1.24.2


