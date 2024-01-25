## Non-interacting particles

We have $d$ 1-dimensional non-interacting bosons in a harmonic oscillator. In this folder, we provide training for DSM only (PINN is available in notebooks folder).

##### Zero initial phase
We consider a harmonic oscillator model with 
$x\in\mathbb{R}^{1}$, 
$V (x) = \frac{1}{2}m\omega^2(x - 0.1)^2$, $t\in [0, 1]$ and where $m=1$ and $\omega=1$. The initial wave function is given as $\psi(x, 0) \propto e^{-x^2/(4\sigma^2)}$. Then $u_0(x) = -\frac{h x}{2 m \sigma^2}$, ${v}_0(x) \equiv 0$. $X(0)$ comes from $X(0) \sim \mathcal{N}(0, \sigma^2),$ where $\sigma^2 = 0.1$. 

##### Non-zero initial phase

We also consider a non-zero initial phase $S_0(x) = -5x$. It corresponds to the initial impulse of a particle. Then ${v}_0(x) \equiv -\frac{5\hbar}{m}$.

To define the initial  phase coefficient $S_0(x) = cx$ (including $c=0$), use an argument `-init_phase -5` in the commend line.

### Code

The DSM training is given in `train-DSM.py`. This file runs DSM training, saves trained models, losses plot, samples with trained NNs and makes density plots after training; it also runs the numerical solution for the specified problem, and saves density and statistics plots. To run it run from the terminal:
```
bash run_dsm.sh
```
Feel free to play with hyperparameters (for example, running training for `-n_epochs=10`epochs to see how it works, changing lr, etc.).

