## Interacting particles

We have two bosons (with interaction) in a harmonic oscillator. For the harm oscillator with a potential $V(x) = 0.5 k x^2 = 0.5 m \omega^2 x^2$, the analytical solution is known for one particle in 1d:

$$
 \Psi_n(x) = \displaystyle \frac{1}{\sqrt{2^n n!}} \Big(\frac{m \omega}{\pi h}\Big)^\frac{1}{4} e^{-\frac{m \omega x^2}{2h}} H_n(\sqrt{\frac{m \omega}{h}}x).
$$

We use this $\Psi_0(x) = (\frac{\omega}{\pi h})^{d/4} e^{\frac{- \omega x^2}{2h}}$ as the initial condition. If there is no interaction term in the potential, the particle system should stay the same (so, for example, X_i mean and variance are the same over $t$). 

If we add an interaction term to the potential $V(x) = 0.5 m \omega^2 x^2 + 0.5 * \frac{ g}{\sqrt{2 \pi s_2}} \exp(-0.5(x_1 - x_2)^2 / s_2).$ $g$ defines the strength of the interaction. In this case, the particle system starts from the ground state mentioned above but there is dynamics (it changes over time). 

### Code

The DSM training is given in `train-DSM-interact-2particles.py`. This file runs DSM training, saves trained models, losses plot, samples with trained NNs and makes density plots after training; it also runs the numerical solution for the specifies problem and saves density plots/statistics plots. To run it run from the terminal:
```
run_dsm.sh
```
Feel free to play with hyperparameters (for example, running training for -n_epochs=10 epochs to see how it works, -invar=0 to use regular NN architecture).

To run PINN:

```
python interacting_PINN.py
```


