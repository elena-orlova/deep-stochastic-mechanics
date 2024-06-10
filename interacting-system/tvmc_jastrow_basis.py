import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import netket as nk
import numpy as np
from netket.experimental import TDVP
from netket.experimental.driver import TDVPSchmitt
from netket.experimental.dynamics import Euler


def hermite_poly_all(n_max, x):
    """
    Calculate the Hermite polynomials from order 0 to n_max for a 2D input x using the recurrence relation.

    Parameters:
    n_max (int): The maximum order of the Hermite polynomials to compute.
    x (array-like): The 2D position values at which to evaluate the Hermite polynomials.
                    The shape of x is (batch, 2).

    Returns:
    array-like: A 3D array containing the values of the Hermite polynomials from order 0 to n_max for each dimension of x.
                The shape of the array is (n_max + 1, batch, 2).
    """
    batch_size = x.shape[0]
    n_particles = x.shape[1]
    H = jnp.zeros((n_max + 1, batch_size, n_particles))
    H = H.at[0, :, :].set(jnp.ones_like(x))

    if n_max >= 1:
        H = H.at[1, :, :].set(2 * x)

        for n in range(2, n_max + 1):
            H = H.at[n, :, :].set(2 * x * H[n - 1] - 2 * (n - 1) * H[n - 2])

    return H


class FiniteHOBasis2Body(nn.Module):
    n_max: int
    m: float = 1.0
    omega: float = 1.0

    def setup(self):
        eps = np.finfo(np.float64).eps

        # 1-body coefficients
        self.coefficients1 = self.param(
            "coefficients1",
            lambda *_: jnp.array(
                [1.0 + 0j if (i == 0) else eps + 0j for i in range(self.n_max + 1)]
            ),
            self.n_max + 1,
        )

        # 2-body coefficients
        self.coefficients2 = self.param(
            "coefficients2",
            lambda *_: jnp.array(
                [
                    1.0 + 0j if (i == 0) else eps + 0j
                    for i in range((self.n_max + 2) ** 2)
                ]
            ),
            (self.n_max + 2) ** 2,
        )

        # Normalization constants
        self.log_norm_consts = (
            -0.5
            * (
                jnp.arange(self.n_max + 1) * jnp.log(2)
                + jnp.log(jax.scipy.special.factorial(jnp.arange(self.n_max + 1)))
            )
            + jnp.log(self.m * self.omega / jnp.pi) / 4
        )

    def __call__(self, x):
        # Rescale x
        x_norm = jnp.sqrt(self.m * self.omega) * x

        # Calculate the Hermite polynomials
        H_n = hermite_poly_all(self.n_max, x_norm)

        # Construct the basis functions
        log_basis_funcs = (
            self.log_norm_consts.reshape(-1, 1, 1)
            - (x_norm**2) / 2
            + jnp.log(jax.lax.complex(H_n, 0.0))
        )

        # 1-body terms, multiply by the coefficients and sum over the basis functions
        log_psi_1 = jax.scipy.special.logsumexp(
            log_basis_funcs + jnp.log(self.coefficients1[:, None, None]), axis=0
        )

        # Add constant basis function
        log_basis_funcs = jnp.concatenate(
            (jnp.zeros_like(log_basis_funcs[:1]), log_basis_funcs), axis=0
        )

        # 2-body terms
        # take the outer product of the basis functions and particles
        log_basis_funcs_outer = (
            log_basis_funcs[:, None, :, :, None] + log_basis_funcs[None, :, :, None, :]
        )
        # Remove the diagonal terms leaving 2-body terms
        log_basis_funcs_outer = log_basis_funcs_outer.at[
            :, :, :, *np.diag_indices(x.shape[-1])
        ].set(0.0j)

        # Symmetrize the 2-body terms
        basis_funcs_outer = jnp.exp(log_basis_funcs_outer)
        log_basis_funcs_outer = jnp.log(
            basis_funcs_outer + basis_funcs_outer.transpose(0, 1, 2, 4, 3)
        )
        # log_basis_funcs_outer = jnp.logaddexp(
        #     log_basis_funcs_outer, log_basis_funcs_outer.transpose(0, 1, 2, 4, 3)
        # )

        # Flatten the 2-body terms
        log_basis_funcs_flat = log_basis_funcs_outer.reshape(
            (self.n_max + 2) ** 2, *log_basis_funcs_outer.shape[2:]
        )

        # Multiply by the coefficients and sum over the basis functions
        log_psi_2 = jax.scipy.special.logsumexp(
            log_basis_funcs_flat + jnp.log(self.coefficients2[:, None, None, None]),
            axis=0,
        )

        # Combine the 1-body and 2-body terms
        log_psi = jnp.sum(log_psi_1, axis=-1) + jnp.sum(log_psi_2, axis=(-1, -2))
        return log_psi


N = 3
m = 1e1
omega = 1.0
g = 1.0
s2 = 0.1
n_max = 20
n_samples = 10**4
dt = 1e-3

# define cont Hilbert space for N particles
# in infinite box with no periodic boundary cond
hilb = nk.hilbert.Particle(N=N, L=np.inf, pbc=False)

sampler = nk.sampler.MetropolisGaussian(hilb, sigma=0.1, n_chains=16, sweep_size=32)


# similar to our DSM model
model = FiniteHOBasis2Body(n_max=n_max, m=m, omega=omega)

# Gauss model
# model = Gaussian_model()

# To initialize the parameters
params = model.init(
    jax.random.PRNGKey(100), hilb.random_state(jax.random.PRNGKey(123), 5)
)
print(jax.tree_map(lambda x: (x.shape, x.dtype), params))
input = hilb.random_state(jax.random.PRNGKey(123), 5)
logpsi_val = model.apply(params, input)
print(f"An input of shape {input.shape=} gives output {logpsi_val.shape=}")
print(logpsi_val)

# choose number of samples that is divided by the number of chains
vstate = nk.vqs.MCState(sampler, model, n_samples=n_samples, n_discard_per_chain=100)

# Hamiltonian
ekin = nk.operator.KineticEnergy(hilb, mass=m)


def v(x):
    v0 = 0.5 * m * omega**2 * jnp.sum(x**2, axis=-1)

    v_int = (
        0.5
        * g
        * m
        / np.sqrt(2 * np.pi * s2)
        * 0.5
        * jnp.sum(
            jnp.exp(-((x[..., None] - x[..., None, :]) ** 2) / (2 * s2)), axis=(-2, -1)
        )
    )
    return v0 + v_int


pot = nk.operator.PotentialEnergy(hilb, v)
H = ekin + pot


integrator = Euler(dt=dt)

# When running, look at R^2 values
# if it converges, this val should be < 1.01 or smth like that
te = TDVPSchmitt(
    H,
    variational_state=vstate,
    integrator=integrator,
    holomorphic=True,
)

T = 1.0
N_t_steps = 101
tstops = np.linspace(0.0, T, N_t_steps)


def compute_density_callback(t, log_data, driver) -> bool:
    # extract the current state
    current_vstate = driver.state

    # put everything you want to log in the log data
    log_data["params"] = flax.core.copy(current_vstate.parameters)
    log_data["samples"] = current_vstate.samples.copy()
    # print(current_vstate.parameters)
    # log_data['other_observable'] = other_observable
    # Always return true, if you return False you stop the driver.
    return True


te.run(
    T=T,
    out=f"interact_HObasis{n_max}_N{N}_dt{dt}_Euler_samples{n_samples}",
    callback=compute_density_callback,
    tstops=tstops,
)
