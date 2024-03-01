import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from qmsolve import Hamiltonian, TwoBosons, TimeSimulation

@torch.jit.script 
def V_x_i(x:torch.Tensor, device:torch.device, d:int, m:float=1, 
                   g:float=1, om:float=1, s2:float=0.1): # grad V for DSM loss
    # X has shape (batch, particles*dim), dim=1 usually
    C = 0.5*g/torch.sqrt(2*3.1415927410125732*s2)
    xv = x[:, :, None].expand(-1, -1, x.shape[-1])
    return om * m * x + C/s2*torch.einsum("kij,kij->kj", torch.exp(-0.5*torch.square(xv - xv.mT) / s2), xv-xv.mT)


@torch.jit.script
def v_0(x:torch.Tensor, h:float=0.1, m:float=1):
    return h / m * torch.zeros_like(x)


@torch.jit.script
def u_0(x:torch.Tensor, om:float=1):
    return -om * x


def V_noninteract(x1, x2, omega2):
    V_trap_1 = 0.5 * omega2 * x1**2
    V_trap_2 = 0.5 * omega2 * x2**2
    return  V_trap_1 + V_trap_2


def V_interact(x1, x2, g=1, s2=0.1):
    V_soft_contact = 0.5 * g / np.sqrt(2*np.pi*s2) * np.exp(-0.5 * (x1 - x2)**2 / s2)
    return V_soft_contact


def potential_sum(g, omega2):
    def _potential_sum(particles, g, omega2):
        return V_noninteract(particles.x1, particles.x2, omega2) + V_interact(particles.x1, particles.x2, g)
    return lambda x: _potential_sum(x, g, omega2)


def potential_noninteract(particles):
    return V_noninteract(particles.x1, particles.x2)


def potential_interact(particles, g=1):
    return V_interact(particles.x1, particles.x2, g)


def ground_state_func(x, omega=1, m=1, d=2, h=0.1):
    a = (omega*m/np.pi/h)**(d/4) * np.exp(-0.5*m *omega* np.einsum("ijk, ijk -> ij", x, x)/h)
    return a


def numerical_sol(spatial_dim, spatial_num, N=1000, g=1, T=1, omega2=1, m=1, d=2, h=0.1):
    particles = TwoBosons()
    H_interact = Hamiltonian(particles = particles,
                potential = potential_sum(g, omega2),
                spatial_ndim = 1, N = spatial_num, extent = int(2*spatial_dim))
    x, y = np.linspace(-spatial_dim, spatial_dim, spatial_num), np.linspace(-spatial_dim, spatial_dim, spatial_num)
    X, Y = np.meshgrid(x, y)

    pos = np.dstack((X, Y))
    Z = ground_state_func(pos, omega=np.sqrt(omega2), m=m, d=d, h=h)

    def init_ground_state(particles):
        return Z

    total_time = T
    sim_inter = TimeSimulation(hamiltonian = H_interact, method = "crank-nicolson")
    sim_inter.run(init_ground_state, total_time = total_time, dt = total_time/N,
                store_steps = N)
    return sim_inter


def make_loss_plot(losses, losses_sm, losses_newton, losses_init, path):
    plt.clf()
    plt.plot(losses_sm, alpha=0.8,  label='Score match')
    plt.plot(losses_newton, alpha=0.8,  label='Newton\'s law')
    plt.plot(losses_init, alpha=0.8,  label='Initial cond')
    plt.plot(losses, alpha=0.8, label='L')
    plt.title('Training losses')
    plt.xlabel('iteration')
    plt.ylabel('loss value')
    plt.yscale('log')
    plt.legend()
    plt.savefig(os.path.join(path, 'losses_train.jpg'), bbox_inches='tight', dpi=300)


def make_stat_plots(mean_trials, var_trials, time_splits, path, d=2, sve=True):
    fig, axs = plt.subplots(d, 2)
    fig.set_size_inches(8, 7)
    x = time_splits.numpy()

    i = 0
    axs[0, 0].plot(x, mean_trials[:, :, i].mean(axis=0), color='dodgerblue', label='DSM', linewidth=1)
    axs[0, 0].fill_between(x, mean_trials[:, :, i].mean(axis=0) - mean_trials[:, :, i].std(axis=0),
                        mean_trials[:, :, i].mean(axis=0) + mean_trials[:, :, i].std(axis=0), color='dodgerblue',
                        alpha=0.5, linewidth=0.8)
    axs[0, 0].set_ylabel('particle 1 value')
    axs[0, 0].legend()
    axs[0, 0].set_title('$X_i$ mean')

    axs[i, 1].plot(x, var_trials[:, :, i].mean(axis=0), color='dodgerblue', label='DSM', linewidth=1)
    axs[i, 1].fill_between(x, var_trials[:, :, i].mean(axis=0) - 2*var_trials[:, :, i].std(axis=0),
                        var_trials[:, :, i].mean(axis=0) + 2*var_trials[:, :, i].std(axis=0), color='dodgerblue',
                        alpha=0.5, linewidth=0.8)
    axs[i, 1].set_title('$X_i$ variance')
    axs[i, 1].legend()

    i = 1
    axs[1, 0].plot(x, mean_trials[:, :, i].mean(axis=0), color='dodgerblue', label='DSM', linewidth=1)
    axs[1, 0].fill_between(x, mean_trials[:, :, i].mean(axis=0) - mean_trials[:, :, i].std(axis=0),
                        mean_trials[:, :, i].mean(axis=0) + mean_trials[:, :, i].std(axis=0), color='dodgerblue',
                        alpha=0.5, linewidth=0.8)
    axs[i, 0].set_ylabel('particle 2 value')
    axs[i, 0].set_xlabel('$t_i$')
    axs[i, 0].legend()
    
    axs[i, 1].plot(x, var_trials[:, :, i].mean(axis=0), color='dodgerblue', label='DSM', linewidth=0.8)
    axs[i, 1].fill_between(x, var_trials[:, :, i].mean(axis=0) - 3*var_trials[:, :, 0].std(axis=0),
                        var_trials[:, :, i].mean(axis=0) + 3*var_trials[:, :, 0].std(axis=0), color='dodgerblue',
                        alpha=0.5, linewidth=0.8)
    axs[i, 1].set_xlabel('$t_i$')
    axs[i, 1].legend()
    if save:
        plt.savefig(os.path.join(path, 'stats_train.png'), bbox_inches='tight', dpi=300)


def make_stat_plots_compare(mean_trials, var_trials, bmeans, bstds, time_splits, path, d=2, save=True):
    fig, axs = plt.subplots(d, 2)
    fig.set_size_inches(8, 7)
    x = time_splits.numpy()

    i = 0
    axs[0, 0].plot(x, mean_trials[:, :, i].mean(axis=0), color='dodgerblue', label='DSM', linewidth=1)
    axs[i, 0].plot(x, bmeans, color='green', label='Truth', linewidth=0.8)
    axs[0, 0].fill_between(x, mean_trials[:, :, i].mean(axis=0) - mean_trials[:, :, i].std(axis=0),
                        mean_trials[:, :, i].mean(axis=0) + mean_trials[:, :, i].std(axis=0), color='dodgerblue',
                        alpha=0.5, linewidth=0.8)
    axs[0, 0].set_ylabel('particle 1 value')
    axs[0, 0].set_ylim(-0.1, 0.1)
    axs[0, 0].legend();
    axs[0, 0].set_title('$X_i$ mean')

    axs[i, 1].plot(x, var_trials[:, :, i].mean(axis=0), color='dodgerblue', label='DSM', linewidth=1)
    axs[i, 1].plot(x, bstds, color='green', label='Truth', linewidth=0.8)
    axs[i, 1].fill_between(x, var_trials[:, :, i].mean(axis=0) - 2*var_trials[:, :, i].std(axis=0),
                        var_trials[:, :, i].mean(axis=0) + 2*var_trials[:, :, i].std(axis=0), color='dodgerblue',
                        alpha=0.5, linewidth=0.8)
    axs[i, 1].set_title('$X_i$ variance')
    axs[i, 1].legend();

    i = 1
    axs[1, 0].plot(x, mean_trials[:, :, i].mean(axis=0), color='dodgerblue', label='DSM', linewidth=1)
    axs[1, 0].fill_between(x, mean_trials[:, :, i].mean(axis=0) - mean_trials[:, :, i].std(axis=0),
                        mean_trials[:, :, i].mean(axis=0) + mean_trials[:, :, i].std(axis=0), color='dodgerblue',
                        alpha=0.5, linewidth=0.8)
    axs[i, 0].plot(x, bmeans, color='green', label='Truth', linewidth=0.8)
    axs[i, 0].set_ylim(-0.1, 0.1)
    axs[i, 0].set_ylabel('particle 2 value')
    axs[i, 0].set_xlabel('$t_i$')
    axs[i, 0].legend()

    axs[i, 1].plot(x, var_trials[:, :, i].mean(axis=0), color='dodgerblue', label='DSM', linewidth=0.8)
    axs[i, 1].fill_between(x, var_trials[:, :, i].mean(axis=0) - 3*var_trials[:, :, 0].std(axis=0),
                        var_trials[:, :, i].mean(axis=0) + 3*var_trials[:, :, 0].std(axis=0), color='dodgerblue',
                        alpha=0.5, linewidth=0.8)
    axs[i, 1].plot(x, bstds, color='green', label='Truth', linewidth=0.8)
    axs[i, 1].set_xlabel('$t_i$')
    axs[i, 1].legend()

    if save:
        plt.savefig(os.path.join(path, 'stats_train_compare.png'), bbox_inches='tight', dpi=300)


def make_density_plot(p, path, low_bound=[-2.0, 0.0], up_bound=[2.0, 1.5], 
                      title='Plot title', vmin_max = [0, 2], save=True):
    vmin_, vmax_ = vmin_max
    fig, ax = plt.subplots(1, 1) #newfig(1.0, 0.9)
    ax.remove()
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])
    h_ = ax.imshow(p, cmap='YlGnBu', extent=[low_bound[1], up_bound[1], low_bound[0], up_bound[0]], origin='lower', aspect='auto')
    h_.set_clim(vmin_, vmax_)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h_, cax=cax)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_title(title, fontsize = 12)
    if "x_1" in title and "pred" in title:
        img_title = 'density_img_{}_{}.pdf'.format("x1", "pred")
    elif "x_2" in title and "pred" in title:
        img_title = 'density_img_{}_{}.pdf'.format("x2", "pred")
    elif "truth" in title:
        img_title = 'density_img_{}.pdf'.format("truth")
    else:
        img_title = 'dens_plot_{}.pdf'.format(np.random.randn(1))
    
    if save:
        plt.savefig(os.path.join(path, img_title), bbox_inches='tight', dpi=300)
        img_title = img_title[:-3] + 'png'
        plt.savefig(os.path.join(path, img_title), bbox_inches='tight', dpi=300)
    