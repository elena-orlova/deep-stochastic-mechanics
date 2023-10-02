import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.autograd import grad
import os
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import qmsolve
from qmsolve import Hamiltonian, TwoBosons, SingleParticle, TimeSimulation, init_visualization
from src.nn import *
from pyDOE import lhs


torch.set_default_tensor_type(torch.DoubleTensor)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'#'0, 1, 2, 3'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

g = 1
omega2 = 1
omega = np.sqrt(1)
m = 1
h = qmsolve.hbar
d = 2
T = 1
N = 1000

spatial_dim = 1.5
spatial_num = 30

# get initial condition values
def psi_0(x, omega=1, m=1, d=2):
    a = (omega*m/np.pi/h)**(d/4) * np.exp(-0.5*m *omega* np.einsum("ijk, ijk -> ij", x, x)/h)
    return a


def psi_0_new(x, omega=1, m=1, d=2):
    a = (omega*m/np.pi/h)**(d/4) * np.exp(-0.5*m *omega* np.einsum("ik, ik -> i", x, x)/h)
    return a


def potential_V(x, omega2=1, g=1, s2=0.1):
    V_trap_1 = 0.5 * omega2 * x[:, 0]**2 + 0.5 * omega2 * x[:, 1]**2
    V_inter = torch.tensor(0.5 * g / np.sqrt(2*np.pi*s2)) * torch.exp(-0.5 * (x[:,0] - x[:,1])**2 / s2)
    return V_trap_1 + V_inter

potential_V(torch.tensor(np.random.randn(10, 2))).shape

class skip_embed(torch.nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim):
        super().__init__()
        self.embed = nn.Sequential(
                               nn.Linear(input_dim, latent_dim),
                               ACT(),
                               nn.Linear(latent_dim, output_dim),
                               ACT(),
                               )

    def forward(self, input):
        out = self.embed(input) + input
        return out

ACT = nn.Tanh

class NN_new(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim):
        super(NN_new, self).__init__()
        self.start = nn.Sequential(nn.Linear(input_dim, hid_dim), nn.Tanh())
        self.middle = skip_embed(hid_dim, hid_dim, hid_dim)
        self.out = nn.Sequential(nn.Tanh(), nn.Linear(hid_dim, output_dim))

    def forward(self, input):
        x, t = input[:, :2], input[:, 2][:, None]
        x, idx_sorted = x.sort(dim=1)
        out = torch.hstack((x, t))
        out = self.out(self.middle(self.start(out)))
        res = torch.take_along_dim(out, idx_sorted, 1)
        return res

# Doman bounds
lb = np.array([-spatial_dim, 0.0])
ub = np.array([spatial_dim, T])

N0 = 1000 # number of init cond points
N_b = 300 # number of boundary cond points
N_f = 60000 # number of collocation points
delta_x = 2* spatial_dim / spatial_num
delta_t = T/(N+1)

x, y = np.linspace(-spatial_dim, spatial_dim, spatial_num), np.linspace(-spatial_dim, spatial_dim, spatial_num)
time_splits = np.linspace(0, T, N+1)
X, Y, time_vec = np.meshgrid(x, y, time_splits)

X_test = np.hstack((X.flatten()[:,None], Y.flatten()[:,None], time_vec.flatten()[:,None]))
print(X_test.shape)

# getting IC points for training
ic_val = np.hstack([np.random.uniform(lb[0], ub[0], size=(N0,2)), np.zeros((N0,1))])
ic_sol = psi_0_new(ic_val[:, :2])
ic_sol_real = np.real(ic_sol)[:, None]
ic_sol_img = np.imag(ic_sol)[:, None]
ic_sol = np.concatenate((ic_sol_real, ic_sol_img), axis=1)
print('Init cond solution and value (input) shape', ic_sol.shape, ic_val.shape)

b_val = []
for i in range(0, d):
    for c in [lb[0], ub[0]]:
        b_val.append(np.hstack([np.random.uniform(-2, 2, size=(N_b//4,2)),
                              np.random.uniform(0, T, size=(N_b//4,1))]))
        for j in range(N_b//4):
            b_val[-1][j][i] = c
b_val = np.vstack(b_val)
b_val.shape

# getting collocation points
lb_ = np.array([-spatial_dim, -spatial_dim, 0.0])
ub_ = np.array([spatial_dim, spatial_dim, T])
X_f = lb_ + (ub_ - lb_) * lhs(3, N_f)
print('Colloc shape', X_f.shape)

X_f = torch.tensor(X_f).requires_grad_(True)
ic_val = torch.tensor(ic_val) #.requires_grad_(False)
ic_sol = torch.tensor(ic_sol)#.requires_grad_(False)
b_val = torch.tensor(b_val)#.requires_grad_(False)

# We have the equation
# $$i h \Psi_t + \frac{h^2}{2m} \Psi_{xx} - V(x) \Psi = 0. $$
# ​
# Let $\Psi = u + i v,$ then we have
# $$i h (u_t + i v_t) + \frac{h^2}{2m} (u_{xx} + i v_{xx}) - V(x) (u + i v) = 0, $$
# $$ih u_t - h v_t + \frac{h^2}{2m} u_{xx} + i \frac{h^2}{2m} v_{xx} - V(x)u - i V(x) v = 0$$
# $$\big(- h  v_t + \frac{h^2}{2m} u_{xx} - V(x)u\big) + i \big(h u_t + \frac{h^2}{2m} v_{xx} -  V(x)v\big) = 0. $$
# It's $L_f $ loss.

def get_Laplacian(net, X_all, d=2):
    du_ddx = torch.zeros(X_all.shape[0], d)
    for j in range(d):
        du_ddx[:, j] = torch.einsum("jii", get_jacobian(lambda x:
                                    get_jacobian(lambda x: net(x)[:, j][:, None],
                                                 x)[:, :, 0], X_all))
    return du_ddx


def get_grad_div(net, X_all):
    du_ddx = get_jacobian(lambda x: torch.einsum("jii", get_jacobian(lambda x: net_u(x),
                                                                     x))[:, None], Xs_all)[:, :, 0]
    return du_ddx

path_save = 'results/pinn'
if not os.path.isdir(path_save):
    os.mkdir(path_save)

def train_pinn(model, lr, n_epochs, f_colloc, b_colloc, ic_colloc, ic, sample_idx=False, N_subs_colloc = 20000):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_list = np.zeros(n_epochs)
    loss_physics_list = np.zeros(n_epochs)
    loss_b_list = np.zeros(n_epochs)
    loss_ic_list = np.zeros(n_epochs)
    resume = True
    if resume:
        checkpoint = torch.load(f'{path_save}/pinn.pth')
        model_pinn.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_start = checkpoint['epoch']
    else:
        epoch_start = 0

    with tqdm(range(epoch_start, n_epochs), unit="epoch") as tepoch:
        for i in tepoch:
            if i % 1000 == 10:
                torch.save({
                            'epoch': i,
                            'model_state_dict': model_pinn.state_dict(),
                            # 'model_v_state_dict': net_v.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'lr': optimizer.param_groups[0]['lr'],
                            }, f'{path_save}/pinn.pth')

            optimizer.zero_grad()

            if sample_idx:
                idx_subsample = np.random.choice(f_colloc.shape[0], N_subs_colloc, replace=False)
                f_colloc_sub = f_colloc[idx_subsample]
                f_colloc_sub = f_colloc_sub.to(device)
                y_pred = model(f_colloc_sub.to(device))
                u = y_pred[:, 0]
                v = y_pred[:, 1]
                grad_u = torch.autograd.grad(u, f_colloc_sub, torch.ones_like(u), create_graph=True)[0]
                grad_v = torch.autograd.grad(v, f_colloc_sub, torch.ones_like(v), create_graph=True)[0]
                # !!!! do not work for 2D particles
                du_dt = grad_u[:, [-1]]#.squeeze()
                dv_dt = grad_v[:, [-1]]#.squeeze()

                duv_dxx = get_Laplacian(model, f_colloc_sub)
                du_dxx = duv_dxx[:, 0].to(device)[:, None]
                dv_dxx = duv_dxx[:, 1].to(device)[:, None]
                loss_u = -h*dv_dt + (h**2 / 2) * du_dxx - potential_V(f_colloc_sub[:, :2])[:, None] * u.view(-1, 1)
                loss_v = h*du_dt + (h**2 / 2) * dv_dxx - potential_V(f_colloc_sub[:, :2])[:, None] * v.view(-1, 1)
            # else:
            #     # !!!needs modifications as above!!!
            #     f_colloc = f_colloc.to(device) # to(device)
            #     y_pred = model(f_colloc)
            #
            #     u = y_pred[:, 0]
            #     v = y_pred[:, 1]
            #
            #     grad_u = torch.autograd.grad(u, f_colloc, torch.ones_like(u), create_graph=True)[0]
            #     grad_v = torch.autograd.grad(v, f_colloc, torch.ones_like(v), create_graph=True)[0]
            #     du_dt = grad_u[:, [1]]
            #     dv_dt = grad_v[:, [1]]
            #     du_dx = grad_u[:, [0]]
            #     dv_dx = grad_v[:, [0]]
            #     du_dxx = torch.autograd.grad(du_dx, f_colloc, torch.ones_like(du_dx), create_graph=True)[0][:, [0]]
            #     dv_dxx = torch.autograd.grad(dv_dx, f_colloc, torch.ones_like(dv_dx), create_graph=True)[0][:, [0]]
            #     loss_u = -h*dv_dt + (h**2 / 2) * du_dxx - potential_V(f_colloc[:, :2])[:, None] * u.view(-1, 1)
            #     loss_v = h*du_dt + (h**2 / 2) * dv_dxx - potential_V(f_colloc[:, :2])[:, None] * v .view(-1, 1)

            loss_physics = (loss_u**2 + loss_v**2).mean()
            y_pred_b = model(b_colloc.to(device))
            y_pred_ic = model(ic_colloc.to(device))

            loss_b = torch.mean(y_pred_b**2)
            loss_ic = torch.mean((y_pred_ic - ic.to(device))**2)

            loss = loss_physics + loss_b + loss_ic #(torch.mean(loss_physics**2) + loss_b + loss_ic)

            loss_list[i] = loss.detach().cpu().numpy()
            loss_physics_list[i] = torch.mean(loss_physics**2).item()
            loss_b_list[i] = loss_b.item()
            loss_ic_list[i] = loss_ic.item()
            tepoch.set_postfix(loss_iter=loss.item(), loss_mean=np.mean(loss_list[i-10:]),
                               loss_std=np.std(loss_list[i-10:]))

            loss.backward()
            optimizer.step()

            if i % 100 == 10:
                plt.plot(loss_list[i], label='loss')
                plt.plot(loss_physics_list[:i], label='l_physics')
                plt.plot(loss_b_list[:i], label='L_bc')
                plt.plot(loss_ic_list[:i], label='L_ic')
                plt.legend()
                plt.yscale('log');
                plt.savefig(f'{path_save}/loss.png')
                plt.close()

    return model, loss_list, loss_physics_list, loss_b_list, loss_ic_list


n_epochs = 30000
learn_rate = 5e-5
model_pinn = NN_new(3, 300, 2).to(device)
model_pinn, loss_pinn, l_physics, l_bc, loss_ic = train_pinn(model_pinn, learn_rate, n_epochs,
                                                             X_f, b_val, ic_val, ic_sol,
                                                             sample_idx=True)


torch.save({
            'epoch': i,
            'model_state_dict': model_pinn.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr': optimizer.param_groups[0]['lr'],
            }, f'{path_save}/pinn.pth')


plt.plot(loss_pinn, label='loss')
plt.plot(l_physics, label='L_f')
plt.plot(l_bc, label='L_bc')
plt.plot(loss_ic, label='L_ic')
plt.legend()
plt.yscale('log');

model_pinn.eval();
with torch.no_grad():
    y_pred = model_pinn(torch.tensor(X_test).to(device))


def get_density(y_pred):
    u = y_pred[:,0]
    v = y_pred[:,1]
    dens = u**2 + v**2
    return dens

dens = get_density(y_pred.cpu().numpy())
dens = dens.reshape(spatial_num, spatial_num, N+1)

dens1 = dens.sum(axis=0) * dx
dens2 = dens.sum(axis=1) * dx

sim_inter = numerical_sol(spatial_dim, spatial_num, N=N, g=g, T=T, omega2=omega2, m=m , d=d, h=h)

dx = 2 * spatial_dim / spatial_num
prob_density_inter = np.abs(sim_inter.Ψ)**2 * dx
density_truth_x1 = prob_density_inter.sum(axis=1).T
density_truth_x2 = prob_density_inter.sum(axis=2).T


sol_t = np.linspace(0, T, N+1)
bmeans1_inter = []
bstds1_inter = []
ts = []
for i, t in enumerate(sol_t):
    ts.append(t)
    bmeans1_inter.append(np.dot(x, dx**d * np.abs(np.sum((sim_inter.Ψ[i])*np.conjugate(sim_inter.Ψ[i]), axis=0))))
    bstds1_inter.append(np.dot((x - bmeans1_inter[-1]) ** 2,
                            dx**2 * np.abs(np.sum((sim_inter.Ψ[i])*np.conjugate(sim_inter.Ψ[i]), axis=0))))

make_density_plot(density_truth_x1, 'results/pinn_interact',  low_bound=lb, up_bound=ub,
                      title='$|\Psi(x_{i}, t)|^2, i = 1, 2$ truth') #vmin_max = [0, v_max_])


make_density_plot(dens2, 'results/pinn_interact',  low_bound=lb, up_bound=ub,
                      title='$|\Psi(x_{1}, t)|^2$ PINN')


make_density_plot(dens1, 'results/pinn_interact',  low_bound=lb, up_bound=ub,
                      title='$|\Psi(x_{1}, t)|^2$ PINN')

x.shape, x[:, 0]

ts = []
bmeans_pinn = []
bstds_pinn = []

for i, time in enumerate(time_splits):
    ts.append(time)
    bmeans_pinn.append(np.dot(x, dx * dens1[ :, i]))
    bstds_pinn.append(np.dot((x - bmeans_pinn[-1]) ** 2, dx * dens1[:, i]))

plt.plot(ts, bmeans1_inter, label='truth', color='black', linestyle='--',)
plt.plot(ts, bmeans_pinn, label='PINN')
plt.legend()
plt.title('Mean of $X_i$')
plt.xlabel('time')
plt.ylabel('value');


plt.plot(ts, bstds1_inter, label='truth', color='black', linestyle='--',)
plt.plot(ts, bstds_pinn, label='PINN')
plt.legend()
plt.title('Variance of $X_i$')
plt.xlabel('time')
plt.ylabel('value');
