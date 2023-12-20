import torch
from torch import nn
from torch.autograd import grad
from tqdm import tqdm
import numpy as np
from src.utils_fast_new import *
# import pdb
from typing import List, Optional, cast
from torch.jit import ScriptModule

class skip_embed(torch.jit.ScriptModule):
    def __init__(self, input_dim, latent_dim, output_dim):
        super().__init__()
        self.embed = nn.Sequential(
                               nn.Linear(input_dim, latent_dim),
                               ACT(),
                               nn.Linear(latent_dim, output_dim),
                               ACT(),
                               )
    
    @torch.jit.script_method
    def forward(self, input):
        out = self.embed(input) + input
        return out

ACT = nn.Tanh

class NN_new(torch.jit.ScriptModule):
    def __init__(self, input_dim, hid_dim, output_dim, h:float, m:float, om:float, w1:float, w2:float):
        super(NN_new, self).__init__()
        self.h = h
        self.m = m
        self.om = om
        self.w1 = w1
        self.w2 = w2
        self.start = nn.Sequential(nn.Linear(input_dim, hid_dim), nn.Tanh())
        self.middle = skip_embed(hid_dim, hid_dim, hid_dim)
        self.out = nn.Sequential(nn.Tanh(), nn.Linear(hid_dim, output_dim))
    
    @torch.jit.script_method
    def forward(self, x, t):
        u0_val = u_0(x, self.om)
        v0_val = v_0(x, h=self.h, m=self.m)
        x, idx_sorted = x.sort(dim=1)
        _, inv_idx = idx_sorted.sort(dim=1)
        out = torch.hstack((x, t))
        out = self.out(self.middle(self.start(out)))
        res = torch.take_along_dim(out, inv_idx, dim=1)
        return t * res + u0_val * self.w1 + v0_val * self.w2


class NN_connect(torch.jit.ScriptModule):
    def __init__(self, input_dim, hid_dim, output_dim):
        super(NN_connect, self).__init__()
        self.start = nn.Sequential(nn.Linear(input_dim, hid_dim), nn.Tanh())
        self.middle = skip_embed(hid_dim, hid_dim, hid_dim)
        self.out = nn.Sequential(nn.Tanh(), nn.Linear(hid_dim, output_dim))

    @torch.jit.script_method
    def forward(self, x, t):
        out = torch.hstack((x, t))
        out = self.out(self.middle(self.start(out)))
        return out


class NN_connect_init(torch.jit.ScriptModule):
    def __init__(self, input_dim, hid_dim, output_dim, h:float, m:float, om:float, w1:float, w2:float):
        super(NN_connect_init, self).__init__()
        self.h = h
        self.m = m
        self.om = om
        self.w1 = w1
        self.w2 = w2
        self.start = nn.Sequential(nn.Linear(input_dim, hid_dim), nn.Tanh())
        self.middle = skip_embed(hid_dim, hid_dim, hid_dim)
        self.out = nn.Sequential(nn.Tanh(), nn.Linear(hid_dim, output_dim))

    @torch.jit.script_method
    def forward(self, x, t):
        out = torch.hstack((x, t))
        out = self.out(self.middle(self.start(out)))
        u0_val = u_0(x, self.om)
        v0_val = v_0(x, h=self.h, m=self.m)
        return t * out + u0_val * self.w1 + v0_val * self.w2


class NN_new_invariant(torch.jit.ScriptModule):
    def __init__(self, input_dim, hid_dim, output_dim):
        super(NN_new_invariant, self).__init__()
        self.fc1 = nn.Linear(input_dim, hid_dim)
        self.relu1 = nn.Tanh()
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.relu2 = nn.Tanh()
        self.fc3 = nn.Linear(hid_dim, hid_dim)
        self.relu3 = nn.Tanh()
        self.fc4 = nn.Linear(hid_dim, output_dim)

    @torch.jit.script_method
    def forward(self, x, t):
        # print(x)
        x, idx_sorted = x.sort(dim=1)
        _, inv_idx = idx_sorted.sort(dim=1)
        # print("After sorting")
        # print(x)
        out = torch.hstack((x, t))
        out = self.relu1(self.fc1(out))
        out = self.relu2(self.fc2(out))
        out = self.relu3(self.fc3(out))
        out = self.fc4(out)
        # print("NN out")
        # print(out)
        res = torch.take_along_dim(out, inv_idx, 1)
        # print("After de-sorting")
        # print(res)
        return res


class NN_old(torch.jit.ScriptModule):
    def __init__(self, input_dim, hid_dim, output_dim):
        super(NN_old, self).__init__()
        self.fc1 = nn.Linear(input_dim, hid_dim)
        self.relu1 = nn.Tanh()
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.relu2 = nn.Tanh()
        self.fc3 = nn.Linear(hid_dim, hid_dim)
        self.relu3 = nn.Tanh()
        self.fc4 = nn.Linear(hid_dim, output_dim)

    @torch.jit.script_method
    def forward(self, x, t):
        out = torch.hstack((x, t))
        out = self.relu1(self.fc1(out))
        out = self.relu2(self.fc2(out))
        out = self.relu3(self.fc3(out))
        out = self.fc4(out)
        return out
    

def unwrap(opt: Optional[torch.Tensor]) -> torch.Tensor:
    if opt is None:
        raise ValueError("Optional value is None")
    return opt


def sample_trajectory_M(mu:float, sig2:float, T:float, net_u, net_v, sampler, device:torch.device, 
                        batch_size:int, d:int, N:int, h:float, m:float, omega:float, 
                        diff_coef:torch.Tensor, time_splits:torch.Tensor, M:int):
    with torch.no_grad():
        X_0_iter = torch.tensor(mu, device=device) + torch.tensor(sig2, device=device) * sampler.sample(sample_shape=(batch_size, d))
        eps = sampler.sample(sample_shape=(N, batch_size, d))
        steps_noise = torch.tensor(T/N, device=device) * torch.rand((M*batch_size, 1), device=device)
        trajectory = [X_0_iter]
        X_i_iter = X_0_iter
        idx, _ = torch.sort(torch.ones(N, device=device).multinomial(M, replacement=False))
        iterator = 0
        trajectory_times = [torch.zeros((batch_size, 1), device=device)]
        for i in range(1, N+1):
            t_prev = time_splits[batch_size*(i-1):batch_size*i]
            t_i = time_splits[batch_size*i:batch_size*(i+1)]
            if idx[iterator] + 1 == i:
                steps = steps_noise[batch_size*iterator:batch_size*(iterator+1)]
                X_i_iter_ = X_i_iter + steps* (net_u(X_i_iter, t_prev) + \
                                                                net_v(X_i_iter, t_prev)) \
                        + diff_coef * torch.sqrt(steps)*eps[i-1]
                trajectory.append(X_i_iter_)
                trajectory_times.append(t_prev + steps * torch.tensor(T / N, device=device))
                iterator += 1
                if iterator >= M:
                    break
            X_i_iter = X_i_iter + (t_i - t_prev)* (net_u(X_i_iter, t_prev) + \
                                                            net_v(X_i_iter, t_prev)) \
                    + diff_coef * torch.sqrt(t_i - t_prev)*eps[i-1]
        return torch.vstack(trajectory), torch.vstack(trajectory_times)


def sample_trajectory_random_grid(mu:float, sig2:float, T:float, net_u, net_v, sampler, device:torch.device, batch_size:int, d:int, N:int, h:float, m:float, omega:float, diff_coef:torch.Tensor, time_splits:torch.Tensor):
    with torch.no_grad():
        X_0_iter = torch.tensor(mu, device=device) + torch.tensor(sig2, device=device) * sampler.sample(sample_shape=(batch_size, d))
        eps = sampler.sample(sample_shape=(N, batch_size, d))
        trajectory = [X_0_iter]
        X_i_iter = X_0_iter
        for i in range(1, N+1):
            t_prev = time_splits[batch_size*(i-1):batch_size*i]
            t_i = time_splits[batch_size*i:batch_size*(i+1)]
            X_i_iter = X_i_iter + (t_i - t_prev)* (net_u(X_i_iter, t_prev) + \
                                                            net_v(X_i_iter, t_prev)) \
                    + diff_coef * torch.sqrt(t_i - t_prev)*eps[i-1]
            trajectory.append(X_i_iter)
        return torch.vstack(trajectory)


def sample_trajectory_fixed(mu:float, sig2:float, T:float, net_u, net_v, sampler, device:torch.device, batch_size:int, d:int, N:int, h:float, m:float, omega:float, diff_coef:torch.Tensor, time_splits:torch.Tensor):
    with torch.no_grad():
        X_0_iter = torch.tensor(mu, device=device) + torch.tensor(sig2, device=device) * sampler.sample(sample_shape=(batch_size, d))
        eps = sampler.sample(sample_shape=(N, batch_size, d))
        trajectory = [X_0_iter]
        X_i_iter = X_0_iter
        for i in range(1, N+1):
            t_prev = time_splits[i-1].clone()
            t_prev_batch = t_prev.to(device).expand(batch_size, 1)
            X_i_iter = X_i_iter + T / N * (net_u(X_i_iter, t_prev_batch) + \
                                                            net_v(X_i_iter, t_prev_batch)) \
                    + diff_coef * eps[i-1]
            trajectory.append(X_i_iter)

        return torch.vstack(trajectory)


@torch.jit.script
def _get_jacobian(y:List[torch.Tensor], x:List[torch.Tensor], device:torch.device) -> torch.Tensor:
    """Computes the Jacobian of f w.r.t x.

    This is according to the reverse mode autodiff rule,

    sum_i v^b_i dy^b_i / dx^b_j = sum_i x^b_j R_ji v^b_i,

    where:
    - b is the batch index from 0 to B - 1
    - i, j are the vector indices from 0 to N-1
    - v^b_i is a "test vector", which is set to 1 column-wise to obtain the correct
        column vectors out ot the above expression.

    :param f: function R^N -> R^N
    :param x: torch.tensor of shape [B, N]
    :return: Jacobian matrix (torch.tensor) of shape [B, N, N]
    """
    B, N = x[0].shape
    jacobian = torch.jit.annotate(List[torch.Tensor], []) #list()
    for i in range(y[0].shape[-1]):
        v_ = torch.zeros_like(y[0], device=device)
        v_[:, i] = 1.0
        v: Optional[torch.Tensor] = v_
        dy_i_dx = grad(y,
                    x,
                    grad_outputs=[v,],
                    retain_graph=True,
                    create_graph=True,
                    allow_unused=True)[0]
        jacobian.append(unwrap(dy_i_dx))
    jacobian = torch.stack(jacobian, dim=2).requires_grad_()
    return jacobian


def get_jacobian(f, x:torch.Tensor, device:torch.device):
    return _get_jacobian([f(x),], [x,], device)


def train_dsm_interact(net_u, net_v,
                       device,
                       optimizer,
                       criterion,
                       alpha, beta, gamma,
                       N, _time_splits,
                       path,
                       save_freq=50,
                       n_iter=2000,
                       batch_size = 100,
                       omega2=1,
                       d=2,
                       mu=0, sig2=1,
                       m=1, h=0.1, T=1,
                       g=1,
                       scheduler=None,
                       epoch_start = 0,
                       sampling_scheme = "random_grid",
                       M = 100
                       ):
    losses = []
    losses_newton = []
    losses_sm = []
    losses_init = []
    omega = np.sqrt(omega2)
    sampler = torch.distributions.normal.Normal(torch.tensor(0.0, device=device), torch.tensor(1.0, device=device))
    diff_coef = torch.sqrt(torch.tensor(h/m, device=device))
    print('g = {}'.format(g))
    with tqdm(range(epoch_start, n_iter), unit="iter") as tepoch:
        Xs_all = torch.zeros((N+1, batch_size, d), device=device)
        # X_0_all = torch.zeros((batch_size, d), device=device)
        for tau in tepoch:
            if tau > 0 and (tau % save_freq ==0 or tau == n_iter-1) and len(losses):
                PATH_check = os.path.join(path, 'net_uv_{}_ep_interact_check.pt'.format(n_iter))
                LOSS, LOSS_NL, LOSS_SM, LOSS_IC = losses[-1], losses_newton[-1], losses_sm[-1], losses_init[-1]

                torch.save({
                            'epoch': tau,
                            'model_u_state_dict': net_u.state_dict(),
                            'model_v_state_dict': net_v.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'lr': optimizer.param_groups[0]['lr'],
                            'loss_criterion': criterion,
                            'loss': LOSS,
                            'losses_last_100': losses[-100:],
                            'loss_nl': LOSS_NL,
                            'loss_sm': LOSS_SM,
                            'loss_ic': LOSS_IC,
                            }, PATH_check)

                make_loss_plot(losses, losses_sm, losses_newton, losses_init, path)

            if sampling_scheme == "fixed":
                with torch.no_grad():
                    diff_coef = torch.sqrt(torch.tensor(h*T/(m*N), device=device))
                    time_splits_batches = _time_splits.to(device).repeat(batch_size).reshape(batch_size, 
                                                                                             N+1).reshape(batch_size*(N+1), 1)
                Xs_all = sample_trajectory_fixed(mu, sig2, T, net_u, net_v, sampler, device, batch_size, 
                                           d, N, h, m, omega, diff_coef, time_splits_batches)
            elif sampling_scheme == "random_grid":
                with torch.no_grad():
                    time_splits_batches = _time_splits.to(device).repeat_interleave(batch_size)[:, None]
                    time_splits_batches[batch_size:] = time_splits_batches[batch_size:] - \
                                                        torch.tensor(T/N, device=device) * torch.rand((N*batch_size, 1), 
                                                                                                      device=device)
                Xs_all = sample_trajectory_random_grid(mu, sig2, T, net_u, net_v, sampler, device, batch_size, 
                                           d, N, h, m, omega, diff_coef, time_splits_batches)
            elif sampling_scheme == "interpolated":
                with torch.no_grad():
                    time_splits_batches = _time_splits.to(device).repeat_interleave(batch_size)[:, None]
                Xs_all, time_splits_batches = sample_trajectory_M(mu, sig2, T, net_u, net_v, sampler, device, batch_size, 
                                                                  d, N, h, m, omega, diff_coef, time_splits_batches, M)
            else: 
                print("No such sampling type:", sampling_scheme)
            
            time_splits_batches.requires_grad = True
            Xs_all.requires_grad = True

            optimizer.zero_grad(set_to_none=True)

            out_u = net_u(Xs_all, time_splits_batches)
            out_v = net_v(Xs_all, time_splits_batches)
            if sampling_scheme == "random_grid" or sampling_scheme == "fixed": 
                du_dt = torch.zeros((batch_size*(N+1), d), device=device)
            elif sampling_scheme == "interpolated":
                du_dt = torch.zeros((batch_size*(M+1), d), device=device)
            for i in range(d):
                vector = torch.zeros_like(out_u, device=device)
                vector[:, i] = 1
                dudt = grad(out_u, time_splits_batches, grad_outputs=vector, create_graph=True)[0]
                du_dt[:, i] = dudt[:, 0]

            if sampling_scheme == "random_grid" or sampling_scheme == "fixed":
                dv_dt = torch.zeros((batch_size*(N+1), d), device=device)
            elif sampling_scheme == "interpolated":
                dv_dt = torch.zeros((batch_size*(M+1), d), device=device)
            for i in range(d):
                vector = torch.zeros_like(out_u, device=device)
                vector[:, i] = 1
                dvdt = grad(out_v, time_splits_batches, grad_outputs=vector, create_graph=True)[0]
                dv_dt[:, i] = dvdt[:, 0]

            d_norm = torch.einsum("ijk,ik->ij", get_jacobian(lambda x: net_u(x, time_splits_batches), Xs_all, device),
                                  net_u(Xs_all, time_splits_batches)) - \
                     torch.einsum("ijk,ik->ij", get_jacobian(lambda x: net_v(x, time_splits_batches), Xs_all, device),
                                  net_v(Xs_all, time_splits_batches))

            dv_ddx = get_jacobian(lambda x: torch.einsum("jii",
                                                get_jacobian(lambda x: net_v(x, time_splits_batches), x, device))[:, None],
                                  Xs_all, device)[:, :, 0]

            du_ddx = get_jacobian(lambda x: torch.einsum("jii",
                                                get_jacobian(lambda x: net_u(x, time_splits_batches), x, device))[:, None],
                                  Xs_all, device)[:, :, 0]
            
            out_uv = net_v(Xs_all, time_splits_batches) * net_u(Xs_all, time_splits_batches)
            dvu_dx = grad(out_uv, Xs_all, grad_outputs=torch.ones_like(out_uv), create_graph=True)[0]

            if sampling_scheme == "interpolated" or sampling_scheme == "random_grid": 
                exp = torch.sqrt(torch.exp((T-time_splits_batches)) - 1)
                exp2 = torch.sqrt(torch.exp(2.0*(T-time_splits_batches))-1)
                L_sm = criterion(exp2*du_dt, exp2*(-(h/(2*m)) * dv_ddx - dvu_dx)) 
                # print("!!!", V_x_i(Xs_all, device, d, m=m, g=g, om=omega2).shape, du_ddx.shape, d_norm.shape)
                # L_nl = criterion(exp2*dv_dt, exp2*(d_norm + (h / 2 / m) * du_ddx - V_x_i(Xs_all, device, d, m=m, g=g, om=omega2)/m)) 
                L_nl = criterion(exp*dv_dt, exp*(d_norm + (h / 2 / m) * du_ddx - V_x_i(Xs_all, device, d, m=m, g=g, om=omega2)/m)) 
                
                # L_sm = criterion(du_dt, -(h/(2*m)) * dv_ddx - dvu_dx) 
                # L_nl = criterion(dv_dt, d_norm + (h / 2 / m) * du_ddx - V_x_i(Xs_all, device, d, m=m, g=g, om=omega2)/m) 
            else:
                L_sm = criterion(du_dt, -(h/(2*m)) * dv_ddx - dvu_dx) 
                # print("!!!", V_x_i(Xs_all, device, d, m=m, g=g, om=omega2).shape, du_ddx.shape, d_norm.shape)
                # print("!!!!", V_x_i(Xs_all, device, d, m=m, g=g, om=omega2).max(),  du_ddx.max(), d_norm.max())
                L_nl = criterion(dv_dt, d_norm + (h / 2 / m) * du_ddx - V_x_i(Xs_all, device, d, m=m, g=g, om=omega2)/m) 

            # u0_val = u_0(Xs_all[:batch_size], omega)
            # v0_val = v_0(Xs_all[:batch_size], h=h, m=m)
            # tss = time_splits_batches[:batch_size]
            L_ic = torch.zeros(1, device=device)
            # L_ic = criterion(net_u(Xs_all[:batch_size], tss), u0_val) \
            #          + criterion(net_v(Xs_all[:batch_size], tss), v0_val)

            loss = (alpha * L_sm + beta * L_nl + gamma * L_ic) / 2.0
            losses.append(loss.item())
            losses_newton.append(L_nl.item())
            losses_sm.append(L_sm.item())
            losses_init.append(L_ic.item())
            tepoch.set_postfix(loss_iter=loss.item(),
                               loss_mean=np.mean(losses[-10:]),
                               loss_std=np.std(losses[-10:]),
                               loss_Newton=losses_newton[-1],
                               losses_sm=losses_sm[-1],
                               losses_init=losses_init[-1])

            loss.backward(retain_graph=True)
            optimizer.step()

            if scheduler:
                scheduler.step()

    return net_u, net_v, losses, losses_newton, losses_sm, losses_init


def sample_w_nn(net_u, net_v, N, d, time_splits, device, mu, sig2, T, m=1.0, h=0.1, nu_s=1.0):
    net_u.eval()
    net_v.eval()

    samples = 10000
    num_trials = 10
    X_test = np.zeros((num_trials, N+1, samples, d))

    # can change nu value for sampling
    # nu = 0 no u and noise during sampling (but still running net_u(x))
    for trial in tqdm(range(num_trials)):
        with torch.no_grad():
            X_0 = torch.Tensor(np.random.multivariate_normal(np.ones(d) * mu, (sig2**2)*np.eye(d), samples)).to(device)
            X_test[trial, 0, :] = X_0.cpu().numpy()
            X_prev = X_0.clone()
            eps = [np.random.multivariate_normal(np.zeros(d), np.eye(d), samples) for i in range(N)]
            for i in range(1, N+1):
                # a = torch.hstack((X_prev, time_splits[i-1].expand(samples, 1).to(device)))
                X_i = torch.Tensor(X_prev).to(device) + T / N * \
                (nu_s * net_u(X_prev, time_splits[i-1].expand(samples, 1).to(device)) \
                + net_v(X_prev, time_splits[i-1].expand(samples, 1).to(device))) \
                        + torch.Tensor(np.sqrt((h*T * nu_s)/(m*N)) * eps[i-1]).to(device)
                X_test[trial, i, :] = X_i.cpu().numpy()
                X_prev = X_i.clone()

    return X_test
