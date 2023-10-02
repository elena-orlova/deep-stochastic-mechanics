import torch
from torch import nn
from torch.autograd import grad
from tqdm import tqdm
import numpy as np
from src.utils import *
import pdb

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

    def forward(self, x, t):
        x, idx_sorted = x.sort(dim=1)
        out = torch.hstack((x, t))
        out = self.out(self.middle(self.start(out)))
        res = torch.take_along_dim(out, idx_sorted, 1)
        return res

class NN_connect(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim):
        super(NN_connect, self).__init__()
        self.start = nn.Sequential(nn.Linear(input_dim, hid_dim), nn.Tanh())
        self.middle = skip_embed(hid_dim, hid_dim, hid_dim)
        self.out = nn.Sequential(nn.Tanh(), nn.Linear(hid_dim, output_dim))

    def forward(self, x, t):
        out = torch.hstack((x, t))
        out = self.out(self.middle(self.start(out)))
        return out

class NN_new_invariant(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim):
        super(NN_new_invariant, self).__init__()
        self.fc1 = nn.Linear(input_dim, hid_dim)
        self.relu1 = nn.Tanh()
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.relu2 = nn.Tanh()
        self.fc3 = nn.Linear(hid_dim, hid_dim)
        self.relu3 = nn.Tanh()
        self.fc4 = nn.Linear(hid_dim, output_dim)

    def forward(self, x, t):
        # print(x)
        x, idx_sorted = x.sort(dim=1)
        # print("After sorting")
        # print(x)
        out = torch.hstack((x, t))
        out = self.relu1(self.fc1(out))
        out = self.relu2(self.fc2(out))
        out = self.relu3(self.fc3(out))
        out = self.fc4(out)
        # print("NN out")
        # print(out)
        # TODO: works only for d=2
        res = torch.take_along_dim(out, idx_sorted, 1)
        # print("After de-sorting")
        # print(res)
        return res

class NN_old(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim):
        super(NN_old, self).__init__()
        self.fc1 = nn.Linear(input_dim, hid_dim)
        self.relu1 = nn.Tanh()
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.relu2 = nn.Tanh()
        self.fc3 = nn.Linear(hid_dim, hid_dim)
        self.relu3 = nn.Tanh()
        self.fc4 = nn.Linear(hid_dim, output_dim)

    def forward(self, x, t):
        out = torch.hstack((x, t))
        out = self.relu1(self.fc1(out))
        out = self.relu2(self.fc2(out))
        out = self.relu3(self.fc3(out))
        out = self.fc4(out)
        return out


# class NN(nn.Module):
    # def __init__(self, input_dim, hid_dim, output_dim):
    #     super(NN, self).__init__()
    #     self.fc1 = nn.Linear(input_dim, hid_dim)
    #     self.relu1 = nn.Tanh()
    #     self.fc2 = nn.Linear(hid_dim, hid_dim)
    #     self.relu2 = nn.Tanh()
    #     self.fc4 = nn.Linear(hid_dim, output_dim)
    #
    # def forward(self, x, t):
    #     out = torch.hstack((x, t))
    #     out = self.fc1(out)
    #     out = self.relu1(out)
    #     out = self.fc2(out)
    #     out = self.relu2(out)
    #     out = self.fc4(out)
    #     return out


def get_jacobian(f, x):
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

    B, N = x.shape
    y = f(x)
    jacobian = list()
    for i in range(y.shape[-1]):
        v = torch.zeros_like(y)
        v[:, i] = 1.
        dy_i_dx = grad(y,
                       x,
                       grad_outputs=v,
                       retain_graph=True,
                       create_graph=True,
                       allow_unused=True)[0]  # shape [B, N]
        jacobian.append(dy_i_dx)

    jacobian = torch.stack(jacobian, dim=2).requires_grad_()

    return jacobian


def train_dsm_interact(net_u, net_v,
                       device,
                       optimizer,
                       criterion,
                       alpha, beta, gamma,
                       N, time_splits,
                       path,
                       n_iter=2000,
                       batch_size = 100,
                       iter_threshhold=4000,
                       frac_1 = 0.8, frac_2=0.6,
                       omega2=1,
                       d=2,
                       mu=0, sig2=1,
                       m=1, h=1, T=1,
                       g=1,
                       flip=False,
                       scheduler=None,
                       epoch_start = 0,
                       ):
    losses = []
    losses_newton = []
    losses_sm = []
    losses_init = []
    N_fast = N
    omega = np.sqrt(omega2)

    with tqdm(range(epoch_start, n_iter), unit="iter") as tepoch:
        for tau in tepoch:
            if tau > 0 and (tau % 10 ==0 or tau == n_iter-1):
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

            if tau == 0 or tau == epoch_start:
                EPOCH = tau

                X_0 = torch.Tensor(np.random.multivariate_normal(np.ones(d) * mu,
                                                                 (sig2**2)*np.eye(d), batch_size)).to(device)
                X_0.requires_grad = True
                u0_val = torch.Tensor(u_0(omega)(X_0)).to(device)
                v0_val = torch.Tensor(v_0(X_0, device, h=h, m=m)).to(device)
                optimizer.zero_grad()
                eps = [np.random.multivariate_normal(np.zeros(d), np.eye(d), batch_size) for i in range(N+1)]
                for i in range(1, N_fast+1): # iterate over time steps
                    t_prev = time_splits[i-1].clone()
                    if i == 1:
                        X_prev = X_0.clone().to(device) # X_{i-1}
                    else:
                        X_prev = X_i.clone().to(device) # X_{i-1}

                    t_i = time_splits[i].clone()

                    t_prev_batch = time_transform(t_prev).expand(batch_size, 1).to(device).clone()
                    X_i = torch.Tensor(X_prev).to(device) + T / N * (net_u(X_prev, t_prev_batch) + \
                                                                     net_v(X_prev, t_prev_batch)) \
                            + torch.Tensor(np.sqrt(h*T/(m*N)) * eps[i-1]).to(device)

                    if i > 1:
                        Xs = torch.hstack((Xs, X_i))
                    else:
                        Xs = X_i.clone()

                X_0_iter0 = X_0.clone() # collect X_0 from the initial iter
                Xs = torch.concat((X_0_iter0, Xs), axis=1)
            elif tau <= iter_threshhold or tau <= epoch_start + 10:
                BATCH_size = int(frac_1* batch_size)

                X_0_iter = torch.Tensor(np.random.multivariate_normal(np.ones(d) * mu,
                                                                 (sig2**2)*np.eye(d), BATCH_size)).to(device)
                X_0_iter.requires_grad = True
                u0_val = torch.Tensor(u_0(omega)(X_0_iter)).to(device)
                v0_val = torch.Tensor(v_0(X_0_iter, device, h=h, m=m)).to(device)
                optimizer.zero_grad()
                eps = [np.random.multivariate_normal(np.zeros(d), np.eye(d), BATCH_size) for i in range(N+1)]
                # get one trajectory
                for i in range(1, N_fast+1):
                    t_prev = time_splits[i-1].clone()
                    if i == 1:
                        X_prev = X_0_iter.clone().to(device)
                    else:
                        X_prev = X_i_iter.clone().to(device)

                    t_i = time_splits[i].clone()

                    t_prev_batch = time_transform(t_prev).expand(BATCH_size, 1).to(device)
                    X_i_iter = torch.Tensor(X_prev).to(device) + T / N * (net_u(X_prev, t_prev_batch) + \
                                                                     net_v(X_prev, t_prev_batch)) \
                            + torch.Tensor(np.sqrt(h*T/(m*N)) * eps[i-1]).to(device)

                    if i > 1:
                        Xs_iter = torch.hstack((Xs_iter, X_i_iter))
                    else:
                        Xs_iter = X_i_iter.clone()

                Xs_iter = torch.concat((X_0_iter, Xs_iter), axis=1)

                # replace the old batch with the new one
                if tau == 1 or tau == epoch_start + 1:
                    Xs_all = torch.vstack((Xs[BATCH_size:].detach(), Xs_iter.detach()) )
                    # Xs_all.requires_grad = True

                    X_0_all = torch.vstack((X_0_iter0[BATCH_size:].detach(), X_0_iter.detach()) )
                    # X_0_all.requires_grad = True
                else:
                    Xs_all = Xs_all.reshape((batch_size, (N+1)*d))
                    X_0_all = X_0_all.reshape((batch_size, d))
                    Xs_all = torch.roll(Xs_all, -BATCH_size, 0)
                    X_0_all = torch.roll(X_0_all, -BATCH_size, 0)
                    Xs_all = torch.vstack((Xs_all[BATCH_size:].detach(), Xs_iter.detach()) )
                    # Xs_all.requires_grad = True

                    X_0_all = torch.vstack((X_0_all[BATCH_size:].detach(), X_0_iter.detach()) )
                    # X_0_all.requires_grad = True

                Xs_all = Xs_all.reshape(batch_size*(N+1), d)
                X_0_all = X_0_all.reshape(batch_size, d)

                if flip:
                    idx_flip = np.random.choice(Xs_all.shape[0], Xs_all.shape[0]//2, replace=False)
                    Xs_all[idx_flip, :] = torch.flip(Xs_all[idx_flip, :], dims=(1, ))

                Xs_all.requires_grad = True
                X_0_all.requires_grad = True

                time_splits_batches = time_splits.repeat(batch_size).reshape(batch_size, N+1).reshape(batch_size*(N+1), 1).to(device)
                time_splits_batches.requires_grad = True

                out_u = net_u(Xs_all, time_splits_batches)
                out_v = net_v(Xs_all, time_splits_batches)
                du_dt = torch.zeros((batch_size*(N+1), d)).to(device)
                for i in range(d):
                    vector = torch.zeros_like(out_u)
                    vector[:, i] = 1

                    dudt = grad(out_u, time_splits_batches, grad_outputs=vector, create_graph=True)[0]
                    du_dt[:, i] = dudt[:, 0]

                dv_dt = torch.zeros((batch_size*(N+1), d)).to(device)
                for i in range(d):
                    vector = torch.zeros_like(out_u)
                    vector[:, i] = 1

                    dvdt = grad(out_v, time_splits_batches, grad_outputs=vector, create_graph=True)[0]
                    dv_dt[:, i] = dvdt[:, 0]

                d_norm = torch.einsum("ijk,ik->ij", get_jacobian(lambda x: net_u(x, time_splits_batches), Xs_all),
                                      net_u(Xs_all, time_splits_batches)) - \
                         torch.einsum("ijk,ik->ij", get_jacobian(lambda x: net_v(x, time_splits_batches), Xs_all),
                                      net_v(Xs_all, time_splits_batches))

                dv_ddx = get_jacobian(lambda x: torch.einsum("jii",
                                                    get_jacobian(lambda x: net_v(x, time_splits_batches), x))[:, None],
                                      Xs_all)[:, :, 0]

                du_ddx = get_jacobian(lambda x: torch.einsum("jii",
                                                    get_jacobian(lambda x: net_u(x, time_splits_batches), x))[:, None],
                                      Xs_all)[:, :, 0]

                out_uv = net_v(Xs_all, time_splits_batches) * net_u(Xs_all, time_splits_batches)
                dvu_dx = grad(out_uv, Xs_all, grad_outputs=torch.ones_like(out_uv), create_graph=True)[0]

                L_sm = criterion(du_dt, -(h/(2*m)) * dv_ddx - dvu_dx) #/N
#                 print(V_x_i(omega2)(Xs_all).shape, du_ddx.shape, dv_dt.shape, d_norm.shape)
                L_nl = criterion(dv_dt, d_norm + (h / 2 / m) * du_ddx - V_x_i(Xs_all, device, m=m, g=g, om=omega2).to(device)/m) #/N

                u0_val = torch.Tensor(u_0(omega)(X_0_all)).to(device)
                v0_val = torch.Tensor(v_0(X_0_all, device, h=h, m=m)).to(device)
                L_ic = criterion(net_u(X_0_all,
                                       time_splits[0].expand(batch_size, 1).to(device)), u0_val) \
                         + criterion(net_v(X_0_all,
                                           time_splits[0].expand(batch_size, 1).to(device)), v0_val)

                loss = (alpha * L_sm + beta * L_nl + gamma * L_ic) / 3.0
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
            elif tau >= iter_threshhold:
                BATCH_size = int(frac_2* batch_size)

                X_0_iter = torch.Tensor(np.random.multivariate_normal(np.ones(d) * mu,
                                                                 (sig2**2)*np.eye(d), BATCH_size)).to(device)
                X_0_iter.requires_grad = True
                u0_val = torch.Tensor(u_0(omega)(X_0_iter)).to(device)
                v0_val = torch.Tensor(v_0(X_0_iter, device, h=h, m=m)).to(device)
                optimizer.zero_grad()
                eps = [np.random.multivariate_normal(np.zeros(d), np.eye(d), BATCH_size) for i in range(N+1)]
                # get one trajectory
                for i in range(1, N_fast+1):
                    t_prev = time_splits[i-1].clone()
                    if i == 1:
                        X_prev = X_0_iter.clone().to(device)
                    else:
                        X_prev = X_i_iter.clone().to(device)

                    t_i = time_splits[i].clone()

                    t_prev_batch = time_transform(t_prev).expand(BATCH_size, 1).to(device)
                    X_i_iter = torch.Tensor(X_prev).to(device) + T / N * (net_u(X_prev, t_prev_batch) + \
                                                                     net_v(X_prev, t_prev_batch)) \
                            + torch.Tensor(np.sqrt(h*T/(m*N)) * eps[i-1]).to(device)

                    if i > 1:
                        Xs_iter = torch.hstack((Xs_iter, X_i_iter))
                    else:
                        Xs_iter = X_i_iter.clone()

                Xs_iter = torch.concat((X_0_iter, Xs_iter), axis=1)

                # replace the old batch with the new one
                Xs_all = Xs_all.reshape((batch_size, (N+1)*d))
                X_0_all = X_0_all.reshape((batch_size, d))
                Xs_all = torch.roll(Xs_all, -BATCH_size, 0)
                X_0_all = torch.roll(X_0_all, -BATCH_size, 0)
                Xs_all = torch.vstack((Xs_all[BATCH_size:].detach(), Xs_iter.detach()) )
                # Xs_all.requires_grad = True

                X_0_all = torch.vstack((X_0_all[BATCH_size:].detach(), X_0_iter.detach()) )
                # X_0_all.requires_grad = True

                Xs_all = Xs_all.reshape(batch_size*(N+1), d)
                X_0_all = X_0_all.reshape(batch_size, d)

                if flip:
                    idx_flip = np.random.choice(Xs_all.shape[0], Xs_all.shape[0]//2, replace=False)
                    Xs_all[idx_flip, :] = torch.flip(Xs_all[idx_flip, :], dims=(1, ))
                Xs_all.requires_grad = True
                X_0_all.requires_grad = True


                time_splits_batches = time_splits.repeat(batch_size).reshape(batch_size, N+1).reshape(batch_size*(N+1), 1).to(device)
                time_splits_batches.requires_grad = True

                out_u = net_u(Xs_all, time_splits_batches)
                out_v = net_v(Xs_all, time_splits_batches)
                du_dt = torch.zeros((batch_size*(N+1), d)).to(device)
                for i in range(d):
                    vector = torch.zeros_like(out_u)
                    vector[:, i] = 1

                    dudt = grad(out_u, time_splits_batches, grad_outputs=vector, create_graph=True)[0]
                    du_dt[:, i] = dudt[:, 0]

                dv_dt = torch.zeros((batch_size*(N+1), d)).to(device)
                for i in range(d):
                    vector = torch.zeros_like(out_u)
                    vector[:, i] = 1

                    dvdt = grad(out_v, time_splits_batches, grad_outputs=vector, create_graph=True)[0]
                    dv_dt[:, i] = dvdt[:, 0]

                d_norm = torch.einsum("ijk,ik->ij", get_jacobian(lambda x: net_u(x, time_splits_batches), Xs_all),
                                      net_u(Xs_all, time_splits_batches)) - \
                         torch.einsum("ijk,ik->ij", get_jacobian(lambda x: net_v(x, time_splits_batches), Xs_all),
                                      net_v(Xs_all, time_splits_batches))

                dv_ddx = get_jacobian(lambda x: torch.einsum("jii",
                                                    get_jacobian(lambda x: net_v(x, time_splits_batches), x))[:, None],
                                      Xs_all)[:, :, 0]

                du_ddx = get_jacobian(lambda x: torch.einsum("jii",
                                                    get_jacobian(lambda x: net_u(x, time_splits_batches), x))[:, None],
                                      Xs_all)[:, :, 0]

                out_uv = net_v(Xs_all, time_splits_batches) * net_u(Xs_all, time_splits_batches)
                dvu_dx = grad(out_uv, Xs_all, grad_outputs=torch.ones_like(out_uv), create_graph=True)[0]

                L_sm = criterion(du_dt, -(h/(2*m)) * dv_ddx - dvu_dx) #/N
#                 print(V_x_i(omega2)(Xs_all).shape, du_ddx.shape, dv_dt.shape, d_norm.shape)
                L_nl = criterion(dv_dt, d_norm + (h / 2 / m) * du_ddx - V_x_i(Xs_all, device, m=m, g=g, om=omega2).to(device)/m) #/N

                u0_val = torch.Tensor(u_0(omega)(X_0_all)).to(device)
                v0_val = torch.Tensor(v_0(X_0_all, device, h=h, m=m)).to(device)
                L_ic = criterion(net_u(X_0_all,
                                       time_splits[0].expand(batch_size, 1).to(device)), u0_val) \
                         + criterion(net_v(X_0_all,
                                           time_splits[0].expand(batch_size, 1).to(device)), v0_val)

                loss = (alpha * L_sm + beta * L_nl + gamma * L_ic) / 3.0
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
            else:
                print('NO CHOICE')

    return net_u, net_v, losses, losses_newton, losses_sm, losses_init



def sample_w_nn(net_u, net_v, N, d, time_splits, device, mu, sig2, T, m=1, h=0.1, nu_s=1):
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
                a = torch.hstack((X_prev, time_splits[i-1].expand(samples, 1).to(device)))
                X_i = torch.Tensor(X_prev).to(device) + T / N * \
                (nu_s * net_u(X_prev, time_splits[i-1].expand(samples, 1).to(device)) \
                + net_v(X_prev, time_splits[i-1].expand(samples, 1).to(device))) \
                        + torch.Tensor(np.sqrt((h*T * nu_s)/(m*N)) * eps[i-1]).to(device)
                X_test[trial, i, :] = X_i.cpu().numpy()
                X_prev = X_i.clone()

    return X_test
