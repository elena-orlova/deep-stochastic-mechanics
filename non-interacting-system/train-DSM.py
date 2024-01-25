from log import setup_logger
setup_logger()
from log import ZeroRankLogger
logger = ZeroRankLogger(__name__)
import numpy as np
from tqdm import tqdm
import argparse

import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
from matplotlib import animation

import numpy as np
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import qmsolve
import json
import torch

from src.utils import *
from src.nn import *
torch.backends.cudnn.benchmark = True


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='DSM model training and eval for 2 interacting particles in harm osc',)

    parser.add_argument('-T', type=float, default=1.5, help="Time [0, T]")
    parser.add_argument('-N', type=int, default=1000, help="Split [0, T] in N steps")
    parser.add_argument('-M', type=int, default=100, help="Sample M points out of N")
    parser.add_argument('-omega2', type=float, default=1, help="Omega^2 for harm oscillator")
    parser.add_argument('-g', type=float, default=2, help="g defines interaction strength")
    parser.add_argument('-d', type=int, default=2, help="number of 1d particles")
    parser.add_argument('-multi_gpu', type=int, default=0, help="use DataParallel?")

    parser.add_argument('-n_epochs', type=int, default=1000, help="Number of epochs")
    parser.add_argument('-batch', type=int, default=100, help="Batch size")
    parser.add_argument('-epoch_threshold', type=int, default=100, help="Epoch threshold (defines for how many epochs we resample most of batches)")
    parser.add_argument('-alpha', type=float, default=1.0, help="Scale score match loss")
    parser.add_argument('-beta', type=float, default=1.0, help="Scale Newton loss")
    parser.add_argument('-gamma', type=float, default=1.0, help="Scale IC loss")
    parser.add_argument('-lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('-dim_hid', type=int, default=500, help="Number of epochs")
    parser.add_argument('-scheduler', type=int, default=0, choices=[0, 1], help="Use lr scheduler?")
    parser.add_argument('-nn_architecture', type=int, default=5, help="nn_architecture")
    parser.add_argument('-sampling_scheme', type=str, default="random_grid",
                         choices=["fixed", "random_grid", "interpolated"], help="Sampling in T")
    parser.add_argument('-init_phase', type=float, default=0,
                         help="Initial phase S_0(x) = c coeff.")
    parser.add_argument('-save_freq', type=int, default=50, help="Save models every ... epochs")

    parser.add_argument('-seed', default=1234, type=int)
    parser.add_argument('-eval', default=False, action='store_true')
    parser.add_argument('-resume', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('-train_results_dir')
    parser.add_argument('-models_dir', type=str, default="", help="extra description")
    arguments = parser.parse_args()

    return arguments

def main(args: argparse.Namespace) -> None:
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Computing on device: {}".format(device))
    # print(args)
    T = args.T # time [0, T]
    N = args.N # split [0, T] into N parts
    h = 1e-2 #qmsolve.hbar
    print("h=", h)
    sampling_type = args.sampling_scheme
    M = args.M

    save_model_freq = args.save_freq
    
    m = 1.0
    omega2 = args.omega2
    omega = np.sqrt(omega2)
    g = args.g

    x0_phase = args.init_phase

    # sampling parameters for X_0
    # sig2 = np.sqrt(h/(2*m*omega))
    sig2 = np.sqrt(0.1) #np.sqrt(h/(2*m*omega))
    mu = 0.0
    d = args.d # dim or num particles (we have two 1d particles)

    time_splits = torch.Tensor(np.linspace(0, T, N+1))

    dim_inp = d + 1
    dim_out = d
    # resample_frac_1 = args.resample_factor_1
    # resample_frac_2 = args.resample_factor_2
    if d == 2:
        if args.models_dir == "":
            models_dir="dsm_harm_osc_interact_g={}_T={}_hdim={}_eps={}_S0={}_batch={}_lr={}_N={}_arch_{}_sample_{}".format(g, T, args.dim_hid, args.n_epochs,
                                                                                    x0_phase, args.batch, args.lr, args.N, args.nn_architecture, sampling_type)
        else:
            models_dir="dsm_harm_osc_interact_g={}_T={}_hdim={}_eps={}_S0={}_batch={}_lr={}_N={}_arch_{}_sample_{}_{}".format(g, T, args.dim_hid, args.n_epochs,
                                                                                    x0_phase, args.batch, args.lr, args.N, args.nn_architecture, sampling_type, args.models_dir)
    else:
        if args.models_dir == "":
            models_dir="dsm_harm_osc_interact_d={}_g={}_T={}_hdim={}_eps={}_S0={}_batch={}_lr={}_N={}_arch_{}_sample_{}".format(d, g, T, args.dim_hid, args.n_epochs,
                                                                                    x0_phase, args.batch, args.lr, args.N, args.nn_architecture, sampling_type)
        else:
            models_dir="dsm_harm_osc_interact__d={}_g={}_T={}_hdim={}_eps={}_S0={}_batch={}_lr={}_N={}_arch_{}_sample_{}_{}".format(d, g, T, args.dim_hid, args.n_epochs,
                                                                                    x0_phase, args.batch, args.lr, args.N, args.nn_architecture, sampling_type, args.models_dir)
    
    dim_hid = args.dim_hid
    # models_dir = models_dir + '_{}'.format(args.nn_architecture)
    if args.resume: 
        print("RESUME TRAIN")

    if args.nn_architecture == 0:
        print('regular fully-connected')
        net_u = NN_old(dim_inp, dim_hid, dim_out)
        net_v = NN_old(dim_inp, dim_hid, dim_out)
    elif args.nn_architecture == 1:
        print('skip invariant')
        net_u = NN_new(dim_inp, dim_hid, dim_out, h, m, omega, 1.0, 0.0)
        net_v = NN_new(dim_inp, dim_hid, dim_out, h, m, omega, 0.0, 1.0)
    elif args.nn_architecture == 2:
        print('normal invariant')
        net_u = NN_new_invariant(dim_inp, dim_hid, dim_out)
        net_v = NN_new_invariant(dim_inp, dim_hid, dim_out)
    elif args.nn_architecture == 3:
        print('fully connected invar')
        net_u = NN_connect(dim_inp, dim_hid, dim_out)
        net_v = NN_connect(dim_inp, dim_hid, dim_out)
    elif args.nn_architecture == 4:
        print('skip not invar')
        net_u = NN_connect_init(dim_inp, dim_hid, dim_out, h, m, omega, 1.0, 0.0)
        net_v = NN_connect_init(dim_inp, dim_hid, dim_out, h, m, omega, 0.0, 1.0)

    elif args.nn_architecture == 5:
        print('Non-interact NN')
        net_u = NN_non_interact(dim_inp, dim_hid, dim_out, h, m, omega, sig2, x0_phase,  1.0, 0.0)
        net_v = NN_non_interact(dim_inp, dim_hid, dim_out, h, m, omega, sig2, x0_phase, 0.0, 1.0)

    if args.multi_gpu:
        logger.info("Running on multiGPU {}".format(torch.cuda.device_count()))
        net_u = nn.DataParallel(net_u)
        net_v = nn.DataParallel(net_v)
    net_u.to(device)
    net_v.to(device)

    for x in [args.train_results_dir, os.path.join(args.train_results_dir, models_dir)]:
        if not os.path.isdir(x):
            # print(x)
            os.mkdir(x)

    path_to_save = os.path.join(args.train_results_dir,  models_dir)
    with open(os.path.join(path_to_save,'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    learning_rate = args.lr
    if sampling_type == "interpolated":
        # optimizer = torch.optim.AdamW([*net_u.parameters(), *net_v.parameters()], lr=learning_rate,
        #                         weight_decay=0, betas=(0.8, 0.99))
        optimizer = torch.optim.Adam([*net_u.parameters(), *net_v.parameters()], lr=learning_rate,
                            weight_decay=0)
    else:
        optimizer = torch.optim.Adam([*net_u.parameters(), *net_v.parameters()], lr=learning_rate,
                            weight_decay=0)
        
    if args.scheduler == 1:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs, eta_min=0)
    else:
        scheduler = None
        
    epochs_old = 0
    if args.eval:
        checkpoint = torch.load(os.path.join(path_to_save, "net_uv_{}_ep_interact_check.pt".format(args.n_epochs)))
        net_u.load_state_dict(checkpoint['model_u_state_dict'])
        net_v.load_state_dict(checkpoint['model_v_state_dict'])
    if args.eval or args.resume:
        checkpoint = torch.load(os.path.join(path_to_save, "net_uv_{}_ep_interact_check.pt".format(args.n_epochs)))
        net_u.load_state_dict(checkpoint['model_u_state_dict'])
        net_v.load_state_dict(checkpoint['model_v_state_dict'])
        logger.info('Previously trained model weights state_dict loaded')
    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info('Previously trained optimizer state_dict loaded')
        epochs_old = checkpoint['epoch'] + 1
        criterion = checkpoint['loss_criterion']
        logger.info('Trained model loss function loaded...')
        logger.info(f"Previously trained for {epochs_old} number of epochs...")
        n_iter = args.n_epochs
        logger.info(f"Train for {n_iter+1-epochs_old} more epochs...")
        optimizer.param_groups[0]['lr'] = learning_rate
    # print("args.eval = {}".format(args.eval))
    if not args.eval:
        criterion = nn.MSELoss()
        alpha = args.alpha # Score
        beta = args.beta # Newton
        gamma = args.gamma # initial
        n_iter = args.n_epochs
        logger.info('Start training...')
        net_u, net_v, losses, losses_newton, losses_sm, losses_init = train_dsm_non_interact(net_u, net_v, device, optimizer,
                                                                                     criterion, alpha, beta, gamma, N, time_splits,
                                                                                     path = path_to_save, save_freq=save_model_freq,
                                                                                     n_iter=n_iter, batch_size=args.batch,
                                                                                     omega2=omega, mu=mu, sig2=sig2,
                                                                                     m=m, h=h, T=T,
                                                                                     g = g, d=d,
                                                                                     scheduler=scheduler,
                                                                                     epoch_start = epochs_old,
                                                                                     sampling_scheme = sampling_type,
                                                                                     M = M, x0_phase=x0_phase
                                                                                     )
        logger.info('Done training...')
        make_loss_plot(losses, losses_sm, losses_newton, losses_init, path_to_save)

    logger.info('Start sampling after training...')
    X_test = sample_w_nn(net_u, net_v, N, d, time_splits, device, mu, sig2, T, h=h, nu_s=1)
    logger.info('Done sampling')
    num_trials = X_test.shape[0]
    mean_trials = np.zeros((num_trials, N + 1, d))
    var_trials = np.zeros((num_trials, N + 1, d))
    for trial in range(num_trials):
        mean_trials[trial] = X_test[trial].mean(axis=1)
        var_trials[trial] = X_test[trial].var(axis=1)

    # numerical solution
    spatial_dim = 1.5
    spatial_num = 100
    dx    = np.sqrt(T / 10000) # spatial separation
    x     = np.arange(-spatial_dim, spatial_dim, dx)       # spatial grid points
    # x = np.linspace(-spatial_dim, spatial_dim, spatial_num)
    # dx = 2 * spatial_dim / spatial_num
    lb = np.array([-spatial_dim, 0.0])
    ub = np.array([spatial_dim, T])
   
    logger.info('Start numerical solution...')
    sim_inter = numerical_sol(spatial_dim, spatial_num, s0_coeff=x0_phase, sig2=sig2, N=N, g=g, T=T, omega2=omega2, m=m , d=d, h=h)

    prob_density_inter = np.abs(sim_inter.y.T)**2 #np.abs(sim_inter.Ψ)**2 * dx
    density_truth_x1 = prob_density_inter.T # prob_density_inter.sum(axis=1).T

    sol_t = np.linspace(0, T, N+1)
    bmeans = []
    bstds = []
    ts = []

    for i, t in enumerate(sim_inter.t):
        ts.append(t)
        bmeans.append(np.dot(x, dx * np.abs(sim_inter.y[:,i])**2)) 
        bstds.append(np.dot((x - bmeans[-1]) ** 2, dx * np.abs(sim_inter.y[:,i])**2))
    
    
    for i in range(d):
        plot_stats(time_splits.numpy(), mean_trials[:, :, i], var_trials[:, :, i], ts, bmeans, bstds, d=i, 
                   path=path_to_save, save=True)
        
    n_bins = 100 #len(x)
    p = np.zeros((d, len(time_splits), n_bins))
    for j in range(d):
        for i in range(len(time_splits)):
            p[j, i, :] = np.histogram(X_test[:, i, :, j].reshape(-1),
                                density=True, bins=n_bins, range=(lb[0], ub[0]))[0]


    if d == 2:
        density_pred_x1 = p[0].T
        density_pred_x2 = p[1].T
        v_max_ = round(max(np.max(density_truth_x1.reshape(-1)), np.max(density_pred_x1.reshape(-1)), np.max(density_pred_x2.reshape(-1))) + 0.1, 1)
        make_density_plot(density_truth_x1,path_to_save, d,  low_bound=lb, up_bound=ub,
                        title='$|\Psi(x_{i}, t)|^2, i = 1, 2$ truth', vmin_max = [0, v_max_])
        make_density_plot(density_pred_x1, path_to_save, d, low_bound=lb, up_bound=ub,
                        title='$|\Psi(x_1, t)|^2$ prediction', vmin_max = [0, v_max_])
        make_density_plot(density_pred_x2, path_to_save, d, low_bound=lb, up_bound=ub,
                        title='$|\Psi(x_2, t)|^2$ prediction', vmin_max = [0, v_max_])
    else: 
        for i in range(d):
            density_pred = p[i].T
            v_max_ = np.max(density_pred.reshape(-1))
            make_density_plot(density_pred, path_to_save, i, low_bound=lb, up_bound=ub,
                            #   title='$|\Psi(x, t)|^2$ prediction, i = {}'.format(i+1), vmin_max = [0, v_max_])
                            title='$|\Psi(x^i, t)|^2$ prediction, i = {}'.format(i+1), vmin_max = [0, v_max_])
        make_density_plot(density_truth_x1, path_to_save, d, low_bound=lb, up_bound=ub,
                        title='$|\Psi(x^i, t)|^2$ truth, i = 1,..,{}'.format(d), vmin_max = [0, v_max_])

    logger.info("Finished")


if __name__ == "__main__":


    args = setup_args()

    FMT = "%(asctime)s:cluster_startup: %(levelname)s - %(message)s"
    TIMEFMT = '%Y-%m-%d %H:%M:%S'

    # if args.debug:
    #     level = logging.DEBUG
    # else:
    #     level = logging.INFO

    # logging.basicConfig(level=level,
    #                 format=FMT,
    #                 datefmt=TIMEFMT)
    main(args)