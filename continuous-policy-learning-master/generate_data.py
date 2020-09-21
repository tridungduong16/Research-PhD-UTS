import scipy.integrate as integrate
from math import exp
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.optimize import minimize
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors.kde import KernelDensity
from sklearn.metrics.pairwise import rbf_kernel
import datetime
import pickle
import sys
# For bandwidth estimation
from scipy.stats import norm 
from sklearn import linear_model
# import numdifftools as nd
from scipy.misc import derivative
from sklearn.neighbors.kde import KernelDensity
from scipy.stats import truncnorm

'''
Different options for generating data
'''


def generate_data_uniform(m, n, d, t_lo, t_hi, x_scheme='unif'):
    """
    # Generate random features
    # n: number of instances
    # m: grid length of treatment
    # d: feature dimension
    # x_scheme: switch to determine dependency structure of x
    """
    xs = np.array(np.random.uniform(0, 2, (n, d)))
    t_fullgrid = np.linspace(t_lo, t_hi, m)
    Z_list = [np.concatenate([xs, np.ones([n, 1]) * (t_lo + 1.0 * i * (t_hi - t_lo) / (m - 1))], axis=1) for i in
              np.arange(m)]
    Z = np.concatenate(Z_list, axis=0)
    K = np.array(gram_matrix(Z)).reshape([m * n, m * n])
    T = Z[:, d]
    # mean_vec = np.asarray([ np.mean(z) for z in Z])
    mean_vec = np.ones([m * n, 1])
    F = np.random.multivariate_normal(mean_vec.flatten(), 7 * K)
    # Ensure outcomes are positive
    if min(F) < 0:
        F = F + abs(min(F))
    Y = F + 0.05 * np.random.randn(m * n)

    return {'y': Y, 'z': Z, 'f': F, 'K': K, 'x': xs}


def generate_data(m, n, d, t_lo, t_hi, mean_vec_f, x_scheme='unif'):
    """
    # Generate random features
    # n: number of instances
    # m: grid length of treatment
    # d: feature dimension
    # x_scheme: switch to determine dependency structure of x
    """
    xs = np.array(np.random.uniform(0, 1, (n, d)))
    t = np.array(np.random.uniform(0, t_hi, size=(n, 1)))
    # change mean vector appropriately
    t_fullgrid = np.linspace(t_lo, t_hi, m)
    Z_list = [np.concatenate((xs, np.ones([n, 1]) * (t_lo + 1.0 * i * (t_hi - t_lo) / (m - 1))), axis=1) for i in
              np.arange(m)]
    Z = np.concatenate(Z_list, axis=0)
    K = np.array(gram_matrix(Z)).reshape([m * n, m * n])
    T = Z[:, d]
    # modify to have T have more of an effect
    mean_vec = np.apply_along_axis(mean_vec_f, 1, Z)
    # mean_vec = 3*np.multiply(T,Z[:,0]) + 2*T + np.multiply(Z[:,0], np.exp(np.multiply(-Z[:,0],T)))
    F = np.random.multivariate_normal(mean_vec, 2 * K)
    # Ensure outcomes are positive
    if min(F) < 0:
        F = F + abs(min(F))
    Y = F + 0.05 * np.random.randn(m * n)

    return {'y': Y, 'z': Z, 'f': F, 'K': K, 'x': xs}


def off_pol_estimator(**params):
    THRESH = params['threshold']
    y_out = params['y'];
    x = params['x'];
    h = params['h'];
    Q = params['Q'];
    n = params['n'];
    t_lo = params['t_lo'];
    t_hi = params['t_hi']
    kernel = params['kernel_func'];
    kernel_int = params['kernel_int_func']
    if ('y_samp' in params.keys()):
        y_out = params['y_samp']
    if ('T_samp' in params.keys()):
        T = params['T_samp']
    else:
        T = params['T']
    if ('x_samp' in params.keys()):
        x = params['x_samp']

    BMI_IND = params.get('BMI_IND')  # propensity score for warfarin data evaluations
    if (params.get('DATA_TYPE') == 'warfarin'):
        x = params['x'][:, BMI_IND]

    loss = 0
    tau = params['tau']
    clip_tau = np.clip(tau, t_lo, t_hi)
    Qs = np.zeros(n)
    for i in np.arange(n):
        Q_i = Q(x[i], T[i], t_lo, t_hi)
        if (abs(clip_tau[i] - t_lo) <= h):
            alpha = kernel_int((t_lo - clip_tau[i]) / h, 1)
        elif (abs(clip_tau[i] - t_hi) <= h):
            alpha = kernel_int(-1, (t_hi - clip_tau[i]) / h)
        else:
            alpha = 1
        Qs[i] = (1.0 / h) * kernel((clip_tau[i] - T[i]) / h) / max(Q_i, THRESH)
        loss += kernel((clip_tau[i] - T[i]) / h) * 1.0 * y_out[i] / max(Q_i, THRESH) * 1.0 / alpha
    norm_sum = np.mean(np.maximum(Qs, THRESH * np.ones(n)))
    return [loss, norm_sum]


def off_policy_variance(**params):
    """
    Takes in a choice of kernel and dictionary of parameters and data required for evaluation
    tau is a vector of treatment values (assumed given)
    If y_samp, T_samp is present, use that instead.
    """
    [loss, norm_sum] = off_pol_estimator(**params)
    h = params['h'];
    n = params['n']
    loss = loss / (norm_sum * 1.0 * n * h)
    loss_mean = np.mean(loss)
    return np.square(loss - loss_mean)


def off_policy_evaluation(**params):
    """
    Takes in a choice of kernel and dictionary of parameters and data required for evaluation
    tau is a vector of treatment values (assumed given)
    If y_samp, T_samp is present, use that instead.
    """
    [loss, norm_sum] = off_pol_estimator(**params)
    h = params['h']
    n = params['n']
    return loss / (norm_sum * 1.0 * h * n)


def off_pol_disc_evaluation(policy, **params):
    THRESH = params['threshold']
    y_out = params['y'];
    x = params['x_samp'];
    h = params['h'];
    Q = params['Q'];
    n = params['n'];
    t_lo = params['t_lo'];
    t_hi = params['t_hi']
    n_bins = params['n_bins']
    if ('y_samp' in params.keys()):
        y_out = params['y_samp'].flatten()
    if ('T_samp' in params.keys()):
        T = params['T_samp'].flatten()
    else:
        T = params['T'].flatten()

    BMI_IND = params.get('BMI_IND')  # propensity score for warfarin data evaluations
    if (params.get('DATA_TYPE') == 'warfarin'):
        x = params['x'][:, BMI_IND]

    t_lo = min(T)
    t_hi = max(T)
    bin_width = t_hi - t_lo
    bins = np.linspace(t_lo, t_hi, n_bins)
    T_binned = np.digitize(T, bins, right=True).flatten()
    bin_means = [T[T_binned == i].mean() for i in range(1, len(bins))]

    loss = 0
    tau_vec = policy(**params).flatten()
    # ! FIXME need to establish whether policy returns discrete bins or means
    treatment_overlap = np.where(np.equal(tau_vec.flatten(), T_binned))[0]

    for ind in treatment_overlap:
        Q_i = Q(x[ind], bin_means[T_binned[ind] - 1], t_lo,
                t_hi) * bin_width * 1.0 / n_bins  # BUG FIX: this is going to have to be integrated against
        loss += y_out[ind] / max(Q_i, THRESH)
    n_overlap = len(treatment_overlap)
    if n_overlap == 0:
        print
        "no overlap"
        return 0
    return loss / (1.0 * n)


def off_pol_gaus_lin_grad(beta, *args):
    """
    Compute a gradient for special case of gaussian kernel and linear policy tau
    """
    params = dict(args[0])
    y_out = params['y'];
    x = params['x'];
    T = params['T'];
    h = params['h'];
    Q = params['Q']
    n = params['n'];
    t_lo = params['t_lo'];
    t_hi = params['t_hi']
    tau = np.dot(x, beta)
    clip_tau = np.clip(tau, t_lo, t_hi)
    d = len(beta)
    grad = np.zeros([d, 1])
    for i in np.arange(n):
        Q_i = Q(x[i], T[i], t_lo, t_hi)
        beta_x_i = np.dot(x[i], beta)
        grad += (gaussian_kernel((beta_x_i - T[i]) / h) * y_out[i] / Q_i) * (-1.0 * x[i] / h ** 2) * (beta_x_i - T[i])
    return grad / (1.0 * h * len(y_out))


def partial_g_n_hat_i(**params):
    '''
    Compute normalization term
    '''


def f_g(**params):
    THRESH = params['threshold']
    y_out = params['y'];
    x = params['x'];
    h = params['h'];
    Q = params['Q'];
    n = params['n'];
    t_lo = params['t_lo'];
    t_hi = params['t_hi']
    kernel = params['kernel_func'];
    kernel_int = params['kernel_int_func']
    if ('y_samp' in params.keys()):
        y_out = params['y_samp']
    if ('T_samp' in params.keys()):
        T = params['T_samp']
    else:
        T = params['T']
    if ('x_samp' in params.keys()):
        x = params['x_samp']
    BMI_IND = params.get('BMI_IND')  # propensity score for warfarin data evaluations

    loss = 0
    g = 0  # also keep track of normalized probability ratio quantity
    partial_f = 0
    partial_g = 0
    tau = params['tau']
    clip_tau = np.clip(tau, t_lo, t_hi)
    Qs = np.zeros(n)
    for i in np.arange(n):
        if (params.get('DATA_TYPE') == 'warfarin'):
            Q_i = Q(x[i, BMI_IND], T[i], t_lo, t_hi)
        else:
            Q_i = Q(x[i], T[i], t_lo, t_hi)
        if (abs(clip_tau[i] - t_lo) <= h):
            alpha = kernel_int((t_lo - clip_tau[i]) / h, 1)
        elif (abs(clip_tau[i] - t_hi) <= h):
            alpha = kernel_int(-1, (t_hi - clip_tau[i]) / h)
        else:
            alpha = 1
        Qs[i] = kernel((clip_tau[i] - T[i]) / h) / max(Q_i, THRESH)
        loss += kernel((clip_tau[i] - T[i]) / h) * 1.0 * y_out[i] / max(Q_i, THRESH) * 1.0 / alpha
        if abs((clip_tau[i] - T[i]) / h) >= 1:
            partial_f += 0  # don't add anything to partial derivatives
        else:
            partial_g += -1.5 * ((clip_tau[i] - T[i]) / h) * 1.0 / max(Q_i, THRESH) * x[i, :]
            partial_f += -1.5 * ((clip_tau[i] - T[i]) / h) * y_out[i] / max(Q_i, THRESH) * x[i, :]
    norm_sum = np.mean(Qs)
    return [loss / (1.0 * h * n), 1.0 * norm_sum / h, partial_f / (1.0 * n * h ** 2), partial_g / (1.0 * n * h ** 2)]


def off_pol_epan_lin_grad(beta, *args):
    """
    Compute a gradient for special case of Epanechnikov kernel and linear policy tau
    """
    # THRESH = 0.001
    d = len(beta)
    params = dict(args[0])
    # ! FIXME x vs xsamp
    tau = np.dot(beta, params['x'].T)
    params['tau'] = tau
    params['beta'] = beta

    THRESH = params['threshold']

    [f, g, nabla_f, nabla_g] = f_g(**params)
    # compute gradient vector via quotient rule
    if g < THRESH:
        g = THRESH
    return np.asarray((g * nabla_f - f * nabla_g) / g ** 2)


def off_pol_var_lin_grad(beta, *args):
    """
    Compute a gradient for special case of Epanechnikov kernel and linear policy tau
    """
    # THRESH = 0.001
    d = len(beta)
    params = dict(args[0])
    # ! FIXME x vs xsamp
    tau = np.dot(beta, params['x'].T)
    params['tau'] = tau
    params['beta'] = beta

    THRESH = params['threshold']

    [f, g, nabla_f, nabla_g] = f_g(**params)
    # compute gradient vector via quotient rule
    if g < THRESH:
        g = THRESH
    return np.asarray((g * nabla_f - f * nabla_g) / g ** 2)


def off_pol_gaus_lin_grad_for_max(beta, *args):
    """Wrapper function which multiplies gradient by -1
    """
    return off_pol_gaus_lin_grad(beta, *args)
