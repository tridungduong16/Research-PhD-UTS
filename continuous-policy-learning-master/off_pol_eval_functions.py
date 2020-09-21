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



# !FIXME Global offset value.

# !FIXME
# Currently when changing data generation distributions, need also to change sampling method in evaluate_subsample
# to generate from the appropriate treatment distribution.
'''
Choices for output function.
'''
def oracle_evaluation(**params):
    X = params['x_samp']; tau = params['tau']
    return 2*pow(np.abs(X - tau),1.5)



'''
    Systematically evaluate over a treatment space defined by a linear treatment policy

With DM 
'''
def off_pol_eval_linear_test( n_max, beta_0, beta_hi, n_trials, n_treatments, n_spacing, n_0, **sub_params):
    '''
    '''
    treatment_space = np.linspace(beta_0, beta_hi, n_treatments)
    off_pol_evals = np.zeros([n_treatments, n_spacing, n_trials])
    oracle_evals = np.zeros([n_treatments, n_spacing, n_trials])
    discrete_off_pol_evals = np.zeros([n_treatments, n_spacing, n_trials])
    t_lo = sub_params['t_lo']; t_hi = sub_params['t_hi']; spl_x = sub_params['z'][:,0]; spl_t = sub_params['z'][:,1]
    # f is positive
    splined_f_tck = interpolate.bisplrep(spl_x,spl_t, sub_params['f'])
    sub_params['spline'] = splined_f_tck
    oracle_func = sub_params['oracle_func']
    n = sub_params['n']; m = sub_params['m']

    for i, n_sub in enumerate(np.linspace(n_0, n_max, n_spacing)): 
        n_rnd = int(np.floor(n_sub))
        print "testing n = " + str(n_rnd)
        for k in np.arange(n_trials):
            for beta_ind, beta in enumerate(treatment_space):
                subsamples_pm = evaluate_subsample( n_rnd, evaluation = False, cross_val = False, **sub_params )
                tau = np.clip(np.dot( subsamples_pm['x_samp'], beta ) , t_lo, t_hi)
                subsamples_pm['tau'] = tau
                oracle_evals[beta_ind, i, k] = np.mean(oracle_func(**subsamples_pm))
                # oracle_evals[beta_ind, i, k] = np.mean(evaluate_oracle_interpolated_outcomes(splined_f_tck,m,n_rnd, subsamples_pm['f'], beta_0, beta_hi, tau, subsamples_pm['x_samp']))
                # off_pol_evals[beta_ind, i, k] = off_policy_evaluation(**subsamples_pm)
                off_pol_evals[beta_ind, i, k] = off_policy_evaluation(**subsamples_pm)
                discrete_off_pol_evals[beta_ind, i, k] = off_pol_disc_evaluation(discretize_tau_policy , **subsamples_pm)

    off_pol_evals.dump( str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")) + 'off_pol_linear_vals.np')
    oracle_evals.dump(str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")) + 'off_pol_linear_oracles.np')
    return [oracle_evals, off_pol_evals, discrete_off_pol_evals]

def plot_off_pol_evals(off_pol_evals, oracle_evals, off_pol_disc_evals, n_0, n, n_trials, n_treatments, n_spacing, t_lo, t_hi, x_label, title_stem, truncate_y = False):
    mean_off_pol_vals = np.mean(off_pol_evals, axis = 2)
    mean_oracle_vals = np.mean(oracle_evals,axis=2)
    sds_off_pol = np.std(off_pol_evals, axis = 2)
    sds_oracle = np.std(oracle_evals, axis = 2)
    mean_off_pol_disc_evals = np.mean(off_pol_disc_evals,axis=2)
    sds_off_pol_disc = np.std(off_pol_disc_evals, axis = 2)

    ts = np.linspace(t_lo, t_hi, n_treatments)

    ns = np.linspace(n_0, n, n_spacing)
    for i in np.arange(n_spacing):
        plt.figure(i+1)
        error_1 = 1.96*sds_off_pol[:,i]/np.sqrt(n_trials)
        error_2 = 1.96*sds_oracle[:,i]/np.sqrt(n_trials)
        error_3 = 1.96*sds_off_pol_disc[:,i]/np.sqrt(n_trials)

        plt.plot(ts, mean_oracle_vals[:,i], c = "blue")
        plt.fill_between(ts, mean_oracle_vals[:,i]-error_2, mean_oracle_vals[:,i]+error_2, alpha=0.5, edgecolor='blue', facecolor='blue')

        plt.scatter(ts, mean_off_pol_disc_evals[:,i], c = "green")
        plt.fill_between(ts, mean_off_pol_disc_evals[:,i]-error_3, mean_off_pol_disc_evals[:,i]+error_3, alpha=0.4, edgecolor='g', facecolor='g')
        plt.scatter(ts, mean_off_pol_vals[:,i], c = "red")
        plt.fill_between(ts, mean_off_pol_vals[:,i]-error_1, mean_off_pol_vals[:,i]+error_1, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')

    #     plt.ylim( (0, 10) )
        plt.title(title_stem+ " with n = " + str(ns[i]))
        plt.ylabel("outcome Y")
        plt.xlabel(x_label)
        if truncate_y: 
            plt.ylim((0,truncate_y))
        plt.show()

