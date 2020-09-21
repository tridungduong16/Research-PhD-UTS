'''
Different option for discrete policy functions
Policy functions take in an x vector and return
'''
def discrete_optimal_central_policy(**params):
    '''
    :param params:
    :return: optimal treatment vector
    '''
    x = params['x_samp']
    T = params['T_samp']
    t_lo = min(T)
    t_hi = max(T)
    n_bins = params['n_bins']
    bins = np.linspace(t_lo, t_hi, n_bins)
    T_binned = np.digitize(T, bins).flatten()
    x_binned = np.digitize(x/2.0, bins).flatten()
    bin_means = [T[T_binned == i].mean() for i in range(1, n_bins)]
    # return np.asarray([bin_means[T_bin - 1] for T_bin in x_binned]).flatten()
    return x_binned

def discretize_tau_policy(**params):
    '''
    Discretize the treatment vector 'tau' according to uniform binning.
    '''
    x = params['x_samp']
    T = params['T_samp']
    n_bins = params['n_bins']
    t_lo = min(T)
    t_hi = max(T)
    bins = np.linspace(t_lo, t_hi, n_bins)
    T_binned = np.digitize(T, bins).flatten()
    bin_means = [T[T_binned == i].mean() for i in range(1, n_bins)]
    tau_binned = np.digitize(params['tau'], bins).flatten()
    return tau_binned