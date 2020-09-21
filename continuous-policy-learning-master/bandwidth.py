'''
Helper functions for (noisy) bandwidth estimation:
'''


def build_linear_model(**samp_params):
    '''
    Fit a linear response model for use in estimation of bandwidth
    Test code for testing linear model of response
    # test_val = np.random.uniform()
    # samp_params['tau'] = test_val * np.ones([n,1])
    # test_data = np.concatenate( [samp_params['x'], samp_params['tau']], axis = 1 )
    # pred = regr.predict(test_data)
    pred_params = {'z' : test_data, 'y' : pred }
    plot_surface(**pred_params)
    plot_surface(**sub_params)
    '''
    n = samp_params['n']
    regr = linear_model.LinearRegression()
    samp_params['z_samp'] = np.concatenate([samp_params['x_samp'], samp_params['T_samp']], axis=1)
    regr.fit(samp_params['z_samp'], samp_params['y_samp'])
    return regr


def scores_cond_f_y_given_tau_x(joint_f_t_x, joint_f_y_t_x, test_point):
    """
    Use the estimates of joint density of F_{T,X} and F_{Y,T,X} to estimate
    the conditional density F_{Y|T,X} at the given test point
    Test point: [y, t, x]
    """
    tp = test_point[1:]
    joint_f_tau_x = joint_f_t_x.score_samples(tp.reshape([1, 2]))
    joint_f_y_tau_x = joint_f_y_t_x.score_samples(test_point.reshape([1, 3]))
    return np.exp(joint_f_y_tau_x - joint_f_tau_x)


# def scores_cond_f_y_given_tau_x_caller(test_point):
#     #FIXME: will look in global scope
#     return scores_cond_f_y_given_tau_x(joint_f_t_x, joint_f_y_t_x, test_point)

def bias_integrand(y, tau, x, hessian):
    x0 = np.asarray([y, tau, x])
    return y ** 2 * hessian([y, tau, x])[1][1] * 0.5


def empirical_exp_second_moment(regr, **params):
    x = params['x']
    tau = params['tau']
    y = params['y_samp']
    T = params['T']

    y_pred = regr.predict(np.concatenate([params['x_samp'], params['tau']], axis=1))
    Q = params['Q']
    Q_vec = np.asarray([Q(x[i], T[i], params['t_lo'], params['t_hi']) for i in range(params['n'])])
    return np.square(y_pred) / Q_vec


def est_h(h_sub, regr, hess, **samp_params):
    R_K = 1.0 / (2 * np.sqrt(np.pi))
    kappa_two = 1.0
    C = R_K / (4.0 * samp_params['n'] * kappa_two ** 2)
    exp_second_moment = np.mean(empirical_exp_second_moment(regr, **samp_params))
    # Assume that tau doesn't change for x_i for now
    bias = 0
    ymin = min(samp_params['y_samp'])
    ymax = max(samp_params['y_samp'])

    for i in range(h_sub):
        print
        i
        bias += \
        integrate.quad(lambda u: bias_integrand(u, samp_params['tau'][i], samp_params['x_samp'][i], hess), ymin, ymax)[
            0]
    mean_bias_sqd = (bias / h_sub) ** 2
    h = np.power(C * exp_second_moment / (mean_bias_sqd * samp_params['n']), 0.2)

    print
    "opt h for this treatment vector: " + str(h)
    return h

    ''' variant of OPE with known propensities
    '''
    ## given Known propensities


def off_policy_evaluation_known_Q(**params):
    """
    Takes in a choice of kernel and dictionary of parameters and data required for evaluation
    tau is a vector of treatment values (assumed given)
    If y_samp, T_samp is present, use that instead.
    """
    [loss, norm_sum] = off_pol_estimator_known_Q(**params)
    h = params['h']
    n = params['n']
    return loss / (norm_sum * 1.0 * h * n)


def off_pol_estimator_known_Q(**params):
    THRESH = params['threshold']
    y_out = params['y'];
    x = params['x'];
    h = params['h'];
    n = params['n'];
    t_lo = params['t_lo'];
    t_hi = params['t_hi']
    kernel = params['kernel_func'];
    kernel_int = params['kernel_int_func']
    Q = params['Q_known'];
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
        Q_i = Q[i]
        if (abs(clip_tau[i] - t_lo) <= h):
            alpha = kernel_int((t_lo - clip_tau[i]) / h, 1)
        elif (abs(clip_tau[i] - t_hi) <= h):
            alpha = kernel_int(-1, (t_hi - clip_tau[i]) / h)
        else:
            alpha = 1
        Qs[i] = (1.0 / h) * kernel((clip_tau[i] - T[i]) / h) / max(Q_i, THRESH)
        loss += kernel((clip_tau[i] - T[i]) / h) * 1.0 * y_out[i] / max(Q_i, THRESH) * 1.0 / alpha

    #         if kernel( (clip_tau[i] - T[i])/h )>0.5:
    #             print y_out[i]
    #             print 'propensity: ' + str(Q_i)
    norm_sum = np.mean(np.maximum(Qs, THRESH * np.ones(n)))
    return [loss, norm_sum]


def bandwidth_selection(n_samp, h_sub, **params):
    '''
    Top-level function for estimating bandwidth. Note that this scales incredibly poorly with the size of the sampled dataset.
    '''

    def scores_cond_f_y_given_tau_x_caller(test_point):
        return scores_cond_f_y_given_tau_x(joint_f_t_x, joint_f_y_t_x, test_point)

    n = params['n']

    samp_params = evaluate_subsample(n_samp, cross_val=False, evaluation=False, **params)
    regr = build_linear_model(**samp_params)

    samp_params['tau'] = 0.5 * np.ones([samp_params['n'], 1])

    samp_params['z_samp'] = np.concatenate([samp_params['x_samp'], samp_params['T_samp']], axis=1)
    bandwidths = {'bandwidth': np.logspace(-1, 1, 20)}
    grid = GridSearchCV(KernelDensity(), bandwidths)
    grid.fit(samp_params['z_samp'])

    bandwidth_est = grid.best_estimator_.bandwidth
    joint_f_t_x = KernelDensity(kernel='gaussian', bandwidth=bandwidth_est).fit(samp_params['z_samp'])
    joint_f_y_t_x = KernelDensity(kernel='gaussian', bandwidth=bandwidth_est).fit(
        np.concatenate([samp_params['y_samp'], samp_params['z_samp']], axis=1))

    cond_dens_hess = nd.Hessian(scores_cond_f_y_given_tau_x_caller)
    h = est_h(h_sub, regr, cond_dens_hess, **samp_params)
    return h