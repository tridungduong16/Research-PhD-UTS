"""
Options for treatment policies
"""


def tau_test(tau_test_value, x):
    return tau_test_value


def linear_tau(x, beta):
    return np.dot(beta, x)


def unif_Q(x, t, t_lo, t_hi):
    return 1.0 / (t_hi - t_lo)


def trunc_norm_Q(x, t, t_lo, t_hi):
    # Get pdf from  truncated normally distributed propensity score (standard normal centered around (x-t)
    sc = 0.5
    mu = x
    a, b = (t_lo - mu) / sc, (t_hi - mu) / sc
    return truncnorm.pdf(t, a, b, loc=mu, scale=sc)


def norm_Q(x, t, t_lo, t_hi):
    OFFSET = 0.1
    std = 0.5
    return 1.0 / std * norm.pdf((t - x - OFFSET) / std)


def exp_Q(x, t, t_lo, t_hi):
    # Sample from an exponential conditional distribution of T on X using Inverse CDF transform
    return x * np.exp(-t * x)


def sample_exp_T(x):
    u = np.random.uniform()
    return -np.log(1 - u) / x


def sample_norm_T(x):
    # ' Sample randomly from uniform normal distribution'
    sc = 0.5
    OFFSET = 0.1
    return np.random.normal(loc=x + OFFSET, scale=sc)


def evaluate_oracle_outcomes(m, n, f, t_lo, t_hi, tau, X):
    """
    Evaluate 'true' outcomes at closest grid point to given tau vector
    """
    j_taus = np.array([int(np.round(1.0 * t * (m - 1) / t_hi)) for t in tau])
    j_taus = np.clip(j_taus, 0, m - 1)
    return np.array([f[j_taus[ind] * n + ind] for ind in np.arange(n)])


def evaluate_oracle_interpolated_outcomes(**params):
    """
    Function is given a spline curve with which to interpolate values at 'tau'
    """
    spline_tck = params['spline'];
    tau = params['tau'];
    X = params['x_samp']
    outcomes = [interpolate.bisplev(X[i], tau[i], spline_tck) for i in np.arange(len(X))]
    return np.array(outcomes)


def sample_T_given_x(x, t_lo, t_hi, sampling="uniform"):
    # Sample from propensity score
    # e.g. exponential distribution
    sc = 0.5
    if (sampling == "exp"):
        sample_exp_T_vec = np.vectorize(sample_exp_T)
        T_sub = sample_exp_T_vec(x / std)
        T_sub = np.clip(T_sub, t_lo, t_hi)
    elif (sampling == "normal"):
        # Unbounded normal sampling
        sample_norm_T_vec = np.vectorize(sample_norm_T)
        T_sub = sample_norm_T_vec(x)
    elif (sampling == "truncated_normal"):
        # Unbounded normal sampling
        # sample_norm_T_vec = np.vectorize(sample_norm_T)
        # T_sub = sample_norm_T_vec(x )
        T_sub = np.zeros([len(x), 1])
        for i in np.arange(len(x)):
            a = (t_lo - x[i]) / sc
            b = (t_hi - x[i]) / sc
            T_sub[i] = truncnorm.rvs(a, b, loc=x[i], scale=sc, size=1)[0]
    else:
        T_sub = np.array([np.random.uniform(low=t_lo, high=t_hi) for x_samp in x])
    return T_sub


def evaluate_subsample(n_sub, verbose=False, evaluation=False, cross_val=True, **param_dict):
    """
    Evaluate off policy evaluation given a subsample of data from full
    Or just subsample data and return subsampled_dictionary
    """
    Z = param_dict['z'];
    X = param_dict['x'];
    t_lo = param_dict['t_lo'];
    t_hi = param_dict['t_hi'];
    m = param_dict['m']
    n = param_dict['n'];
    Y = param_dict['y'];
    d = param_dict['d'];
    f = param_dict['f'];
    data_gen = param_dict['data_gen']
    sampling = param_dict['sampling'];
    sub_params = param_dict.copy()
    # Subsample data
    if (data_gen == "grid"):
        X_sub = np.random.choice(n - 1, n_sub)
        T_sub = sample_T_given_x(X[X_sub], t_lo, t_hi, sampling)
        # Round T to grid values
        j_s = np.array([int(np.round(1.0 * t * (m - 1) / t_hi)) for t in T_sub]).flatten()
        T_grid = np.array([t_lo + 1.0 * np.round(1.0 * t * (m - 1) / t_hi) * (t_hi - t_lo) / (m - 1) for t in T_sub])
        Y_sub = np.array([Y[j_s[ind] * n + x] for (ind, x) in enumerate(X_sub)])
        sub_params['n'] = n_sub
        sub_params['y_samp'] = Y_sub.flatten()
        # ! FIXME flattening possibly multidimensional data
        sub_params['x_samp'] = X[X_sub, :]
        sub_params['T_samp'] = T_grid.flatten()

    else:
        # Uniform sampling
        X_sub = np.random.choice(m * n - 1, n_sub)
        sub_params['n'] = n_sub
        sub_params['x_samp'] = X[X_sub, :].reshape([n_sub, 1])
        # Toggle how sampling is drawn
        if sampling != "uniform":
            sub_params['T_samp'] = sample_T_given_x(X[X_sub, :], t_lo, t_hi, sampling).reshape([n_sub, 1])
        else:  # assume uniform otherwise
            sub_params['T_samp'] = Z[:, d][X_sub].reshape([n_sub, 1])
        # Toggle how oracle values are drawn
        if (sub_params['oracle_func']):
            # temporary setting of tau to
            sub_params['tau'] = sub_params['T_samp']
            # adding noise to 'y' values
            sub_params['y_samp'] = oracle_evaluation(**sub_params)  # + np.random.randn(n_sub,1)*0.05
            sub_params['f_samp'] = oracle_evaluation(**sub_params)
            del sub_params['tau']
        else:  # Oracle fnc parameter not set
            sub_params['y_samp'] = Y[X_sub].reshape([n_sub, 1])
            sub_params['f_samp'] = f[X_sub].reshape([n_sub, 1])

    if 'tau' in param_dict.keys():
        sub_params['tau'] = param_dict['tau'][X_sub]
    else:
        if verbose:
            print
            "No taus given"
    if cross_val:
        h_opt = find_best_h(cv_func, res, **sub_params)
        sub_params['h'] = h_opt

    return sub_params


def plot_surface(plot_sample=False, **params):
    fig = plt.figure(figsize=plt.figaspect(.2))
    ax = fig.add_subplot(1, 3, 1, projection='3d')

    if not plot_sample:
        x = params['z'][:, 0]
        t = params['z'][:, 1]
        y = params['y']
    else:
        x = params['x_samp']
        t = params['T_samp']
        y = params['y_samp']

    ax.scatter(x, t, y, s=0.06)
    ax.set_xlabel('x Label')
    ax.set_ylabel('t Label')
    ax.set_zlabel('y Label')
    ax = fig.add_subplot(1, 3, 2, projection='3d')
    ax.scatter(x, t, y, s=0.06)
    # Add best beta vector
    # ax1.scatter(x[40:],y[40:], s=10, c='r', marker="o", label='second')
    ax.azim = 240
    ax.elev = 20
    ax.set_xlabel('x ')
    ax.set_ylabel('t ')
    ax.set_zlabel('y ')
    plt.show()


def lin_off_policy_loss_evaluation(beta, *args):
    arg_dict = dict(args[0])
    t_lo = arg_dict['t_lo']
    t_hi = arg_dict['t_hi']
    x = arg_dict['x_samp']
    arg_dict['tau'] = np.clip(np.dot(x, beta), t_lo, t_hi)
    return off_policy_evaluation(**arg_dict)


def constant_off_policy_loss_evaluation(const, *args):
    arg_dict = dict(args[0])
    x = arg_dict['x_samp']
    arg_dict['tau'] = const * np.ones(arg_dict['n'])
    return off_policy_evaluation(**arg_dict)


def eval_interpolated_oracle_tau(beta, *args):
    params = dict(args[0])
    t_lo = params['t_lo']
    t_hi = params['t_hi']
    spline_tck = params['spline']
    tau_candidate = np.clip(np.dot(beta, params['x_samp'].T), t_lo, t_hi)
    params['tau'] = tau_candidate
    return np.mean(evaluate_oracle_interpolated_outcomes(**params))


def eval_const_interpolated_oracle_tau(const, *args):
    params = dict(args[0])
    t_lo = params['t_lo']
    t_hi = params['t_hi']
    spline_tck = params['spline']
    tau_candidate = const * np.ones(params['n'])
    params['tau'] = tau_candidate
    return np.mean(evaluate_oracle_interpolated_outcomes(**params))


def eval_oracle_tau(beta, *args):
    params = dict(args[0])
    t_lo = params['t_lo']
    t_hi = params['t_hi']
    tau_candidate = np.clip(np.dot(beta, params['x'].T), t_lo, t_hi)
    # !FIXME graceful handling of loss function of y_i
    params['tau'] = tau_candidate
    return np.mean(evaluate_oracle_interpolated_outcomes(**params))


def eval_oracle_tau_evaluation(beta, *args):
    params = dict(args[0])
    t_lo = params['t_lo']
    t_hi = params['t_hi']
    tau_candidate = np.clip(np.dot(beta, params['x'].T), t_lo, t_hi)
    # !FIXME graceful handling of loss function of y_i
    params['tau'] = tau_candidate
    return np.mean(oracle_evaluation(**params))


def pol_opt(verbose=True, samp_func=lin_off_policy_loss_evaluation, oracle_eval=eval_interpolated_oracle_tau,
            **samp_params):
    """
    Run a policy optimization test, comparing performance of empirical minimizer against the true counterfactual outcomes.
    """
    d = samp_params['d']
    n = samp_params['n']
    t_lo = samp_params['t_lo']
    t_hi = samp_params['t_hi']
    beta_d = [np.random.uniform() for i in np.arange(d)]
    if samp_params['kernel_func'] == gaussian_kernel:
        res = minimize(samp_func, x0=beta_d, jac=off_pol_gaus_lin_grad_for_max,
                       bounds=((0, t_hi / max(samp_params['x'])),), args=samp_params.items())
    else:
        res = minimize(samp_func, x0=beta_d, jac=off_pol_epan_lin_grad,
                       bounds=((t_lo / max(samp_params['x_samp']), t_hi / max(samp_params['x_samp'])),),
                       args=samp_params.items())
    emp_best_tau = np.clip(np.dot(res.x, samp_params['x'].T), t_lo, t_hi)
    if verbose:
        print
        "Optimization results"
        print
        res
        print
        "Policy treatments:"
        print
        emp_best_tau
        print
        "Observed treatments: "
        print
        samp_params['T_samp']
    # print "Deviation in treatment vector: "
    # print np.linalg.norm(emp_best_tau - samp_params['T_samp'])
    print
    'x: ' + str(res.x)
    print
    'off pol evaluation value '
    print
    res.fun
    """
    Optimize a treatment policy over oracle outcomes f 
    """
    # spl_x = samp_params['z'][:,0]
    # spl_t = samp_params['z'][:,1]
    # # f is positive
    # splined_f_tck = interpolate.bisplrep(spl_x,spl_t, samp_params['f'])
    # samp_params['spline'] = splined_f_tck
    samp_params['tau'] = emp_best_tau
    oracle_outcomes = samp_params['oracle_func'](**samp_params)
    ## Evaluate the 'true' performance of this treatment vector
    print
    'oracle mean of empirically best feature vector  \n'
    print
    np.mean(oracle_outcomes)

    # print 'Computing oracle best-in-class linear policy via interpolation of true response surface: \n'
    beta_d = [np.random.uniform() for i in np.arange(d)]
    # print "initial condition: " + str(beta_d)
    # print 'val of initial condition: '
    # print oracle_func(beta_d, samp_params.items())

    oracle_res = minimize(oracle_eval, x0=beta_d, bounds=((0, 1.0 / np.mean(samp_params['x'])),),
                          args=samp_params.items())
    if verbose:
        print
        oracle_res
        print
        'beta'
        print
        oracle_res.x
        print
        'oracle best linear treatment policy value '
        print
        oracle_res.fun

    return [res, oracle_res, splined_f_tck]


def off_pol_opt_test(n_max, n_trials, n_spacing, n_0, t_lo_sub, t_hi_sub, **sub_params):
    n = sub_params['n'];
    m = sub_params['m'];
    t_lo = t_lo_sub;
    t_hi = t_hi_sub
    d = sub_params['d']
    n_space = np.linspace(n_0, n_max, n_spacing)
    best_beta = np.zeros([len(n_space), n_trials])
    best_oracle_beta = np.zeros([len(n_space), n_trials])
    OOS_OPE = np.zeros([len(n_space), n_trials])
    OOS_oracle = np.zeros([len(n_space), n_trials])
    # discrete_off_pol_evals = np.zeros([n_treatments, n_spacing, n_trials])
    oracle_func = sub_params['oracle_func']
    h_orig = sub_params['h']
    TEST_N = 250
    TEST_SET = evaluate_subsample(250, evaluation=False, cross_val=False, **sub_params)

    for i, n_sub in enumerate(np.linspace(n_0, n_max, n_spacing)):
        # sub_params['h'] = h_orig * (np.power(n_sub,0.2))/np.power(n_0,0.2)
        n_rnd = int(np.floor(n_sub))
        print
        "testing with n = " + str(n_rnd)
        for k in np.arange(n_trials):
            subsamples_pm = evaluate_subsample(n_rnd, evaluation=False, cross_val=False, **sub_params)
            # oracle_evals[t_ind, i, k] = np.mean(evaluate_oracle_interpolated_outcomes(splined_f_tck, m,n_rnd, subsamples_pm['f'], t_lo, t_hi, subsamples_pm['tau'], subsamples_pm['x_samp']))
            ### Compute best betas with random restarts
            oracle_betas = np.zeros([n_restarts, d]);
            eval_vals = np.zeros([n_restarts, d]);
            emp_betas = np.zeros([n_restarts, d]);
            emp_eval_vals = np.zeros([n_restarts, d])
            for i_restart in np.arange(n_restarts):
                beta_d = [np.random.uniform() for i in np.arange(d)]
                res = minimize(lin_off_policy_loss_evaluation, x0=beta_d, jac=off_pol_epan_lin_grad,
                               bounds=((t_lo / max(samp_params['x_samp']), t_hi / max(samp_params['x_samp'])),),
                               args=samp_params.items())
                emp_betas[i_restart] = res.x;
                emp_eval_vals[i_restart] = res.fun

                oracle_res = minimize(oracle_func, x0=beta_d, bounds=((0, 1.0 / np.mean(samp_params['x'])),),
                                      args=samp_params.items())
                oracle_betas[i_restart] = oracle_res.x;
                eval_vals[i_restart] = oracle_res.fun

            emp_best_tau = np.clip(np.dot(res.x, samp_params['x_samp'].T), t_lo, t_hi)
            # get best beta value from random restarts
            best_ind = np.argmin(emp_eval_vals)
            best_beta[i, k] = emp_betas[best_ind, :]

            best_oracle_ind = np.argmin(eval_vals)
            best_oracle_beta[i, k] = oracle_betas[oracle_betas, :]
            TEST_SET['tau'] = best_beta[i, k] * TEST_SET['x_samp']
            OOS_OPE[i, k] = off_policy_evaluation(**TEST_SET)
            OOS_oracle[i, k] = np.mean(oracle_func(**TEST_SET))

    return [best_beta, best_oracle_beta, OOS_OPE, OOS_oracle]


def off_pol_eval_cons_test(n_max, n_trials, n_treatments, n_spacing, n_0, t_lo_sub, t_hi_sub, **sub_params):
    n = sub_params['n'];
    m = sub_params['m'];
    t_lo = t_lo_sub;
    t_hi = t_hi_sub
    treatment_space = np.linspace(t_lo, t_hi, n_treatments)
    off_pol_evals = np.zeros([n_treatments, n_spacing, n_trials])
    oracle_evals = np.zeros([n_treatments, n_spacing, n_trials])
    discrete_off_pol_evals = np.zeros([n_treatments, n_spacing, n_trials])
    oracle_func = sub_params['oracle_func']
    splined_f_tck = sub_params['spline']
    h_orig = sub_params['h']
    for i, n_sub in enumerate(np.linspace(n_0, n_max, n_spacing)):
        # sub_params['h'] = h_orig * (np.power(n_sub,0.2))/np.power(n_0,0.2)
        n_rnd = int(np.floor(n_sub))
        print
        "testing with n = " + str(n_rnd)
        for k in np.arange(n_trials):
            for t_ind, t in enumerate(treatment_space):
                subsamples_pm = evaluate_subsample(n_rnd, evaluation=False, cross_val=False, **sub_params)
                subsamples_pm['tau'] = t * np.ones(n_sub)
                oracle_evals[t_ind, i, k] = np.mean(oracle_func(**subsamples_pm))
                # oracle_evals[t_ind, i, k] = np.mean(evaluate_oracle_interpolated_outcomes(splined_f_tck, m,n_rnd, subsamples_pm['f'], t_lo, t_hi, subsamples_pm['tau'], subsamples_pm['x_samp']))
                off_pol_evals[t_ind, i, k] = off_policy_evaluation(**subsamples_pm)
                discrete_off_pol_evals[t_ind, i, k] = off_pol_disc_evaluation(discretize_tau_policy, **subsamples_pm)

    off_pol_evals.dump(str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")) + 'off_pol_vals.np')
    oracle_evals.dump(str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")) + 'off_pol_oracles.np')
    return [oracle_evals, off_pol_evals, discrete_off_pol_evals]


def off_pol_eval_linear_test(n_max, beta_0, beta_hi, n_trials, n_treatments, n_spacing, n_0, **sub_params):
    '''
    Systematically evaluate over a treatment space defined by a linear treatment policy
    '''
    treatment_space = np.linspace(beta_0, beta_hi, n_treatments)
    off_pol_evals = np.zeros([n_treatments, n_spacing, n_trials])
    oracle_evals = np.zeros([n_treatments, n_spacing, n_trials])
    discrete_off_pol_evals = np.zeros([n_treatments, n_spacing, n_trials])
    t_lo = sub_params['t_lo'];
    t_hi = sub_params['t_hi'];
    spl_x = sub_params['z'][:, 0];
    spl_t = sub_params['z'][:, 1]
    # f is positive
    splined_f_tck = interpolate.bisplrep(spl_x, spl_t, sub_params['f'])
    sub_params['spline'] = splined_f_tck
    oracle_func = sub_params['oracle_func']
    n = sub_params['n'];
    m = sub_params['m']

    for i, n_sub in enumerate(np.linspace(n_0, n_max, n_spacing)):
        n_rnd = int(np.floor(n_sub))
        print
        "testing n = " + str(n_rnd)
        for k in np.arange(n_trials):
            for beta_ind, beta in enumerate(treatment_space):
                subsamples_pm = evaluate_subsample(n_rnd, evaluation=False, cross_val=False, **sub_params)
                tau = np.clip(np.dot(subsamples_pm['x_samp'], beta), t_lo, t_hi)
                subsamples_pm['tau'] = tau
                oracle_evals[beta_ind, i, k] = np.mean(oracle_func(**subsamples_pm))
                # oracle_evals[beta_ind, i, k] = np.mean(evaluate_oracle_interpolated_outcomes(splined_f_tck,m,n_rnd, subsamples_pm['f'], beta_0, beta_hi, tau, subsamples_pm['x_samp']))
                # off_pol_evals[beta_ind, i, k] = off_policy_evaluation(**subsamples_pm)
                off_pol_evals[beta_ind, i, k] = off_policy_evaluation(**subsamples_pm)
                discrete_off_pol_evals[beta_ind, i, k] = off_pol_disc_evaluation(discretize_tau_policy, **subsamples_pm)

    off_pol_evals.dump(str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")) + 'off_pol_linear_vals.np')
    oracle_evals.dump(str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")) + 'off_pol_linear_oracles.np')
    return [oracle_evals, off_pol_evals, discrete_off_pol_evals]