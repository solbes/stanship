import numpy as np


def invcdf(x, cdf, nr, cont=True):
    """
    Inverse CDF random sampler
    :param x: points where the CDF is defined
    :param cdf: CDF values
    :param nr: number of samples to draw
    :param cont: continous / discrete distribution
    :return: (r, inds), where r are the random samples and inds are the
    sampled indices of x
    """

    # normalize cdf
    _cdf = cdf / cdf[-1]

    rr = np.random.random(nr)
    r = np.zeros(nr)
    inds = np.zeros(nr)

    for i in range(nr):
        ind = np.sum(rr[i] > _cdf)
        if ind == 0 or not cont:
            r[i] = x[ind]
        else:
            r[i] = x[ind] + (x[ind] - x[ind - 1]) / (
                        _cdf[ind] - _cdf[ind - 1]) * (rr[i] - _cdf[ind - 1])
        inds[i] = ind

    return r, inds.astype('int')


def hist2d_sample(h, xe, ye, nr, cont=True):
    """
    Sample random values from a 2d histogram via inverse-CDF. First sample
    value from the marginal for the "x-direction", and then sample from the
    conditionals in the "y-direction".
    :param h: 2-dimensional array containing the histogram values (as
    returned by np.histogram2d)
    :param xe: x-direction bin endpoints as returned by np.histogram2d
    :param ye: y-direction bin endpoints as returned by np.histogram2d
    :param nr: number of samples to draw
    :param cont: continuous / discrete distribution
    :return: (xr, yr), where xr and yr are samples in x- and y-directions
    """

    # marginal CDF in the x-direction
    cdf_x = np.cumsum(np.sum(h, axis=1))

    # samples from the marginal
    xr, ix = invcdf(xe, cdf_x, nr, cont=cont)

    # all CDFs in the y-direction
    cdf_y = np.cumsum(h, axis=1)

    # sample from the conditional CDFs
    yr = np.zeros(nr)
    for i, ind_x in enumerate(ix):
        yr[i] = invcdf(ye, cdf_y[ind_x], 1, cont=cont)[0][0]

    return xr, yr


def generate_ship_data(gt, n_data, pars):
    """
    Helper function to generate random ship data
    :param gt: Gross tonnages for the ships
    :param n_data: number of data points per ship to generate
    :param pars: dict of true parameter values
    :return: (power, speed, wind) -data
    """

    assert len(gt) == len(n_data), 'GT and N_data lengths must match.'

    # load the pre-generated 2d histogram
    h = np.loadtxt('data/hist_values.txt')
    xe, ye = np.loadtxt('data/hist_bins.txt')

    n_ships = len(gt)

    a = pars['alp0'] + pars['alp1']*gt + \
        pars['sig_alp']*np.random.standard_normal(n_ships)
    b = pars['beta0'] + pars['beta1']*gt + \
        pars['sig_beta']*np.random.standard_normal(n_ships)

    # sample random values according to the 2d-histrogram
    inputs = [hist2d_sample(h, xe, ye, n) for n in n_data]
    speeds = [x[0] for x in inputs]
    winds = [x[1] for x in inputs]

    powers = [
        ai*speed + bi*wind +
        pars['sig_obs']*np.random.standard_normal(len(speed))
        for ai, bi, speed, wind in zip(a, b, speeds, winds)
    ]

    return powers, speeds, winds
