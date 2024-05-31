import numpy as np
import scipy
import osqp
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
from tqdm import tqdm
from metrics import gaussian_grammat
from functools import partial
from representations import first_dimensions_compression, direct_covariate_compression, fitter_compression, fitter_compression_with_score, pca_compression, AutoDMLBalancingScoreFitter

from real_weights import propensity_score_compression, propensity_score_ground_truth_fitter, GROUND_TRUTH_FITTERS



def mahalanobis_kernel(z, degree, scale=1, higher_order_importance=1):
    mean = np.mean(z, axis=0).reshape(1, -1)
    cov = np.cov(z.T)
    if z.shape[1] != 1:
        inverse_cov = np.linalg.inv(cov)
    else:
        cov = cov.reshape(1, 1)
        inverse_cov = np.linalg.inv(cov)
    centered_z = z - mean
    products = np.matmul(centered_z, np.matmul(inverse_cov, centered_z.T))
    assert products.shape[0] == z.shape[0]
    assert products.shape[1] == z.shape[0]

    return scale * np.power(1 + higher_order_importance * products, degree)



def kernel2semimetric(kernel_mat):
    assert (kernel_mat.shape[0] == kernel_mat.shape[1])
    n = kernel_mat.shape[0]
    diag = np.diag(kernel_mat).flatten()
    diag_tiled = np.tile(diag, (n, 1))
    diag_tiled_transposed = diag_tiled.T
    return diag_tiled + diag_tiled_transposed - 2*kernel_mat



def independence_weights(x, a, kernel_x='energy', kernel_a='energy', eps=1e-6, lmb=0., normalisation_x=False, normalisation_a=False):
    n = x.shape[0]

    if normalisation_x:
        x = (x - x.mean(axis=0)) / x.std(axis=0)
    if normalisation_a:
        a = (a - a.mean(axis=0)) / a.std(axis=0)

    kernels = {
        'energy': (lambda z: scipy.spatial.distance_matrix(z, z)),
        'rbf': (lambda z: kernel2semimetric(gaussian_grammat(z))),
    }
    kernels.update({
        f'{degree}-polynomial': (lambda z: kernel2semimetric(np.power(np.matmul(z, z.T), degree)))  for degree in range(10)
    })
    distances = {
        'a': kernels[kernel_a](a),
        'x': kernels[kernel_x](x),
    }

    Cs = {}
    for name, z in zip(['a', 'x'], [a, x]):
        ckl = distances[name]
        ckx = np.mean(ckl, axis=1).reshape(-1, 1)
        cxl = np.mean(ckl, axis=0).reshape(1, -1)
        cxx = np.mean(ckl)
        Cs[name] = ckl - ckx - cxl + cxx
    P = (2 / (n * n)) * scipy.sparse.csr_matrix(
        Cs['a'] * Cs['x'] - distances['a'] - distances['x'] + lmb * np.identity(n))

    q = 2 / (n * n) * (distances['a'] + distances['x']).sum(axis=1)

    A = scipy.sparse.csr_matrix(np.vstack((np.identity(n), np.ones((1, n)))))
    l = np.hstack((eps * np.ones(n), n))
    u = np.hstack((np.inf * np.ones(n), n))

    m = osqp.OSQP()
    m.setup(P=P, q=q, A=A, l=l, u=u, verbose=False)
    res = m.solve()
    w = res.x.reshape(-1, 1)
    w[w < 0] = 0
    return w


# Inspired from https://github.com/ngreifer/WeightIt
def kernel_balancing(x, a, kernel='energy', normalisation_x=False, eps=1e-6, lmb=0):
    n_a_s = a.sum(axis=0)
    n = x.shape[0]


    if normalisation_x:
        x = (x - x.mean(axis=0)) / x.std(axis=0)

    P = np.zeros((n, n))
    q = np.zeros((n,))
    if kernel == 'rbf':
        D_all_all = scipy.spatial.distance_matrix(x, x)
        mdist = np.median(D_all_all[D_all_all != 0])

    for a_val, n_a in enumerate(n_a_s):
        x_a = x[a[:, a_val] == 1, :]
        if kernel == 'energy':
            D_a_a = scipy.spatial.distance_matrix(x_a, x_a)
            D_a_all = scipy.spatial.distance_matrix(x_a, x)
            K_a_a = -D_a_a
            K_a_all = -D_a_all
        elif kernel == 'rbf':
            D_a_a = scipy.spatial.distance_matrix(x_a, x_a)
            D_a_all = scipy.spatial.distance_matrix(x_a, x)
            sigma = np.sqrt(mdist * 0.5)
            K_a_a = gaussian_grammat(xnorm=D_a_a * D_a_a, sigma=sigma)
            K_a_all = gaussian_grammat(xnorm=D_a_all * D_a_all, sigma=sigma)
        elif 'polynomial' in kernel or kernel == 'linear':
            degree = 1 if kernel == 'linear' else int(kernel[0])
            K_a_a = np.power(np.matmul(x_a, x_a.T), degree)
            K_a_all = np.power(np.matmul(x_a, x.T), degree)
        else:
            print(f'Kernel {kernel} not supported')

        Pa = 2 * K_a_a / (n_a ** 2)
        qa = - 2 / (n * n_a) * K_a_all.sum(axis=1)

        indices = np.nonzero((a[:, a_val] == 1).flatten())[0]
        for i_a, i in enumerate(indices):
            q[i] = qa[i_a]
            for j_a, j in enumerate(indices):
                P[i, j] = Pa[i_a, j_a]
    P = scipy.sparse.csr_matrix(P + 2 * lmb * np.identity(n))

    A = scipy.sparse.csr_matrix(np.vstack((np.identity(n), a.T)))
    l = np.hstack((eps * np.ones(n), n_a_s))
    u = np.hstack((np.inf * np.ones(n), n_a_s))

    m = osqp.OSQP()
    m.setup(P=P, q=q, A=A, l=l, u=u, verbose=False)
    res = m.solve()
    w = res.x.reshape(-1, 1)
    w[w < 0] = 0
    return w


def kernel_balancing_with_score(x, a, score, kernel='energy', normalisation_x=False, eps=1e-6, lmb=0):
    n_a_s = a.sum(axis=0)
    n = x.shape[0]


    if normalisation_x:
        x = (x - x.mean(axis=0)) / x.std(axis=0)

    P = np.zeros((n, n))
    q = np.zeros((n,))

    for a_val, n_a in enumerate(n_a_s):
        x_a = x[a[:, a_val] == 1, :]
        a_here = a[a[:, a_val] == 1]
        a_this_val = np.zeros(a.shape)
        a_this_val[:, a_val] = 1
        if kernel == 'energy':
            D_a_a = scipy.spatial.distance_matrix(score(x_a, a_here), score(x_a, a_here))
            D_a_all = scipy.spatial.distance_matrix(score(x_a, a_here), score(x, a_this_val))
            K_a_a = -D_a_a
            K_a_all = -D_a_all
        else:
            print(f'Kernel {kernel} not supported')

        Pa = 2 * K_a_a / (n_a ** 2)
        qa = - 2 / (n * n_a) * K_a_all.sum(axis=1)

        indices = np.nonzero((a[:, a_val] == 1).flatten())[0]
        for i_a, i in enumerate(indices):
            q[i] = qa[i_a]
            for j_a, j in enumerate(indices):
                P[i, j] = Pa[i_a, j_a]
    P = scipy.sparse.csr_matrix(P + 2 * lmb * np.identity(n))

    A = scipy.sparse.csr_matrix(np.vstack((np.identity(n), a.T)))
    l = np.hstack((eps * np.ones(n), n_a_s))
    u = np.hstack((np.inf * np.ones(n), n_a_s))

    m = osqp.OSQP()
    m.setup(P=P, q=q, A=A, l=l, u=u, verbose=False)
    res = m.solve()
    w = res.x.reshape(-1, 1)
    w[w < 0] = 0
    return w


def compression_then_balancing(x, a, compression_method, balancing_method, compression_kwargs={}, balancing_kwargs={}):
    repr_x = compression_method(x, a, **compression_kwargs)
    w = balancing_method(repr_x, a, **balancing_kwargs)
    return w

def compression_then_balancing_with_score(x, a, compression_method, balancing_method, compression_kwargs={}, balancing_kwargs={}):
    score = compression_method(x, a, **compression_kwargs)
    w = balancing_method(x, a, score=score, **balancing_kwargs)
    return w



from scipy.optimize import root

def entropy_balancing_weights(x, a, normalisation_x=False, normalisation_a=False):
    n = x.shape[0]

    if normalisation_x:
        x = (x - x.mean(axis=0)) / x.std(axis=0)
    if normalisation_a:
        a = (a - a.mean(axis=0)) / a.std(axis=0)

    def g(x, a):
        x_normalised = x - x.mean(axis=0).reshape((1, -1))
        a_normalised = a - a.mean(axis=0).reshape((1, -1))
        corrs = []
        for a_value in range(a.shape[1]):
            corrs.append(a_normalised[:, a_value].reshape((-1, 1)) * x_normalised)
        return np.hstack([x_normalised, a_normalised] + corrs)

    gxa = g(x, a)

    def exps_func(gamma):
        return np.exp(np.matmul(gxa, gamma.reshape(-1, 1))).reshape((-1, 1))

    def dual_gradient(gamma):
        exps = exps_func(gamma)
        return (np.matmul(gxa.T, exps) / np.sum(exps)).flatten()

    gamma_prime = root(dual_gradient, np.zeros((gxa.shape[1],))).x
    exps = exps_func(gamma_prime)
    w = exps / exps.sum()

    return w

def ipw(x, a, algorithm='propensity', **kwargs):
    return GROUND_TRUTH_FITTERS[algorithm](x, a, **kwargs)

def autodml(x, a, model='neural_different_outputs', **kwargs):
    fitter = AutoDMLBalancingScoreFitter(model=model, **kwargs)
    fitter.fit(x,a)
    return fitter.predict_riesz(x, a)

def unweighted(x, a):
    n = x.shape[0]
    return np.ones((n,1))



ALGORITHMS = {
    'IW': independence_weights,
    'KB': kernel_balancing,
    'KBScore': kernel_balancing_with_score,
    "AutoDML": autodml,
    'IPW': ipw,
    'EntB': entropy_balancing_weights,
    'unweighted': unweighted,
}

COMPRESSION_METHODS = {
    'PS': propensity_score_compression,
    'F': first_dimensions_compression,
    'C': direct_covariate_compression,
    'Fitter': fitter_compression,
    'FitterScore': fitter_compression_with_score,
    'PCA': pca_compression,
}

for algo in list(ALGORITHMS.keys()):
    for compression in COMPRESSION_METHODS:
        if 'Score' not in algo and 'Score' not in compression:
            ALGORITHMS[compression + '+' + algo] = partial(compression_then_balancing,
                                                           compression_method=COMPRESSION_METHODS[compression],
                                                           balancing_method=ALGORITHMS[algo])
        elif 'Score' in algo and 'Score' in compression:
            ALGORITHMS[compression + '+' + algo] = partial(compression_then_balancing_with_score,
                                                           compression_method=COMPRESSION_METHODS[compression],
                                                           balancing_method=ALGORITHMS[algo])

ALGORITHMS_NOT_CONTINUOUS = ['KB', 'KBS']