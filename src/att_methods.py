import numpy
import scipy
import osqp
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
from tqdm import tqdm
from metrics import gaussian_grammat
from functools import partial
from representations import first_dimensions_compression, direct_covariate_compression, fitter_compression, pca_compression, AutoDMLBalancingScoreFitter

from real_weights import propensity_score_compression, propensity_score_compression_xp_xq, propensity_score_ground_truth_fitter, GROUND_TRUTH_FITTERS, from_xp_xq_to_x_labels, from_x_labels_to_xp_xq



# Inspired from https://github.com/ngreifer/WeightIt
def kernel_balancing(xp, xq, kernel='energy', eps=1e-6, lmb=0, rep=None):
    np = xp.shape[0]
    nq = xq.shape[1]

    if rep is not None:
        xp = rep(xp)
        xq = rep(xq)

    if kernel == 'energy':
        Dpp = scipy.spatial.distance_matrix(xp, xp)
        Dpq = scipy.spatial.distance_matrix(xp, xq)
        Kpp = -Dpp
        Kpq = -Dpq
    elif 'polynomial' in kernel or kernel == 'linear':
        degree = 1 if kernel == 'linear' else int(kernel[0])
        Kpp = numpy.power(numpy.matmul(xp, xp.T), degree)
        Kpq = numpy.power(numpy.matmul(xp, xq.T), degree)
    else:
        print(f'Kernel {kernel} not supported')

    S = 2 * Kpp / (np ** 2)
    v = - 2 / (nq * np) * Kpq.sum(axis=1)

    S = scipy.sparse.csr_matrix(S + 2 * lmb * numpy.identity(np))

    A = scipy.sparse.csr_matrix(numpy.vstack((numpy.identity(np), numpy.ones((1,np)))))
    l = numpy.hstack((eps * numpy.ones(np), np*numpy.ones(1)))
    u = numpy.hstack((numpy.inf * numpy.ones(np), np*numpy.ones(1)))

    m = osqp.OSQP()
    m.setup(P=S, q=v, A=A, l=l, u=u, verbose=False)
    res = m.solve()
    w = res.x.reshape(-1, 1)
    w[w < 0] = 0
    return w



def compression_then_balancing(xp, xq, compression_method, balancing_method, compression_kwargs={}, balancing_kwargs={}):
    repxp, repxq = compression_method(xp, xq, **compression_kwargs)
    w = balancing_method(repxp, repxq, **balancing_kwargs)
    return w


from scipy.optimize import root


def entropy_balancing_weights(xp, xq, C=None):
    np = xp.shape[0]
    C = xp.T - xq.mean(axis=0).reshape(-1,1) if C is None else C(xp)

    def exps_func(lmb):
        return numpy.exp(numpy.matmul(-C.T, lmb.reshape(-1, 1))).reshape((-1, 1))
    
    def W(lmb):
        exps = exps_func(lmb)
        return exps / exps.sum()


    def gradient(lmb):
        return -numpy.matmul(C,W(lmb)).flatten()

    def hessian(lmb):
        w = W(lmb)
        left = C
        right = C.T
        middle = numpy.diag(w) - numpy.matmul(w, w.T)
        return numpy.matmul(numpy.matmul(left, middle), right)



    lmb0 = numpy.zeros((xp.shape[1],))
    lmb_prime = root(gradient, lmb0, jac=hessian).x
    w = W(lmb_prime)

    return w

def ipw(xp, xq, algorithm='density_ratio', **kwargs):
    x, labels = from_xp_xq_to_x_labels(xp, xq)
    w = GROUND_TRUTH_FITTERS[algorithm](x, labels, **kwargs)
    w = w.flatten()
    labels = labels.flatten()
    w = w[labels == 0]
    return w

def autodml(xp, xq, model='neural_pq', **kwargs):
    fitter = AutoDMLBalancingScoreFitter(model=model, **kwargs)
    fitter.fit(xp,xq)
    return fitter.predict_riesz(xp)

def unweighted(xp, xq):
    np = xp.shape[0]
    return numpy.ones((np,1))



ALGORITHMS = {
    'KB': kernel_balancing,
    "AutoDML": autodml,
    'IPW': ipw,
    'EntB': entropy_balancing_weights,
    'unweighted': unweighted,
}

COMPRESSION_METHODS = {
    'PS': propensity_score_compression_xp_xq,
    'Fitter': fitter_compression,
    'PCA': pca_compression,
}

for algo in list(ALGORITHMS.keys()):
    for compression in COMPRESSION_METHODS:
        ALGORITHMS[compression + '+' + algo] = partial(compression_then_balancing,
                                                       compression_method=COMPRESSION_METHODS[compression],
                                                       balancing_method=ALGORITHMS[algo])

