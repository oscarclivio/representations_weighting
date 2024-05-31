import numpy as np
import scipy
import osqp
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
from tqdm import tqdm


# Parts of the file come from https://github.com/strumke/hsic_python/blob/master/hsic.py

def gaussian_kernel(z):
    return np.exp(np.sum(-z * z, axis=-1))

def exact_kernel(z, eps=1e-4):
    return (np.max(np.abs(z), axis=-1) < eps)


def l1_error(w, a, y, taus, categorical=False, h=1, normalise_weights=True):


    w = w.reshape(-1, 1)
    y = y.reshape(-1, 1)

    d_a = a.shape[1]
    n = a.shape[0]

    nw_kernel = exact_kernel if categorical else gaussian_kernel

    # Estimated
    nw_weights = nw_kernel((a.reshape(n, 1, d_a) - a.reshape(1, n, d_a)) / h)
    weights = w * nw_weights
    tau_estimated = np.sum(weights * y, axis=0)
    if normalise_weights:
        tau_estimated = tau_estimated / np.sum(weights, axis=0)
    tau_estimated = tau_estimated.flatten()

    return np.mean(np.abs(tau_estimated - taus))


def l2_error(w, a, y, taus, categorical=False, h=1, normalise_weights=True):


    w = w.reshape(-1, 1)
    y = y.reshape(-1, 1)

    d_a = a.shape[1]
    n = a.shape[0]

    nw_kernel = exact_kernel if categorical else gaussian_kernel

    # Estimated
    nw_weights = nw_kernel((a.reshape(n, 1, d_a) - a.reshape(1, n, d_a)) / h)
    weights = w * nw_weights
    tau_estimated = np.sum(weights * y, axis=0)
    if normalise_weights:
        tau_estimated = tau_estimated / np.sum(weights, axis=0)
    tau_estimated = tau_estimated.flatten()

    return np.sqrt(np.mean(np.power(tau_estimated - taus, 2)))
def scalar_error(w, y, tau, normalise_weights = True):
    w = w.flatten()
    y = y.flatten()
    tau_estimated = np.sum(w * y)
    if normalise_weights:
        tau_estimated = tau_estimated / np.sum(w)
    return np.abs(tau_estimated - tau)

def centering(M, w):
    """
    Calculate the centering matrix
    """
    n = M.shape[0]
    w = w.flatten()
    n2 = w.shape[0]
    assert n == n2
    prods = w * w.reshape(-1,1)
    diag = np.diag(w)
    H = diag - prods/n

    return np.matmul(M, H)

def gaussian_grammat(x=None, xnorm=None, sigma=None):
    """
    Calculate the Gram matrix of x using a Gaussian kernel.
    If the bandwidth sigma is None, it is estimated using the median heuristic:
    ||x_i - x_j||**2 = 2 sigma**2
    """
    if x is not None:
        try:
            x.shape[1]
        except IndexError:
            x = x.reshape(x.shape[0], 1)

    if xnorm is None:
        xxT = np.matmul(x, x.T)
        xnorm = np.diag(xxT) - xxT + (np.diag(xxT) - xxT).T
    if sigma is None:
        mdist = np.median(xnorm[xnorm!= 0])
        sigma = np.sqrt(mdist*0.5)


   # --- If bandwidth is 0, add machine epsilon to it
    if sigma==0:
        eps = 7./3 - 4./3 - 1
        sigma += eps

    KX = - 0.5 * xnorm / sigma / sigma
    np.exp(KX, KX)
    return KX

def dHSIC_calc(K_list):
    """
    Calculate the HSIC estimator in the general case d > 2, as in
    [2] Definition 2.6
    """
    if not isinstance(K_list, list):
        K_list = list(K_list)

    n_k = len(K_list)

    length = K_list[0].shape[0]
    term1 = 1.0
    term2 = 1.0
    term3 = 2.0/length

    for j in range(0, n_k):
        K_j = K_list[j]
        term1 = np.multiply(term1, K_j)
        term2 = 1.0/length/length*term2*np.sum(K_j)
        term3 = 1.0/length*term3*K_j.sum(axis=0)

    term1 = np.sum(term1)
    term3 = np.sum(term3)
    dHSIC = (1.0/length)**2*term1+term2-term3
    return dHSIC

def HSIC(w, x, y):
    """
    Calculate the HSIC estimator for d=2, as in [1] eq (9)
    """
    n = x.shape[0]
    w = n * w / w.sum()
    assert n == w.size
    return np.trace(np.matmul(centering(gaussian_grammat(x), w),centering(gaussian_grammat(y), w)))/n/n

def dHSIC(*argv):
    assert len(argv) > 1, "dHSIC requires at least two arguments"

    if len(argv) == 2:
        x, y = argv
        return HSIC(x, y)

    K_list = [gaussian_grammat(_arg) for _arg in argv]

    return dHSIC_calc(K_list)


def true_weights_mse(w_pred, w_real, a=None):
    if w_real is None:
        return None
    w_pred = w_pred.flatten() if not isinstance(w_pred, float) else w_pred
    w_real = w_real.flatten()
    if a is not None:
        n = w_pred.size
        res = 0.
        for j in range(a.shape[1]):
            res += np.sqrt(np.sum(np.power(w_pred - w_real,2).flatten()*a[:,j].flatten())) * np.mean(a[:,j])
        return res
    else:
        return np.sqrt(np.sum(np.power(w_pred - w_real,2)))