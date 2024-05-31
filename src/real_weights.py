
from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingClassifier


from lightning_modules import neural_network_fitter, ClassificationNet, NSMBalancingScoreNet, BalancingScoreRieszNetDifferentOutputs
import torch
import numpy as np
import scipy

from functools import partial

def check_integers(a):
    return len(a.shape) == 2 and a.shape[1] == 1 and np.all(a >= 0) and np.all(a == a.astype(int))

def check_one_hot(a):
    return len(a.shape) == 2 and np.all(np.isin(a, [0, 1])) and np.all(np.sum(a, axis=1) == 1)

def a_to_integers(a):
    return a if check_integers(a) else np.sum(np.repeat(np.arange(a.shape[1]).reshape(1, -1), a.shape[0], axis=0) * a, axis=1).astype(int)



def from_xp_xq_to_x_labels(xp, xq):
    x = np.concatenate((xp, xq), axis=0)
    labels = np.concatenate((np.zeros((xp.shape[0],1)), np.ones((xq.shape[0],1))), axis=0).astype(int)
    return x, labels

def from_x_labels_to_xp_xq(x, labels):
    xp = x[labels==0]
    xq = x[labels==1]
    return xp, xq



class ClassificationNetWrapper(object):

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, z, labels, **kwargs):
        self.model = neural_network_fitter(z, labels, model_class=ClassificationNet, regression=False, **kwargs)

    def predict_proba(self, z):
        self.model.eval()
        self.model.to('cpu')
        z = torch.from_numpy(z)
        probas = self.model.predict_proba(z).detach().numpy()
        return probas


class NSMClassificationNetWrapper(object):

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, x, a):
        if len(a.shape) > 1:
            a = np.sum(np.repeat(np.arange(a.shape[1]).reshape(1, -1), a.shape[0], axis=0) * a, axis=1).astype(int)
        self.model = neural_network_fitter(x, a, model_class=NSMBalancingScoreNet, regression=False,
                                           **self.kwargs)

    def predict_proba(self, x):
        self.model.eval()
        self.model.to('cpu')
        x = torch.from_numpy(x)
        probas = self.model.predict_proba(x).detach().numpy()
        return probas


class AutoDMLWrapper(object):

    def __init__(self, path='', balancingscore_fitter_model='riesz_different_outputs', **balancingscore_fitter_kwargs):
        self.groundtruth_fitter_kwargs = groundtruth_fitter_kwargs
        self.balancingscore_fitter_kwargs = balancingscore_fitter_kwargs

    def fit(self, x, a):
        self.model = AUTODML_LEARNERS[self.balancingscore_fitter_model](x, a, **self.balancingscore_fitter_kwargs)

    def predict_proba(self, x):
        self.model.eval()
        self.model.to('cpu')
        x = torch.from_numpy(x)
        probas = self.model.predict_proba(x).detach().numpy()
        return probas



def polynomial_regression(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree=degree), LogisticRegression(**kwargs))

def propensity_score_compression_xp_xq(xp, xq, **kwargs):
    x, labels = from_xp_xq_to_x_labels(xp, xq)
    repr = propensity_score_compression(x, labels, **kwargs)
    reprxp = repr[labels.flatten() == 0,:]
    reprxq = repr[labels.flatten() == 1,:]
    return reprxp, reprxq
def propensity_score_compression(x, a, model='2-PolynomialRegression', normalisation_x=False,  logits=False, **kwargs):
        if normalisation_x:
            x = (x - x.mean(axis=0)) / x.std(axis=0)



        model = BASE_CLASSIFIERS[model](random_state=0, **kwargs)

        a_integers = a_to_integers(a)
        model.fit(x, a_integers)
        repr = model.predict_proba(x)
        if logits:
            repr = scipy.special.logit(np.clip(repr, 1e-10, 1 - 1e-10))
        return repr

def propensity_score_ground_truth_fitter(x, a, **kwargs):
    assert 'logits' not in kwargs
    repr = propensity_score_compression(x, a, **kwargs)
    true_weights = (a.mean(axis=0) / repr * a).sum(axis=1)
    return true_weights

def density_ratio_ground_truth_fitter(x, labels, **kwargs):
    assert 'logits' not in kwargs
    repr = propensity_score_compression(x, labels, **kwargs)
    true_weights = repr[:,1] / repr[:,0]
    return true_weights


def pw(x, a, model='LogisticRegression', normalisation_x=False, normalisation_a=False, n_permutations=1, seed=0, **kwargs):

    if normalisation_x:
        x = (x - x.mean(axis=0)) / x.std(axis=0)
    if normalisation_a:
        a = (a - a.mean(axis=0)) / a.std(axis=0)

    model = BASE_CLASSIFIERS[model]

    distr_reg = np.hstack((x, a))
    distr_indep = None
    rng = np.random.default_rng(seed=seed)
    if n_permutations > 0:
        distr_indeps = []
        for _ in range(n_permutations):
            a_permuted = rng.permutation(a)
            distr_indeps.append(np.hstack((x, a_permuted)))
        distr_indep = np.vstack(distr_indeps)
    else:  # cross-product
        raise NotImplemented

    n = x.shape[0]
    n_indep = distr_indep.shape[0]

    data = np.vstack((distr_reg, distr_indep))
    is_indep = np.hstack([np.zeros((n,)), np.ones((n_indep,))]).astype(int)

    clf = model(random_state=0, **kwargs)
    clf.fit(data, is_indep)
    probas = clf.predict_proba(distr_reg)

    w = probas[:, 1].reshape(-1, 1) / probas[:, 0].reshape(-1, 1)
    return w

def permutation_weighting_ground_truth_fitter(x, a, **kwargs):
    w = pw(x,a,**kwargs)
    if 'n_permutations' in kwargs:
        w = w / kwargs['n_permutations']
    return w

def by_treatment_density_ratio_ground_truth_fitter(x, a, model='2-PolynomialRegression', **kwargs):
    true_weights = np.zeros((x.shape[0], 1))

    model = BASE_CLASSIFIERS[model]

    for a_value in range(a.shape[1]):
        mask = (a[:,a_value] == 1)
        x_a = x[mask]
        is_indep = np.hstack([np.zeros((x_a.shape[0],)), np.ones((x.shape[0],))]).astype(int)
        x_concat = np.vstack([x_a, x])
        clf = model(random_state=0, **kwargs)
        clf.fit(x_concat, is_indep)
        probas = clf.predict_proba(x_a)
        true_weights[mask] = (x_a.shape[0] / x.shape[0]) * (probas[:, 1].reshape(-1, 1) / probas[:, 0].reshape(-1, 1))
    return true_weights



def real_weights_ground_truth_fitter(x, a, dataset_class, dataset_hyperparams):
    from datasets import DATASET_CLASSES
    true_weights = ((a.mean(axis=0) / DATASET_CLASSES[dataset_class](**dataset_hyperparams).p_a) * a).sum(axis=1)
    return true_weights


GROUND_TRUTH_FITTERS = {
    'density_ratio': density_ratio_ground_truth_fitter,

    'propensity': propensity_score_ground_truth_fitter,
    'permutation': permutation_weighting_ground_truth_fitter,
    'densities': by_treatment_density_ratio_ground_truth_fitter,
    'real': real_weights_ground_truth_fitter,
}

BASE_CLASSIFIERS = {
    'LogisticRegression': LogisticRegression,
    '2-PolynomialRegression': polynomial_regression,
    'GradientBoostingClassifier': GradientBoostingClassifier,
    'ClassificationNetWrapper': ClassificationNetWrapper,
    'NSMClassificationNetWrapper': NSMClassificationNetWrapper,
    'NSMClassificationNetWrapperES' : partial(NSMClassificationNetWrapper, early_stopping=True),
    'NSMClassificationNetWrapperNoES': partial(NSMClassificationNetWrapper, early_stopping=False),
}