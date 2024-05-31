import os
import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers
import torch
import numpy as np

from functools import partial
import time
from lightning_modules import neural_network_fitter, BalancingScoreNet, NSMBalancingScoreNet, BalancingScoreNetDifferentOutputs, BalancingScoreNetDifferentScores, BalancingScoreRieszNetPQ, BalancingScoreRieszNetDifferentOutputs
from real_weights import GROUND_TRUTH_FITTERS, from_xp_xq_to_x_labels, check_integers, check_one_hot
from sklearn.decomposition import PCA



def first_dimensions_compression(x, a, k=5):
    repr = x[:, :k]
    return repr

def pca_compression(x, y, k=5, **kwargs):
    print(y)
    if check_integers(y) or check_one_hot(y):
        reprx = PCA(k, random_state=0, **kwargs).fit_transform(x)
        return reprx
    else:
        pca = PCA(k, random_state=0, **kwargs)
        pca.fit(np.concatenate((x,y), axis=0))
        reprx = pca.transform(x)
        repry = pca.transform(y)
        return reprx, repry

def direct_covariate_compression(x, a, score=(lambda x: x)):
    repr = score(x)
    return repr

def fitter_compression(x, y, fitter_class='GroundTruthThenBalancingScoreFitter', **kwargs):
    fitter = REPRESENTATION_FITTERS[fitter_class](**kwargs)
    fitter.fit(x, y)
    if check_integers(y) or check_one_hot(y):
        reprx = fitter.representation(x, y)
        return reprx
    else:
        reprx = fitter.representation(x)
        repry = fitter.representation(y)
        return reprx, repry

def fitter_compression_with_score(x, a, fitter_class='GroundTruthThenBalancingScoreFitter', **kwargs):
    fitter = REPRESENTATION_FITTERS[fitter_class](**kwargs)
    fitter.fit(x, a)
    return fitter.representation






BALANCING_SCORE_WITH_GROUND_TRUTH_FITTERS = {
    'neural_shared_all': partial(neural_network_fitter, model_class=BalancingScoreNet),
    'neural_different_outputs': partial(neural_network_fitter, model_class=BalancingScoreNetDifferentOutputs),
    'neural_different_scores': partial(neural_network_fitter, model_class=BalancingScoreNetDifferentScores),
}
BALANCING_SCORE_WITH_AUTODML_LOSS_FITTERS = {
    'neural_pq': partial(neural_network_fitter, model_class=BalancingScoreRieszNetPQ),
    'neural_different_outputs': partial(neural_network_fitter, model_class=BalancingScoreRieszNetDifferentOutputs)
}

class GroundTruthThenBalancingScoreFitter(object):

    def __init__(self, path='', groundtruth_fitter_model='propensity', groundtruth_fitter_kwargs={}, balancingscore_fitter_model='neural_different_outputs', balancingscore_fitter_kwargs={}):
        self.balancingscore_fitter_model = balancingscore_fitter_model
        if os.path.exists(path):
            self.load(path)
        else:
            self.groundtruth_fitter_model = groundtruth_fitter_model
            self.groundtruth_fitter_kwargs = groundtruth_fitter_kwargs
            self.balancingscore_fitter_kwargs = balancingscore_fitter_kwargs



    def fit(self, x, y):
        if not check_integers(y) and not check_one_hot(y):
            x, a = from_xp_xq_to_x_labels(x, y)
        elif not check_one_hot(y):
            raise NotImplementedError
        else:
            a = y
        pseudo_groundtruth_weights = GROUND_TRUTH_FITTERS[self.groundtruth_fitter_model](x, a, **self.groundtruth_fitter_kwargs)
        self.model = BALANCING_SCORE_WITH_GROUND_TRUTH_FITTERS[self.balancingscore_fitter_model](x, a, pseudo_groundtruth_weights, **self.balancingscore_fitter_kwargs)


    def score_x(self, x, a=None):
        self.model.eval()
        self.model.to('cpu')
        x = torch.from_numpy(x)
        if a is not None:
            a = torch.from_numpy(a)
        if self.balancingscore_fitter_model != 'neural_different_scores':
            return self.model.score_x(x).detach().numpy()
        else:
            return self.model.score_x(x, a).detach().numpy()


    def score_a(self, a):
        self.model.eval()
        self.model.to('cpu')
        a = torch.from_numpy(a)
        return self.model.score_a(a).detach().numpy()


class AutoDMLBalancingScoreFitter(object):

    def __init__(self, path='', model='neural_different_outputs', **kwargs):
        self.balancingscore_fitter_model = model
        self.balancingscore_fitter_kwargs = kwargs



    def fit(self, x, y):
        if not check_integers(y) and not check_one_hot(y):
            x, labels = from_xp_xq_to_x_labels(x, y)
        else:
            labels = y
        self.model = BALANCING_SCORE_WITH_AUTODML_LOSS_FITTERS[self.balancingscore_fitter_model](x, labels, **self.balancingscore_fitter_kwargs)


    def representation(self, x, a=None):
        self.model.eval()
        self.model.to('cpu')
        x = torch.from_numpy(x)
        if a is not None:
            a = torch.from_numpy(a)

        if self.balancingscore_fitter_model != 'neural_different_scores':
            return self.model.score_x(x).detach().numpy()
        else:
            return self.model.score_x(x, a).detach().numpy()

    def predict_riesz(self, x, a=None):
        self.model.eval()
        self.model.to('cpu')
        x = torch.from_numpy(x)
        if self.balancingscore_fitter_model == 'neural_pq':
            return self.model.riesz(x).detach().numpy()
        else:
            a = torch.from_numpy(a)
            return self.model.riesz(x, a).detach().numpy()


class NSMBalancingScoreFitter(object):

    def __init__(self, return_ps=False, **kwargs):
        self.return_ps = return_ps
        self.kwargs = kwargs



    def fit(self, x, y):
        if not check_integers(y) and not check_one_hot(y):
            x, labels = from_xp_xq_to_x_labels(x, y)
        elif check_one_hot(y):
            labels = np.sum(np.repeat(np.arange(y.shape[1]).reshape(1, -1), y.shape[0], axis=0) * y, axis=1).astype(int)
        else:
            labels = y
        self.model = neural_network_fitter(x, labels, model_class=NSMBalancingScoreNet, regression=False, **self.kwargs)


    def representation(self, x, a=None):
        self.model.eval()
        self.model.to('cpu')
        x = torch.from_numpy(x)
        repr = None
        if self.return_ps == 'logs':
            repr = self.model.presoftmax(x).detach().numpy()
        elif self.return_ps:
            repr = self.model.softmax(self.model.presoftmax(x)).detach().numpy()
        else:
            repr = self.model.score_x(x).detach().numpy()
        return repr


    def score_a(self, a):
        return a
    
    def predict_proba(self, x):
        self.model.eval()
        self.model.to('cpu')
        x = torch.from_numpy(x)
        probas = self.model.predict_proba(x).detach().numpy()
        return probas

REPRESENTATION_FITTERS = {
    'GroundTruthThenBalancingScoreFitter': GroundTruthThenBalancingScoreFitter,
    'AutoDMLBalancingScoreFitter': AutoDMLBalancingScoreFitter,
    'NSMBalancingScoreFitter': NSMBalancingScoreFitter,
}