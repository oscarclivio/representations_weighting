from tuning import Tuner
from ate_methods import ALGORITHMS
from datasets import DATASET_CLASSES
from copy import copy
import numpy as np
from functools import partial
import os
import logging
log = logging.getLogger("lightning_fabric")
log.propagate = False
log.setLevel(logging.ERROR)

name = os.path.basename(__file__).replace('.py','').split('_part')[0]
file_path = f'../outputs/{name}'

# Methods


kernel_list = ['energy','linear']

base_method_list = ['propensity']
base_classifier_list = ['GradientBoostingClassifier']

lmb_list = [0.0001]


weighting_methods = {}
for kernel in kernel_list:
    for lmb in lmb_list:
        weighting_methods[f'KB ({kernel}, lmb{lmb})'] = ('KB', {'kernel': kernel, 'lmb': lmb})

methods = {}
for base_method in base_method_list:
    for base_classifier in base_classifier_list:
        methods[f'IPW {base_method, base_classifier}'] = ('IPW', {'algorithm': base_method, 'model': base_classifier})

# dataset and methods and iterator!!!

iterator = []

range_h_categorical = [1]
range_h_continuous = [10,1,0.1]

datasets = {}


hyperparams = {}
dataset_class = f'News'
dataset_name = f'{dataset_class}'
hyperparams_list = []
for seed in range(50):
    hyperparams_list.append(dict(**hyperparams, seed=seed))
datasets[dataset_name] = (dataset_class, hyperparams_list)


hyperparams = {}
dataset_class = f'IHDP'
dataset_name = f'{dataset_class}'
hyperparams_list = []
for seed in range(50):
    hyperparams_list.append(dict(**hyperparams, seed=seed))
datasets[dataset_name] = (dataset_class, hyperparams_list)


#base_method_list = ['propensity', 'permutation', 'densities']
#base_classifier_list = ['LogisticRegression', '2-PolynomialRegression', 'GradientBoostingClassifier','ClassificationNetWrapper', 'NSMClassificationNetWrapperES', 'NSMClassificationNetWrapperNoES']
compression_kwargs_list = [
    {'model': 'neural_different_outputs', 'balancing_x_dim': 10, 'batch_size': 100000, 'early_stopping': True, 'init_lr': 0.01, 'layer_dims_balancing_x': [200], 'layer_dims_output': [200], 'weight_decay': 0},
]

compression_kwargs_list = compression_kwargs_list[:1]

for original_method_key in weighting_methods:
    algo = weighting_methods[original_method_key][0]
    algo_kwargs = weighting_methods[original_method_key][1]

    for k in [10]:

        # PCA
        score_method_key = f'PCA+{original_method_key} (k{k})'
        methods[score_method_key] \
            = (
            f'PCA+{algo}',
            {
                'compression_kwargs': {'k': k},
                'balancing_kwargs': algo_kwargs
            }
        )

        # PS compression
        for logits in [False]:
            for base_classifier in base_classifier_list:
                score_method_key = f'PS+{original_method_key} ({base_classifier}, logits{logits})'
                methods[score_method_key] \
                    = (
                    f'PS+{algo}',
                    {
                        'compression_kwargs': {'model': base_classifier, 'logits': logits},
                        'balancing_kwargs': algo_kwargs
                    }
                )

    for compression_kwargs in compression_kwargs_list:

        k = compression_kwargs['balancing_x_dim']

        score_method_key = f'ABS+{original_method_key} (k{k}'
        for value in compression_kwargs.values():
            score_method_key = score_method_key + f' {value}'
        score_method_key = score_method_key + ')'
        methods[score_method_key] \
            = (
            f'Fitter+{algo}',
            {
                'compression_kwargs': {
                    'fitter_class': 'AutoDMLBalancingScoreFitter',
                    **compression_kwargs
                },
                'balancing_kwargs': algo_kwargs
            }
        )

        # NSM
        score_method_key = f'NSM+{original_method_key} (k{k}'
        for value in compression_kwargs.values():
            score_method_key = score_method_key + f' {value}'
        score_method_key = score_method_key + ')'
        methods[score_method_key] \
            = (
            f'Fitter+{algo}',
            {
                'compression_kwargs': {'fitter_class': 'NSMBalancingScoreFitter',
                                       **compression_kwargs},
                'balancing_kwargs': algo_kwargs
            }
        )

        # AutoDML head
        score_method_key = f'AutoDML head (k{k}'
        for value in compression_kwargs.values():
            score_method_key = score_method_key + f' {value}'
        score_method_key = score_method_key + ')'
        methods[score_method_key] \
            = (
            f'AutoDML', compression_kwargs
        )

methods['unweighted'] = ('unweighted', {})
methods.update(weighting_methods)
methods['EntB'] = ('EntB', {})


for dataset_key in datasets:
    range_h = copy(range_h_categorical)
    for method_key in methods.keys():
        print(method_key)
        iterator.append((method_key, dataset_key, range_h))


for method in methods:
    print(method)

tuner = Tuner(file_path=file_path, methods=methods, datasets=datasets, clean=True, hard_clean=True, ate=True)
tuner.tune(iterator)
