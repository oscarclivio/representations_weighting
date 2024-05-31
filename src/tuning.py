import os
import traceback
from tqdm import tqdm
import warnings
import glob
import pandas as pd
import time

import ate_methods
import att_methods
from datasets import DATASET_CLASSES
from metrics import l1_error, l2_error, true_weights_mse, scalar_error
from real_weights import GROUND_TRUTH_FITTERS

warnings.filterwarnings('ignore')


class Tuner(object):

    def __init__(self, file_path, methods, datasets, ate=True, debug=False, normalise_weights=True, clean=False, hard_clean=False):
        self.file_path = file_path
        self.methods = methods
        self.datasets = datasets
        self.ate = ate
        self.debug = debug
        self.normalise_weights = normalise_weights
        self.algorithms = att_methods.ALGORITHMS if not ate else ate_methods.ALGORITHMS
        if not os.path.isdir(file_path):
            os.mkdir(file_path)

    def _to_csv_name(self, string):
        string = string.replace(' ','_').replace('(','').replace(')','').replace('=','-').replace(';','')
        return os.path.join(self.file_path, f'{string}.csv')

    def _load_files(self):
        return list(glob.glob(self._to_csv_name('*')))


    def write(self, key, value):
        value.to_csv(self._to_csv_name(key))

    def _check_and_lock(self, key, **kwargs):
        filenames = self._load_files()
        free = (self._to_csv_name(key) not in filenames)
        if free:
            self.write(key, pd.DataFrame([dict(status = 'Locked')]))
        return free

    def tune(self, iterator):
        taus_dict = {}

        for key in tqdm(iterator):
            (method_key, dataset_key, range_h) = key
            key_string = ' ; '.join(key[:2])
            print(key_string)

            if self._check_and_lock(key_string):

                try:


                    res = []

                    algorithm, method_hyperparameters = self.methods[method_key]
                    dataset_class, dataset_hyperparameters_list = self.datasets[dataset_key]

                    for dataset_hyperparameters in dataset_hyperparameters_list:

                        start_time = time.time()

                        dataset = DATASET_CLASSES[dataset_class](**dataset_hyperparameters)
                        dataset_key_extended = dataset_key + str(dataset_hyperparameters)

                        if self.ate:
                            x, y = dataset.x, dataset.a
                        else:
                            x, y = dataset.xp, dataset.xq
                        w = self.algorithms[algorithm](x, y, **method_hyperparameters)

                        if not self.ate:
                            res_iter_dict = dict(
                                method_key=method_key,
                                dataset_key=dataset_key,
                                **method_hyperparameters,
                                **dataset_hyperparameters,
                                time1=time.time() - start_time,
                                status='Done',
                                checksum=dataset.checksum,
                                bias=scalar_error(w, dataset.pseudo_y, tau=dataset.tau_real, normalise_weights=self.normalise_weights),
                            )
                            res.append(res_iter_dict)
                        else:
                            if dataset_key_extended not in taus_dict:
                                taus_dict[dataset_key_extended] = dataset.tau(dataset.a).flatten()

                            res_iter_dict = dict(
                                method_key=method_key,
                                dataset_key=dataset_key,
                                **method_hyperparameters,
                                **dataset_hyperparameters,
                                time=time.time() - start_time,
                                status='Done',
                                checksum=dataset.checksum,
                            )
                            for h in range_h:
                                res_iter_dict.update({
                                    f'bias_h{h}': l1_error(w, dataset.a, dataset.y,
                                                           taus=taus_dict[dataset_key_extended],
                                                           categorical=dataset.categorical, h=h,
                                                           normalise_weights=self.normalise_weights),
                                    f'bias_l2_h{h}': l2_error(w, dataset.a, dataset.y,
                                                              taus=taus_dict[dataset_key_extended],
                                                              categorical=dataset.categorical, h=h,
                                                              normalise_weights=self.normalise_weights),
                                })
                            res_iter_dict['time2'] = time.time() - start_time
                            res.append(res_iter_dict)
                    res = pd.DataFrame(res)
                    self.write(key_string, res)

                except Exception:
                    traceback.print_exc()
                    print(f'Error! Not saving {key_string} which remains locked.')
                    assert not self.debug



class TrueWeightsTuner(Tuner):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def tune(self, iterator):
        taus_dict = {}

        for key in tqdm(iterator):
            (method_key, dataset_key) = key
            key_string = ' ; '.join(key[:2])
            print(key_string)

            if self._check_and_lock(key_string):

                try:


                    res = []

                    algorithm, method_hyperparameters = self.methods[method_key]

                    dataset_class, dataset_hyperparameters_list = self.datasets[dataset_key]

                    for dataset_hyperparameters in dataset_hyperparameters_list:

                        start_time = time.time()

                        dataset = DATASET_CLASSES[dataset_class](**dataset_hyperparameters)

                        w = GROUND_TRUTH_FITTERS[algorithm](dataset.x, dataset.a, **method_hyperparameters)

                        res_iter_dict = dict(
                            method_key=method_key,
                            dataset_key=dataset_key,
                            **method_hyperparameters,
                            **dataset_hyperparameters,
                            time1=time.time() - start_time,
                            mse=true_weights_mse(w, dataset.true_w, a=dataset.a),
                            status='Done',
                            checksum=dataset.checksum,
                        )
                        res.append(res_iter_dict)

                    res = pd.DataFrame(res)
                    self.write(key_string, res)

                except Exception:
                    traceback.print_exc()
                    print(f'Error! Not saving {key_string} which remains locked.')