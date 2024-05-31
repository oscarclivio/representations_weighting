import numpy as np
import scipy
import osqp
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
from tqdm import tqdm
import os
from copy import copy
import pickle
import pyreadr
import urllib
import zipfile

# These datasets are taken from https://github.com/oscarclivio/neuralscorematching or https://github.com/BenedicteColnet/IPSW-categorical
def one_hot(input_array):
    # Determine the value of k (number of categories)
    k = input_array.max() + 1

    # Convert to one-hot encoding
    one_hot_encoding = np.eye(k)[input_array.flatten()]

    # Reshape to (n, k)
    one_hot_encoding = one_hot_encoding.reshape(-1, k)

    return one_hot_encoding


class IHDP(object):
    def __init__(self, seed, path_data="../data"):
        self.path_data = path_data

        attrs = {}
        data = np.loadtxt(os.path.join(self.path_data, 'IHDP/csv/ihdp_npci_' + str(seed + 1) + '.csv'), delimiter=',')
        t, y, y_cf = data[:, 0][:, np.newaxis], data[:, 1][:, np.newaxis], data[:, 2][:, np.newaxis]
        mu_0, mu_1, x = data[:, 3][:, np.newaxis], data[:, 4][:, np.newaxis], data[:, 5:]
        x[:, 13] -= 1  # binary variable with values 1 and 2
        attrs['t'] = t
        attrs['y'] = y
        attrs['y_cf'] = y_cf
        attrs['mu0'] = mu_0
        attrs['mu1'] = mu_1
        attrs['cate_true'] = mu_1 - mu_0
        attrs['x'] = x

        # Find binary covariates
        self.binary = []
        for ind in range(attrs['x'].shape[1]):
            self.binary.append(len(np.unique(attrs['x'][:, ind])) == 2)
        self.binary = np.array(self.binary)

        # Subsample data and convert to torch.Tensor with the right device
        for key, value in attrs.items():
            setattr(self, key, value)
        self.t = self.t.astype(int)

        # TODO : set a in one hot format
        self.a = one_hot(self.t)

        # TODO : self.taus_real
        self.taus_real = np.array([self.mu0.mean(), self.mu1.mean()]).flatten()

        # TODO : create true_w
        self.true_w = None

        # ATT format
        self.xp = self.x[self.t.flatten() == 0, :]
        self.xq = self.x[self.t.flatten() == 1, :]
        self.pseudo_y = self.y[self.t.flatten() == 0, :]
        self.tau_real = self.mu0[self.t.flatten() == 1, :].mean()

        # print('IHDP SIZE :', self.xp.shape, self.xq.shape)

        self.checksum = self.x.mean() + self.t.sum() + self.y.mean()
        self.categorical = True

    def tau(self, a):
        return np.matmul(a, self.taus_real.reshape(2, 1))



class TBI(object):
    def __init__(self, seed=0, n=1000, m=1000, separate=True, dataset='train', path_data="../data", pi=0.5):
        self.path_data = os.path.join(path_data, f'Colnet/')

        data = pyreadr.read_r(os.path.join(self.path_data, 'semi-synthetic-DGP.rds'))['total.with.overlap']
        data = data.astype(float)

        self.seed = seed
        self.rng = np.random.default_rng(seed=self.seed)
        rct = data[data.S == 1].sample(n=n, replace=True, random_state=self.rng)
        obs = data[data.S == 0].sample(n=m, replace=True, random_state=self.rng)
        total = pd.concat([rct, obs])

        # Outcome model
        total['outcome.baseline'] = (10 - total['Glasgow.initial']) - 5 * total['gender']

        total['outcome.cate'] = 15 * (6 - total['time_to_treatment.categorized']) + 3 * (
                total['systolicBloodPressure.categorized'] - 1) ** 2

        total['outcome.Y_0'] = total['outcome.baseline']
        total['outcome.Y_1'] = total['outcome.baseline'] + total['outcome.cate']

        # add gaussian noise
        total['outcome.Y_0'] = total['outcome.Y_0'] + self.rng.normal(size=n + m)
        total['outcome.Y_1'] = total['outcome.Y_1'] + self.rng.normal(size=n + m)

        # Covariates
        total_with_covs = pd.DataFrame(total)
        for col in total.columns:
            if len(pd.unique(total[col])) <= 5:
                for el in np.unique(total[col])[1:]:
                    name = 'x.' + col + '.' + str(el)
                    total_with_covs[name] = 0
                    total_with_covs[name][total[col] == el] = 1
            else:
                total_with_covs['x.' + col] = total[col]

            if col == 'systolicBloodPressure':
                break

        self.x = total_with_covs[[col for col in total_with_covs.columns if len(col) >= 2 and col[:2] == 'x.']]

        # random treatment assignment within the RCT / Bernoulli trial
        self.t = self.rng.binomial(n=1, p=pi, size=n)
        self.y = self.t * total['outcome.Y_1'][total.S == 1] + (1 - self.t) * total['outcome.Y_0'][total.S == 1]
        q_mask = (total.S <= (not separate) + 0.5)
        p_mask = (total.S == 1)

        # P/Q format
        self.xp = np.array(self.x[p_mask])
        self.xq = np.array(self.x[q_mask])
        self.pseudo_y = np.array(self.t * self.y / pi - (1 - self.t) * self.y / (1 - pi))
        self.tau_real = np.mean(total['outcome.cate'][q_mask])

        self.total = total

        self.checksum = self.x.mean().mean() + self.t.sum() + self.y.mean()
        self.categorical = True


class News():

    def __init__(self, seed, data_folder=None):

        if data_folder is None:
            data_folder = '../data'

        # Create data if it does not exist
        if not os.path.isdir(os.path.join(data_folder, 'News/numpy_dicts/')):
            self._create_data(data_folder)

        with open(os.path.join(data_folder, 'News/numpy_dicts/data_as_dicts_with_numpy_seed_{}'.format(seed + 1)),
                  'rb') as file:
            data = pickle.load(file)
        data['cate_true'] = data['mu1'] - data['mu0']

        # Subsample data and convert to torch.Tensor with the right device
        for key, value in data.items():
            setattr(self, key, value)
        self.x = self.x.astype(np.double)
        self.t = self.t.astype(int)

        # TODO : set a in one hot format
        self.a = one_hot(self.t)

        # TODO : self.taus_real
        self.taus_real = np.array([self.mu0.mean(), self.mu1.mean()]).flatten()

        # TODO : create true_w
        self.true_w = None

        self.checksum = self.x.mean() + self.t.sum() + self.y.mean()
        self.categorical = True

        # ATT format
        self.xp = self.x[self.t.flatten() == 0, :]
        self.xq = self.x[self.t.flatten() == 1, :]
        self.pseudo_y = self.y[self.t.flatten() == 0, :]
        self.tau_real = self.mu0[self.t.flatten() == 1, :].mean()

    def tau(self, a):
        return np.matmul(a, self.taus_real.reshape(2, 1))

    @staticmethod
    def _create_data(data_folder):

        print('News : no data, creating it')
        print('Downloading zipped csvs')
        urllib.request.urlretrieve('http://www.fredjo.com/files/NEWS_csv.zip',
                                   os.path.join(data_folder, 'News/csv.zip'))

        print('Unzipping csvs with sparse data')
        with zipfile.ZipFile(os.path.join(data_folder, 'News/csv.zip'), 'r') as zip_ref:
            zip_ref.extractall(os.path.join(data_folder, 'News'))

        print('Densifying the sparse data')
        os.mkdir(os.path.join(data_folder, 'News/numpy_dicts/'))

        for f_index in range(1, 50 + 1):
            mat = pd.read_csv(
                os.path.join(data_folder, 'News/csv/topic_doc_mean_n5000_k3477_seed_{}.csv.x'.format(f_index)))
            n_rows, n_cols = int(mat.columns[0]), int(mat.columns[1])
            x = np.zeros((n_rows, n_cols)).astype(int)
            for i, j, val in zip(mat.iloc[:, 0], mat.iloc[:, 1], mat.iloc[:, 2]):
                x[i - 1, j - 1] = val
            data = {}
            data['x'] = x
            meta = pd.read_csv(
                os.path.join(data_folder, 'News/csv/topic_doc_mean_n5000_k3477_seed_{}.csv.y'.format(f_index)),
                names=['t', 'y', 'y_cf', 'mu0', 'mu1'])
            for col in ['t', 'y', 'y_cf', 'mu0', 'mu1']:
                data[col] = np.array(meta[col]).reshape((-1, 1))
            with open(os.path.join(data_folder, 'News/numpy_dicts/data_as_dicts_with_numpy_seed_{}'.format(f_index)),
                      'wb') as file:
                pickle.dump(data, file)

        print('Done!')


DATASET_CLASSES = {
    'IHDP': IHDP,
    'News': News,
    'TBI': TBI
}
