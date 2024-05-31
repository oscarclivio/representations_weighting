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

# Many of these datasets taken from https://github.com/oscarclivio/neuralscorematching or https://github.com/BenedicteColnet/IPSW-categorical
def one_hot(input_array):
    # Determine the value of k (number of categories)
    k = input_array.max() + 1

    # Convert to one-hot encoding
    one_hot_encoding = np.eye(k)[input_array.flatten()]

    # Reshape to (n, k)
    one_hot_encoding = one_hot_encoding.reshape(-1, k)

    return one_hot_encoding


class Synthetic1Categorical(object):

    def __init__(self, **kwargs):

        self.categorical = True

        self.n = kwargs['n']
        self.d_a = kwargs['d_a']
        self.d_x = kwargs['d_x']
        self.seed = kwargs['seed']

        self.rng = np.random.default_rng(seed=self.seed)
        self.x = self.rng.normal(size=(self.n, self.d_x))

        self.W_a = self.rng.normal(size=(self.d_x + self.d_x ** 2, self.d_a))
        self.p_a = np.matmul(self.second_order_features(self.x), self.W_a)
        self.p_a = self.p_a / (self.d_x + self.d_x ** 2)
        self.p_a = scipy.special.softmax(self.p_a, axis=1)
        self.a = np.zeros((self.n, self.d_a))
        for i in range(self.n):
            self.a[i] = self.rng.multinomial(n=1, pvals=self.p_a[i])
        self.W_y = self.rng.normal(size=(self.d_x + self.d_a + self.d_x * self.d_x + self.d_x * self.d_a, 1))
        self.y = self.rng.normal(loc=self.m(self.x, self.a), scale=1)

        # Real outcomes
        taus_real = []
        size_of_a = self.d_a
        for a_val in range(size_of_a):
            a_val_mat = np.zeros((self.n, size_of_a))
            a_val_mat[:, a_val] = 1
            y_a = self.m(self.x, a_val_mat)
            taus_real.append(y_a.mean())
        self.taus_real = np.array(taus_real)

        if np.sum(np.sum(self.a, axis=0) == 0) > 0:
            raise Exception("Not enough overlap!")

        self.a_integers = np.sum(np.repeat(np.arange(self.a.shape[1]).reshape(1, -1), self.a.shape[0], axis=0) * self.a,
                                 axis=1)
        self.checksum = self.x.mean() + self.a_integers.mean() + self.y.mean()

    def m(self, x, a):
        return self.outcome(x, a, self.W_y)

    def tau(self, a):
        return np.matmul(a, self.taus_real.reshape(self.d_a, 1))

    def second_order_features(self, x):
        return np.hstack(
            (x, np.hstack([np.reshape(x[:, i] * x[:, j], (-1, 1)) for i in range(self.d_x) for j in range(self.d_x)])))

    def joint_second_order_features(self, x, a):
        return np.hstack((self.second_order_features(x), a, np.hstack(
            [np.reshape(x[:, i] * a[:, j], (-1, 1)) for i in range(self.d_x) for j in range(self.d_a)])))

    def outcome(self, x, a, W):
        return np.matmul(self.joint_second_order_features(x, a), W)


class Synthetic2Categorical(object):

    def __init__(self, **kwargs):

        self.categorical = True

        self.n = kwargs['n']
        self.d_a = kwargs['d_a']
        self.d_x = kwargs['d_x']
        self.seed = kwargs['seed']

        self.rng = np.random.default_rng(seed=self.seed)
        self.z = self.rng.normal(size=(self.n, self.d_x))

        self.x = np.zeros(self.z.shape)

        self.x[:, 0] = np.exp(self.z[:, 0] / 2)
        self.x[:, 1] = self.z[:, 1] / (1 + np.exp(self.z[:, 0])) + 10
        self.x[:, 2] = np.power(self.z[:, 0] * self.z[:, 2] / 25 + 0.6, 3)
        self.x[:, 3] = np.power(self.z[:, 1] + self.z[:, 3] + 20, 2)

        self.W_a = self.rng.normal(size=(self.d_x + self.d_x ** 2, self.d_a))
        self.p_a = np.matmul(self.second_order_features(self.z), self.W_a)
        self.p_a = self.p_a / (self.d_x + self.d_x ** 2)
        self.p_a = scipy.special.softmax(self.p_a, axis=1)
        self.a = np.zeros((self.n, self.d_a))
        for i in range(self.n):
            self.a[i] = self.rng.multinomial(n=1, pvals=self.p_a[i])
        self.W_y = self.rng.normal(size=(self.d_x + self.d_a + self.d_x * self.d_x + self.d_x * self.d_a, 1))
        self.y = self.rng.normal(loc=self.m(self.z, self.a), scale=1)

        # Real outcomes
        taus_real = []
        size_of_a = self.d_a
        for a_val in range(size_of_a):
            a_val_mat = np.zeros((self.n, size_of_a))
            a_val_mat[:, a_val] = 1
            y_a = self.m(self.z, a_val_mat)
            taus_real.append(y_a.mean())
        self.taus_real = np.array(taus_real)

        if np.sum(np.sum(self.a, axis=0) == 0) > 0:
            raise Exception("Not enough overlap!")

        self.a_integers = np.sum(np.repeat(np.arange(self.a.shape[1]).reshape(1, -1), self.a.shape[0], axis=0) * self.a,
                                 axis=1)
        self.checksum = self.z.mean() + self.a_integers.mean() + self.y.mean()

    def m(self, x, a):
        return self.outcome(x, a, self.W_y)

    def tau(self, a):
        return np.matmul(a, self.taus_real.reshape(self.d_a, 1))

    def second_order_features(self, x):
        return np.hstack(
            (x, np.hstack([np.reshape(x[:, i] * x[:, j], (-1, 1)) for i in range(self.d_x) for j in range(self.d_x)])))

    def joint_second_order_features(self, x, a):
        return np.hstack((self.second_order_features(x), a, np.hstack(
            [np.reshape(x[:, i] * a[:, j], (-1, 1)) for i in range(self.d_x) for j in range(self.d_a)])))

    def outcome(self, x, a, W):
        return np.matmul(self.joint_second_order_features(x, a), W)


class SyntheticFirstDimensionsCategorical(object):

    def __init__(self, **kwargs):

        self.categorical = True

        self.n = kwargs['n']
        self.k = kwargs['k']
        self.d_a = kwargs['d_a']
        self.d_x = kwargs['d_x']
        self.seed = kwargs['seed']

        self.rng = np.random.default_rng(seed=self.seed)
        self.x = self.rng.normal(size=(self.n, self.d_x))

        self.z = self.x[:, :self.k]

        self.W_a = self.rng.normal(size=(self.k + self.k ** 2, self.d_a))
        self.p_a = np.matmul(self.second_order_features(self.z), self.W_a)
        self.p_a = self.p_a / (self.k + self.k ** 2)
        self.p_a = scipy.special.softmax(self.p_a, axis=1)
        self.a = np.zeros((self.n, self.d_a))
        for i in range(self.n):
            self.a[i] = self.rng.multinomial(n=1, pvals=self.p_a[i])
        self.W_y = self.rng.normal(size=(self.k + self.d_a + self.k * self.k + self.k * self.d_a, 1))
        self.y = self.rng.normal(loc=self.m(self.z, self.a), scale=1)

        # Real outcomes
        taus_real = []
        size_of_a = self.d_a
        for a_val in range(size_of_a):
            a_val_mat = np.zeros((self.n, size_of_a))
            a_val_mat[:, a_val] = 1
            y_a = self.m(self.z, a_val_mat)
            taus_real.append(y_a.mean())
        self.taus_real = np.array(taus_real)

        if np.sum(np.sum(self.a, axis=0) == 0) > 0:
            raise Exception("Not enough overlap!")

        self.a_integers = np.sum(np.repeat(np.arange(self.a.shape[1]).reshape(1, -1), self.a.shape[0], axis=0) * self.a,
                                 axis=1)
        self.checksum = self.z.mean() + self.a_integers.mean() + self.y.mean()

    def m(self, x, a):
        return self.outcome(x, a, self.W_y)

    def tau(self, a):
        return np.matmul(a, self.taus_real.reshape(self.d_a, 1))

    def second_order_features(self, x):
        return np.hstack(
            (x, np.hstack([np.reshape(x[:, i] * x[:, j], (-1, 1)) for i in range(self.k) for j in range(self.k)])))

    def joint_second_order_features(self, x, a):
        return np.hstack((self.second_order_features(x), a, np.hstack(
            [np.reshape(x[:, i] * a[:, j], (-1, 1)) for i in range(self.k) for j in range(self.d_a)])))

    def outcome(self, x, a, W):
        return np.matmul(self.joint_second_order_features(x, a), W)


class SyntheticBalancingScoreCategorical(object):

    def __init__(self, **kwargs):

        self.categorical = True

        self.n = kwargs['n']
        self.score = kwargs['score']
        self.d_a = kwargs['d_a']
        self.d_x = kwargs['d_x']
        self.seed = kwargs['seed']
        self.pacoef = kwargs.get('pacoef', 1.)

        self.rng = np.random.default_rng(seed=self.seed)
        self.x = self.rng.normal(size=(self.n, self.d_x))

        self.z = self.score(self.x)
        self.k = self.z.shape[1]

        self.W_a = self.rng.normal(size=(self.k + self.k ** 2, self.d_a))
        self.p_a = np.matmul(self.second_order_features(self.z), self.W_a)
        self.p_a = self.pacoef * self.p_a / (self.k + self.k ** 2)
        self.p_a = scipy.special.softmax(self.p_a, axis=1)
        self.a = np.zeros((self.n, self.d_a))
        for i in range(self.n):
            self.a[i] = self.rng.multinomial(n=1, pvals=self.p_a[i])
        self.true_w = ((self.a.mean(axis=0) / self.p_a) * self.a).sum(axis=1)
        self.W_y = self.rng.normal(size=(self.k + self.d_a + self.k * self.k + self.k * self.d_a, 1))
        self.y = self.rng.normal(loc=self.m(self.z, self.a), scale=1)

        # Real outcomes
        taus_real = []
        size_of_a = self.d_a
        for a_val in range(size_of_a):
            a_val_mat = np.zeros((self.n, size_of_a))
            a_val_mat[:, a_val] = 1
            y_a = self.m(self.z, a_val_mat)
            taus_real.append(y_a.mean())
        self.taus_real = np.array(taus_real)

        if np.sum(np.sum(self.a, axis=0) == 0) > 0:
            raise Exception("Not enough overlap!")

        self.a_integers = np.sum(np.repeat(np.arange(self.a.shape[1]).reshape(1, -1), self.a.shape[0], axis=0) * self.a,
                                 axis=1)
        self.checksum = self.z.mean() + self.a_integers.mean() + self.y.mean()

    def m(self, x, a):
        return self.outcome(x, a, self.W_y)

    def tau(self, a):
        return np.matmul(a, self.taus_real.reshape(self.d_a, 1))

    def second_order_features(self, x):
        return np.hstack(
            (x, np.hstack([np.reshape(x[:, i] * x[:, j], (-1, 1)) for i in range(self.k) for j in range(self.k)])))

    def joint_second_order_features(self, x, a):
        return np.hstack((self.second_order_features(x), a, np.hstack(
            [np.reshape(x[:, i] * a[:, j], (-1, 1)) for i in range(self.k) for j in range(self.d_a)])))

    def outcome(self, x, a, W):
        return np.matmul(self.joint_second_order_features(x, a), W)


class Synthetic1Continuous(object):

    def __init__(self, **kwargs):
        self.categorical = False

        self.n = kwargs['n']
        self.d_a = kwargs['d_a']
        self.d_x = kwargs['d_x']
        self.seed = kwargs['seed']

        self.rng = np.random.default_rng(seed=self.seed)
        self.x = self.rng.normal(size=(self.n, self.d_x))

        self.W_a = self.rng.normal(size=(self.d_x + self.d_x ** 2, self.d_a))
        self.mu_a = np.matmul(self.second_order_features(self.x), self.W_a)
        self.mu_a = self.mu_a / (self.d_x + self.d_x ** 2)
        self.mu_a = scipy.special.softmax(self.mu_a, axis=1)
        self.a = self.rng.normal(loc=self.mu_a)
        self.W_y = self.rng.normal(size=(self.d_x + self.d_a + self.d_x * self.d_x + self.d_x * self.d_a, 1))
        self.y = self.rng.normal(loc=self.m(self.x, self.a), scale=1)

        self.checksum = self.x.mean() + self.a.mean() + self.y.mean()

    def m(self, x, a):
        return self.outcome(x, a, self.W_y)

    def tau(self, a):
        n_a = a.shape[0]
        n = self.x.shape[0]
        a_reshaped = a.reshape(n_a, 1, a.shape[1])
        a_duplicated = np.concatenate(n * [a_reshaped], axis=1)
        x_reshaped = self.x.reshape(1, n, self.x.shape[1])
        x_duplicated = np.concatenate(n_a * [x_reshaped], axis=0)

        outcomes = self.m(x_duplicated, a_duplicated)
        return outcomes.mean(axis=1)

    def second_order_features(self, x):
        return np.concatenate(
            (x, np.stack([x[..., i] * x[..., j] for i in range(self.d_x) for j in range(self.d_x)], axis=-1)), axis=-1)

    def joint_second_order_features(self, x, a):
        return np.concatenate((self.second_order_features(x), a,
                               np.stack([x[..., i] * a[..., j] for i in range(self.d_x) for j in range(self.d_a)],
                                        axis=-1)), axis=-1)

    def outcome(self, x, a, W):
        if len(a.shape) > 2:
            W = np.stack(a.shape[0] * [W], axis=0)
        return np.matmul(self.joint_second_order_features(x, a), W)


class Synthetic2Continuous(object):

    def __init__(self, **kwargs):
        self.categorical = False

        self.n = kwargs['n']
        self.d_a = kwargs['d_a']
        self.d_x = kwargs['d_x']
        self.seed = kwargs['seed']

        self.rng = np.random.default_rng(seed=self.seed)
        self.z = self.rng.normal(size=(self.n, self.d_x))

        self.x = np.zeros(self.z.shape)

        self.x[:, 0] = np.exp(self.z[:, 0] / 2)
        self.x[:, 1] = self.z[:, 1] / (1 + np.exp(self.z[:, 0])) + 10
        self.x[:, 2] = np.power(self.z[:, 0] * self.z[:, 2] / 25 + 0.6, 3)
        self.x[:, 3] = np.power(self.z[:, 1] + self.z[:, 3] + 20, 2)

        self.W_a = self.rng.normal(size=(self.d_x + self.d_x ** 2, self.d_a))
        self.mu_a = np.matmul(self.second_order_features(self.z), self.W_a)
        self.mu_a = self.mu_a / (self.d_x + self.d_x ** 2)
        self.mu_a = scipy.special.softmax(self.mu_a, axis=1)
        self.a = self.rng.normal(loc=self.mu_a)
        self.W_y = self.rng.normal(size=(self.d_x + self.d_a + self.d_x * self.d_x + self.d_x * self.d_a, 1))
        self.y = self.rng.normal(loc=self.m(self.z, self.a), scale=1)

        self.checksum = self.z.mean() + self.a.mean() + self.y.mean()

    def m(self, x, a):
        return self.outcome(x, a, self.W_y)

    def tau(self, a):
        n_a = a.shape[0]
        n = self.z.shape[0]
        a_reshaped = a.reshape(n_a, 1, a.shape[1])
        a_duplicated = np.concatenate(n * [a_reshaped], axis=1)
        x_reshaped = self.z.reshape(1, n, self.z.shape[1])
        x_duplicated = np.concatenate(n_a * [x_reshaped], axis=0)

        outcomes = self.m(x_duplicated, a_duplicated)
        return outcomes.mean(axis=1)

    def second_order_features(self, x):
        return np.concatenate(
            (x, np.stack([x[..., i] * x[..., j] for i in range(self.d_x) for j in range(self.d_x)], axis=-1)), axis=-1)

    def joint_second_order_features(self, x, a):
        return np.concatenate((self.second_order_features(x), a,
                               np.stack([x[..., i] * a[..., j] for i in range(self.d_x) for j in range(self.d_a)],
                                        axis=-1)), axis=-1)

    def outcome(self, x, a, W):
        if len(a.shape) > 2:
            W = np.stack(a.shape[0] * [W], axis=0)
        return np.matmul(self.joint_second_order_features(x, a), W)


class SyntheticFirstDimensionsContinuous(object):

    def __init__(self, **kwargs):
        self.categorical = False

        self.n = kwargs['n']
        self.d_a = kwargs['d_a']
        self.d_x = kwargs['d_x']
        self.k = kwargs['k']
        self.seed = kwargs['seed']

        self.rng = np.random.default_rng(seed=self.seed)
        self.x = self.rng.normal(size=(self.n, self.d_x))

        self.z = self.x[:, :self.k]

        self.W_a = self.rng.normal(size=(self.k + self.k ** 2, self.d_a))
        self.mu_a = np.matmul(self.second_order_features(self.z), self.W_a)
        self.mu_a = self.mu_a / (self.k + self.k ** 2)
        self.mu_a = scipy.special.softmax(self.mu_a, axis=1)
        self.a = self.rng.normal(loc=self.mu_a)
        self.W_y = self.rng.normal(size=(self.k + self.d_a + self.k * self.k + self.k * self.d_a, 1))
        self.y = self.rng.normal(loc=self.m(self.z, self.a), scale=1)

        self.checksum = self.z.mean() + self.a.mean() + self.y.mean()

    def m(self, x, a):
        return self.outcome(x, a, self.W_y)

    def tau(self, a):
        n_a = a.shape[0]
        n = self.z.shape[0]
        a_reshaped = a.reshape(n_a, 1, a.shape[1])
        a_duplicated = np.concatenate(n * [a_reshaped], axis=1)
        x_reshaped = self.z.reshape(1, n, self.z.shape[1])
        x_duplicated = np.concatenate(n_a * [x_reshaped], axis=0)

        outcomes = self.m(x_duplicated, a_duplicated)
        return outcomes.mean(axis=1)

    def second_order_features(self, x):
        return np.concatenate(
            (x, np.stack([x[..., i] * x[..., j] for i in range(self.k) for j in range(self.k)], axis=-1)), axis=-1)

    def joint_second_order_features(self, x, a):
        return np.concatenate((self.second_order_features(x), a,
                               np.stack([x[..., i] * a[..., j] for i in range(self.k) for j in range(self.d_a)],
                                        axis=-1)), axis=-1)

    def outcome(self, x, a, W):
        if len(a.shape) > 2:
            W = np.stack(a.shape[0] * [W], axis=0)
        return np.matmul(self.joint_second_order_features(x, a), W)


class IHDP1Categorical(object):

    def __init__(self, data_path="../data", **kwargs):

        self.categorical = True

        self.d_a = kwargs['d_a']
        self.seed = kwargs['seed']

        self.rng = np.random.default_rng(seed=self.seed)
        self.x = np.array(pd.read_csv(os.path.join(data_path, 'ihdp/ihdp.csv')))
        self.n = self.x.shape[0]
        self.d_x = self.x.shape[1]
        self.x_cont = self.x[:, :6]
        self.x_cat = self.x[:, 6:]
        self.d_x_cont = self.x_cont.shape[1]
        self.d_x_cat = self.x_cat.shape[1]

        n_dimensions = self.d_x_cat + self.d_x_cont + int(
            self.d_x_cont * (self.d_x_cont + 1) / 2) + self.d_x_cont * self.d_x_cat
        self.W_a = self.rng.normal(size=(n_dimensions, self.d_a))
        self.p_a = np.matmul(self.joint_second_order_features(self.x_cont, self.x_cat), self.W_a)
        self.p_a = self.p_a / (self.d_x + self.d_x ** 2)
        self.p_a = scipy.special.softmax(self.p_a, axis=1)
        self.a = np.zeros((self.n, self.d_a))
        for i in range(self.n):
            self.a[i] = self.rng.multinomial(n=1, pvals=self.p_a[i])

        n_dimensions_single_covariate = self.d_x_cat + 2 * self.d_x_cont
        n_dimensions_interactions = +int(self.d_x_cont * (self.d_x_cont - 1) / 2) + self.d_x_cont * self.d_x_cat
        self.W_y_single_covariate = self.rng.choice([0, 1, 2], p=[.6, .3, .1],
                                                    size=(n_dimensions_single_covariate, self.d_a))
        self.W_y_interactions = self.rng.choice([0, .5, 1], p=[.8, .15, .05],
                                                size=(n_dimensions_interactions, self.d_a))
        self.W_y = np.concatenate([self.W_y_single_covariate, self.W_y_interactions], axis=0)
        self.y = self.rng.normal(loc=self.m(self.x_cont, self.x_cat, self.a), scale=1).reshape(self.n, 1)
        # Real outcomes
        ctes = self.ctes(self.x_cont, self.x_cat)
        self.taus_real = ctes.mean(axis=0)

        if np.sum(np.sum(self.a, axis=0) == 0) > 0:
            raise Exception("Not enough overlap!")

        self.a_integers = np.sum(np.repeat(np.arange(self.a.shape[1]).reshape(1, -1), self.a.shape[0], axis=0) * self.a,
                                 axis=1)
        self.checksum = self.x.mean() + self.a_integers.mean() + self.y.mean()

    def ctes(self, x_cont, x_cat):
        return self.outcome(x_cont, x_cat, self.W_y)

    def m(self, x_cont, x_cat, a):
        ctes = self.ctes(x_cont, x_cat)
        return np.sum(ctes * a, axis=-1)

    def tau(self, a):
        return np.matmul(a, self.taus_real.reshape(self.d_a, 1))

    def second_order_features(self, x_cont):
        d_x_cont = x_cont.shape[1]
        return np.hstack((x_cont, np.hstack(
            [np.reshape(x_cont[:, i] * x_cont[:, j], (-1, 1)) for i in range(d_x_cont) for j in range(i, d_x_cont)])))

    def joint_second_order_features(self, x_cont, x_cat):
        d_x_cont = x_cont.shape[1]
        d_x_cat = x_cat.shape[1]
        return np.hstack((x_cat, self.second_order_features(x_cont), np.hstack(
            [np.reshape(x_cont[:, i] * x_cat[:, j], (-1, 1)) for i in range(d_x_cont) for j in range(d_x_cat)])))

    def outcome(self, x_cont, x_cat, W):
        return np.matmul(self.joint_second_order_features(x_cont, x_cat), W)


class IHDP1Continuous(object):

    def __init__(self, data_path="../data", **kwargs):
        self.categorical = False

        self.d_a = kwargs['d_a']
        self.seed = kwargs['seed']

        self.rng = np.random.default_rng(seed=self.seed)
        self.x = np.array(pd.read_csv(os.path.join(data_path, 'ihdp/ihdp.csv')))
        self.n = self.x.shape[0]
        self.d_x = self.x.shape[1]
        self.x_cont = self.x[:, :6]
        self.x_cat = self.x[:, 6:]
        self.d_x_cont = self.x_cont.shape[1]
        self.d_x_cat = self.x_cat.shape[1]

        n_dimensions = self.d_x_cat + self.d_x_cont + int(
            self.d_x_cont * (self.d_x_cont + 1) / 2) + self.d_x_cont * self.d_x_cat
        self.W_a_beta_min = self.rng.normal(size=(n_dimensions, self.d_a))
        self.W_a_beta_max = self.rng.normal(size=(n_dimensions, self.d_a))

        self.beta_min = 1 + np.exp(
            np.matmul(self.joint_second_order_features(self.x_cont, self.x_cat), self.W_a_beta_min))
        self.beta_max = 1 + np.exp(
            np.matmul(self.joint_second_order_features(self.x_cont, self.x_cat), self.W_a_beta_max))
        self.a = self.rng.beta(self.beta_min, self.beta_max)

        n_dimensions_single_covariate = self.d_x_cat + 2 * self.d_x_cont
        n_dimensions_interactions = +int(self.d_x_cont * (self.d_x_cont - 1) / 2) + self.d_x_cont * self.d_x_cat
        self.W_y_single_covariate = self.rng.choice([0, 1, 2], p=[.6, .3, .1],
                                                    size=(n_dimensions_single_covariate, self.d_a))
        self.W_y_interactions = self.rng.choice([0, .5, 1], p=[.8, .15, .05],
                                                size=(n_dimensions_interactions, self.d_a))
        self.W_y = np.concatenate([self.W_y_single_covariate, self.W_y_interactions], axis=0)
        self.y = self.rng.normal(loc=self.m(self.x_cont, self.x_cat, self.a), scale=1).reshape(self.n, 1)

        if np.sum(np.sum(self.a, axis=0) == 0) > 0:
            raise Exception("Not enough overlap!")

        self.checksum = self.x.mean() + self.a.mean() + self.y.mean()

    def ctes(self, x_cont, x_cat):
        return self.outcome(x_cont, x_cat, self.W_y)

    def m(self, x_cont, x_cat, a):
        ctes = self.ctes(x_cont, x_cat)
        return np.sum(ctes * a, axis=-1)

    def tau(self, a):
        n_a = a.shape[0]
        n = self.x.shape[0]
        a_reshaped = a.reshape(n_a, 1, a.shape[1])
        a_duplicated = np.concatenate(n * [a_reshaped], axis=1)
        x_cont_reshaped = self.x_cont.reshape(1, n, self.x_cont.shape[1])
        x_cont_duplicated = np.concatenate(n_a * [x_cont_reshaped], axis=0)
        x_cat_reshaped = self.x_cat.reshape(1, n, self.x_cat.shape[1])
        x_cat_duplicated = np.concatenate(n_a * [x_cat_reshaped], axis=0)

        outcomes = self.m(x_cont_duplicated, x_cat_duplicated, a_duplicated)
        return outcomes.mean(axis=1)

    def second_order_features(self, x_cont):
        return np.concatenate((x_cont, np.stack(
            [x_cont[..., i] * x_cont[..., j] for i in range(self.d_x_cont) for j in range(i, self.d_x_cont)], axis=-1)),
                              axis=-1)

    def joint_second_order_features(self, x_cont, x_cat):
        return np.concatenate((x_cat, self.second_order_features(x_cont), np.stack(
            [x_cont[..., i] * x_cat[..., j] for i in range(self.d_x_cont) for j in range(self.d_x_cat)], axis=-1)),
                              axis=-1)

    def outcome(self, x_cont, x_cat, W):
        return np.matmul(self.joint_second_order_features(x_cont, x_cat), W)


class ACIC2016(object):
    def __init__(self, setting=4, seed=1, one_hot_factors=True, path_data="../data", scale=False):
        self.path_data = path_data
        self.one_hot_factors = one_hot_factors
        self.covariates = pd.read_csv(os.path.join(path_data, 'ACIC2016/covariates.csv'))
        self.info = pd.read_csv(os.path.join(path_data, 'ACIC2016/info.csv'))
        self.dgp_data = pd.read_csv(
            os.path.join(path_data, f'ACIC2016/dgp_data/setting{setting}_dataset{seed + 1}.csv'))
        self.X_df = self._process_covariates(self.covariates)  # turn factor variables into one-hot binary variables
        self.n = len(self.X_df)

        attrs = {}
        attrs['x'] = np.array(self.X_df)
        attrs['t'] = np.array(self.dgp_data['z']).reshape((-1, 1))
        attrs['y'] = np.array(self.dgp_data['y']).reshape((-1, 1))
        attrs['y0'] = np.array(self.dgp_data['y.0']).reshape((-1, 1))
        attrs['y1'] = np.array(self.dgp_data['y.1']).reshape((-1, 1))
        attrs['mu0'] = np.array(self.dgp_data['mu.0']).reshape((-1, 1))
        attrs['mu1'] = np.array(self.dgp_data['mu.1']).reshape((-1, 1))
        attrs['cate_true'] = attrs['mu1'] - attrs['mu0']
        attrs['ps'] = np.array(self.dgp_data['e']).reshape((-1, 1))

        # Find binary covariates
        self.binary = []
        for ind in range(attrs['x'].shape[1]):
            self.binary.append(len(np.unique(attrs['x'][:, ind])) == 2)
        self.binary = np.array(self.binary)

        # Normalise - continuous data
        self.scale = scale
        self.xm = np.zeros(self.binary.shape)
        self.xs = np.ones(self.binary.shape)
        if self.scale:
            raise NotImplementedError
            # self.xm[~self.binary] = np.mean(attrs['x'][itrva][:,~self.binary], axis=0)
            # self.xs[~self.binary] = np.std(attrs['x'][itrva][:,~self.binary], axis=0)
        attrs['x'] -= self.xm
        attrs['x'] /= self.xs

        # Subsample data and convert to torch.Tensor with the right device
        for key, value in attrs.items():
            setattr(self, key, value)

        # TODO : set a in one hot format
        self.a = one_hot(self.t)

        # TODO : self.taus_real
        self.taus_real = np.array([self.mu0.mean(), self.mu1.mean()]).flatten()

        # TODO : create true_w
        self.true_ws = np.zeros((self.n, 2))
        self.true_ws[:, 0] = self.a[:, 0].mean() / self.ps.flatten()
        self.true_ws[:, 1] = self.a[:, 1].mean() / (1 - self.ps).flatten()
        self.true_w = (self.true_ws * self.a).sum(axis=1)

        self.checksum = self.x.mean() + self.t.sum() + self.y.mean()
        self.categorical = True

        # ATT format
        self.xp = self.x[self.t.flatten() == 0, :]
        self.xq = self.x[self.t.flatten() == 1, :]
        self.pseudo_y = self.y[self.t.flatten() == 0, :]
        self.tau_real = self.mu0[self.t.flatten() == 1, :].mean()

    def tau(self, a):
        return np.matmul(a, self.taus_real.reshape(2, 1))

    def _process_covariates(self, covariates):
        covariates_done = {}
        for ind, covariate_name in enumerate(covariates.columns):
            if not 'x_' in covariate_name:
                continue
            if pd.to_numeric(covariates[covariate_name], errors='coerce').notnull().all():
                covariates_done[covariate_name] = covariates[covariate_name]
            else:
                if self.one_hot_factors:
                    for item in sorted(pd.unique(covariates[covariate_name])):
                        covariates_done[covariate_name + '_' + item] = (covariates[covariate_name] == item).astype(int)
                else:
                    covariates_done[covariate_name] = pd.Series([0] * len(covariates[covariate_name]))
                    for idx, item in sorted(enumerate(pd.unique(covariates[covariate_name]))):
                        covariates_done[covariate_name][covariates[covariate_name] == item] = idx

        return pd.DataFrame(covariates_done)


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


class Jobs(object):
    def __init__(self, seed=0, dataset='train', path_data="../data"):
        self.path_data = os.path.join(path_data, f'Jobs/{dataset}')

        attrs = {}

        for attr in ['x', 't', 'yf', 'e']:
            attrs[attr] = np.load(os.path.join(self.path_data, f'{attr}.npy'))
        for key, value in attrs.items():
            value = value[..., seed]
            if len(value.shape) == 1:
                value = value.reshape((-1, 1))
            setattr(self, key, value)
        self.t = self.t.astype(int)

        # ATT format
        self.xp = self.x[self.t.flatten() == 0, :]
        self.xq = self.x[self.t.flatten() == 1, :]
        self.pseudo_y = self.yf[self.t.flatten() == 0, :]
        self.tau_real = self.yf[(self.t.flatten() == 1) & (self.e.flatten() == 1), :].mean() - self.yf[(
                                                                                                                   self.t.flatten() == 0) & (
                                                                                                                   self.e.flatten() == 1),
                                                                                               :].mean()

        self.checksum = self.x.mean() + self.t.sum() + self.yf.mean()
        self.categorical = True


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
    'Synthetic1Categorical': Synthetic1Categorical,
    'Synthetic1Continuous': Synthetic1Continuous,
    'Synthetic2Categorical': Synthetic2Categorical,
    'Synthetic2Continuous': Synthetic2Continuous,
    'IHDP1Categorical': IHDP1Categorical,
    'IHDP1Continuous': IHDP1Continuous,

    'SyntheticFirstDimensionsCategorical': SyntheticFirstDimensionsCategorical,
    'SyntheticFirstDimensionsContinuous': SyntheticFirstDimensionsContinuous,

    'SyntheticBalancingScoreCategorical': SyntheticBalancingScoreCategorical,

    'ACIC2016': ACIC2016,
    'IHDP': IHDP,
    'News': News,
    'Jobs': Jobs,
    'TBI': TBI
}
