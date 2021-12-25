# encoding: utf-8


from utils import *
from Tree import *
import numpy as np
from multiprocessing import Pool
import math


class RandomForest:

    def __init__(self, n_estimators=100, min_sample_leaf=5, n_features="sqrt", q_method="fix",
                 q_value=2, q_min=None, q_max=None, q_mean=2, n_jobs=4,
                 silent=False, softmax=False, softmax_factor=1, softmax_method="fix", cutoff=None):
        self.n_estimators = n_estimators
        self.min_sample_leaf = min_sample_leaf
        self.n_features = n_features
        self.q_method = q_method
        self.q_value = q_value
        self.q_min = q_min
        self.q_max = q_max
        self.q_mean = q_mean
        self.q_list = None
        self.n_jobs = n_jobs
        self.silent = silent
        self.tree_list = None
        self.features_attr = None
        self.softmax_factor = softmax_factor
        self.softmax = softmax
        self.softmax_method = softmax_method
        self.cutoff = cutoff

    def fit(self, X, y, features_attr=None):
        # train with the features X and labels y

        if features_attr is None:
            features_attr = []
            for feature in np.transpose(X):
                if isinstance(feature[0], float):
                    features_attr.append('c')
                    continue
                if isinstance(feature[0], str):
                    features_attr.append('d')
                    continue
                unique_feature = np.unique(feature)
                if len(unique_feature) > 0.1 * len(feature):
                    features_attr.append('c')
                else:
                    features_attr.append('d')
        self.features_attr = features_attr

        self.tree_list = np.array([])

        self.__generate_q_list()
        self.__generate_ind_factor_list()

        pool = Pool(processes=self.n_jobs)
        jobs_set = []
        for i in range(self.n_estimators):
            sample_index, unsample_index = bootstrap(X.shape[0])
            new_X, new_y = X[sample_index], y[sample_index]
            tree = TreeClassifier(q_value=self.q_list[i], n_features=self.n_features,
                                  min_sample_leaf=self.min_sample_leaf,
                                  softmax=self.softmax, softmax_factor=self.softmax_factor_list[i],
                                  cutoff=self.cutoff)
            jobs_set.append(pool.apply_async(self.train_one_tree,
                                             (i, tree, new_X, new_y, self.features_attr, )))
        pool.close()
        pool.join()

        for job in jobs_set:
            self.tree_list = np.append(self.tree_list, job.get())

    @staticmethod
    def train_one_tree(id, tree, X_train, y_train, features_attr=None):

        tree.fit(X_train, y_train, features_attr)

        return tree

    def predict(self, X):
        # predict with feature(s) X
        tree_pred_res = []
        for tree in self.tree_list:
            tree_pred_res.extend([tree.predict(X)])
        tree_pred_res = np.array(tree_pred_res).T

        if np.ndim(tree_pred_res) == 1:
            return voting(cal_label_dic(tree_pred_res))
        else:
            return np.array([voting(cal_label_dic(res)) for res in tree_pred_res])

    def __generate_q_list(self):
        # for every tree, generate one q of Tsallis entropy
        if self.q_method == "exp":
            #self.q_list = np.random.exponential(self.q_mean-1, self.n_estimators)+1
            self.q_list = self.exponential_rand_list(self.q_mean, 1, self.n_estimators)
        elif self.q_method == "uniform":
            self.q_list = [np.random.uniform(self.q_min, self.q_max) for i in range(self.n_estimators)]
        elif self.q_method == "fix":
            self.q_list = [self.q_value for i in range(self.n_estimators)]
        else:
            self.q_list = [2 for i in range(self.n_estimators)]

    def __generate_ind_factor_list(self):
        if self.softmax_method == "exp":
            self.softmax_factor_list = np.random.exponential(self.softmax_factor, self.n_estimators)
        elif self.softmax_method == "fix":
            self.softmax_factor_list = [self.softmax_factor for i in range(self.n_estimators)]
        elif self.softmax_method == "rec_exp":
            mean_value = 1 / self.softmax_factor
            self.softmax_factor_list = 1 / np.random.exponential(mean_value, self.n_estimators)

    def exponential_rand(self, lam, lower_bound):
        if lam <= 0:
            return -1
        U = random.uniform(0.0, 1.0)
        return lower_bound + (-1.0 / lam) * math.log(U)

    def exponential_rand_list(self, lam, lower_bound, num):
        expo_list = []
        for i in range(num):
            expo_tmp = self.exponential_rand(lam, lower_bound)
            expo_list.append(expo_tmp)
        return expo_list
