# encoding: utf-8


from __future__ import print_function
from utils import *
from DecisionNode import *
import numpy as np


class TreeClassifier:

    def __init__(self, n_features=None, min_sample_leaf=5):
        self.root = None
        self.min_sample_leaf = min_sample_leaf
        self.n_features = n_features
        self.features_attr = None

    def fit(self, X, y, features_attr=None):

        # feature_atrr: an array, its size is same as the number of features,
        # 'd' 'is discrete', 'c' is continuous

        np.random.seed()
        self.features_attr = features_attr

        if self.n_features is None or self.n_features == "all":
            self.n_features = X.shape[1]
        elif self.n_features == "sqrt":
            self.n_features = int(np.sqrt(X.shape[1]))
        elif self.n_features == "half":
            self.n_features = int(0.5 * X.shape[1])
        elif self.n_features == "sqrt(nlogn)":
            self.n_features = int(np.sqrt(X.shape[1]*np.log(X.shape[1])))

        dataset = np.concatenate((np.array(X), np.array([y]).T), axis=1)
        self.root = self.__build_ind_tree_rec(dataset.astype(np.double))

    def predict(self, X):

        if np.ndim(X) == 1:
            return self.__predict_rec(X, self.root)
        else:
            result = []
            for sample in X:
                result.append(self.__predict_rec(sample, self.root))
            return np.array(result)

    def __predict_rec(self, X, node):
        if node.label is not None:
            return node.label
        else:
            feat_value = X[node.feature]
            feat_attr = self.features_attr[node.feature]
            threshold = node.threshold

            if feat_value is None or feat_value is np.nan:
                choice = np.random.randint(1, 3)
                if choice == 1:
                    return self.__predict_rec(X, node.true_branch)
                else:
                    return self.__predict_rec(X, node.false_branch)
            else:
                if feat_attr == 'd':
                    if feat_value == threshold:
                        return self.__predict_rec(X, node.true_branch)
                    else:
                        return self.__predict_rec(X, node.false_branch)
                elif feat_attr == 'c':
                    if feat_value >= threshold:
                        return self.__predict_rec(X, node.true_branch)
                    else:
                        return self.__predict_rec(X, node.false_branch)

    def __split(self, dataset, split_feature, threshold):

        true_index = []
        false_index = []

        if self.features_attr[split_feature] == 'd':
            for i in range(len(dataset)):
                if dataset[i][split_feature] == threshold:
                    true_index.append(i)
                else:
                    false_index.append(i)
        elif self.features_attr[split_feature] == 'c':
            for i in range(len(dataset)):
                if dataset[i][split_feature] >= threshold:
                    true_index.append(i)
                else:
                    false_index.append(i)

        return true_index, false_index

    def __split_pair(self, dataset, candidate_features):

        current_gini = cal_gini(dataset[:, -1])

        ret = []

        for feat in candidate_features:
            col = dataset[:, feat]
            unique_col = np.unique(col)
            attr = self.features_attr[feat]

            threshold_list = []
            if attr == 'd' or unique_col.shape == 1:
                threshold_list = unique_col
            elif attr == 'c':
                threshold_list = [(unique_col[i]+unique_col[i+1]) / 2 for i in range(len(unique_col)-1)]

            for t in threshold_list:
                true_index, false_index = self.__split(dataset, feat, t)
                p = float(len(true_index)) / len(dataset)
                next_gini = p * cal_gini(dataset[true_index, -1]) + \
                           (1-p) * cal_gini(dataset[false_index, -1])
                gain = current_gini - next_gini
                ret.append([gain, feat, t])
        ret = np.array(ret)
        return ret[np.argsort(-ret[:, 0])]

    def __build_ind_tree_rec(self, dataset):

        y_dict = cal_label_dic(dataset[:, -1])

        if len(y_dict) == 1:
            d = y_dict
            l = voting(y_dict)
            return DecisionNode(label_dict=d, label=l)

        candidate_features = []
        for i in range(dataset.shape[1]-1):
            if len(np.unique(dataset[:, i])) > 1:
                candidate_features.append(i)
        if candidate_features == []:
            d = y_dict
            l = voting(y_dict)
            return DecisionNode(label_dict=d, label=l)

        candidate_features = np.random.choice(candidate_features,
                                              min(self.n_features, len(candidate_features)), replace=False)

        split_pair = self.__split_pair(dataset, candidate_features)

        split_feature, threshold = int(split_pair[0][1]), split_pair[0][2]

        true_index, false_index = self.__split(dataset, split_feature, threshold)

        if len(true_index) == 0 or len(false_index) == 0:
            d = y_dict
            l = voting(y_dict)
            return DecisionNode(label_dict=d, label=l)

        if len(true_index) <= self.min_sample_leaf:
            y_true_dict = cal_label_dic(dataset[true_index, -1])
            d = y_true_dict
            l = voting(y_true_dict)
            true_branch = DecisionNode(label_dict=d, label=l)
        else:
            true_branch = self.__build_ind_tree_rec(dataset[true_index])

        if len(false_index) <= self.min_sample_leaf:
            y_false_dict = cal_label_dic(dataset[false_index, -1])
            d = y_false_dict
            l = voting(y_false_dict)
            false_branch = DecisionNode(label_dict=d, label=l)
        else:
            false_branch = self.__build_ind_tree_rec(dataset[false_index])

        return DecisionNode(feature=split_feature, threshold=threshold, label_dict=y_dict,
                            true_branch=true_branch, false_branch=false_branch)
    def show_tree(self,model_path):
        path=[]
        attr_map=[]
        self.root.show_tree(path, self.features_attr, attr_map,model_path)


