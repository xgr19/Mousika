from __future__ import print_function
from utils import *
from DecisionNode import *
import numpy as np

FEATURE_THRESHOLD = 1e-7

class SoftTreeClassifier:
    def __init__(self, class_num=2, n_features=None, min_sample_leaf=5):
        self.root = None
        self.min_sample_leaf = min_sample_leaf
        self.n_features = n_features
        self.features_attr = None
        self.class_num = class_num

    def fit(self, X, y, features_attr=None):
        np.random.seed()
        self.features_attr = features_attr

        if self.n_features is None or self.n_features == "all":
            self.n_features = X.shape[1]
        elif self.n_features == "sqrt":
            self.n_features = int(np.sqrt(X.shape[1]))
        elif self.n_features == "half":
            self.n_features = int(0.5 * X.shape[1])
        elif self.n_features == "sqrt(nlogn)":
            self.n_features = int(np.sqrt(X.shape[1] * np.log(X.shape[1])))

        X, y = np.array(X), np.array(y)
        self.root = self.__build_tree(X, y)

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
                if feat_value == threshold:
                    return self.__predict_rec(X, node.true_branch)
                else:
                    return self.__predict_rec(X, node.false_branch)

    def __split(self, dataset, split_feature, threshold):
        true_index = []
        false_index = []
        for i in range(len(dataset)):
            if dataset[i][split_feature] == threshold:
                true_index.append(i)
            else:
                false_index.append(i)

        return true_index, false_index

    def __split_pair(self, X, y, candidate_features):
        attr = self.features_attr[0]
        ret = []
        for feat in candidate_features:
            dataset = np.concatenate((X, y), axis=1)
            end = len(dataset)
            index = np.where(dataset[:, feat] == 0)[0]
            left_samples = len(index)
            right_samples = end - left_samples
            left_sum, right_sum = 0, 0
            for class_id in range(self.class_num):
                left_class_sum = np.sum(dataset[index, class_id - self.class_num])
                left_sum += left_class_sum ** 2
                right_sum += np.sum(dataset[:, class_id - self.class_num]) - left_class_sum
            left_gini = 1 - left_sum / (left_samples ** 2)
            right_gini = 1 - right_sum / (right_samples ** 2)
            gain = - (left_samples / end * left_gini + right_samples / end * right_gini)
            ret.append([gain, feat, 0])
            ret.append([gain, feat, 1])
        ret = np.array(ret)

        return ret[np.argsort(-ret[:, 0])]

    def __build_tree(self, X, y):

        y_dict = soft_label_dic(y)

        if np.sum((y > (1.0 / y.shape[1])).astype(int), axis=0).max() == y.shape[0]:
            d = y_dict
            l = soft_voting(y_dict)
            return DecisionNode(label_dict=d, label=l)

        candidate_features = []
        for i in range(X.shape[1]):
            if len(np.unique(X[:, i])) > 1:
                candidate_features.append(i)

        if candidate_features == []:
            d = y_dict
            l = soft_voting(y_dict)
            return DecisionNode(label_dict=d, label=l)

        candidate_features = np.random.choice(candidate_features,
                                              min(self.n_features, len(candidate_features)), replace=False)

        split_pair = self.__split_pair(X, y, candidate_features)

        split_feature, threshold = int(split_pair[0][1]), split_pair[0][2]

        true_index, false_index = self.__split(X, split_feature, threshold)

        if len(true_index) == 0 or len(false_index) == 0:
            d = y_dict
            l = soft_voting(y_dict)
            return DecisionNode(label_dict=d, label=l)

        if len(true_index) <= self.min_sample_leaf:
            y_true_dict = soft_label_dic(y[true_index])
            d = y_true_dict
            l = soft_voting(y_true_dict)
            true_branch = DecisionNode(label_dict=d, label=l)
        else:
            true_branch = self.__build_tree(X[true_index], y[true_index])

        if len(false_index) <= self.min_sample_leaf:
            y_false_dict = soft_label_dic(y[false_index])
            d = y_false_dict
            l = soft_voting(y_false_dict)
            false_branch = DecisionNode(label_dict=d, label=l)
        else:
            false_branch = self.__build_tree(X[false_index], y[false_index])

        return DecisionNode(feature=split_feature, threshold=threshold, label_dict=y_dict,
                            true_branch=true_branch, false_branch=false_branch)

    def find_path(self, data, attr_map):

        path = []
        for d in data:
            p = []
            path.append(self.root.find_path(d, p, self.features_attr, attr_map))
        return path

    def show_tree(self, model_path):
        path = []
        attr_map = []
        self.root.show_tree(path, self.features_attr, attr_map, model_path)

    def get_tree_max_depth_and_nodes_count(self, depth=0):
        max_depth = self.root.get_tree_max_depth_and_nodes_count(depth)
        return max_depth
