import random
import numpy as np
import pandas as pd
import time
#import Cython
import copy
from sklearn.model_selection import train_test_split


def cal_label_dic(label_col):
    # return a dic, key is label, value is count of label
    dic = {}
    for label in label_col:
        if label not in dic:
            dic[label] = 0
        dic[label] += 1
    return dic


def split_train_test(df, train_percent=0.8,bin=True):
    """
    @description  : divide train set and test set according to flow
    @param        : df(dtype=np.int16), tran set percent
    @Returns      : training sets and test sets that contain binary features
    """
    drop_cols = ["srcPort", "dstPort", "protocol", 'srcIP', 'dstIP',
                 "ip_ihl", "ip_tos", "ip_flags", "ip_ttl", "tcp_dataofs", "tcp_flag", "tcp_window",
                 "udp_len",
                 "length",
                 'srcAddr1', 'srcAddr2', 'srcAddr3', 'srcAddr4', 'dstAddr1', 'dstAddr2', 'dstAddr3',
                 'dstAddr4']
    for col_names in ['srcAddr{}'.format(i) for i in range(1, 5)]:
        df[col_names] = df[col_names].astype('str')
    for col_names in ['dstAddr{}'.format(i) for i in range(1, 5)]:
        df[col_names] = df[col_names].astype('str')
    df['srcIP'] = df['srcAddr1'].str.cat([df['srcAddr2'], df['srcAddr3'], df['srcAddr4']], sep='.')
    df['dstIP'] = df['dstAddr1'].str.cat([df['dstAddr2'], df['dstAddr3'], df['dstAddr4']], sep='.')
    group = df.groupby(["srcIP", "srcPort", "dstIP", "dstPort", "protocol"])

    # ngroups: the number of groups
    total_index = np.arange(group.ngroups)
    print('total flow number', len(total_index))
    np.random.seed(1234)
    np.random.shuffle(total_index)
    split_index = int(len(total_index) * train_percent)
    # ngroup(): Number each group from 0 to the number of groups - 1.
    df_train = df[group.ngroup().isin(total_index[: split_index])]
    df_test = df[group.ngroup().isin(total_index[split_index:])]
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    if bin:
        df_train.drop(drop_cols, axis=1, inplace=True)
        df_test.drop(drop_cols, axis=1, inplace=True)


    return df_train, df_test


def cal_tsallis_entropy(label_column, q):
    # Tsallis entropy as a impurity criterion
    total = len(label_column)
    label_dic = cal_label_dic(label_column)
    tsa = 0

    if q != 1:
        tmp = 0
        for k in label_dic:
            p = float(label_dic[k]) / total
            tmp += p ** q
        tsa = (tmp - 1) / (1 - q)
    else:
        for k in label_dic:
            p = float(label_dic[k]) / total
            tsa += -p*np.log(p)

    return tsa


def cal_entropy_from_histogram(h):

    l_num, r_num, l_gini, r_gini = 0, 0, 1, 1
    for c, n in h['L'].items():
        l_num += n
    for c, n in h['R'].items():
        r_num += n

    if l_num == 0:
        l_gini = 0
    else:
        for c, n in h['L'].items():
            l_gini -= (n/l_num)**2

    if r_num == 0:
        r_gini = 0
    else:
        for c, n in h['R'].items():
            r_gini -= (n / r_num) ** 2

    return (r_num/(r_num+l_num)) * r_gini + (l_num/(r_num+l_num)) * l_gini


def cal_entropy_from_histogram2(h):

    l_list = np.array(list(h['L'].items()))
    l_num = np.sum(l_list)
    l_gini = 0
    l_gini = 1-np.sum(np.square(l_list / l_num))

    r_list = np.array(list(h['R'].items()))
    r_num = np.sum(r_list)
    r_gini = 0
    r_gini = 1-np.sum(np.square(r_list / r_num))

    return (r_num/(r_num+l_num)) * r_gini + (l_num/(r_num+l_num)) * l_gini


def voting(label_dic, voting_rule="random", priority=None, random_seed=None):
    # return majority label, counts is a dic,key is label,value is counts of label
    np.random.seed(random_seed)
    winner_key = list(label_dic.keys())[0]
    for key in label_dic:
        if label_dic[key] > label_dic[winner_key]:
            winner_key = key
        elif label_dic[key] == label_dic[winner_key]:
            if voting_rule == "random":
                winner_key = np.random.choice([key, winner_key], 1)[0]  # return a list with len 1
            elif voting_rule == "prior":
                if priority[winner_key] < priority[key]:
                    winner_key = key
                elif priority[winner_key] == priority[key]:
                    winner_key = np.random.choice([key, winner_key], 1)[0]  # return a list with len 1

    return winner_key


def bootstrap(n_samples, random_seed = None):
    # generate indices of samples for training a tree

    if random_seed is not None:
        np.random.seed(random_seed)
    sample_indices = np.random.randint(0, n_samples, n_samples)
    # sample_indices = np.unique(sample_indices)
    # all_sample = np.array([i for i in range(n_samples)])
    # unsample_indices = np.delete(all_sample, sample_indices)
    unsample_indices = None
    return sample_indices, unsample_indices


def accuracy(pred, true_value):

    true_num = 0
    for i in range(len(pred)):
        if pred[i] == true_value[i]:
            true_num += 1

    return float(true_num) / len(pred)


def k_statistic(labels, r1, r2):

    dict1 = {}
    dict2 = {}
    for i in labels:
        dict1[i] = []
        dict2[i] = []

    for i in range(len(r1)):
        dict1[r1[i]].append(i)
    for i in range(len(r2)):
        dict2[r2[i]].append(i)

    c_table = {}
    for i in labels:
        t = {}
        for j in labels:
            t[j] = 0
        c_table[i] = t
    for k1 in dict1:
        for v1 in dict1[k1]:
            for k2 in dict2:
                if v1 in dict2[k2]:
                    c_table[k1][k2] += 1
                    continue
    # print(c_table)

    theta1 = 0
    for i in labels:
        theta1 += c_table[i][i]
    theta1 = theta1/len(r1)
    theta2 = 0
    for i in labels:
        factor1 = 0
        factor2 = 0
        for j in labels:
            factor1 += c_table[i][j]
            factor2 += c_table[j][i]
        theta2 += (factor1*factor2) / (len(r1)*len(r1))
    return (theta1 - theta2) / (1 - theta2)


def load_data(data_name):
    seed = 5
    feature_number = 112

    features_attr = []
    if data_name == 'univ':
        inputName = './Dataset/train.csv'
        df = pd.read_csv(inputName)

        iot_feature_names = ['srcPort', 'dstPort', 'protocol',
                             'ip_ihl', 'ip_tos', 'ip_flags', 'ip_ttl', 'tcp_dataofs', 'tcp_flag', 'tcp_window',
                             'udp_len',
                             'length',
                             'srcAddr1', 'srcAddr2', 'srcAddr3', 'srcAddr4', 'dstAddr1', 'dstAddr2', 'dstAddr3',
                             'dstAddr4']
        df.drop(columns=iot_feature_names, inplace=True)
        data = df.values
        from imblearn.under_sampling import RandomUnderSampler

        rus = RandomUnderSampler(random_state=seed)
        X, y = rus.fit_resample(data[:, :-1], data[:, -1])
        data = np.column_stack((X, y))
        data = np.random.permutation(data)
        for i in range(data.shape[1] - 1):
            features_attr.append('d')

    if data_name == 'univ_test':
        inputName = "./Dataset/test.csv"
        df = pd.read_csv(inputName)

        iot_feature_names = ['srcPort', 'dstPort', 'protocol',
                             'ip_ihl', 'ip_tos', 'ip_flags', 'ip_ttl', 'tcp_dataofs', 'tcp_flag', 'tcp_window',
                             'udp_len',
                             'length',
                             'srcAddr1', 'srcAddr2', 'srcAddr3', 'srcAddr4', 'dstAddr1', 'dstAddr2', 'dstAddr3',
                             'dstAddr4']
        df.drop(columns=iot_feature_names, inplace=True)
        data = df.values
        for i in range(data.shape[1] - 1):
            features_attr.append('d')

    for i in range(data.shape[1]):
        if isinstance(data[0, i], str):
            col = data[:, i]
            new_col = []
            for k in range(len(col)):
                if col[k] is np.nan:
                    data[k, i] = -1
                else:
                    new_col.append(col[k])
            unique_val = np.unique(new_col)
            for num in range(len(unique_val)):
                for k in range(data.shape[0]):
                    if data[k, i] == unique_val[num]:
                        data[k, i] = num
    label = dict(zip(np.unique(data[:, -1]), list(range(len(np.unique(data[:, -1]))))))
    for i in range(data.shape[0]):
        data[i][-1] = label[data[i][-1]]

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isnan(data[i][j]):
                data[i][j] = -1.0

    return data, features_attr


def get_thres(flowSize, elePercent):
    # param flowSize is DataFrame
    np_flowSize = np.array(flowSize)
    quantile = 1 - elePercent
    thres = np.quantile(np_flowSize, quantile)
    return thres


class SplitPair(list):
    def __hash__(self):
        return hash(self[0])

    def __eq__(self, other):
        return self[0] == other[0] and self[1] == other[1]


def addtodict2(thedict, key_a, key_b, val):
    if key_a in thedict:
        thedict[key_a].update({key_b: val})
    else:
        thedict.update({key_a:{key_b: val}})


def max_min_normalization(arr):
    min_ = np.min(arr)
    max_ = np.max(arr)
    if max_ - min_ == 0:
        return np.zeros(np.shape(arr))
    return (arr - min_) / (max_-min_)


def add_noise(data, percent):
    noise_index = np.random.choice(len(data), int(percent * len(data)), replace=False)
    ret_data = copy.deepcopy(data)
    labels = np.unique(data[:, -1])

    for i in noise_index:
        ret_data[i][-1] = np.random.choice(labels[labels!=data[i][-1]], 1)[0]

    return ret_data


def soft_gini(label):
    assert np.ndim(label) == 2
    label = np.array(label)
    sum = 0
    for i in range(label.shape[1]):
        sum += (np.sum(label[:, i]) / label.shape[0]) ** 2
    return 1-sum


def soft_label_dic(label):
    # return a dic, key is label, value is sum of label (软标签为属于该类的概率)
    assert np.ndim(label) == 2
    label = np.array(label)

    label_dict = {}
    for i in range(label.shape[1]):
        label_dict[i] = np.sum(label[:, i])

    return label_dict


def soft_voting(label_dic):
    winner_key = list(label_dic.keys())[0]
    for key in label_dic:
        if label_dic[key] > label_dic[winner_key]:
            winner_key = key
        elif label_dic[key] == label_dic[winner_key]:
            winner_key = np.random.choice([key, winner_key], 1)[0]

    return winner_key


def cal_gini(label_column):
    total = len(label_column)
    label_dic = cal_label_dic(label_column)
    sum = 0
    for k1 in label_dic:
        sum += (float(label_dic[k1])/total) ** 2

    return 1 - sum


def tree_node_num_our(node):
    node_num = 1
    if node.label is not None:
        return node_num
    else:
        node_num += tree_node_num_our(node.true_branch)
        node_num += tree_node_num_our(node.false_branch)
    return node_num


def get_c_avg(c):
    """
    @description  : from classification_report(ans, pred, digits=4, output_dict=True)
    @param        : classification_report
    @Returns      : macro avg:[precision, recall, f1-score], weighted avg:[precision, recall, f1-score]
    """
    m_avg_values = list(c['macro avg'].values())[:-1]
    w_avg_values = list(c['weighted avg'].values())[:-1]
    m_avg_values.extend(w_avg_values)
    return m_avg_values
