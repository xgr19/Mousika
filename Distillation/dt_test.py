# encoding: utf-8

# @author: Dongyutao


import warnings

warnings.filterwarnings('ignore')
from Tree import *
from utils import *
from RandomForest import *
from SoftTree import *
import time
import numpy as np
from sklearn.metrics import classification_report


if __name__ == "__main__":

    ROUND_NUM = 5
    TEST_SIZE = 0.3
    MAX_K = 1

    for data_name in ['univ_C']:  #
        data_train, feature_attr = load_data(data_name)

        data_eval, feature_attr = load_data('univ_test_C')
        print('training data:')

        print('\n testing data:')
        print(data_eval.shape)

        output = []
        for kk in range(MAX_K):
            print("--------------", kk, "--------------")
            acc_hdt, hdt_time, hdt_test_time = [], [], []
            hdt_report = []
            for i in range(ROUND_NUM):
                print("ROUND:", str(i))

                # hand decision tree
                begin_time = time.time()
                clf = TreeClassifier(n_features="all", min_sample_leaf=5)
                clf.fit(data_train[:, :-1], data_train[:, -1], feature_attr)

                end_time = time.time()
                t1 = end_time - begin_time
                hdt_time.append(t1)
                print('training dt needs {:}s'.format(t1))

                begin_time = time.time()
                pred = clf.predict(data_eval[:, :-1])
                acc = accuracy(pred, data_eval[:, -1].astype(int))
                end_time = time.time()
                t1 = end_time - begin_time
                hdt_test_time.append((t1))
                print('dt accuracy {:}%'.format(acc))
                acc_hdt.append(acc)

                model_path = './rule_tree/{}_round{}_DT.txt'.format(data_name, str(i))
                clf.show_tree(model_path)

                hdt_round = get_c_avg(classification_report(data_eval[:, -1], pred, digits=4, output_dict=True))
                hdt_report.append(hdt_round)

            print('acc_hdt', acc_hdt)
            output.append([kk, np.mean(acc_hdt),  np.mean(hdt_time), np.mean(hdt_test_time)])

        np.savetxt("./result_acc/" + data_name + "_DT_Acc.txt", np.array(output))
        np.savetxt("./result_acc/" + data_name + "_DT_report.txt", np.mean(hdt_report, axis=0))
