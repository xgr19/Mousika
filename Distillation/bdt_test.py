# encoding: utf-8



import warnings
warnings.filterwarnings('ignore')
import imp
from Tree import *
from utils import *
from RandomForest import *
import torch
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import argparse
import time
from train_teacher import *
from SoftTree import *
from models import *
import time
from sklearn.metrics import classification_report
import torch.nn.functional as F


def softmax(x):
    """
    对输入x的每一行计算softmax。

    该函数对于输入是向量（将向量视为单独的行）或者矩阵（M x N）均适用。

    代码利用softmax函数的性质: softmax(x) = softmax(x + c)

    参数:
    x -- 一个N维向量，或者M x N维numpy矩阵.

    返回值:
    x -- 在函数内部处理后的x
    """
    orig_shape = x.shape

    # 根据输入类型是矩阵还是向量分别计算softmax
    if len(x.shape) > 1:
        # 矩阵
        tmp = np.max(x, axis=1)  # 得到每行的最大值，用于缩放每行的元素，避免溢出。 shape为(x.shape[0],)
        x -= tmp.reshape((x.shape[0], 1))  # 利用性质缩放元素
        x = np.exp(x)  # 计算所有值的指数
        tmp = np.sum(x, axis=1)  # 每行求和
        x /= tmp.reshape((x.shape[0], 1))  # 求softmax
    else:
        # 向量
        tmp = np.max(x)  # 得到最大值
        x -= tmp  # 利用最大值缩放数据
        x = np.exp(x)  # 对所有元素求指数
        tmp = np.sum(x)  # 求元素和
        x /= tmp  # 求somftmax
    return x


def produce_soft_labels(data, round_num, fold_num, k=1,model='rf'):

    soft_label = np.zeros([data.shape[0], len(np.unique(data[:, -1]))])

    for i in range(round_num):
        kf = KFold(n_splits=fold_num)
        for train_index, test_index in kf.split(X=data[:, :-1], y=data[:, -1], groups=data[:, -1]):
            train_set, test_set = data[train_index], data[test_index]
            train_X,train_Y=train_set[:, :-1],train_set[:, -1].astype(int)
            test_X=test_set[:, :-1]
            if model=='rf':
                clf = RandomForestClassifier(300, min_samples_leaf=5, criterion="gini")
            clf.fit(train_X, train_Y)

            pred_prob = clf.predict_proba(test_X)
            soft_label[test_index] += pred_prob

    soft_label /= round_num

    hard_label = np.zeros([data.shape[0], len(np.unique(data[:, -1]))])
    for i in range(np.shape(data)[0]):
        hard_label[i][int(data[i, -1])] = 1

    soft_label = (soft_label + hard_label*k) / (k+1)

    return soft_label

def NN_produce_soft_kf(data, round_num, fold_num, k=1,T=1,model='mlp',data_name='iot'):
    #input_size = data.shape[1] - 1
    output_size = len(np.unique(data[:, -1]))
    soft_label = np.zeros([data.shape[0], output_size])


    for i in range(round_num):
        kf = KFold(n_splits=fold_num)
        for train_index, test_index in kf.split(X=data[:, :-1], y=data[:, -1], groups=data[:, -1]):
            train_set, test_set = data[train_index], data[test_index]
            train_X, train_Y = train_set[:, :-1], train_set[:, -1].astype(int)
            test_X = test_set[:, :-1]
            NN,best_acc=Train_Teacher(train_X,train_Y,model,data_name)

            test_X = torch.tensor(test_X, dtype=torch.float32)

            temp = NN(test_X[0:1000].cuda())
            pred_prob = temp.detach().cpu().numpy()
            for i in range(1000, test_X.shape[0], 1000):
                temp = NN(test_X[i:i + 1000].cuda())
                pred_prob = np.append(pred_prob, temp.detach().cpu().numpy(), axis=0)

            soft_label[test_index] += pred_prob

    soft_label /= round_num

    hard_label = np.zeros([data.shape[0], len(np.unique(data[:, -1]))])
    for i in range(np.shape(data)[0]):
        hard_label[i][int(data[i, -1])] = 1

    soft_label = softmax(soft_label / T)
    soft_label = (soft_label + hard_label*k) / (k+1)

    return soft_label



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch SDT Training')
    parser.add_argument(
        '--teacher', default='rf', choices=['rf', 'gru'], type=str, help='teacher model selection')
    parser.add_argument(
        '--cuda', default=1, choices=[0, 1, 2, 3], type=int, help='cuda selection')
    parser.add_argument(
        '--K', default=1,  type=int, help='the proportion of hard label')
    parser.add_argument(
        '--T', default=1, type=int, help='the temperature of soft label')

    args = parser.parse_args()
    # torch.cuda.set_device(args.cuda)

    ROUND_NUM = 5
    TEST_SIZE = 0.3
    #MAX_T=[1,2,3,4,5,6]
    MAX_K = 1

    for data_name in ['univ']:
        data_train, feature_attr = load_data(data_name)
        data_eval, feature_attr=load_data('univ_test')
        print('training data:')
        print(data_name, data_train.shape)

        #kk=args.K
        T=1
        output = []
        teacher_output = []
        sdt_output = []
        for kk in range(MAX_K):
            print("--------------", kk, "--------------")
            acc_sdt, acc_dt, acc_Teacher, sdt_time, dt_time,   sdt_test_time,  NN_time = [], [], [], [], [], [], []
            teacher_report, sdt_report = [], []
            for i in range(ROUND_NUM):
                print("ROUND:", str(i))


                begin_time = time.time()
                if args.teacher == 'rf':
                    soft_label = produce_soft_labels(data_train, round_num=1, fold_num=2, k=kk, model='rf')
                else:
                    soft_label = NN_produce_soft_kf(data_train, round_num=1, fold_num=2, k=kk, T=T, model=args.teacher,
                                                    data_name=data_name)
                end_time = time.time()
                print('produce soft label needs {:}s'.format(end_time - begin_time))

                if args.teacher == 'rf':
                    # random forest
                    clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, criterion="gini")
                    clf.fit(data_train[:, :-1], data_train[:, -1].astype(int))
                    pred = clf.predict(data_eval[:, :-1])
                    acc_Teacher.append(accuracy(pred, data_eval[:, -1].astype(int)))
                else:  # NN
                    begin_time = time.time()
                    NN, best_acc = Train_Teacher(data_train[:, :-1], data_train[:, -1].astype(int), args.teacher,
                                                 data_name)
                    end_time = time.time()
                    t1 = end_time - begin_time
                    NN_time.append(t1)
                    print('training {:} needs {:}s'.format(args.teacher, t1))
                    test_X = data_eval[:, :-1]
                    test_X = torch.tensor(test_X, dtype=torch.float32)
                    test_Y = data_eval[:, -1].astype(int)
                    test_Y = torch.tensor(test_Y)

                    test_datasets = Data.TensorDataset(test_X, test_Y)
                    test_loader = Data.DataLoader(dataset=test_datasets, batch_size=128, shuffle=False, num_workers=2)
                    correct = 0
                    pred = np.array([])
                    for batch_idx, (data_X, target) in enumerate(test_loader):
                        data_X = data_X.cuda()
                        test_output = NN(data_X)
                        pred_y = torch.max(test_output, 1)[1].cpu().data.numpy()
                        pred = np.concatenate([pred, pred_y], axis=0)
                        correct += (pred_y == np.array(target.view(-1).data)).sum()
                    acc = float(correct) / len(test_loader.dataset)
                    print(args.teacher + '| test accuracy: %.4f' % acc)
                    train_acc = 0.0
                    acc_Teacher.append(acc)
                teacher_round = get_c_avg(classification_report(data_eval[:, -1], pred, digits=4, output_dict=True))
                teacher_report.append(teacher_round)

                # soft decision tree
                begin_time = time.time()
                clf = SoftTreeClassifier(n_features="all", min_sample_leaf=5)
                clf.fit(data_train[:, :-1], soft_label, feature_attr)
                end_time = time.time()
                t = end_time - begin_time
                sdt_time.append(t)
                print('training sdt needs {:}s'.format(t))

                begin_time = time.time()
                pred = clf.predict(data_eval[:, :-1])
                acc = accuracy(pred, data_eval[:, -1])
                end_time = time.time()
                t = end_time - begin_time
                sdt_test_time.append(t)
                print('sdt testing needs {:}s'.format(t))
                print("  soft decision tree:", acc)
                acc_sdt.append(acc)



                sdt_round = get_c_avg(classification_report(data_eval[:, -1], pred, digits=4, output_dict=True))
                sdt_report.append(sdt_round)

                model_path = './rule_tree/{}_{}_kk{}_round{}.txt'.format(data_name, args.teacher, str(kk),str(i))
                clf.show_tree(model_path)
                print("teacher_report", teacher_round)
                print("sdt_report", sdt_round)

                begin_time = time.time()
                clf = TreeClassifier(n_features="all", min_sample_leaf=5)
                clf.fit(data_train[:, :-1], data_train[:, -1], feature_attr)

                end_time = time.time()
                t1 = end_time - begin_time
                dt_time.append(t1)
                print('training BDT needs {:}s'.format(t1))

                pred = clf.predict(data_eval[:, :-1])
                acc = accuracy(pred, data_eval[:, -1].astype(int))
                print('BDT accuracy {:}%'.format(acc))
                acc_dt.append(acc)

                if data_name == "mnist" and i == 2:
                    break
            print("       {:} Classifier:".format(args.teacher), np.mean(acc_Teacher))
            print("  binary decision tree:", np.mean(acc_dt))
            print("  soft decision tree mean acc:", np.mean(acc_sdt))
            print("  soft decision tree mean time cost:", np.mean(sdt_time))
            print('  BDT mean time cost ', np.mean(dt_time))
            print('  NN mean time cost ', np.mean(NN_time))

            output.append(
                [kk, np.mean(acc_Teacher), np.mean(acc_dt), np.mean(acc_sdt), np.mean(sdt_time),
                 np.mean(dt_time),  np.mean(NN_time)])
            teacher_output.append(np.mean(teacher_report, axis=0))
            sdt_output.append(np.mean(sdt_report, axis=0))

        np.savetxt("./result_acc/" + data_name + "_" + args.teacher + "_acc.txt", np.array(output))
        np.savetxt("./result_acc/" + data_name + "_" + args.teacher + "_teacher.txt", np.array(teacher_output))
        np.savetxt("./result_acc/" + data_name + "_" + args.teacher + "_sdt.txt", np.array(sdt_output))
