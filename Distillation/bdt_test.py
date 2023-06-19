import pickle
import argparse
from collections import Counter
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')
from SoftTree import *
from train_teacher import *

def out_test_metrics(acc, report):
    report = np.mean(report, axis=0)
    print('accuracy: %6f' % np.mean(acc))
    print('precision: %6f' % report[3])
    print('recall: %6f' % report[4])
    print('f1-score: %6f' % report[5])

def softmax(x):
    if len(x.shape) > 1:
        tmp = np.max(x, axis=1)
        x -= tmp.reshape((x.shape[0], 1))
        x = np.exp(x)
        tmp = np.sum(x, axis=1)
        x /= tmp.reshape((x.shape[0], 1))
    else:
        tmp = np.max(x)
        x -= tmp
        x = np.exp(x)
        tmp = np.sum(x)
        x /= tmp
    return x

def produce_soft_labels(data, k):
    soft_label = np.zeros([data.shape[0], len(np.unique(data[:, -1]))])

    kf = KFold(n_splits=2, shuffle=True)
    for train_index, test_index in kf.split(X=data[:, :-1], y=data[:, -1], groups=data[:, -1]):
        train_set, test_set = data[train_index], data[test_index]
        train_X, train_Y, test_X = train_set[:, :-1], train_set[:, -1].astype(int), test_set[:, :-1]
        # you can change the sklearn model, e.g., GradientBoostingClassifier(...) for gbdt, MLPClassifier(...) for mlp
        clf = RandomForestClassifier(n_estimators=10, max_depth=10, max_features='sqrt')  # for rf
        clf.fit(train_X, train_Y)
        pred_prob = clf.predict_proba(test_X)
        soft_label[test_index] += pred_prob

    hard_label = np.zeros([data.shape[0], len(np.unique(data[:, -1]))])
    for i in range(np.shape(data)[0]):
        hard_label[i][int(data[i, -1])] = 1
    soft_label = soft_label * (1 - k) + hard_label * k

    return soft_label

def NN_produce_soft_labels(data, k, T, model, data_name):
    soft_label = np.zeros([data.shape[0], len(np.unique(data[:, -1]))])

    kf = KFold(n_splits=2, shuffle=True)
    for train_index, test_index in kf.split(X=data[:, :-1], y=data[:, -1], groups=data[:, -1]):
        train_set, test_set = data[train_index], data[test_index]
        train_X, train_Y = train_set[:, :-1], train_set[:, -1].astype(int)
        test_X, test_Y = test_set[:, :-1], test_set[:, -1].astype(int)
        NN = Train_Teacher(train_X, train_Y, test_X, test_Y, model, data_name)
        test_X = torch.tensor(test_X, dtype=torch.float32)
        temp = NN(test_X[0:200].cuda())
        pred_prob = temp.detach().cpu().numpy()
        for i in range(200, test_X.shape[0], 200):
            temp = NN(test_X[i:i+200].cuda())
            pred_prob = np.append(pred_prob, temp.detach().cpu().numpy(), axis=0)
        soft_label[test_index] += pred_prob

    hard_label = np.zeros([data.shape[0], len(np.unique(data[:, -1]))])
    for i in range(np.shape(data)[0]):
        hard_label[i][int(data[i, -1])] = 1
    soft_label = softmax(soft_label / T)
    soft_label = soft_label * (1 - k) + hard_label * k

    return soft_label


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch SDT Training')
    # you can use different teacher models, e.g., rf, gbdt, mlp, gru and lstm
    parser.add_argument('--teacher', default='rf', type=str, help='teacher model selection')
    parser.add_argument('--cuda', default=0, type=int, help='cuda selection')
    args = parser.parse_args()
    torch.cuda.set_device(args.cuda)
    data_name, K, T, class_num = 'univ', 0.75, 1, 2

    with open('train/dataset/path', 'rb') as tf:
        data_train = pickle.load(tf)
    with open('test/dataset/path', 'rb') as tf:
        data_eval = pickle.load(tf)
    with open('feature/attribution/path', 'rb') as tf:
        feature_attr = pickle.load(tf)  # c: continuous, d: discrete
    print('training data:', data_name, data_train.shape)
    print('testing data:', data_name, data_eval.shape)
    print('feature attr:', feature_attr[0], len(feature_attr))
    print('label of data_train:', Counter(data_train[:, -1]))
    print('label of data_eval:', Counter(data_eval[:, -1]))

    acc_sdt, sdt_report, acc_Teacher, teacher_report = [], [], [], []
    # if you are using sklearn to train teacher models, e.g., rf, gbdt and mlp,
    # you will need to specify a model in produce_soft_labels().
    # if you are using gru or lstm as the teacher model,
    # you will need to modify the model and hyperparameter setting in train_teacher.py.
    if args.teacher in ['rf', 'gbdt', 'mlp']:
        soft_label = produce_soft_labels(data_train, k=K)
    else:
        soft_label = NN_produce_soft_labels(data_train, k=K, T=T, model=args.teacher, data_name=data_name)
    
    print('==========here is sdt==========')
    clf = SoftTreeClassifier(class_num=class_num, n_features='sqrt', min_sample_leaf=20)
    clf.fit(data_train[:, :-1], soft_label, feature_attr)
    pred = clf.predict(data_eval[:, :-1])
    acc = accuracy(pred, data_eval[:, -1].astype(int))
    acc_sdt.append(acc)
    sdt_round = get_c_avg(classification_report(data_eval[:, -1], pred, output_dict=True))
    sdt_report.append(sdt_round)
    out_test_metrics(acc_sdt, sdt_report)
    model_path = './mousika_v2/{}/{}_{}_sdt.txt'.format(data_name, data_name, args.teacher)
    clf.show_tree(model_path)
