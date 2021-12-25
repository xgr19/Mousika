# encoding: utf-8

from models import *
import torch
import torch.utils.data as Data
import numpy as np
from sklearn.model_selection import train_test_split
from utils import *
# torch.manual_seed(1)    # reproducible
import torch.nn.functional as F



def select_model(T,input_size,output_size):
    return GRU(input_size, output_size)

def Train_Teacher(X,y,model_name,data_name):
    input_size = X.shape[1]
    output_size = len(np.unique(y))

    model = select_model(model_name, input_size, output_size)
    model = model.cuda()
    BATCH_SIZE=256
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=20, test_size=0.2)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test)

    train_datasets = Data.TensorDataset(X_train, y_train)
    test_datasets = Data.TensorDataset(X_test, y_test)

    train_loader = Data.DataLoader(dataset=train_datasets, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = Data.DataLoader(dataset=test_datasets, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    # Hyper Parameters
    EPOCH = 10             # train the training data n times, to save time, we just train 1 epoch
    LR = 0.001              # learning rate
    best_testing_acc = 0.0
    best_epoch=0

    train_acc=0.0


    model_path = "../params/Teacher_"+data_name+"_model_"+model_name+".pkl"
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)   # optimize all model parameters
    loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted
    # training and testing

    for epoch in range(EPOCH):
        for step, (b_x, b_y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
            b_x=b_x.cuda()
            b_y=b_y.cuda()
            output = model(b_x)               # model output
            output= F.softmax(output,dim=1)
            loss = loss_func(output, b_y)   # cross entropy loss
            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients

            train_pred_y = torch.max(output, 1)[1].cpu().data.numpy()
            train_acc += 100.0 * float((train_pred_y == np.array(b_y.cpu().view(-1).data)).sum())/ len(b_y)

            if step % 100 == 0:
                correct = 0.
                for batch_idx, (data, target) in enumerate(test_loader):
                    data=data.cuda()
                    test_output = model(data)
                    pred_y = torch.max(test_output, 1)[1].cpu().data.numpy()

                    correct +=  (pred_y==np.array(target.view(-1).data)).sum()

                accuracy = 100.0 * float(correct) / len(test_loader.dataset)

                #print('Epoch: ', epoch, '| train loss: %.4f' % loss.cpu().data.numpy(),'| train acc: %.4f' % float(train_acc/100), '| test accuracy: %.2f' % accuracy)
                train_acc=0.0

                if accuracy > best_testing_acc:
                    best_epoch=epoch
                    best_testing_acc = accuracy
                    torch.save(model.state_dict(), model_path)
        #print('epoch| {:} best training acc {:}'.format(epoch,best_testing_acc))
        '''
        correct = 0
        for batch_idx, (data, target) in enumerate(eval_loader):
            data = data.cuda()
            test_output = model(data)
            pred_y = torch.max(test_output, 1)[1].cpu().data.numpy()

            correct += (pred_y == np.array(target.view(-1).data)).sum()

        acc = float(correct) / len(eval_loader.dataset)

        print(model_name + '| test accuracy: %.4f' % acc)
        #train_acc = 0.0
        '''

    print('epoch| {:} best training acc {:}'.format(best_epoch, best_testing_acc))
    model.load_state_dict(torch.load(model_path))
    return  model, best_testing_acc

#BATCH_SIZE = 50
# Load data
#load=LoadData(BATCH_SIZE)
#train_loader,test_loader=load.get_univ_data()

#cnn = CNN()
#mlp=MLP(128,2)
#mlp=mlp.cuda()
#print(cnn)  # net architecture
#Train_Teacher(train_loader,test_loader,mlp)
'''
data_name='iot'
model_name='lstm'
data_train, feature_attr = load_data('iot_bin')

data_eval, feature_attr=load_data('iot_test_bin')


test_X = data_eval[:, :-1]
test_X = torch.tensor(test_X, dtype=torch.float32)
test_Y=data_eval[:, -1].astype(int)
test_Y = torch.tensor(test_Y)

test_datasets = Data.TensorDataset(test_X, test_Y)
eval_loader = Data.DataLoader(dataset=test_datasets, batch_size=128, shuffle=True, num_workers=2)
NN,best_acc=Train_Teacher(data_train[:, :-1], data_train[:, -1].astype(int),model_name,data_name)
'''