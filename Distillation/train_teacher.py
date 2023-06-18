from models import *
import torch
import torch.utils.data as Data
from utils import *
import torch.nn.functional as F

def select_model(model_name, input_size, output_size):
    if model_name == 'gru':
        return GRU(input_size, output_size)

def Train_Teacher(X_train, y_train, X_test, y_test, model_name, data_name):
    input_size, output_size = X_train.shape[1], len(np.unique(y_train))
    model = select_model(model_name, input_size, output_size)

    model = model.cuda()
    BATCH_SIZE = 16
    EPOCH = 10
    LR = 0.001

    X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train)
    X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test)
    train_datasets, test_datasets = Data.TensorDataset(X_train, y_train), Data.TensorDataset(X_test, y_test)
    train_loader = Data.DataLoader(dataset=train_datasets, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = Data.DataLoader(dataset=test_datasets, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    best_epoch = 0
    best_test_acc = 0.0
    model_path = 'params/Teacher_' + data_name + '_model_' + model_name + '.pkl'
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(EPOCH):
        train_acc = 0.0
        for step, (b_x, b_y) in enumerate(train_loader):
            b_x, b_y = b_x.cuda(), b_y.cuda()
            output = model(b_x)
            output = F.softmax(output, dim=1)
            loss = loss_func(output, b_y.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_pred_y = torch.max(output, 1)[1].cpu().data.numpy()
            train_acc += 100.0 * float((train_pred_y == np.array(b_y.cpu().view(-1).data)).sum()) / len(b_y)

            if step > 0 and step % 100 == 0:
                correct = 0
                for batch_idx, (data, target) in enumerate(test_loader):
                    data, target = data.cuda(), target.cuda()
                    test_output = model(data)
                    pred_y = torch.max(test_output, 1)[1].cpu().data.numpy()
                    correct += (pred_y == np.array(target.cpu().view(-1).data)).sum()
                test_acc = 100.0 * float(correct) / len(test_loader.dataset)

                print('Epoch: ', epoch, '| train loss: %.6f' % loss.item(),
                      '| train acc: %.6f' % float(train_acc / 100), '| test acc: %.6f' % test_acc)
                train_acc = 0.0

                if test_acc > best_test_acc:
                    best_epoch = epoch
                    best_test_acc = test_acc
                    torch.save(model.state_dict(), model_path)
    print('Epoch: ', best_epoch, '| best test acc: %.6f' % float(best_test_acc / 100))
    model.load_state_dict(torch.load(model_path))
    return model
