import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.utils.data as data
from time import perf_counter as t
import cycleprediction

class S_DNN(nn.Module):
    def __init__(self, size, batch_normalization=False, use_cuda=False):
        super(S_DNN, self).__init__()
        self.size = size
        self.bn = batch_normalization
        self.bn_input = nn.BatchNorm1d(size[0], momentum=0.5)
        self.hids = []
        self.bns = []

        for i in range(len(size)-2):
            hid = nn.Linear(size[i], size[i+1])
            setattr(self, 'hid%i' % i, hid)
            self._set_ini(hid)
            self.hids.append(hid)
            if self.bn:
                bn = nn.BatchNorm1d(size[i+1], momentum=0.99)
                setattr(self, 'bn%i' % i, bn)
                self.bns.append(bn)
        self.drop = nn.Dropout(0.1)
        self.pre = nn.Linear(size[-2], size[-1])
        self._set_ini(self.pre)

    def _set_ini(self, layer):
        nn.init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='relu')

    def forward(self, y):
        prea = [y]
        if self.bn:
            y = self.bn_input(y)
        layer_input = [y]
        for i in range(len(self.size)-2):
            y = self.hids[i](y)  #
            prea.append(y)
            if self.bn:
                y = self.bns[i](y)
            y = F.relu(y)
            layer_input.append(y)
            if i == 0:
                y = self.drop(y)
        y = self.pre(y)
        return y, layer_input, prea

def dnn_main(mean, scale, num_pred, num_repeat, data_n, use_cuda):
    batch_size = 128  
    training_epochs = 300 
    start_lr = 0.001

    out_dim = 3 

    train_x = data_n[:600, :] 
    train_y = data_n[1:601, -3:]
    test_x = data_n[601:700, :].type(torch.FloatTensor) 
    test_y = data_n[602:701, -3:].type(torch.FloatTensor)
    
    torch_dataset = data.TensorDataset(train_x.type(torch.FloatTensor), train_y.type(torch.FloatTensor))
    torch_dataset1 = data.TensorDataset(test_x.type(torch.FloatTensor), test_y.type(torch.FloatTensor))
    loader = data.DataLoader(
        dataset=torch_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    
    dnn_size = [train_x.size(1), 256, 128, out_dim]
    dnn = S_DNN(size=dnn_size, batch_normalization=True, use_cuda=use_cuda)

    optimizer = torch.optim.Adam(dnn.parameters(), lr=start_lr, weight_decay=1e-3)
    loss_func = nn.MSELoss()
    if torch.cuda.is_available():
        dnn.cuda()

    loss = 0
    g_step = 0

    import time

    t1 = time.time()
    loss_list = []
    for n_epoch in range(training_epochs):
        train_loss, train_acc = 0.0, 0.0
        for step, (batch_x, batch_y) in enumerate(loader):
            g_step = g_step + 1
            if use_cuda:
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
            res, layer_in, pre_act = dnn(batch_x) 
            loss = loss_func(res, batch_y)
            if torch.cuda.is_available():
                loss.cuda()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += np.sum(np.argmax(res.cpu().data.numpy(), axis=1) == np.argmax(batch_y.cpu().numpy(), 1))
            
        print('[%03d/%03d] %2.2f sec(s)  Acc: %3.6f Loss: %3.6f'
            % (n_epoch + 1, training_epochs, time.time() - t1, train_acc / torch_dataset.__len__(),
                 train_loss / torch_dataset.__len__()))
        loss_list.append(train_loss)
        
    dnn = dnn.eval()
    cycleprediction.loss_curve(data_n, mean, scale, num_pred, num_repeat, 1, True, dnn, use_cuda)
        