import argparse
import scipy.io
from time import perf_counter as t
import yaml
from yaml import SafeLoader
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.utils.data as data
import numpy as np
from torch.nn import init
from torch.autograd import Variable
from sklearn import preprocessing
import cycleprediction
import DNNprediction


class GRU(nn.Module):
    def __init__(self, input_dim, hidden_num, output_dim, layer_num, use_cuda):
        super(GRU, self).__init__()
        self.use_cuda = use_cuda
        self.layer_num = layer_num
        self.hidden_num = hidden_num
        self.input_dim = input_dim
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_num,  
            num_layers= layer_num,  
            dropout= 0.1,
            batch_first=True,  
        )
        self.out = nn.Linear(hidden_num, output_dim)
        optim_range = np.sqrt(1.0 / hidden_num) 
        self.weightInit(optim_range)

    def forward(self, x):
        h0 = torch.zeros(self.layer_num, x.size(0), self.hidden_num).requires_grad_()
        h0 = h0.cuda() if self.use_cuda else h0
        gru_out, _ = self.gru(x, h0.detach())
        p_out = self.out(gru_out)
        return p_out

    def weightInit(self, gain=1):
        for name, param in self.named_parameters():
            if 'gru.weight' in name:
                init.orthogonal_(param, gain)

               
def predict(net, testX, use_cuda=False):
    net = net.eval()
    testX = testX.view(-1, outputSteps, train_x.size(1))
    if use_cuda:
        testX = testX.cuda()
        net = net.cuda()
    return net(testX)


if __name__ == '__main__':
    use_cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

    torch.manual_seed(125)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(125)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ST-GRU')
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    config = yaml.load(open(args.config), Loader=SafeLoader)[args.model]

    learning_rate = config['learning_rate']
    num_hidden = config['num_hidden']
    outputSteps = config['outputSteps']
    data_type = config['data_type']
    numStartPoints = config['numStartPoints']
    num_repeat = config['numRepeat']
    max_epochs = config['maxEpochs']
    numPredSteps = config['numPredSteps']
    weight_decay = config['weight_decay']
    batch_size = config['batch_size']

    ori_data = scipy.io.loadmat('./measuresPhantom.mat')
    ori = torch.from_numpy(ori_data[data_type]).T
    scaler = preprocessing.StandardScaler().fit(ori)
    mean = scaler.mean_[-3:]
    scale = scaler.scale_[-3:]
    data_n = torch.from_numpy(scaler.transform(ori))

    type = 2
    fig = plt.figure(1, figsize=(12, 5))
    plt.ion()
    for ty in range(type):
        train_x = train_x = data_n[0:600, :] if ty else data_n[0:600, -3:]
        train_y = data_n[1:601, -3:]
        out_dim = train_y.size(1)

        torch_dataset = data.TensorDataset(train_x.type(torch.FloatTensor), train_y.type(torch.FloatTensor))
        loader = data.DataLoader(
            dataset=torch_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        net = GRU(train_x.size(1), num_hidden, out_dim, layer_num=2, use_cuda=use_cuda)
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.MSELoss()  
        if torch.cuda.is_available():
            net.cuda()
    
        start = t()
        prev = start
        
        for epoch in range(max_epochs):
            train_acc = 0.0
            val_acc = 0.0
            for batch_idx, (x, y) in enumerate(loader):
                
                x, y = Variable(x), Variable(y)
                x = x.view(-1, outputSteps, train_x.size(1))
                y = y.view(-1, outputSteps, out_dim)
                y_actual = y.cpu().view(-1, out_dim) * scale + mean
                if use_cuda:
                    x = x.cuda()
                    y = y.cuda()
    
                prediction = net(x)
                optimizer.zero_grad()
                loss = criterion(prediction, y)
                if torch.cuda.is_available():
                    loss.cuda()
    
                loss.backward()
                optimizer.step()
    
                now = t()
                loss_val = loss.item()
                print(f'(T) | Epoch={epoch:03d}, loss={loss_val:.4f}, '
                      f'this epoch {now - prev:.4f}, total {now - start:.4f}')
                
                prev = now
                
            print("%d epoch is finished!" % (batch_idx + 1))
            
        net = net.eval()

        cycleprediction.loss_curve(data_n, mean, scale, numPredSteps, num_repeat, ty, 0, net, use_cuda)
        
        color = 'g-'
        la = 'T_GRU'
        if ty:
            color = 'r-'
            la = 'ST_GRU'
            DNNprediction.dnn_main(mean, scale, numPredSteps, num_repeat, data_n, use_cuda)
        plt.draw();
        plt.legend()
        
    plt.xlabel('Predicted steps (n)')
    plt.ylabel('MSE (mm)')
    plt.title('Phantom dataset')
    plt.savefig("phantom.png")
    
    
    
   

