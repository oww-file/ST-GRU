import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error as mse

def predict(net, testX, n_input, dnn, use_cuda=False):
    outputSteps, in_dim = 1, n_input
    testX = testX.view(-1, in_dim) if dnn else testX.view(-1, outputSteps, in_dim)
    if use_cuda:
        net = net.cuda()
        testX = testX.cuda()
    if dnn:
        pred, layer_in, pre_act = net(testX)
        return pred
    else:
        return net(testX)

def to_torch(a, new_start, batch_size, data_n, st):
    x_t = torch.from_numpy(a[-1].cpu().data.numpy())
    x_st = torch.cat((data_n[new_start + batch_size, :-3], a[-1].cpu().flatten()), 0).type(torch.FloatTensor)
    return x_st if st else x_t

def to_actual(i, scale, mean, out_dim):
    return i.cpu().view(-1, out_dim).data.numpy() * scale + mean

def initial(new_start, s, batch_size, n_input, data_n, net, st, dnn, use_cuda):
    test_x = data_n[new_start:new_start + batch_size, -n_input:].type(torch.FloatTensor)
    p = predict(net, test_x, n_input, dnn, use_cuda)
    test_x = to_torch(p, new_start, batch_size, data_n, st)
    test_y = data_n[new_start + batch_size + 1:700 + s, -3:].type(torch.FloatTensor)
    return test_x, test_y, p, test_y.size(1)

def cal_mean(val):
    return np.mean(np.array(val),axis=0)

def loss_curve(data_n, mean_0, scale_0, num_pred, num_repeat, st, dnn, net, use_cuda=False):
    if st == 0:
        plt.figure(1, figsize=(12, 5))
        plt.ion()

    mean_draw, mean_pre = [], []
    start_pos, batch_size = 0, 599   
    cycle_loss = 0
    for s in range(num_pred):
        new_start = start_pos + s
        n_input = 12 if st else 3 
        
        test_x, test_y, p, out_dim = initial(new_start, s, batch_size, n_input, data_n, net, st, dnn, use_cuda)
        test_y = torch.from_numpy(to_actual(test_y, scale_0, mean_0, out_dim))
        steps = np.linspace(1, test_y.size(0), test_y.size(0), dtype=np.float32, endpoint=False)
        loss_draw_all, mean_prep = [], []
        loss_avg = 0

        p_0 = predict(net, test_x, n_input, dnn, use_cuda)
        p_val = torch.from_numpy(to_actual(p_0, scale_0, mean_0, out_dim))
        for n in range(num_repeat):
            p_pred = to_actual(p, scale_0, mean_0, out_dim)
            mean_pred = np.mean(p_pred, axis=0)
            p_1 = p_0 

            loss = mse(p_val, test_y[0].view(-1, out_dim)).item()
            mean_predt = mse(mean_pred, test_y[0].view(-1, 1))
            sum_loss = loss
            loss_draw, mean_prer = [], []
            
            for i in range(test_y.size(0) - 1):
                loss_draw.append(loss)
                mean_prer.append(mean_predt)
                cycle_x = to_torch(p_1, new_start + i + 1, batch_size, data_n, st)
                cycle_x = cycle_x.cuda() if use_cuda else cycle_x
                p_1 = predict(net, cycle_x, n_input, dnn, use_cuda)
                p_2 = to_actual(p_1, scale_0, mean_0, out_dim)
                p_pred = np.concatenate((p_pred, p_2), axis=0)
                p_2 = torch.from_numpy(p_2)
                loss = mse(p_2, test_y[i + 1].view(-1, out_dim)).item()
                
                mean_pred = np.mean(p_pred, axis=0)
                mean_predt = mse(mean_pred, test_y[i + 1].view(-1, 1))
                sum_loss += loss

            loss_avg += sum_loss / test_y.size(0)
            mean_prer.append(mean_predt)
            loss_draw.append(loss)
            loss_draw_all.append(loss_draw)
            mean_prep.append(mean_prer)

        cycle_loss += loss_avg / num_repeat
        mean_loss = cal_mean(loss_draw_all)
        mean_draw.append(mean_loss)
        mean_prep = cal_mean(mean_prep)
        mean_pre.append(mean_prep)
    
    mse_draw = cal_mean(mean_draw)
    mean_mse = cal_mean(mean_pre)
    mean_loss = cal_mean(mean_mse)
    print('MSE loss', cycle_loss / num_pred, 'mean prediction', mean_loss)
    
    if st:
        if dnn:
            plt.plot(steps, mse_draw, 'y-', label='S-DNN')
        else:
            plt.plot(steps, mse_draw, 'g-', label='ST_GRU')
            plt.plot(steps, mean_mse, 'r-', label='Mean Prediction') 
    else:
        plt.plot(steps, mse_draw, 'b-', label='T_GRU')
    




