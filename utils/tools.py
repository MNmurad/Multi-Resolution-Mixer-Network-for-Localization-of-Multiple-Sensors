import numpy as np
from scipy.interpolate import interp1d
import torch
import torch.nn as nn
import random
import pandas as pd
import matplotlib.pyplot as plt
torch.set_printoptions(precision = 10)

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj=='type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch-1) // 1))}
    elif args.lradj=='type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    
    elif args.lradj == 'type4':
        lr_adjust = {epoch: args.learning_rate if epoch < 2 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'type5':
        lr_adjust = {epoch: args.learning_rate if epoch % 10 == 0 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
         
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout: print('Updating learning rate to {}'.format(lr))

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'\tEarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'\tValidation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'/'+'checkpoint.pth')
        # torch.save(model, './'+'checkpoint.pth')
        self.val_loss_min = val_loss

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

   
class StandardScalerTensor(nn.Module):
    def __init__(self):
        super(StandardScalerTensor, self).__init__()
        # self.device = device
        self.mean = torch.tensor([0.0])
        self.std = torch.tensor([1.0])
        self.c = torch.tensor([1e-7])
        # self.register_buffer('mean', torch.tensor([0.0]))
        # self.register_buffer('std', torch.tensor([1.0]))
        # self.register_buffer('c', torch.tensor([0.0]))
        
    def fit(self, data):
        # self.mean = data.mean(0)
        # self.std = data.std(0)
        self.mean = torch.from_numpy(data.mean(0)).to(dtype = torch.float)
        self.std = torch.from_numpy(data.std(0)).to(dtype = torch.float)
        

    def transform(self, data):
        return (data - self.mean) / (self.std + self.c)

    def inverse_transform(self, data):
        return data * (self.std + self.c) + self.mean

class StandardScaler():
    def __init__(self):
        self.mean = torch.tensor([0.0])
        self.std = torch.tensor([1.0])
        self.c = torch.tensor([0.0])
        
    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)
        self.c = self.c.numpy() if not torch.is_tensor(data) else self.c

    # def transform(self, data):
    #     mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
    #     std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
    #     c = self.c if torch.is_tensor(data) else self.c.item()
    #     return (data - mean) / (std + c)

    # def inverse_transform(self, data):
    #     mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
    #     std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
    #     c = self.c if torch.is_tensor(data) else self.c.item()
    #     if data.shape[-1] != mean.shape[-1]:
    #         mean = mean[-1:]
    #         std = std[-1:]
    #     return data * (std + c) + mean
    def transform(self, data):
        return (data - self.mean) / (self.std + self.c)

    def inverse_transform(self, data):
        return data * (self.std + self.c) + self.mean
    

def save_to_csv(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    data = pd.DataFrame({'true': true, 'preds': preds})
    data.to_csv(name, index=False, sep=',')


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def visual_weights(weights, name='./pic/test.pdf'):
    """
    Weights visualization
    """
    fig, ax = plt.subplots()
    # im = ax.imshow(weights, cmap='plasma_r')
    im = ax.imshow(weights, cmap='YlGnBu')
    fig.colorbar(im, pad=0.03, location='top')
    plt.savefig(name, dpi=500, pad_inches=0.02)
    plt.close()


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


class Permute(nn.Module):
    def __init__(self, *dims):
        super(Permute, self).__init__()
        self.dims = dims  # The new order of dimensions

    def forward(self, x):
        return x.permute(*self.dims)
    
class Reshape(nn.Module):
    def __init__(self, *dims):
        super(Reshape, self).__init__()
        self.dims = dims  # The new order of dimensions

    def forward(self, x):
        return x.reshape(*self.dims)

class PreProcessing:
    def __init__(self, rejection_points = 0, interPred = 'linear', envelope = 'upper'):
        # rejection_points: how many conjucative peak/trough points should be rejected to get the next one
        # interPred type: ‘linear’, ‘nearest’, ‘nearest-up’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, or ‘next’
        # envelope: 'upper', 'lower'
        self.rejection_points = rejection_points
        self.interPred = interPred
        self.envelope = envelope
        
    def process(self, data):
        # data: 2d array: (samples, channel)
        # output: 2d array: (samples, channel)
        n, c = data.shape # n: samples, c: channel
        output = np.zeros_like(data)
        
        for i in range(c):
            output[:, i] = self.getEnvelopeSeries(data[:, i])[0] if self.envelope == 'upper' else self.getEnvelopeSeries(data[:, i])[1]
        return output
        
    def getEnvelopeSeries(self, data):   
        # x: 1d array. like list
        # upped envelope
        upper_x = [0,]
        upper_y = [data[0],]    
        lastPeak = 0;
        
        # lower envelope
        lower_x = [0,]
        lower_y = [data[0],]
        lastTrough = 0;
           
        for k in range(1,len(data)-1):     
            if (np.sign(data[k]-data[k-1])==1) and (np.sign(data[k]-data[k+1])==1) and ((k-lastPeak)>self.rejection_points ):
                upper_x.append(k)
                upper_y.append(data[k])    
                lastPeak = k;
                
            #Mark troughs
            if (np.sign(data[k]-data[k-1])==-1) and ((np.sign(data[k]-data[k+1]))==-1) and ((k-lastTrough)>self.rejection_points ):
                lower_x.append(k)
                lower_y.append(data[k])
                lastTrough = k
        
        upper_x.append(len(data)-1)
        upper_y.append(data[-1])
        
        lower_x.append(len(data)-1)
        lower_y.append(data[-1])
        
        #Fit suitable models to the data. Here cubic splines.    
        u_p = interp1d(upper_x, upper_y, kind = self.interPred, bounds_error = False, fill_value=0.0) # interpreted model
        l_p = interp1d(lower_x, lower_y, kind = self.interPred, bounds_error = False, fill_value=0.0) # interpreted model
        
        data_index = range(0, len(data))
        output_upper = u_p(data_index) # upper envelope series: 1d array
        output_lower = l_p(data_index) # lower envelope series: 1d array
        return output_upper, output_lower


def set_random_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
