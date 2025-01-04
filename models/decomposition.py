
import torch
import torch.nn as nn
from pytorch_wavelets import DWT1DForward, DWT1DInverse
from utils.RevIN import RevIN

class Decomposition(nn.Module):
    def __init__(self,
                 input_length = [], 
                 pred_length = [],
                 wavelet_name = [],
                 level = [],
                 batch_size = [],
                 channel_in = [],
                 channel_out = [],
                 d_model = [],
                 tfactor = [],
                 dfactor = [],
                 device = [],
                 no_decomposition = [],
                 use_amp = []):
        super(Decomposition, self).__init__()
        self.input_length = input_length
        self.pred_length = pred_length
        self.wavelet_name = wavelet_name
        self.level = level
        self.batch_size = batch_size
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.d_model = d_model
        self.device = device
        self.no_decomposition = no_decomposition
        self.use_amp = use_amp
        self.eps = 1e-5
        
        self.dwt = DWT1DForward(wave = self.wavelet_name, J = self.level, use_amp = self.use_amp).cuda() if self.device.type == 'cuda' else DWT1DForward(wave = self.wavelet_name, J = self.level, use_amp = self.use_amp)
        self.idwt = DWT1DInverse(wave = self.wavelet_name, use_amp = self.use_amp).cuda() if self.device.type == 'cuda' else DWT1DInverse(wave = self.wavelet_name, use_amp = self.use_amp)

        self.input_w_dim = self._dummy_forward(self.input_length) if not self.no_decomposition else [self.input_length] # length of the input seq after decompose
        self.pred_w_dim = self._dummy_forward(self.pred_length) if not self.no_decomposition else [self.pred_length] # required length of the pred seq after decom
        
        self.tfactor = tfactor
        self.dfactor = dfactor
        
            
    def transform(self, x):
        # input: x shape: batch, channel, seq
        if not self.no_decomposition:
            yl, yh = self._wavelet_decompose(x)
        else:
            yl, yh = x, [] # no decompose: returning the same value in yl
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot(x[0, 0, :].detach().cpu())
        # plt.plot(x[0, 1, :].detach().cpu())
        # plt.figure()
        # plt.plot(yh[0][0, 1, :].detach().cpu())
        # plt.draw()
        # plt.figure()
        # plt.plot(yh[1][0, 1, :].detach().cpu())
        # plt.draw()
        # plt.figure()
        # plt.plot(yh[2][0, 1, :].detach().cpu())
        # plt.draw()
        # plt.figure()
        # plt.plot(yh[3][0, 1, :].detach().cpu())
        # plt.draw()
        # plt.figure()
        # plt.plot(yh[4][0, 1, :].detach().cpu())
        # plt.draw()
        # plt.figure()
        # plt.plot(yl[0, 1, :].detach().cpu())
        # plt.draw()
        return yl, yh
    
    def inv_transform(self, yl, yh):
        if not self.no_decomposition:
            x = self._wavelet_reverse_decompose(yl, yh)
        else:
            x = yl # no decompose: returning the same value in x
        return x
           
    def _dummy_forward(self, input_length):
        dummy_x = torch.ones((self.batch_size, self.channel_out, input_length)).to(self.device)
        yl, yh = self.dwt(dummy_x)
        l = []
        l.append(yl.shape[-1])
        for i in range(len(yh)):
            l.append(yh[i].shape[-1])
        return l
    
    def _wavelet_decompose(self, x):
        # input: x shape: batch, channel, seq
        yl, yh = self.dwt(x)
        return yl, yh
    
    def _wavelet_reverse_decompose(self, yl, yh):
        x = self.idwt((yl, yh))
        return x # shape: batch, channel, seq
