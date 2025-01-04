import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_printoptions(precision = 10)

from utils.tools import Permute, Reshape
from utils.RevIN import RevIN

import matplotlib.pyplot as plt
import numpy as np
from models.wavelet_patch_mixer import WPMixerCore
    

class BridgeModel(nn.Module): # single pwoer model for all nodes
    def __init__(self,
                  c_in = [], 
                  c_out = [],
                  seq_len = [],
                  out_len = [], 
                d_model = [],  
                dropout = [], 
                embedding_dropout = [],
                device = [],
                batch_size = [],
                tfactor = [],
                dfactor = [],
                wavelet = [],
                level = [],
                patch_len = [],
                stride = [],
                no_decomposition = [],
                use_amp = [],
                n_nodes = [],
                n_interrogators = [],
                positionScaler = [],
                powerScaler = []
                ):
        super(BridgeModel, self).__init__()
        self.pred_len = out_len
        self.channel_in = c_in
        self.channel_out = c_out
        self.patch_len = patch_len
        self.stride = stride
        self.seq_len = seq_len
        self.d_model = d_model
        self.dropout = dropout
        self.embedding_dropout = embedding_dropout
        self.batch_size = batch_size # not required now
        self.tfactor = tfactor
        self.dfactor = dfactor
        self.wavelet = wavelet
        self.level = level
        # patch predictior
        self.actual_seq_len = seq_len
        self.no_decomposition = no_decomposition
        self.use_amp = use_amp
        self.device = device
        self.n_nodes = n_nodes
        self.n_interrogators = n_interrogators
        
        self.positionModel = WPMixer(c_in = self.channel_in,
                                      c_out = self.channel_out,
                                      seq_len = self.seq_len,
                                      out_len = self.pred_len, 
                                      d_model = self.d_model,  
                                      dropout = self.dropout, 
                                      embedding_dropout = self.embedding_dropout,
                                      device = self.device,
                                      batch_size = self.batch_size,
                                      tfactor = self.tfactor,
                                      dfactor = self.dfactor,
                                      wavelet = self.wavelet,
                                      level = self.level,
                                      patch_len = self.patch_len,
                                      stride = self.stride,
                                      no_decomposition = self.no_decomposition,
                                      use_amp = self.use_amp,
                                      )
        
        self.corr_factor_weight = torch.nn.Parameter(torch.ones(self.n_nodes * 2, ).to(self.device)) # considering only (x, y) coordinates of nodes
        self.corr_factor_bias = torch.nn.Parameter(torch.zeros(self.n_nodes * 2, ).to(self.device)) # considering only (x, y) coordinates of nodes
        
        self.scaler_for_position = positionScaler
        
        self.position_scaler_mean = nn.Parameter(torch.tensor(self.scaler_for_position.mean).to(dtype = torch.float), requires_grad=False) 
        self.position_scaler_std = nn.Parameter(torch.tensor(self.scaler_for_position.std).to(dtype = torch.float), requires_grad=False)
        self.position_scaler_c = nn.Parameter(torch.tensor(self.scaler_for_position.c).to(dtype = torch.float), requires_grad=False)
        
    def forward(self, x):
        position_scaled = self.positionModel(x)
        revised_position_scaled = position_scaled * self.corr_factor_weight + self.corr_factor_bias
        return position_scaled, revised_position_scaled, None # estimated_power_scaled
    
    def transform(self, x, mean, std, c):
        return (x - mean) /(std + c)
    
    def inverse_transform(self, x, mean, std, c):
        return x * (std + c) + mean

    
class WPMixer(nn.Module):
    def __init__(self,
                 c_in = [], 
                 c_out = [],
                 seq_len = [],
                 out_len = [], 
                d_model = [],  
                dropout = [], 
                embedding_dropout = [],
                device = [],
                batch_size = [],
                tfactor = [],
                dfactor = [],
                wavelet = [],
                level = [],
                patch_len = [],
                stride = [],
                no_decomposition = [],
                use_amp = []):
        
        super(WPMixer, self).__init__()
        self.pred_len = out_len
        self.channel_in = c_in
        self.channel_out = c_out
        self.patch_len = patch_len
        self.stride = stride
        self.seq_len = seq_len
        self.d_model = d_model
        self.dropout = dropout
        self.embedding_dropout = embedding_dropout
        self.batch_size = batch_size # not required now
        self.tfactor = tfactor
        self.dfactor = dfactor
        self.wavelet = wavelet
        self.level = level
        # patch predictior
        self.actual_seq_len = seq_len
        self.no_decomposition = no_decomposition
        self.use_amp = use_amp
        self.device = device
        
        self.wpmixerCore = WPMixerCore(input_length = self.actual_seq_len,
                                                      pred_length = self.pred_len,
                                                      wavelet_name = self.wavelet,
                                                      level = self.level,
                                                      batch_size = self.batch_size,
                                                      channel_in = self.channel_in, 
                                                      channel_out = self.channel_out,
                                                      d_model = self.d_model, 
                                                      dropout = self.dropout, 
                                                      embedding_dropout = self.embedding_dropout,
                                                      tfactor = self.tfactor, 
                                                      dfactor = self.dfactor, 
                                                      device = self.device,
                                                      patch_len = self.patch_len, 
                                                      patch_stride = self.stride,
                                                      no_decomposition = self.no_decomposition,
                                                      use_amp = self.use_amp)
        
        
    def forward(self, x):
        pred = self.wpmixerCore(x)
        pred = pred[:, :, -self.channel_out:]
        return pred 

