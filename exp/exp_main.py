from exp.exp_basic import Exp_Basic
from models.model import BridgeModel
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
from utils.misc_functions import plot_custom, plot_custom2
from data_provider.data_factory import data_provider

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import os
import time
import optuna
import copy
torch.set_printoptions(precision = 10)

import warnings
warnings.filterwarnings('ignore')
from thop import profile
import pandas as pd
import matplotlib.pyplot as plt


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.min_test_loss = np.inf
        self.epoch_for_min_test_loss = 0
        
    def _build_model(self):
        traindata, _ = self._get_data('train')
        model = BridgeModel(self.args.c_in,
                            self.args.c_out, 
                            self.args.seq_len, 
                            self.args.pred_len, 
                            self.args.d_model, 
                            self.args.dropout, 
                            self.args.embedding_dropout, 
                            self.device,
                            self.args.batch_size,
                            self.args.tfactor,
                            self.args.dfactor,
                            self.args.wavelet,
                            self.args.level,
                            self.args.patch_len,
                            self.args.stride,
                            self.args.no_decomposition,
                            self.args.use_amp,
                            self.args.n_nodes,
                            self.args.n_interrogators,
                            traindata.scaler_output,
                            traindata.scaler_input
                            ).float()
        
            
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader
    
    
    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        return model_optim
    
    def _select_criterion(self):
        criterion = {'mse': torch.nn.MSELoss(), 'smoothL1': torch.nn.SmoothL1Loss()}
        
        try:
            return criterion[self.args.loss]
        except KeyError as e:
            raise ValueError(f"Invalid argument: {e} (loss: {self.args.loss})")

 
    def trainPlus(self, setting, optunaTrialReport = None, corr_factor_optimization = False):
        print('train: {}'.format(corr_factor_optimization))
        # Datasets
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        # paths
        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok = True)
        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        # Optimizer
        if corr_factor_optimization == True:
            model_optim = optim.Adam(list(self.model.positionModel.parameters()) + [self.model.corr_factor_weight, self.model.corr_factor_bias],
                                     lr = self.args.learning_rate, weight_decay = self.args.weight_decay)
        elif corr_factor_optimization == False:
            model_optim = optim.Adam(self.model.positionModel.parameters(),
                                     lr = self.args.learning_rate, weight_decay = self.args.weight_decay)
        # criterion
        criterion =  self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.to(dtype = torch.float, device = self.device) # 
                batch_y =  batch_y.to(dtype = torch.float, device = self.device) 
                iter_count += 1
                model_optim.zero_grad(set_to_none = True)
                position_scaled, revised_position_scaled, _ = self.model(batch_x)
                loss = criterion(batch_y, revised_position_scaled)
                train_loss.append(loss) 
                loss.backward()
                model_optim.step()

            print("Epoch {}: cost time: {:.2f} sec".format(epoch + 1, time.time()-epoch_time))
            train_loss = torch.tensor(train_loss).mean()
            vali_loss, vali_mae = self.valiPlus(vali_data, vali_loader, criterion)

            if vali_loss <  self.min_test_loss:
                self.min_test_loss = vali_loss
                self.min_test_mae = vali_mae
                self.epoch_for_min_test_loss = epoch            

            ########################### this part is just for optuna ###########
            # no other modifications have been done except this for optuna in this code
            if optunaTrialReport is not None:
                optunaTrialReport.report(vali_loss, epoch)
                if optunaTrialReport.should_prune():
                    raise optuna.exceptions.TrialPruned()
            #############################################################
            
            print("\tEpoch {0}: Steps- {1} | Train Loss: {2:.5f} Vali.MSE: {3:.5f} Vali.MAE: {4:.5f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, vali_mae))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("\tEarly stopping")
                break
            if torch.isnan(train_loss):
                print("\stopping: train-loss-nan")
                break
            adjust_learning_rate(model_optim, None, epoch+1, self.args)
            
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def valiPlus(self, vali_data, vali_loader, criterion):
        self.model.eval()        
        preds_mean, trues = [], []

        with torch.no_grad():
            for batch_x, batch_y in vali_loader:
                batch_x = batch_x.to(dtype = torch.float, device = self.device) # non-scaled
                batch_y =  batch_y.to(dtype = torch.float, device = self.device) 
                position_scaled, revised_position_scaled, _ = self.model(batch_x)
                
                preds_mean.append(revised_position_scaled)
                trues.append(batch_y)
            
            # non scaled
            preds_mean = vali_data.scaler_output.inverse_transform(torch.cat(preds_mean).cpu()) # non-scaled
            trues = torch.cat(trues).cpu() # non-scaled
            
            preds_mean = preds_mean.reshape(-1, preds_mean.shape[-2], preds_mean.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            
            mae, mse, rmse, mape, mspe = metric(preds_mean.numpy(), trues.numpy()) # non-scaled loss
            self.model.train()
            return mse, mae

    def testPlus(self, corr_factor_optimization = False):
        if corr_factor_optimization == False:
            test_data, test_loader = self._get_data(flag='test')
            self.model.eval()        
            preds_mean, trues = [], []
        
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    batch_x = batch_x.to(dtype = torch.float, device = self.device) # 
                    batch_y =  batch_y.to(dtype = torch.float, device = self.device) 
                    position_scaled, revised_position_scaled, _ = self.model(batch_x)
                    preds_mean.append(revised_position_scaled)
                    trues.append(batch_y)
        
                preds_mean = test_data.scaler_output.inverse_transform(torch.cat(preds_mean).cpu()) # non-scaled
                trues = torch.cat(trues).cpu() # non-scaled
                
                fig_name = 's{}_p{}s{}_trf{}_tsf{}_dec{}'.format(self.args.seq_len, self.args.patch_len, self.args.stride, self.args.train_cf_optimization, self.args.test_cf_optimization, not self.args.no_decomposition)
                plot_custom(preds_mean.shape[-1]/2, np.asarray(preds_mean.squeeze()), np.asarray(trues.squeeze()), x_label = 'steps', y_label = 'position', path = './outputs/mixer/', name = fig_name, title = self.args)
                
                preds_mean = preds_mean.reshape(-1, preds_mean.shape[-2], preds_mean.shape[-1])
                trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
                
                mae, mse, rmse, mape, mspe = metric(preds_mean.numpy(), trues.numpy()) # non-scaled
                self.model.train()
                print('Test-mse: {}, mae: {}'.format(mse, mae))
                
        elif corr_factor_optimization == True:
            model_optim = optim.Adam([self.model.corr_factor_weight, self.model.corr_factor_bias], lr = 1e-2)
            criterion =  self._select_criterion() 
            test_data, test_loader = self._get_data(flag='test')
            self.model.eval()
            preds, trues = [], []
            factor1_list = []
            factor2_list = []
            
            power_true = []
            power_pred = []
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.to(dtype = torch.float, device = self.device) # scaled
                true =  batch_y.to(dtype = torch.float, device = self.device) # non-scaled

                model_optim.zero_grad(set_to_none = True)
                position_scaled, revised_position_scaled, power2_scaled = self.model(batch_x)
                loss_power = criterion(power2_scaled, batch_x[:, -1, :])
                loss_power.backward()
                model_optim.step()
                
                preds.append(revised_position_scaled)
                trues.append(true)
                factor1_list.append(self.model.corr_factor_weight.repeat(self.args.batch_size,1, 1))
                factor2_list.append(self.model.corr_factor_bias.repeat(self.args.batch_size,1, 1))
                
                power_true.append(batch_x[:, -1, :])
                power_pred.append(power2_scaled)
                
            preds = test_data.scaler_output.inverse_transform(torch.cat(preds).detach().cpu()) # non-scaled
            trues = torch.cat(trues).cpu() # non-scaled
            factor1_list = torch.cat(factor1_list).detach().cpu()
            factor2_list = torch.cat(factor2_list).detach().cpu()
            
            power_true = torch.cat(power_true).detach().cpu()
            power_pred = torch.cat(power_pred).detach().cpu()
            
            fig_name = 's{}_p{}s{}_trf{}_tsf{}_dec{}'.format(self.args.seq_len, self.args.patch_len, self.args.stride, self.args.train_cf_optimization, self.args.test_cf_optimization, not self.args.no_decomposition)
            plot_custom(preds.shape[-1]/2, np.asarray(preds.squeeze()), np.asarray(trues.squeeze()), x_label = 'steps', y_label = 'position', path = './outputs/mixer/', name = fig_name, title = self.args)
            
            plot_custom2(factor1_list[:, :, 0::2].squeeze(), preds.shape[-1]/2, title = 'corr_fac_weight_x', sub_title = 'node', x_label = 'steps', y_label = 'value', path = './outputs/mixer/', name = fig_name + '_weight_x')
            plot_custom2(factor1_list[:, :, 1::2].squeeze(), preds.shape[-1]/2, title = 'corr_fac_weight_y', sub_title = 'node', x_label = 'steps', y_label = 'value', path = './outputs/mixer/', name = fig_name + '_weight_y')
            
            plot_custom2(factor2_list[:, :, 0::2].squeeze(), preds.shape[-1]/2, title = 'corr_fac_bias_x', sub_title = 'node', x_label = 'steps', y_label = 'value', path = './outputs/mixer/', name = fig_name + '_bias_x')
            plot_custom2(factor2_list[:, :, 1::2].squeeze(), preds.shape[-1]/2, title = 'corr_fac_bias_y', sub_title = 'node', x_label = 'steps', y_label = 'value', path = './outputs/mixer/', name = fig_name + '_bias_y')
            
            plot_custom2(power_true.squeeze(), 2, title = 'true_power', sub_title = 'int', x_label = 'steps', y_label = 'value', path = './outputs/mixer/', name = fig_name + '_true_pow')
            plot_custom2(power_pred.squeeze(), 2, title = 'power_pred', sub_title = 'int', x_label = 'steps', y_label = 'value', path = './outputs/mixer/', name = fig_name + '_power_pred')
            
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            
            mae, mse, rmse, mape, mspe = metric(preds.numpy(), trues.numpy()) # non-scaled
            print('Test-mse: {}, mae: {}'.format(mse, mae))
        return mse, mae
  

    def get_gflops(self):
        batch = self.args.batch_size
        seq = self.args.seq_len
        channel = self.args.c_in
        
        input_tensor = torch.randn(batch, seq, channel).to('cuda')
        
        # ############
        self.model.eval()
        macs, params = profile(self.model, inputs=(input_tensor, ), verbose = True)
        gflops = 2 * macs / 1e9  # convert to GFLOPs
        print(f"Total GFLOPs: {gflops:.4f}")
        return gflops




