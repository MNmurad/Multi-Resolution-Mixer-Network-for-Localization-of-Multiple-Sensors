
import os
import torch
import argparse
import optuna

torch.set_printoptions(precision = 10)
from exp.exp_main import Exp_Main

import numpy as np
import random 
from utils.tools import dotdict
import gc
from utils.Tuner import Tuner # additional line for optuna
from utils.output_database import Output_database
from utils.tools import set_random_seed
import time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='[WPMixer] Long Sequences Forecasting')
    # frequent changing hy.params
    parser.add_argument('--model', type=str, required=False, choices=['BridgeModel'], default='BridgeModel',help='model of experiment')
    parser.add_argument('--task_name', type=str, required=False, choices=['long_term_forecast'], default='long_term_forecast')
    parser.add_argument('--root_path', type = str, default = './data/sensor/', help = 'path to dataset')
    parser.add_argument('--data_path', type = str, default = 'test-1.csv', choices = ['test-1.csv', 'test-2.csv', 'test-3.csv', 'test-4.csv'], help = 'test datasets')
    parser.add_argument('--load_checkpoint', type=bool, default=False, help='load the checkpoint')
    
    parser.add_argument('--use_hyperParam_optim', action = 'store_true', default = False, help = 'True: HyperParameters optimization using optuna, False: no optimization')
    parser.add_argument('--no_decomposition', action = 'store_true', default = False, help = 'whether to use wavelet decomposition')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--n_jobs', type = int, required = False, choices = [1, 2, 3, 4], default = 1, help = 'number_of_jobs for optuna')
    parser.add_argument('--seed', type = int, required = False, default = 42, help = 'random seed')
    parser.add_argument('--n_nodes', type = int, default = 9, help = 'number of nodes')
    parser.add_argument('--n_interrogators', type = int, default = 2, help = 'number of interrogators')
    
    parser.add_argument('--test_cf_optimization', type = int, default = 0, choices = [0, 0], help='correction factor optimization in test: 1= use, 0= do not')
    parser.add_argument('--train_cf_optimization', type = int, default = 1, choices = [1, 0], help='correction factor optimization in training: 1= use, 0= do not')
    
    # WPMixer
    parser.add_argument('--seq_len', type = int, default = 256, help = 'length of the look back window')
    parser.add_argument('--pred_len', type = int, default = 1, choices = [1], help = 'prediction length')
    parser.add_argument('--d_model', type = int, default = 16, help = 'embedding dimension')
    parser.add_argument('--tfactor', type = int, default = 7, help = 'expansion factor in the patch mixer')
    parser.add_argument('--dfactor', type = int, default = 7, help = 'expansion factor in the embedding mixer')
    parser.add_argument('--wavelet', type = str, default = 'sym3', help = 'wavelet type for wavelet transform')
    parser.add_argument('--level', type = int, default = 1, help = 'level for multi-level wavelet decomposition')
    parser.add_argument('--patch_len', type = int, default = 16, help = 'Patch size')
    parser.add_argument('--stride', type = int, default = 8, help = 'Stride')
    parser.add_argument('--batch_size', type = int, default = 32, help = 'batch size')
    parser.add_argument('--learning_rate', type = float, default = 0.00982476146087077, help = 'initial learning rate')    
    parser.add_argument('--dropout', type = float, default = 0.2, help = 'dropout for mixer')
    parser.add_argument('--embedding_dropout', type = float, default = 0.1, help = 'dropout for embedding layer')
    parser.add_argument('--weight_decay', type = float, default = 0.05, help = 'pytorch weight decay factor')
    parser.add_argument('--patience', type = int, default = 12, help = 'patience')
    parser.add_argument('--train_epochs', type = int, default = 60, help = 'train epochs')
    
    
    # rare changing hy.params
    parser.add_argument('--label_len', type=int, default=0, help='label length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M]; M:multivariate predict multivariate')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--cols', type=str, nargs='+', default = None, help='certain cols from the data files as the input features')
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--lradj', type=str, default='type3',help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--devices', type=str, default='0,1',help='device ids of multile gpus')
    parser.add_argument('--embed', type=str, default=0)
    parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'smoothL1'])
    parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
    parser.add_argument('--inverse', type=bool, default=True, help='inverse the model output for original scale')
    
    # Optuna Search Region: 
    ''' if you don't pass the argument, then value form the hyperparameters_optuna.py will be considered as search region'''
    parser.add_argument('--optuna_seq_len', type = int, nargs = '+', required = False, default = None, help = 'Optuna seq length list')
    parser.add_argument('--optuna_lr', type = float, nargs = '+', required = False, default = None, help = 'Optuna lr: first-min, 2nd-max')
    parser.add_argument('--optuna_batch', type = int, nargs = '+', required = False, default = None, help = 'Optuna batch size list')
    parser.add_argument('--optuna_wavelet', type = str, nargs = '+', required = False, default = None, help = 'Optuna wavelet type list')
    parser.add_argument('--optuna_tfactor', type = int, nargs = '+', required = False, default = None, help = 'Optuna tfactor list')
    parser.add_argument('--optuna_dfactor', type = int, nargs = '+', required = False, default = None, help = 'Optuna dfactor list')
    parser.add_argument('--optuna_epochs', type = int, nargs = '+', required = False, default = None, help = 'Optuna epochs list')
    parser.add_argument('--optuna_dropout', type = float, nargs = '+', required = False, default = None, help = 'Optuna dropout list')
    parser.add_argument('--optuna_embedding_dropout', type = float, nargs = '+', required = False, default = None, help = 'Optuna embedding_dropout list')
    parser.add_argument('--optuna_patch_len', type = int, nargs = '+', required = False, default = None, help = 'Optuna patch len list')
    parser.add_argument('--optuna_stride', type = int, nargs = '+', required = False, default = None, help = 'Optuna stride len list')
    parser.add_argument('--optuna_lradj', type = str, nargs = '+', required = False, default = None, help = 'Optuna lr adjustment type list')
    parser.add_argument('--optuna_dmodel', type = int, nargs = '+', required = False, default = None, help = 'Optuna dmodel list')
    parser.add_argument('--optuna_weight_decay', type = float, nargs = '+', required = False, default = None, help = 'Optuna weight_decay list')
    parser.add_argument('--optuna_patience', type = int, nargs = '+', required = False, default = None, help = 'Optuna patience list')
    parser.add_argument('--optuna_level', type = int, nargs = '+', required = False, default = None, help = 'Optuna level list')    
    parser.add_argument('--optuna_trial_num', type = int, required = False, default = None, help = 'Optuna trial number')            
    args = parser.parse_args()
    args.c_in = args.n_interrogators
    args.c_out = args.n_nodes * 2
    
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ','')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]


    if args.use_hyperParam_optim == False: # this block is not for hyper param tuning
        print('Args in experiment: {}'.format(args))
        setting = '{}_dec-{}_sl{}_pl{}_dm{}_bt{}_wv{}_tf{}_df{}_ptl{}_stl{}_sd{}'.format(args.model, not args.no_decomposition, args.seq_len, args.pred_len, args.d_model, args.batch_size, args.wavelet, args.tfactor, args.dfactor, args.patch_len, args.stride, args.seed)
        set_random_seed(args.seed)
  
        Exp = Exp_Main
        exp = Exp(args) # set experiments

        if not args.load_checkpoint:
            exp.trainPlus(setting, corr_factor_optimization = args.train_cf_optimization)
            exp.testPlus(corr_factor_optimization = args.test_cf_optimization)
        elif args.load_checkpoint:
            exp.model.load_state_dict(torch.load('checkpoints/' + setting + '/checkpoint.pth'))
            exp.testPlus(corr_factor_optimization = args.test_cf_optimization)
            
    elif args.use_hyperParam_optim: # this is for tuner only
        tuner = Tuner(42, args.n_jobs)
        tuner.tune(args)
        