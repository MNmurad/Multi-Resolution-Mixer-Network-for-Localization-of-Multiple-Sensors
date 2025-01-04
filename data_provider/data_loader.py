import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset

from utils.tools import StandardScaler, PreProcessing


import warnings
warnings.filterwarnings('ignore')


class Dataset_Sensor(Dataset):
    def __init__(self, args, flag):
        # info
        self.args = args
        self.seq_len = self.args.seq_len
        self.pred_len = self.args.pred_len
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        self.flag = flag
        
        self.root_path = self.args.root_path
        self.data_path = self.args.data_path # test data path
        self.data_input, self.data_target = self.__read_data__()
        
    def __read_data__(self):
        self.scaler_input = StandardScaler() # for power
        self.scaler_output = StandardScaler() # for position
        num_node = self.args.n_nodes
        num_int = self.args.n_interrogators
        
        train_dataset_list = os.listdir(os.path.join(self.root_path, 'train'))
        vali_dataset_list = os.listdir(os.path.join(self.root_path, 'vali'))
        
        df_vali = pd.read_csv(os.path.join(self.root_path, 'vali', vali_dataset_list[0])) # test dataframe
        df_test = pd.read_csv(os.path.join(self.root_path, 'test', self.data_path)) # test dataframe
        
        pos_index = df_test.columns[0:num_node*2]
        power_index = df_test.columns[num_node*2:num_node*2+num_int]
        vel_index = df_test.columns[num_node*2+num_int : num_node*2+num_int + num_node*2]
        final_index = list(pos_index) + list(power_index)
        
        def create_sequences(data, look_back):
            data_seq = []
            for i in range(len(data) - look_back):
                data_seq.append(data[i:i + look_back])
            return np.array(data_seq)
        
        train_datasets = []
        for i in range(len(train_dataset_list)):
            df_train = pd.read_csv(os.path.join(self.root_path, 'train', train_dataset_list[i]))
            data_train = df_train[final_index].values
                
            if i == 0:
                self.scaler_input.fit(data_train[:, -num_int:])
                self.scaler_output.fit(data_train[:, :num_node*2])
            data_train[:, :num_node*2], data_train[:, -num_int:] = self.scaler_output.transform(data_train[:, :num_node*2]), self.scaler_input.transform(data_train[:, -num_int:]) # target, input
            train_datasets.append(data_train)
      
        data_vali = df_vali[final_index].values
        data_test = df_test[final_index].values

        # scaling
        data_vali[:, :num_node*2], data_vali[:, -num_int:] = data_vali[:, :num_node*2], self.scaler_input.transform(data_vali[:, -num_int:]) 
        data_test[:, :num_node*2], data_test[:, -num_int:] = data_test[:, :num_node*2], self.scaler_input.transform(data_test[:, -num_int:] ) 
        
        train_data_seq = []
        for dataset in train_datasets:
            train_data_seq.append(create_sequences(dataset, self.seq_len))
        
        test_datasets = [data_test]
        test_data_seq = []
        for dataset in test_datasets:
            test_data_seq.append(create_sequences(dataset, self.seq_len))
        
        vali_datasets = [data_vali]
        vali_data_seq = []
        for dataset in vali_datasets:
            vali_data_seq.append(create_sequences(dataset, self.seq_len))
        
        train_data_seq = np.vstack(train_data_seq)
        test_data_seq = np.vstack(test_data_seq)
        vali_data_seq = np.vstack(vali_data_seq)
        # Convert to PyTorch tensors
        train_data_seq = torch.tensor(train_data_seq, dtype=torch.float32)
        test_data_seq = torch.tensor(test_data_seq, dtype=torch.float32)
        vali_data_seq = torch.tensor(vali_data_seq, dtype=torch.float32)
        
        train_data_input, train_data_target = train_data_seq[:, :, num_node*2:num_node*2+num_int], train_data_seq[:, :, :num_node*2]
        test_data_input, test_data_target = test_data_seq[:, :, num_node*2:num_node*2+num_int], test_data_seq[:, :, :num_node*2]
        vali_data_input, vali_data_target = vali_data_seq[:, :, num_node*2:num_node*2+num_int], vali_data_seq[:, :, :num_node*2]
        
        if self.flag == 'train':
            return train_data_input, train_data_target[:, -1:,:]
        elif self.flag == 'test':
            return test_data_input, test_data_target[:, -1:,:]
        elif self.flag == 'val':
            return vali_data_input, vali_data_target[:, -1:,:]
    
    def __getitem__(self, index):
        return self.data_input[index], self.data_target[index]
    
    def __len__(self):
        return len(self.data_input)
    
    def inverse_transform(self, data):
        return self.scaler_output.inverse_transform(data)  
 
