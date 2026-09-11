import numpy as np
import torch

from torch.utils.data import Dataset
import os
from os.path import join


class agc_dataset(Dataset):
    def __init__(self, stage, feature_dir='../feature/dataset/'):
        if stage=='train':
            self.data_path='train'
        else:
            self.data_path='valid'

        load_path = join(feature_dir, self.data_path)
        # load input
        self.pathi_time = load_path + '/input_time'
        self.pathi_freq = load_path + '/input_freq'
        
        self.obji = os.scandir(self.pathi_time)

        self.filei=[]
        for entry in self.obji :
            if '.npy' in entry.name:
                self.filei.append(entry.name)
        
        self.file_num = len(self.filei)
        
        # load output
        self.patho = load_path + '/output'
        
    def __len__(self):
        return self.file_num
    
    def __getitem__(self, index):
        self.x_time_buf = np.load(join(self.pathi_time, self.filei[index]))
        self.x_freq_buf = np.load(join(self.pathi_freq, self.filei[index].replace('input_time', 'input_freq')))
        self.y_buf = np.load(join(self.patho, self.filei[index].replace('input_time', 'output')))
        # return torch.Tensor(self.x_time_buf), torch.Tensor(self.x_freq_buf), torch.Tensor(self.y_buf)
        return {'x_time': torch.Tensor(self.x_time_buf),'x_freq': torch.Tensor(self.x_freq_buf),'y': torch.Tensor(self.y_buf)}
