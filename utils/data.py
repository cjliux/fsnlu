#coding: utf-8
import os
import torch.utils.data as thdata


class Dataset(thdata.Dataset):

    def __init__(self, data):
        super().__init__()
        self.data = data

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    
    