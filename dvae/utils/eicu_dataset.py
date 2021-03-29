#!/usr/bin/env python3
"""
Simon Bing, 2021
MPI for Intelligent Systems

Class eICUSequence():
-generate pytorch dataloader
-time series dataset from eICU data
"""
import os
import numpy as np
import torch
from torch.utils import data

class eICUSequence(data.Dataset):
    """
    Dataset from eICU time series data.

    TODO: Implement actually using eICU data. Randomly sampling for now.
    """
    def __init__(self, sequence_len, shuffle, name='eICU'):
        super.__init__()

        # Dataset parameters
        self.sequence_len = sequence_len
        self.shuffle = shuffle
        self.name = name

        # TODO: generate random np array
        self.len = 500
        self.dim = 15
        self.data = np.random.rand(self.len, self.sequence_len, self.dim)

    def __len__(self):
        """
        arguments should not be modified
        Return the total number of samples
        """
        return self.len

    def __getitem__(self, index):
        """
        input arguments should not be modified
        torch data loader will use this function to read ONE sample of data from a list that can be indexed by
        parameter 'index'
        """
        return torch.to_numpy(self.data[index, :, :])

