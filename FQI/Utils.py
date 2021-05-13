"""
INFO8003-1
2020 - 2021
Final Project
Inverted Double Pendulum
François LIEVENS - 20103816
Julien HUBAR - 10152485

Some tools used by others classes
"""

import os
import torch
import pandas as pd
import numpy as np


def get_highest(path):
    """
    Return highest tensor idx in the directory
    """
    tensor_lst = os.listdir(path)
    tensor_idx = []

    if len(tensor_lst) != 0:
        for itm in tensor_lst:
            if '.pt' in itm:
                tensor_idx.append(int(itm.replace('.pt', '')))
        return max(tensor_idx)
    else:
        return None

def test_states():

    # Load states data
    state_header = ['pos', 'speed', 'theta', 'theta_speed', 'phi', 'phi_speed', 'sx', 'sy', 'wtf',
                    'action', 'reward',
                    'next_pos', 'next_speed', 'next_theta', 'next_theta_speed', 'next_phi', 'next_phi_speed', 'next_sx',
                    'next_sy',
                    'next_wtf', 'done']
    data = pd.read_csv('Data/test_states.csv', sep=';', header=None, names=state_header, index_col=False)
    np_data = data.to_numpy()

    track_tensor = torch.zeros((np_data.shape[0], 9))
    for i in range(np_data.shape[0]):
        track_tensor[i, :] = torch.Tensor(np_data[i, 0:9])
    return track_tensor