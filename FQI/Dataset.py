"""
INFO8003-1
2020 - 2021
Final Project
Inverted Double Pendulum
François LIEVENS - 20103816
Julien HUBAR - 10152485

This class manage the dataset who contain
transition's tuple that we are using during
the FQI learning process.
"""

import torch
import numpy as np
import pandas as pd
import tqdm
from tqdm import tqdm
import pickle
import gym
import pybullet_envs
import random

NB_ACTIONS = 20

class FQI_Dataset(torch.utils.data.Dataset):

    def __init__(self, device='cuda:0',
                 size=1000000
                 ):

        super(FQI_Dataset, self).__init__()

        self.size = size
        self.device = device

        # Data storing
        self.state = []
        self.action = []
        self.reward = []
        self.next_state = []
        self.done = []

        self.q_val = torch.zeros(size).to(self.device)

        # Index to shuffle
        np.random.seed(0)
        self.index = np.arange(self.size)
        np.random.shuffle(self.index)

        # Actions values
        self.action_values = np.linspace(-1, 1, NB_ACTIONS)

        # Restore serialized if exists
        try:
            print('Dataset resotring')
            self.restore()
            print('Dataset successfully restored')
        except:
            print('No dataset find, build a new')
            self.simulator()
            print('Simulation: Done')
            print('Serialization: ')
            self.serialize()
            print('Done')

    def simulator(self):

        # Get env
        env = gym.make("InvertedDoublePendulumBulletEnv-v0")
        obs = torch.Tensor(env.reset()).to(self.device)

        # Init tensors
        self.state = torch.zeros((self.size, 9)).to(self.device)
        self.action = torch.zeros((self.size, 1)).to(self.device)
        self.reward = torch.zeros((self.size, 1)).to(self.device)
        self.next_state = torch.zeros((self.size, 9)).to(self.device)
        self.done = torch.zeros((self.size, 1)).to(self.device)

        with tqdm(total=self.size, position=0, leave=True) as pbar:
            for i in range(self.size):

                # Store the state
                self.state[i, :] = obs
                # Randomly select an action
                act_idx = random.randint(0, NB_ACTIONS-1)
                act = self.action_values[act_idx]

                # Perform the action
                obs, reward, done, _ = env.step([act])

                # Tensorize
                obs = torch.Tensor(obs).to(self.device)
                self.next_state[i, :] = obs
                self.action[i, :] = torch.Tensor([act_idx]).to(self.device)
                self.reward[i, :] = torch.Tensor([reward]).to(self.device)
                self.done[i, :] = torch.Tensor([int(done)]).to(self.device)

                if done:
                    obs = torch.Tensor(env.reset()).to(self.device)
                pbar.update(1)


    def serialize(self):

        torch.save(self.state, 'Data/state.pt')
        torch.save(self.action, 'Data/action.pt')
        torch.save(self.reward, 'Data/reward.pt')
        torch.save(self.next_state, 'Data/next_state.pt')
        torch.save(self.done, 'Data/done.pt')

    def restore(self):

        self.state = torch.load('Data/state.pt')
        self.action = torch.load('Data/action.pt')
        self.reward = torch.load('Data/reward.pt')
        self.next_state = torch.load('Data/next_state.pt')
        self.done = torch.load('Data/done.pt')

    def __len__(self):

        return len(self.action)

    def __getitem__(self, index):

        return self.state[self.index[index]], self.action[self.index[index]], self.reward[self.index[index]], \
                self.next_state[self.index[index]], self.done[self.index[index]], self.q_val[self.index[index]], \
                self.index[index]









if __name__ == '__main__':

    test = FQI_Dataset()






