"""
INFO8003-1
2020 - 2021
Final Project
Inverted Double Pendulum
François LIEVENS - 20103816
Julien HUBAR - 10152485

This class implement the cyclic buffer
who store our last simulation's transitions
"""
import torch

class ReplayMemory(torch.utils.data.Dataset):
    """
    A cyclic buffer who contain already palyed
    state/action/reward/next-state tuples
    """
    def __init__(self, state, action, reward, next_state, done, device='cuda:0'):
        super().__init__()

        # The number of tuples in the memory
        self.capacity = action.size(0)
        # The memory
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.final = done
        # Store next states predictions
        self.tmp_target = torch.zeros(self.reward.size()).to(device)

        self.index = 0
        self.device = device

    def __len__(self):
        return self.capacity

    def __getitem__(self, index):

        return self.state[index, :], self.action[index, :], self.reward[index, :], \
               self.next_state[index, :], self.final[index, :], self.tmp_target[index, :], index

    def push(self, state, action, reward, next_state, final):

        # Number of element to add
        start_idx = self.index
        size = state.size(0)
        end_idx = self.index + size
        if end_idx >= self.capacity:
            end_idx = self.capacity
            size = end_idx - self.index

        self.action[start_idx:end_idx, :] = action[0:size, :]
        self.state[start_idx:end_idx, :] = state[0:size, :]
        self.reward[start_idx:end_idx, :] = reward[0:size, :]
        self.next_state[start_idx:end_idx, :] = next_state[0:size, :]
        self.final[start_idx:end_idx, :] = final[0:size, :]

        self.index = end_idx
        if self.index >= self.capacity:
            self.index = 0



