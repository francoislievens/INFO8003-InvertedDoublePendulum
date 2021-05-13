"""
INFO8003-1
2020 - 2021
Final Project
Inverted Double Pendulum
François LIEVENS - 20103816
Julien HUBAR - 10152485

In this file we are implementing a simple Neural
network using fully connected layers in order to
approximate our non-linear Q-value function.
"""
import torch
import torch.nn as nn
import numpy as np


class NeuralNet(nn.Module):

    def __init__(self,
                 name='Test_A',         # Name of the model
                 lr=0.00001,            # The learning rate to use
                 device='cuda:0',       # Device to use
                 nb_actions=20,         # Discretization
                 ):
        super(NeuralNet, self).__init__()

        # Parameters:
        self.nb_actions = nb_actions
        self.name = name
        self.device = device

        # Input layer: take as input an array with position - speed - action index
        self.fc1 = nn.Linear(9, 200)
        # Others fully connected layers
        self.fc2 = nn.Linear(200, 5000)
        self.fc3 = nn.Linear(5000, 5000)
        self.fc4 = nn.Linear(5000, 5000)
        self.fc5 = nn.Linear(5000, 5000)
        self.fc6 = nn.Linear(5000, 5000)
        self.fc7 = nn.Linear(5000, nb_actions)

        # Use batch normalization
        self.bn1 = nn.BatchNorm1d(200)
        self.bn2 = nn.BatchNorm1d(5000)
        self.bn3 = nn.BatchNorm1d(5000)
        self.bn4 = nn.BatchNorm1d(5000)

        # Activations
        self.ac1 = nn.ReLU()
        self.ac2 = nn.ReLU()
        self.ac3 = nn.Tanh()
        self.ac4 = nn.ReLU()
        self.ac5 = nn.ReLU()
        self.ac6 = nn.ReLU()


        # Store device
        self.to(device)

        # Actions values
        self.actions = np.linspace(-1, 1, nb_actions)

    def forward(self, state):

        # Feed the MLP:
        out = self.ac1(self.bn1(self.fc1(state)))
        out = self.ac2(self.bn2(self.fc2(out)))
        out = self.ac3(self.bn3(self.fc3(out)))
        out = self.fc4(self.bn4(self.fc4(out)))
        out = self.ac5(self.fc5(out))
        out = self.ac6(self.fc6(out))
        out = self.fc7(out)

        return out

    def get_action(self, idx):

        return self.actions[idx]

    def save(self, add=None):
        """
        Save weights of the model
        """
        if add is not None:
            torch.save(self.state_dict(), '{}/model/{}.pt'.format(self.name, add))
        else:
            torch.save(self.state_dict(), '{}/model/{}.pt'.format(self.name, 'model_params'))

    def restore(self, add=None):
        """
        Restore model's weights
        """
        if add is not None:
            self.load_state_dict(torch.load('{}/model/{}.pt'.format(self.name, add)))
        else:
            self.load_state_dict(torch.load('{}/model/{}.pt'.format(self.name, 'model_params')))

