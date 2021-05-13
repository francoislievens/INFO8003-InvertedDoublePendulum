"""
INFO8003-1
2020 - 2021
Final Project
Inverted Double Pendulum
François LIEVENS - 20103816
Julien HUBAR - 10152485

This class implement the double neural
network that we are using for our A2C
approach of the reinforcement learning
of the double inverted pendulum
"""

import torch
import torch.nn as nn
import numpy as np


class ActorCriticNeural(nn.Module):

    def __init__(self,
                 name='Test_A',         # Name of the model
                 lr=0.001,            # The learning rate to use
                 device='cuda:0',       # Device to use
                 nb_actions=20,         # Discretization
                 state_size=9,
                 dropout=0.2
                 ):
        super(ActorCriticNeural, self).__init__()

        # Parameters:
        self.nb_actions = nb_actions
        self.name = name
        self.device = device
        self.state_size = state_size

        # Input layer: take as input an array with position - speed - action index
        self.actor_fc1 = nn.Linear(state_size, 200)
        self.actor_fc2 = nn.Linear(200, 5000)
        self.actor_fc3 = nn.Linear(5000, 5000)
        self.actor_fc4 = nn.Linear(5000, nb_actions)

        # Initialization
        torch.nn.init.xavier_uniform(self.actor_fc1.weight)
        torch.nn.init.xavier_uniform(self.actor_fc2.weight)
        torch.nn.init.xavier_uniform(self.actor_fc3.weight)
        torch.nn.init.xavier_uniform(self.actor_fc4.weight)

        # Activations
        self.actor_ac1 = nn.ReLU()
        self.actor_ac2 = nn.ReLU()
        self.actor_ac3 = nn.ReLU()
        self.actor_ac4 = nn.Softmax(dim=1)

        # And for the critic network
        self.critic_fc1 = nn.Linear(state_size, 200)
        self.critic_fc2 = nn.Linear(200, 5000)
        self.critic_fc3 = nn.Linear(5000, 5000)
        self.critic_fc4 = nn.Linear(5000, 1)

        # Initialization
        torch.nn.init.xavier_uniform(self.critic_fc1.weight)
        torch.nn.init.xavier_uniform(self.critic_fc2.weight)
        torch.nn.init.xavier_uniform(self.critic_fc3.weight)
        torch.nn.init.xavier_uniform(self.critic_fc4.weight)

        # Activations
        self.critic_ac1 = nn.ReLU()
        self.critic_ac2 = nn.ReLU()
        self.critic_ac3 = nn.ReLU()

        self.dropout = nn.Dropout(dropout)

        # Store device
        self.to(device)

        # Actions values
        self.actions = np.linspace(-1, 1, nb_actions)
        self.actions_idx = np.arange(0, nb_actions)

        # Learning stuffs
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, state):

        # Get the actor probabilities
        out = self.actor_ac1(self.actor_fc1(state))
        out = self.dropout(self.actor_ac2(self.actor_fc2(out)))
        out = self.dropout(self.actor_ac3(self.actor_fc3(out)))
        probs = self.actor_ac4(self.actor_fc4(out))

        # Get the critic value
        out = self.critic_ac1(self.critic_fc1(state))
        out = self.dropout(self.critic_ac2(self.critic_fc2(out)))
        out = self.dropout(self.critic_ac3(self.critic_fc3(out)))
        value = self.critic_fc4(out)

        return value, probs

    def get_action(self, idx):

        return self.actions[idx]

    def save(self, add=None):
        """
        Save weights of the model
        """
        if add is not None:
            torch.save(self.state_dict(), '{}/model/{}.pt'.format(self.name, add))
            torch.save(self.optimizer.state_dict(), '{}/optimizer/{}.pt'.format(self.name, add))
        else:
            torch.save(self.state_dict(), '{}/model/{}.pt'.format(self.name, 'model_params'))
            torch.save(self.optimizer.state_dict(), '{}/optimizer/{}.pt'.format(self.name, 'otpimizer_weights'))

    def restore(self, add=None):
        """
        Restore model's weights
        """
        if add is not None:
            self.load_state_dict(torch.load('{}/model/{}.pt'.format(self.name, add)))
            self.optimizer.load_state_dict(torch.load('{}/optimizer/{}.pt'.format(self.name, add)))
        else:
            self.load_state_dict(torch.load('{}/model/{}.pt'.format(self.name, 'model_params')))
            self.optimizer.load_state_dict(torch.load('{}/optimizer/{}.pt'.format(self.name, 'optimizer_weights')))


