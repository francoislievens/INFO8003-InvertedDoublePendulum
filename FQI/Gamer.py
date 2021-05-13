"""
INFO8003-1
2020 - 2021
Final Project
Inverted Double Pendulum
François LIEVENS - 20103816
Julien HUBAR - 10152485

This class perform the simulations
in order to build the dataset used
by the FQI process
"""
import gym
import pybullet_envs
import numpy as np
import torch
from NeuralNet import NeuralNet
from tqdm import tqdm
import time
from scipy.stats import norm

class Gamer():

    def __init__(self, model, graphical=False, device='cuda:0'):

        # Environment of the game
        self.env = gym.make("InvertedDoublePendulumBulletEnv-v0")

        # The neural network to use as policy
        self.model = model

        # Show graphical game
        self.graphical = graphical

        self.device = device
        self.env_speed = (1./500.)

        # Normal distribution to select random moves
        x = np.linspace(-1, 1, len(model.actions))
        self.norm_dist = norm.pdf(x, 0, 1)
        self.norm_dist = self.norm_dist / np.sum(self.norm_dist)
        self.action_idx = np.arange(0, len(x))

    def test_it(self, nb_test=30):

        cum_rewards = []
        tot_duration = []
        print('Testing process: ')
        with tqdm(total=nb_test, position=0, leave=True) as pbar:
            for i in range(nb_test):
                rew, duration = self.one_game()
                cum_rewards.append(rew)
                tot_duration.append(duration)
                pbar.update(1)

        return np.mean(cum_rewards), np.mean(tot_duration)

    def one_game(self):

        # Restart env
        obs = torch.Tensor(self.env.reset()).to(self.device)
        # Store rewards
        cum_rew = 0
        game_duration = 0
        # Game loop
        while True:
            if self.graphical:
                time.sleep(self.env_speed)
            self.model.eval()
            # Make prediction
            prd = self.model(obs.view(1, 9))
            # Get the betst action index
            action_idx = torch.argmax(prd).cpu().detach().item()
            # Make the move
            obs, reward, done, _ = self.env.step([self.model.get_action(action_idx)])
            # Save the reward
            cum_rew += reward
            game_duration += 1
            # Update the observations
            obs = torch.Tensor(obs).to(self.device)

            # Check if terminal state
            if done:
                break

        # Return the cumulative reward
        return cum_rew, game_duration




