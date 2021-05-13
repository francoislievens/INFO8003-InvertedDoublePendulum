"""
INFO8003-1
2020 - 2021
Final Project
Inverted Double Pendulum
François LIEVENS - 20103816
Julien HUBAR - 10152485

This file contain the implementation an agent
who perform games in order to generate transition's
data for the learning process
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

    def __init__(self, model, graphical=False, device='cuda:0', stochastic=False):

        # Environment of the game
        self.env = gym.make("InvertedDoublePendulumBulletEnv-v0")

        # If stochastic env
        self.stochastic = stochastic

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

        # Softmax object of stochastic agent
        self.sm = torch.nn.Softmax(dim=0)


    def training_game(self, eps=0.5, nb_transitions=10000):
        """
        This method perform games episodes in order to obtain
        the asked number of transitions tuples. Games are play
        following an epsilon greedy aproach. An epsilon value of
        1 consist in a fully random game while an epsilon value
        of zero always follow the policy provided by the model
        """
        # A tensor to store
        data_pos = torch.zeros((nb_transitions, 9)).to(self.device)
        data_next_pos = torch.zeros((nb_transitions, 9)).to(self.device)
        data_actions = torch.zeros((nb_transitions, 1)).to(self.device)
        data_reward = torch.zeros((nb_transitions, 1)).to(self.device)
        data_done = torch.zeros((nb_transitions, 1)).to(self.device)

        # Init the env
        obs = torch.Tensor(self.env.reset()).to(self.device)
        if self.graphical:
            self.env.render(mode='human')
        with tqdm(total=nb_transitions, position=0, leave=True) as pbar:
            idx = 0
            while idx < nb_transitions:
                if self.graphical:
                    time.sleep(self.env_speed)
                # Store previous observation:
                data_pos[idx, :] = obs
                # Chose the action
                action = self.epsilon_greedy_action(obs, eps)
                data_actions[idx, :] = torch.Tensor([action]).to(self.device)
                # Apply the action:
                obs, reward, done, _ = self.env.step([self.model.get_action(action)])
                # Store data
                int_done = int(done)
                obs = torch.Tensor(obs).to(self.device)
                data_next_pos[idx, :] = obs
                data_reward[idx, :] = torch.Tensor([reward]).to(self.device)
                data_done[idx, :] = torch.Tensor([int_done]).to(self.device)

                # If end of the game
                if done:
                    obs = torch.Tensor(self.env.reset()).to(self.device)
                idx += 1
                pbar.update(1)


        return data_pos, data_actions, data_reward, data_next_pos, data_done

    def test_it(self, nb_test=30):
        """
        This method test the policy provided by the model by playing
        30 games episodes.
        return: the average cumulative reward and the average game duration
        """
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
        """
        This method perform one game episode and
        return the duration and the cumulative reward
        """
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

    def epsilon_greedy_action(self, state, eps=0.5):
        """
        This method implement the epsilon greedy approach.
        While the agent follow the random policy, the next
        action is chosen following a normal distribution
        centered in 0 with a std of 1
        """
        # Get action space
        discretization = len(self.model.actions)

        # Epsilon greedy approach
        rd = np.random.uniform(0, 1)
        if rd < eps:
            # Normal distribution aproach
            if self.stochastic:
                return self.stochastic_agent(state)
            else:
                return np.random.choice(self.action_idx, p=self.norm_dist)

            #return np.random.randint(0, discretization)

        # If follow the policy of the model:
        # Make a prediction
        with torch.no_grad():
            self.model.eval()
            prd = self.model(state.view(1, 9)).view(1, discretization)

        # Return max indice
        return torch.argmax(prd).cpu().detach().item()

    def stochastic_agent(self, obs):
        """
        This method chose the next action by generating a
        probability distribution by performing a softmax
        on the q-value distribution.
        """
        # Get proba:
        self.model.eval()
        with torch.no_grad():
            prd = self.model(obs.view(1, 9)).flatten()
            probs = self.sm(prd).cpu().detach().numpy()
        a = np.random.choice(self.action_idx, p=probs)
        return a

