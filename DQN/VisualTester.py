"""
INFO8003-1
2020 - 2021
Final Project
Inverted Double Pendulum
François LIEVENS - 20103816
Julien HUBAR - 10152485

This class permit us to visually test
our models on the pybullet environment
"""
import gym
import pybullet_envs
import torch
import time
import matplotlib.pyplot as plt
from NeuralNet import NeuralNet
import Utils
import numpy as np
import os

MODEL_NAME = 'Test_E'
DEVICE = 'cuda:0'
NB_ACTIONS = 20         # Discretization
TIMER = (1./60.)

if __name__ == "__main__":

    # Load the model
    model = NeuralNet(device=DEVICE, nb_actions=NB_ACTIONS, name=MODEL_NAME)
    # Recover it
    #max_model = Utils.get_highest('{}/model'.format(MODEL_NAME))
    max_model = 150
    if max_model is not None:
        try:
            model.restore(add=max_model)
            model.to(DEVICE)
            gen_idx = max_model + 1
            print('Model index {} successfully recovered'.format(max_model))
            loaded = True
        except:
            print('ERROR during model index {} loading. Maybe corrupted'.format(max_model))

    # Environment of the game
    env = gym.make("InvertedDoublePendulumBulletEnv-v0")
    env.render(mode="human")
    obs = torch.Tensor(env.reset()).to(DEVICE)

    hist_idx = 0
    # Game loop
    while True:
        time.sleep(TIMER)

        # Predict actions
        model.eval()
        with torch.no_grad():
            prds = model(obs.view(1, 9))

        max_action = torch.argmax(prds).cpu().detach().item()

        # Plot probabilities
        np_prds = prds.flatten().cpu().detach().numpy()
        x = np.arange(20)
        plt.plot(x, np_prds)
        plt.savefig('TestHistograms/{}.png'.format(hist_idx))
        hist_idx += 1
        plt.close()

        action = model.get_action(max_action)

        # Make the move
        obs, reward, done, _ = env.step([action])
        obs = torch.Tensor(obs).to(DEVICE)


        print('chosen action index: {} - applyed value: {} - reward: {}'.format(max_action, action, reward))

        input('press key')

        if done:
            break





