"""
INFO8003-1
2020 - 2021
Final Project
Inverted Double Pendulum
François LIEVENS - 20103816
Julien HUBAR - 10152485

This class was use to train our
neural network during the policy gradient learning
approach. Since the basis policy gradient approach was
very noisy, we have move towards an actor-critic variant
and this class is not yet used.
"""
import torch
import numpy as np


class Learner():

    def __init__(self, model,
                discounter=0.8,
                eps_mach=1e-8           # improve numerical stability, avoid to devide by zero
                ):


        self.model = model
        self.discounter = discounter
        self.optimizer = model.optimizer
        self.eps_mach = eps_mach
        self.device = model.device

    def learn(self, rewards, lg_prbs):

        # Compute discounted reward for each elements of the trajectory
        discounted_rew = []
        for i in range(len(rewards)):
            dr = 0
            idx = 0
            for rew in rewards[i:]:
                dr += self.discounter**idx * rew
                idx += 1
            discounted_rew.append(dr)

        # Transform it into a tensor
        discounted_rew = torch.Tensor(discounted_rew).to(self.device)
        # Normalize it
        discounted_rew = (discounted_rew - discounted_rew.mean()) / (discounted_rew.std() + self.eps_mach)

        # Now compute gradients D
        plc_grad = []
        for i in range(len(discounted_rew)):
            plc_grad.append(lg_prbs[i] * (-discounted_rew[i]))
        plc_grad = torch.stack(plc_grad)

        # Optimization step
        self.optimizer.zero_grad()
        grad = plc_grad.sum()
        grad.backward()
        self.optimizer.step()


