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
import gym
import pybullet_envs
import numpy as np
from ActorCriticNeural import ActorCriticNeural
import time
import os
from Learner import Learner
import pandas as pd
from torch.distributions import Categorical


GRAPHICAL = False
TIME_STEP = 1/1000
NB_EPISODE = 10000
MAX_EPI_LEN = 1000
DEVICE = 'cuda:0'
DISCOUNT = 0.8
SAVE_INTERVAL = 1000
STATE_SIZE = 9
NB_ACTONS = 50
LEARNING_RATE = 2e-6
EPSILON = 0.2
DROPOUT = 0


def trainer(name='PGRAD_A'):

    # Get the environment
    env = gym.make("InvertedDoublePendulumBulletEnv-v0")
    # Get the policy network
    AC_Network = ActorCriticNeural(name=name, device=DEVICE, state_size=STATE_SIZE,
                           nb_actions=NB_ACTONS, lr=LEARNING_RATE,
                                   dropout= DROPOUT).to(DEVICE)
    # Try to restore more recent model if exist

    mod_lst = os.listdir('{}/model'.format(name))
    restored = False
    if len(mod_lst) > 0:
        mod_lst_idx = []
        for itm in mod_lst:
            try:
                mod_lst_idx.append(int(itm.replace('.pt', '')))
            except:
                pass
        max_idx = max(mod_lst_idx)
        print('Already existing model found, try to restore idx {}'.format(max_idx))
        try:
            AC_Network.restore(add=max_idx)
            print('Model successfully restored')
        except:
            print('Fail to restore existing model. Maybe corrupted? ')
            exit(-1)

    if GRAPHICAL:
        env.render(mode='human')

    start_epi_idx = 0
    # Retore last index from tracking if restored model
    if restored:
        df = pd.read_csv('{}/tracking.csv', sep=';', header=None).numpy()
        start_epi_idx = int(df[:, 0])

    end_epi_idx = start_epi_idx + NB_EPISODE

    # A learner object to perform learning steps
    learner_obj = Learner(model=AC_Network, discounter=DISCOUNT)

    # Main loop
    track_duration = []
    track_reward = []
    entropy_term = 0
    for e in range(start_epi_idx, end_epi_idx):
        print('======= Episode {}/{} ========='.format(e, end_epi_idx))

        # Store actions log probs and rewards
        tot_log_prbs = []
        tot_rewards = []
        tot_critic_val = []
        tot_entropy = []

        # Reinit the env
        obs = torch.Tensor(env.reset()).to(DEVICE)

        # Episode loop
        epi_idx = 0
        while epi_idx < MAX_EPI_LEN:

            if GRAPHICAL:
                time.sleep(TIME_STEP)

            # Get actions probs
            AC_Network.train()
            critic_val, prbs = AC_Network.forward(obs.view(1, STATE_SIZE))

            # To numpy
            np_prbs = prbs.cpu().detach().numpy()
            #np_val = prbs.cpu().detach().numpy()
            tot_critic_val.append(critic_val)

            # Chose the next action according to probability distribution
            action = np.random.choice(NB_ACTONS, p=np.squeeze(np_prbs))
            # Get actuator value
            actuator_val = AC_Network.actions[action]

            # Get log probs of the chosen action
            log_prbs = torch.log(prbs.squeeze(0)[action])
            tot_log_prbs.append(log_prbs)
            # Compute the entropy of the prediction
            entropy = Categorical(probs=prbs).entropy()
            tot_entropy.append(entropy)
            #entropy = -np.sum(np.mean(np_prbs) * np.log(np_prbs))
            #entropy_term += entropy

            # Perform the action
            new_obs, r, done, _ = env.step([actuator_val])
            obs = torch.Tensor(new_obs).to(DEVICE)
            tot_rewards.append(r)

            # If end of the episode:
            if done or epi_idx == MAX_EPI_LEN - 1:

                track_duration.append(epi_idx)
                track_reward.append(np.sum(tot_rewards))

                file = open('{}/tracking.csv'.format(name), 'a')
                file.write('{};{};{};{}\n'.format(e, track_duration[-1], track_reward[-1], entropy.cpu().detach().item()))
                file.close()

                print('Episode {} : Duration {} - Total reward {} - Entropy: {}'.format(e, track_duration[-1],
                                                                                        track_reward[-1],
                                                                                        entropy.cpu().detach().item()))
                if len(track_duration) > 10:
                    print('AVG 10 last duration: {} / Rewards: {}'.format(np.mean(track_duration[-10:-1]),
                                                                          np.mean(track_reward[-10:-1])))

                break
            epi_idx += 1

        # Computing cumulative rewards
        cum_rew = np.zeros(len(tot_rewards))
        idx = len(tot_rewards) - 1
        cum_rew[idx] = tot_rewards[idx]
        while idx > 0:
            idx -= 1
            cum_rew[idx] = tot_rewards[idx] + DISCOUNT * cum_rew[idx + 1]

        # Actor critic update:
        val = torch.stack(tot_critic_val).to(DEVICE)
        cum_rew = torch.Tensor(cum_rew).to(DEVICE)
        log_prbs = torch.stack(tot_log_prbs).to(DEVICE)

        # Get the advantage
        adv = cum_rew - val

        # Compute the general loss
        final_entropy = torch.stack(tot_entropy).mean()
        print('entropy: ', final_entropy.cpu().detach().item())
        loss = torch.mean((-1 * log_prbs) * adv) + (0.5 * torch.mean(adv ** 2)) + (100 / (final_entropy+1e-8))
        # Optimize
        AC_Network.optimizer.zero_grad()
        loss.backward()
        AC_Network.optimizer.step()
        print('loss: ', loss.cpu().detach().item())
        entropy_term = 0


        # Model saving
        if e % SAVE_INTERVAL == 0 and e != 0:
            AC_Network.save(add=e)



if __name__ == "__main__":

    trainer()