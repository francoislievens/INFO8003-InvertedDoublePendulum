"""
INFO8003-1
2020 - 2021
Final Project
Inverted Double Pendulum
François LIEVENS - 20103816
Julien HUBAR - 10152485

This file implement the training process
of our neural network on the inverted double
pendulum by using the deep Q-Learning approach
"""
from NeuralNet import NeuralNet
from ReplayMemory import ReplayMemory
from Gamer import Gamer
import torch
import numpy
import os
import time
import gym
from tqdm import tqdm
import pybullet_envs
import Utils
import numpy as np



def trainer(name='Test_H',
            batch_size=10000,
            buff_capacity=25000,
            lr=0.0005,
            discount=0.8,
            epsilon_start=0.2,
            epsilon_end=0.1,
            epsilon_decay=1000,      # Down from start to end in this nb of steps
            nb_epoch=3,
            nb_simu=5000,
            grad_clamp=0.8,
            discretization_size=20,
            graphical=False,
            nb_step=10000,        # The number of step to perform
            stochastic=True,
            device='cuda:0'):

    # The general index
    gen_idx = 0

    # Get neural networks
    model_plc = NeuralNet(device=device, nb_actions=discretization_size, name=name).to(device)

    # Recover if already exist
    loaded = False
    max_model = Utils.get_highest('{}/model'.format(name))
    if max_model is not None:
        try:
            model_plc.restore(add=max_model)
            model_plc.to(device)
            gen_idx = max_model + 1
            print('Model index {} successfully recovered'.format(max_model))
            loaded = True
        except:
            print('ERROR during model index {} loading. Maybe corrupted'.format(max_model))
            return -1
    # Get target model
    model_trg = NeuralNet(device=device, nb_actions=discretization_size, name='{}_trg'.format(name)).to(device)
    model_trg.load_state_dict(model_plc.state_dict())

    # Game object to perform simulations
    gamer = Gamer(model_plc, graphical=graphical, device=device, stochastic=stochastic)

    # Replay memory Buffer
    tmp_eps = epsilon_start
    if not loaded:
        # Purely random filling
        tmp_eps = 1
    print('Buffer filling with epsilon value of {}'.format(tmp_eps))
    pos, action, reward, next_pos, done = gamer.training_game(eps=tmp_eps, nb_transitions=buff_capacity)
    # Cretate the buffer
    buffer = ReplayMemory(pos, action, reward, next_pos, done, device=device)
    # Get the data loader
    data_loader = torch.utils.data.DataLoader(buffer,
                                              batch_size=batch_size,
                                              shuffle=True)
    # Track Q-value predictions of some states
    q_track_states = Utils.test_states().to(device)

    # Loss object
    #loss_fn = torch.nn.MSELoss()
    loss_fn = torch.nn.L1Loss()
    # Optimizer
    optimizer = torch.optim.Adam(model_plc.parameters(), lr=lr)
    # Restore optmizer state if exists
    try:
        optimizer.load_state_dict(torch.load('{}/optimizer/{}.pt'.format(name, max_model)))
        print('Successfull optimizer restoration')
    except:
        print('No optmizer state to restore')
    #optimizer = torch.optim.RMSprop(model_plc.parameters())

    gen_end = gen_idx + nb_step
    eps_idx=0
    eps_decay_value = (epsilon_start - epsilon_end) / epsilon_decay
    eps = epsilon_start
    # Main Loop
    for i in tqdm(range(gen_idx, gen_end)):
        torch.cuda.empty_cache()
        print('=======================================')
        print('Learning process, model idx: {}'.format(i))

        # Make target predictions:
        print('Target prediction...')
        for idx, (state, action, reward, next_state, done, target, index) in enumerate(data_loader):
            # Make predictions for each next states
            next_state = next_state.to(device)
            reward = reward.to(device)
            with torch.no_grad():
                model_trg.eval()
                prds = model_trg(next_state)
            # Get max prediction:
            max_prds = torch.max(prds, 1)[0]
            # Update Q value estimation
            tmp_q = max_prds * discount
            tmp_q = tmp_q.view(tmp_q.size(0), 1) + reward

            # Set future reward of terminal state to zero
            mask = torch.abs(done - 1).type(torch.FloatTensor).to(device)
            tmp_q = tmp_q * mask

            # Update targergets
            buffer.tmp_target[index, :] = tmp_q

        # Update the loader
        data_loader = torch.utils.data.DataLoader(buffer,
                                                  batch_size=batch_size,
                                                  shuffle=True)

        # The learning step on new targets
        tmp_loss = []
        print('Learning step...')
        with tqdm(total=nb_epoch, position=0, leave=True) as pbar:
            for e in range(nb_epoch):
                for idx, (state, action, reward, next_state, done, target, index) in enumerate(data_loader):
                    # Make predictions
                    model_plc.train()
                    state = state.to(device)
                    target = target.to(device)
                    action = action.to(device)
                    prds = model_plc(state)
                    # Only keep chosen action value
                    if device == 'cuda:0':
                        chosen_prds = torch.gather(prds, 1, action.type(torch.cuda.LongTensor))
                    else:
                        chosen_prds = torch.gather(prds, 1, action.type(torch.LongTensor))
                    # Compute the loss
                    loss = loss_fn(chosen_prds, target)
                    # Opimization step
                    optimizer.zero_grad()
                    loss.backward()
                    # Clip gradients:
                    for param in model_plc.parameters():
                        param.grad.data.clamp_(-grad_clamp, grad_clamp)
                    optimizer.step()
                    # Track the loss
                    tmp_loss.append(loss.cpu().detach().item())
                pbar.update(1)

        # Store loss evolution
        string = []
        for itm in tmp_loss:
            string.append(str(itm))
        while len(string) < 30:
            string.append('0')
        string = ';'.join(string)
        file = open('{}/Loss_tracking.csv'.format(name), 'a')
        file.write('{};{}\n'.format(i, string))
        file.close()
        print('\t Average Loss: {}'.format(np.mean(tmp_loss)))

        # Save the model
        model_plc.save(add=i)
        # Save optimizer state
        torch.save(optimizer.state_dict(), '{}/optimizer/{}.pt'.format(name, i))
        # Delete old models
        mdl_lst = os.listdir('{}/model'.format(name))
        mdl_lst_idx = []
        for itm in mdl_lst:
            mdl_lst_idx.append(int(itm.replace('.pt', '')))
        for itm in mdl_lst_idx:
            if itm < i - 3 and itm <= i and itm % 20 != 0:
                try:
                    os.remove('{}/model/{}.pt'.format(name, itm))
                except:
                    print('Fail to remove old model index {}'.format(itm))
                try:
                    os.remove('{}/optimizer/{}.pt'.format(name, itm))
                except:
                    pass

        # Refresh target network
        model_trg.load_state_dict(model_plc.state_dict())

        # Make predictions on states to track
        model_plc.eval()
        with torch.no_grad():
            track_prds = model_plc(q_track_states[:, 0:-1])
        chosen_prds = torch.gather(track_prds, 1, q_track_states[:, -1].unsqueeze(1).type(torch.cuda.LongTensor)).cpu().detach().numpy()
        track_tmp = []
        for itm in chosen_prds:
            track_tmp.append(str(float(itm)))
        string = ';'.join(track_tmp)
        file = open('{}/Q_conv_tracking.csv'.format(name), 'a')
        file.write('{}\n'.format(string))
        file.close()
        # Test the policy:
        avg_cum_rew, avg_duration = gamer.test_it()
        # Save it in a file
        file = open('{}/policy_perf.csv'.format(name), 'a')
        string = '{};{};{}\n'.format(i, avg_cum_rew, avg_duration)
        file.write(string)
        file.close()
        print('\t Average test duration: {}'.format(avg_duration))
        print('\t Average test reward: {}'.format(avg_cum_rew))
        print('=======================================')


        # Buffer updating
        state, action, reward, next_state, done = gamer.training_game(eps=eps, nb_transitions=nb_simu)
        buffer.push(state, action, reward, next_state, done)

        if avg_cum_rew > 900:
            torch.save(model_plc.state_dict(), 'special_model_{}_idx{}.pt'.format(name, i))

        # Epsilon decay
        if eps > epsilon_end:
            eps -= eps_decay_value
        print('epsilon value: {}'.format(eps))






if __name__ == "__main__":

    trainer()