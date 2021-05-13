import torch
from NeuralNet import NeuralNet
from Dataset import FQI_Dataset
import pandas as pd
import numpy as np
from Gamer import Gamer
import Utils


def FQI(name='Test_FQI_A',
        device=torch.device('cuda:0'),
        batch_size=1000,
        nb_epoch=1,
        N=1000,
        discount=0.8,
        lr=0.0001
        ):

    # Build the model
    model = NeuralNet()
    # Restore if exists
    try:
        model.restore()
        print('Model Successfully restored')
    except:
        print('No existing model to restore')
    # Get the dataset
    dataset = FQI_Dataset()
    # The data handler
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=True)
    # Learning index's
    start_idx = 0
    # Find actual index
    try:
        track = pd.read_csv('{}/model/Loss_tracking.csv', sep=';', header=None)
        track = track.to_numpy()
        start_idx = max(track[:, 0])
    except:
        pass
    end_idx = start_idx + N

    # Loss function
    loss_fn = torch.nn.L1Loss()
    # Optimizer
    #optimizer = torch.optim.RMSprop(model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # A gamer object to test the model
    tester = Gamer(model=model,
                   graphical=False,
                   device=device)

    # Track Q-value predictions of some states
    q_track_states = Utils.test_states().to(device)

    for n in range(start_idx, end_idx):
        print('============ Iter {} / {} ================='.format(n, end_idx))
        # Make predictions on next states
        model.eval()
        with torch.no_grad():
            # Read all the dataset
            for idx, (state, action, reward, next_state, done, q_val, index) in enumerate(data_loader):

                # Put next states to device
                next_state = next_state.to(device)
                reward = reward.to(device)
                # Make predictions on next states
                preds = model(next_state)
                # Get highest prediction
                max_pred = torch.max(preds, 1, keepdim=True)[0]
                # Compute the target
                target = max_pred * discount
                target += reward
                # Store it in the dataset
                dataset.q_val[index.view(-1, 1).type(torch.LongTensor)] = target.view(-1, 1)

        # Learning step
        for e in range(0, nb_epoch):
            tmp_loss = []
            model.train()
            for idx, (state, action, reward, next_state, done, q_val, index) in enumerate(data_loader):

                # Get data
                state = state.to(device)
                action = action.to(device)
                target = q_val.to(device)

                # Make predictions
                preds = model(state)
                # Get the good predictions
                preds = torch.gather(preds, 1, action.type(torch.cuda.LongTensor))
                # Compute the loss
                loss = loss_fn(preds, target)

                # Optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tmp_loss.append(loss.cpu().detach().item())
                print('N: {}/{} - Epoch {}/{} - Batch {} - Loss {}'.format(n, N, e, nb_epoch, idx, tmp_loss[-1]))

            # Store loss tracking
            file = open('{}/model/Loss_tracking.csv'.format(name), 'a')
            string = '{};{};{}\n'.format(n, e, np.mean(tmp_loss))
            file.write(string)
            file.close()

        # Testing part
        model.eval()
        with torch.no_grad():
            # On static states/actions values
            trk_pred = model(q_track_states).cpu().detach().numpy()
            for l in range(0, trk_pred.shape[0]):
                string = []
                for itm in trk_pred[l, :]:
                    string.append(str(itm))
                file = open('{}/model/track_state_{}.csv'.format(name, l), 'a')
                file.write('{}\n'.format(';'.join(string)))
                file.close()
        avg_cum_rew, avg_duration = tester.test_it(nb_test=30)
        print('AVG cumulative reward: {} - AVG duration: {} time steps, over 30 random games'.format(avg_cum_rew,
                                                                                                     avg_duration))
        file = open('{}/model/performances.csv'.format(name), 'a')
        file.write('{};{};{}\n'.format(n, avg_cum_rew, avg_duration))
        file.close()

        # Save the model
        model.save(add=n)



if __name__ == '__main__':

    test = FQI()