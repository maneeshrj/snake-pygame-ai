# Imports
import matplotlib.pyplot as plt
import numpy as np
from collections import deque, namedtuple
import random
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
import math
import time
import argparse
import csv

from itertools import count
from Snake import Game, GameState, Trial

from dqn import DQN, ReplayMemory, tensor_to_action, Transition

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def select_action(state, valid_actions, cur_epoch, network):
    sample = random.random()
    if cur_epoch >= 0:
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * cur_epoch / EPS_DECAY)
    else:
        eps_threshold = 0.0

    action_as_str = None
    actionDict = {}
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            #print("NETWORK OUTPUT", policy_net(state))
            network_output = network(state)
            output_np = network_output.cpu().detach().numpy()[0]

            actionDict['UP'] = output_np[0]
            actionDict['DOWN'] = output_np[1]
            actionDict['LEFT'] = output_np[2]
            actionDict['RIGHT'] = output_np[3]
            actionDict['CONTINUE'] = output_np[4]

            # shuffle the valid actions
            random.shuffle(valid_actions)
            maxActionValue = actionDict[valid_actions[0]]
            action_as_str = valid_actions[0]
            for action in valid_actions[1:]:
                if actionDict[action] >= maxActionValue:
                    maxActionValue = actionDict[action]
                    action_as_str = action
    else:
        # print("Taking random action")
        # pick a random action from the list of valid actions
        action_as_str = random.choice(valid_actions)
        # print(action_as_str)
    
    action_num = -1
    if action_as_str == 'UP':
        action_num = 0
    elif action_as_str == 'DOWN':
        action_num = 1
    elif action_as_str == 'LEFT':
        action_num = 2
    elif action_as_str == 'RIGHT':
        action_num = 3
    elif action_as_str == 'CONTINUE':
        action_num = 4
    else:
        # Throw error, invalid action
        print("Invalid action")
    action = torch.tensor([[action_num]], device=device, dtype=torch.long)
    
    return action

# Optimizer
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    # state_batch = torch.cat(batch.state)
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    # print('calling policy net')
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    
    # print('calling target net')
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return loss.item()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--episodes', type=int, default=100, help='number of episodes to train for')
    parser.add_argument('-g', '--graphics', action='store_true', help='show graphics')
    parser.add_argument('-s', '--save', action='store_true', help='save model')
    parser.add_argument('-r', '--random_food', action='store_true', help='random food')
    parser.add_argument('-d', '--record_data', action='store_true', help='record run data')

    args = parser.parse_args()
    num_episodes = args.episodes
    show_graphics = args.graphics
    save_model = args.save
    random_food = args.random_food
    record_data = args.record_data

    if torch.cuda.is_available():
        print(f"{torch.cuda.device_count()} GPU(s) available.")
    else:
        print('No GPUs available, using CPU.')

    # Training Setup
    BATCH_SIZE = 128    # Originally 128
    GAMMA = 0.8
    EPS_START = 0.90
    EPS_END = 0.05
    EPS_DECAY = 200
    # EPS_DECAY = num_episodes // 2
    TARGET_UPDATE = 10

    grid_height = grid_width = 10

    # Number of actions
    n_actions = 5

    policy_net = DQN((grid_height, grid_width, 1), n_actions).to(device)
    target_net = DQN((grid_height, grid_width, 1), n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.RMSprop(policy_net.parameters())
    memory = ReplayMemory(20000)

    # steps_done = 0

    learningTrial = Trial()
    moreThanZeroScores = []
    run_data = []
    print("Starting Training")
    
    intervalLengths, intervalScores, epochTimes = [], [], []
    for ep in range(1, num_episodes+1):
        start_time = time.time()
        
        # Initialize the environment and state
        gameState = GameState(pos=[[30, 20], [20, 20], [10, 20]], direction='RIGHT',
                            frameSizeX=100, frameSizeY=100)
        
        # If halfway through the epochs, then run the graphics
        trainGraphics = False
        # Mute for training on Argon:
        if show_graphics:
            if ep % (num_episodes // 5) == 0:
                trainGraphics = True
        
        if random_food:
            game = Game(gameState, graphics=trainGraphics, plain=True, randomFood=True, framerate=5)
        else:
            game = Game(gameState, graphics=trainGraphics, plain=True, 
                        foodPosList=learningTrial.getFoodPosList(), framerate=5)
        
        learningTrial.setCurrentGame(game)
        game.setFoodPos()
        
        gameOver = False

        # The state is the game frame stacked on top of the next state frame
        start_matrix = game.getCurrentState().getAsMatrix()
        next_matrix = game.getNextState("CONTINUE").getAsMatrix()
        state = torch.tensor(np.dstack((start_matrix, next_matrix)), dtype=torch.float)
        # state = torch.tensor(start_matrix, dtype=torch.float).unsqueeze(-1)
        state = state.unsqueeze(0)

        t = 0
        while not gameOver:
            valid_actions = game.gameState.getValidActions()
            action_tensor = select_action(state, valid_actions, ep, policy_net)
            action_str = tensor_to_action(action_tensor)
            
            reward = game.getReward(action_str)
            gameOver, score = game.playStep(action_str)
            reward = torch.tensor([reward], device=device, dtype=torch.float)

            # last_matrix = next_matrix
            next_matrix = game.getCurrentState().getAsMatrix()

            if not gameOver:
                # print(t)
                # next_state = torch.tensor(np.dstack((start_matrix, next_matrix)), dtype=torch.float)
                next_state = torch.tensor(next_matrix, dtype=torch.float).unsqueeze(-1)
                next_state = next_state.unsqueeze(0)
                # print(next_state.shape)
            else:
                next_state = None
                # Mute for training on ARGON:
                """if score > 0:
                    moreThanZeroScores.append(score)
                if ep % (num_episodes // 10) == 0:
                    print(moreThanZeroScores)
                    moreThanZeroScores = []"""

            # Save the experience to our memory
            memory.push(state, action_tensor, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            loss = optimize_model()
            t += 1
        
        game.gameOver()
        
        intervalScores.append(score)
        intervalLengths.append(t)
        epochTimes.append((time.time() - start_time))
        
        #episode_durations.append(t + 1)
        if ep % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
        curr_eps = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * ep / EPS_DECAY)
        # Time, score, eps
        if record_data:
            run_data.append([epochTimes[-1], score, curr_eps, loss])
            
        if ep % (num_episodes // 10) == 0:
            epSummary = '\nEpoch {:<3d}\tAvg_score={:<3.2f}\tNonzeros={:d}\tMax={:<3d}'.format(ep, np.mean(intervalScores), np.count_nonzero(intervalScores), max(intervalScores))
            epSummary += '\nAvg_len={:<.2f}\tMax_len={:<4d}\teps={:<.2f}\t({:<.2f} sec/ep)'.format(np.mean(intervalLengths),max(intervalLengths),curr_eps, np.mean(epochTimes))
            # print('Scores:', intervalScores)
            intervalScores, intervalLengths, epochTimes = [], [], []
            print(epSummary)
            
    
    print('Training Complete')
    
    if save_model:
        torch.save(target_net.state_dict(), f'DQN_{num_episodes}_epochs.pth')
        print(f"Saved model to DQN_{num_episodes}_epochs.pth")
    
    if record_data:
        # Save the run data
        with open('run_data.csv', 'w') as csvfile:
            writer = csv.writer(csvfile)
            # Write the header
            writer.writerow(['time', 'score', 'eps', 'loss'])
            writer.writerows(run_data)
            print("Saved run data to run_data.csv")