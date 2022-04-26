#%% Imports
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

from itertools import count
from Snake import Game, GameState, Trial

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Moves last dimension of 3D tensor to the front
def b2f(inp):
    if inp.ndim==3: return inp.permute(2,0,1)
    elif inp.ndim==4: return inp.permute(0,3,1,2)
    else: print('wrong dimensions')

# Moves first dimension of 3D tensor to the back
def f2b(inp):
    if inp.ndim==3: return inp.permute(1,2,0)
    else: print('wrong dimensions')

#%% DQN Model Setup
# Represents a transition from one state to another
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Replay Memory stores the last <capacity> experiences to sample from
# for training the DQN
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# DQN
# Should take a tensor of shape (10, 10, 2) and return a tensor of shape (5)
class DQN(nn.Module):
    
        def __init__(self, input_shape, n_actions):
            super(DQN, self).__init__()
            self.conv1 = nn.Conv2d(input_shape[2], 32, kernel_size=2, stride=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=2, stride=1)
            self.conv3 = nn.Conv2d(64, 64, kernel_size=2, stride=1)
            self.fc1 = nn.Linear(3136, 512)
            self.fc2 = nn.Linear(512, n_actions)
    
        def forward(self, x):
            x = b2f(x)#.unsqueeze(0)
            # print('1',x.shape)
            x = F.relu(self.conv1(x))
            # print('2',x.shape)
            x = F.relu(self.conv2(x))
            # print('3',x.shape)
            x = F.relu(self.conv3(x))
            # print('4',x.shape)
            x = x.view(x.size(0), -1)
            # print('5',x.shape)
            x = F.relu(self.fc1(x.view(x.size(0), -1)))
            # print('6',x.shape)
            x = self.fc2(x)
            # print('7',x.shape)
            # print()
            return x

#%% Training Setup
BATCH_SIZE = 128    # Originally 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

grid_height = grid_width = 10

# Number of actions
n_actions = 5

policy_net = DQN((grid_height, grid_width, 2), n_actions).to(device)
target_net = DQN((grid_height, grid_width, 2), n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            a = policy_net(state).max(dim=1)[1].view(1, 1)
            # print(a.shape)
            a = a.type(torch.LongTensor)
            # print(a)
            return a
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


episode_durations = []

def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated

#%% Optimizer
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
    print('calling policy net')
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    
    print('calling target net')
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

#%% Training Loop
# Want to loop over the number of epidodes
# For each episode, start by getting the initial input to the network 
# which is the frame from the game in the current state frame stacked on top of the
# next state frame
# Then we play the game in steps and at each step, select an action, take the
# action, and then observe the next state and get our new input to the network
# which is the frame from the game in the current state frame stacked on top of the
# next state frame
# Store the experience (state, action, reward, next_state) in the memory
# Move to the next state
# Optimize the model
# if the episode is done, then we reset the game and start a new episode updating the 
# target network with the policy network every TARGET_UPDATE steps

learningTrial = Trial()

num_episodes = 20
for ep in range(num_episodes):
    print('Epoch', ep)
    if (ep == 18):
        print("here")
    # Initialize the environment and state
    gameState = GameState(pos=[[30, 20], [20, 20], [10, 20]], direction='RIGHT',
                          frameSizeX=100, frameSizeY=100)
    game = Game(gameState, graphics=False, plain=True, 
                foodPosList=learningTrial.getFoodPosList())
    game.setFoodPos()

    gameOver = False

    # The state is the game frame stacked on top of the next state frame
    start_matrix = game.getCurrentState().getAsMatrix()
    next_matrix = game.getNextState("CONTINUE").getAsMatrix()
    state = torch.tensor(np.dstack((start_matrix, next_matrix)), device=device, dtype=torch.float)
    state = state.unsqueeze(0)

    t = 0
    while not gameOver:

        action = select_action(state)
        
        reward = game.getReward(action)
        gameOver, score = game.playStep(action)
        reward = torch.tensor([reward], device=device, dtype=torch.float)

        last_matrix = next_matrix
        next_matrix = game.getCurrentState().getAsMatrix()

        if not gameOver:
            # print(t)
            next_state = torch.tensor(np.dstack((start_matrix, next_matrix)), device=device, dtype=torch.float)
            next_state = next_state.unsqueeze(0)
        else:
            next_state = None

        # Save the experience to our memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        t += 1
    episode_durations.append(t + 1)
    if ep % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Episode durations:',episode_durations)
print('Complete')
# plt.ioff()
plt.show()