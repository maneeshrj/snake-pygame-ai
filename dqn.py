import random
from collections import namedtuple, deque

import torch.nn as nn
import torch.nn.functional as F

# Moves last dimension of 3D tensor to the front
def b2f(inp):
    if inp.ndim==3: return inp.permute(2,0,1)
    elif inp.ndim==4: return inp.permute(0,3,1,2)
    else: print('wrong dimensions')

# Moves first dimension of 3D tensor to the back
def f2b(inp):
    if inp.ndim==3: return inp.permute(1,2,0)
    else: print('wrong dimensions')

# Convert tensor to action
def tensor_to_action(tensor):
    action_num = tensor.item()
    if action_num == 0:
        return "UP"
    elif action_num == 1:
        return "DOWN"
    elif action_num == 2:
        return "LEFT"
    elif action_num == 3:
        return "RIGHT"
    elif action_num == 4:
        return "CONTINUE"

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
