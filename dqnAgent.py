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

from itertools import count
from Snake import Game, GameState, Trial

from dqn import DQN, ReplayMemory, tensor_to_action, Transition

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNAgent:
    '''
    A DQN agent uses DQN and convolutional neural networks to learn the optimal policy for the game.
    '''
    def __init__(self):
        self.gameState = None
        self.training = False
        self.net = None

    def __str__(self):
        return "DQNAgent"

    def loadNetwork(self, net):
        self.net = net

    def startEpisode(self, gameState):
        self.gameState = gameState
        self.lastAction = None
        
    def stopTraining(self):
        self.training = False
        
    def stopEpisode(self):
        if self.training:
            self.epsilon -= (self.initEpsilon)*(1.0/(self.numTraining+1))
            if self.epsilon < 0: self.epsilon = 0
        else:
            self.epsilon = 0.0
            self.alpha = 0.0

    def getNextAction(self):
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            #print("NETWORK OUTPUT", policy_net(state))
            start_matrix = self.gameState.getAsMatrix()
            next_matrix = self.gameState.getSuccessor('CONTINUE').getAsMatrix()
            state = torch.tensor(np.dstack((start_matrix, next_matrix)), device=device, dtype=torch.float)
            state = state.unsqueeze(0)

            network_output = self.net(state)
            output_np = network_output.cpu().detach().numpy()[0]

            action_as_str = None
            actionDict = {}
            
            actionDict['UP'] = output_np[0]
            actionDict['DOWN'] = output_np[1]
            actionDict['LEFT'] = output_np[2]
            actionDict['RIGHT'] = output_np[3]
            actionDict['CONTINUE'] = output_np[4]

            # shuffle the valid actions
            valid_actions = self.gameState.getValidActions()
            random.shuffle(valid_actions)
            maxActionValue = actionDict[valid_actions[0]]
            action_as_str = valid_actions[0]
            for action in valid_actions[1:]:
                if actionDict[action] >= maxActionValue:
                    maxActionValue = actionDict[action]
                    action_as_str = action

            return action_as_str
