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

from dqnTrain import select_action
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

            start_matrix = self.gameState.getAsMatrix()
            # next_matrix = self.gameState.getSuccessor('CONTINUE').getAsMatrix()
            # state = torch.tensor(np.dstack((start_matrix, next_matrix)), device=device, dtype=torch.float)
            state = torch.tensor(start_matrix, device=device, dtype=torch.float).unsqueeze(-1)
            state = state.unsqueeze(0)
            return tensor_to_action(select_action(state, self.gameState.getValidActions(), -1, self.net))
