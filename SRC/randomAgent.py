# Imports
import random

class RandomAgent:
    '''
    A Random agent picks a random action at each step
    '''
    def __init__(self):
        self.accumulatedRewards = 0
        self.gameState = None

    def __str__(self):
        return "RandomAgent"

    def startEpisode(self, gameState):
        self.gameState = gameState

    def stopEpisode(self):
        self.accumulatedRewards += self.gameState.score
        self.gameState = None

    # Choose a random action out of the valid action set
    def getNextAction(self):
        validActions = self.gameState.getValidActions()
        return random.choice(validActions)
