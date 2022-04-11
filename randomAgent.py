import random

# RANDOM AGENT
# At each time step, randomly picks an action from the valid choices.
class RandomAgent:
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
        
    
    def getNextAction(self):
        validActions = self.gameState.getValidActions()
        return random.choice(validActions)