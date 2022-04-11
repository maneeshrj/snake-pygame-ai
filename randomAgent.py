import random

# RANDOM AGENT
# At each time step, randomly picks an action from the valid choices.
class RandomAgent:
    def __init__(self, gameState, env):
        self.gameState = gameState
        self.env = env
    
    def __str__(self):
        return "RandomAgent"
    
    def getNextAction(self):
        validActions = self.gameState.getValidActions()
        return random.choice(validActions)