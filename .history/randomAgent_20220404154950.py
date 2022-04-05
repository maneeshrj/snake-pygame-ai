import random

# RANDOM AGENT
# At each time step, randomly picks an action from the valid choices.
class RandomAgent:
    def __init__(self, snake, env):
        self.snake = snake
        self.env = env
    
    def getNextAction(self):
        validActions = self.snake.get_valid_actions()
        return random.choice(validActions)