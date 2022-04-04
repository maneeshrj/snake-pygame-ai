from Snake import Snake, Game
import random
from math import sqrt
import Counter

def generateFixedActions():
    actions = ['RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'LEFT', 'LEFT','LEFT','LEFT', 'UP','UP','UP','UP']
    #actions = ['RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'DOWN', 'DOWN', 'LEFT', 'UP','UP','UP','UP','UP']
    #actions = ['RIGHT']*63
    return actions

def distance(a, b):
    return sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

# RANDOM AGENT
# At each time step, randomly picks an action from the valid choices.
class RandomAgent:
    def __init__(self, snake, env):
        self.snake = snake
        self.env = env
    
    def getNextAction(self):
        validActions = self.snake.get_valid_actions()
        return random.choice(validActions)

# REFLEX AGENT
# At each time step, takes the action that most decreases the distance
# between the head of the snake and the food.
class ReflexAgent:
    def __init__(self, snake, env):
        self.snake = snake
        self.env = env
    
    def getNextAction(self):
        validActions = self.snake.get_valid_actions()
        currentPos = self.snake.pos
        foodPos = self.env.food_pos
        newPos = None
        bestDist, bestAction = float('inf'), None
        for action in validActions:
            newPos = [self.snake.pos[0][0], self.snake.pos[0][1]] 
            if action == 'UP':
                newPos = [self.snake.pos[0][0], self.snake.pos[0][1] - 10] 
            if action == 'DOWN':
                newPos = [self.snake.pos[0][0], self.snake.pos[0][1] + 10] 
            if action == 'LEFT':
                newPos = [self.snake.pos[0][0] - 10, self.snake.pos[0][1]] 
            if action == 'RIGHT':
                newPos = [self.snake.pos[0][0] + 10, self.snake.pos[0][1]] 
            if action == 'CONTINUE':
                newPos = self.snake.pos[0]

            if distance(newPos, foodPos) < bestDist:
                bestDist = distance(newPos, foodPos)
                bestAction = action                
        return bestAction
    
# APPROXIMATE Q-LEARNING AGENT
class ApproxQAgent:
    def __init__(self, snake, env):
        self.snake = snake
        self.env = env
        self.counter = Counter.Counter()
    
    def getNextAction(self):
        # validActions = self.snake.get_valid_actions()
        # return random.choice(validActions)
        pass
    # Implement approximative Q-learning
    def getQValue(self, state, action):
        pass
    def update(self, state, action, nextState, reward):
        pass

    def rewards(self):
        pass
    

### DEFINED AGENTS:
randomAgent = lambda snake, env : RandomAgent(snake, env)
reflexAgent = lambda snake, env : ReflexAgent(snake, env)

def getAgentName(agent):
    if agent==randomAgent:
        return 'Random Agent'
    if agent==reflexAgent:
        return 'Reflex Agent'
    return 'unknown agent'

if __name__ == "__main__":
    snake = Snake()
    env = Game(snake)
    agent = ReflexAgent(snake, env)
    
    timestep = 0
    while True:
        timestep += 1
        if (timestep % 50) == 0:
            print('step', timestep)
        action = agent.getNextAction()
        is_over, score = env.play_step(action)
        # print(is_over)
        if is_over:
            print("Game Over")
            print("Score: ", score)
            env.game_over()
            break
