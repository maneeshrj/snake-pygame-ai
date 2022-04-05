from Snake import Snake, Game
import random
from math import sqrt
from Counter import Counter
import featureExtractors as feat

def generateFixedActions():
    actions = ['RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'LEFT', 'LEFT','LEFT','LEFT', 'UP','UP','UP','UP']
    #actions = ['RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'DOWN', 'DOWN', 'LEFT', 'UP','UP','UP','UP','UP']
    #actions = ['RIGHT']*63
    return actions

def distance(a, b):
    return sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)    

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
