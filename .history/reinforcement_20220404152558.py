from Snake import Snake, Game
import random
from math import sqrt
import reflexAgent
import randomAgent

### DEFINED AGENTS:
randomAgent = lambda snake, env : reflexAgent.RandomAgent(snake, env)
reflexAgent = lambda snake, env : reflexAgent.ReflexAgent(snake, env)
approxQAgent = lambda snake, env : reflexAgent.ReflexAgent(snake, env)

def getAgentName(agent):
    if agent== randomAgent:
        return 'Random Agent'
    if agent == reflexAgent:
        return 'Reflex Agent'
    if agent == approxQAgent:
        return 'approxQAgent'
    return 'unknown agent'

if __name__ == "__main__":
    snake = Snake()
    env = Game(snake)
    agent = reflexAgent(snake, env)
    
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