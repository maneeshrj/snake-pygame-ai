from Snake import Snake, Game
import random
from reflexAgent import ReflexAgent
from randomAgent import RandomAgent
from qLearningAgent import ApproxQAgent

### DEFINED AGENTS:
randomAgent = lambda snake, env : RandomAgent(snake, env)
reflexAgent = lambda snake, env : ReflexAgent(snake, env)
approxQAgent = lambda snake, env : ApproxQAgent(snake, env)

def getAgentName(agent):
    if agent == randomAgent:
        return 'Random Agent'
    if agent == reflexAgent:
        return 'Reflex Agent'
    if agent == approxQAgent:
        return 'Approx-Q Agent'
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
