from Snake import Game, GameState
import random
from reflexAgent import ReflexAgent
from randomAgent import RandomAgent
from qLearningAgent import QLearningAgent

# from qLearningAgent import ApproxQAgent


AGENT_MAP = {
    # 'approxq': ag.approxQAgent,
    'reflex': ReflexAgent,
    'random': RandomAgent,
    'exactq': QLearningAgent,
}


### DEFINED AGENTS:
def getAgentName(agent):
    if agent == RandomAgent:
        return 'Random Agent'
    if agent == ReflexAgent:
        return 'Reflex Agent'
    if agent == QLearningAgent:
        return 'Q-Learning Agent'
    # if agent==approxQAgent:
    #     return 'Approximate Q-Learning Agent'
    return 'unknown agent'
