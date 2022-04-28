# Imports
from Snake import Game, GameState
import random
from reflexAgent import ReflexAgent
from randomAgent import RandomAgent
from qLearningAgent import QLearningAgent, ApproxQAgent
from dqn import DQN

# Defined Agents:
AGENT_MAP = {
    'reflex': ReflexAgent,
    'random': RandomAgent,
    'exactq': QLearningAgent,
    'approxq': ApproxQAgent,
    'dqn': DQN
}

# Return the name of the agent
def getAgentName(agent):
    if agent == RandomAgent:
        return 'Random Agent'
    if agent == ReflexAgent:
        return 'Reflex Agent'
    if agent == QLearningAgent:
        return 'Q-Learning Agent'
    if agent == ApproxQAgent:
        return 'Approximate Q-Learning Agent'
    if agent == DQN:
        return 'Deep Q-Learning Agent'
    return 'unknown agent'
