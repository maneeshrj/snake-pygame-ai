from Snake import Game, GameState
import random
from reflexAgent import ReflexAgent
from randomAgent import RandomAgent
from qLearningAgent import ApproxQAgent, QLearningAgent

### DEFINED AGENTS:
randomAgent = lambda gameState, env : RandomAgent(gameState, env)
reflexAgent = lambda gameState, env : ReflexAgent(gameState, env)
approxQAgent = lambda gameState, env : ApproxQAgent(gameState, env)
qLearnAgent = lambda gameState, env : QLearningAgent(gameState, env)