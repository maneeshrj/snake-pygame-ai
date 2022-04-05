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
    
# APPROXIMATE Q-LEARNING AGENT
class ApproxQAgent: 
    def __init__(self, snake, env, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0):
        self.snake = snake
        self.env = env
        self.weights = Counter()
        self.featExtractor = feat.SimpleExtractor()
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.numTraining = numTraining
        self.episodesSoFar = 0
        self.accumTrainRewards = 0.0
        self.accumTestRewards = 0.0

    def getWeights(self):
        return self.weights
    
    def isInTraining(self):
        return self.episodesSoFar < self.numTraining

    def isInTesting(self):
        return not self.isInTraining()
    
    def getNextAction(self):
        validActions = self.env.getValidActions()
        action = None
        r = random.random()
        if (r < self.epsilon) and self.isInTraining():
            action = random.choice(validActions)
        else:
            action = self.computeActionFromQValues()
        
        self.observeTransition(action)
        return action
    
    def computeActionFromQValues(self, state):
        bestValue, bestAction = float("-inf"), None
        for action in state.getValidActions():
            qVal = self.getQValue(state, action)
            if qVal > bestValue:
                bestValue, bestAction = qVal, action
        return bestAction
    
    def stopEpisode(self):
        if self.episodesSoFar < self.numTraining:
            self.accumTrainRewards += self.episodeRewards
        else:
            self.accumTestRewards += self.episodeRewards
            self.epsilon = 0.0    # no exploration
            self.alpha = 0.0      # no learning
        self.episodesSoFar += 1
        self.episodeRewards = 0.0
        
    def computeValueFromQValues(self, state):
        # if len(self.getLegalActions(state)) == 0:
        #     return 0.0
        bestValue, bestAction = float("-inf"), None
        for action in state.getValidActions():
            qVal = self.getQValue(action)
            if qVal > bestValue:
                bestValue, bestAction = qVal, action
        return bestValue
    
    def getValue(self, state):
        return self.computeValueFromQValues()
    
    # Implement approximative Q-learning
    def getQValue(self, state, action):
        feats = self.featExtractor.getFeatures(state, action)
        q = self.weights * feats
        return q
    
    def observeTransition(self, action):
        reward = self.env.getReward(action)
        self.update(self.env.getCurrentState(), action, self.env.getNextState, reward)
    
    def update(self, state, action, nextState, reward):
        feats = self.featExtractor.getFeatures(state, action)
        diff = (reward + self.discount*self.getValue(nextState)) - self.getQValue(state,action)
        for i in feats:
            self.weights[i] += (self.alpha*diff)*feats[i]
    

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
