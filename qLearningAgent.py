import random
from util import Counter
import featureExtractors as feat

# APPROXIMATE Q-LEARNING AGENT
class ApproxQAgent: 
    def __init__(self, epsilon=0.05,gamma=0.5, alpha=0.7):
        self.snake = None
        self.env = None
        self.weights = Counter()
        self.featExtractor = feat.SimpleExtractor()
        self.epsilon = epsilon
        self.discount = gamma
        self.alpha = alpha
        self.training = True
        self.episodesSoFar = 0
        self.accumTrainRewards = 0.0
        self.accumTestRewards = 0.0

    def getWeights(self):
        return self.weights
    
    def setWeights(self, weights):
        self.weights = weights
    
    def isInTraining(self):
        return self.training

    def stopTraining(self):
        self.training = False
    
    def getNextAction(self):
        if(self.env == None): return None
        validActions = self.snake.get_valid_actions()
        action = None
        r = random.random()
        if (r < self.epsilon) and self.isInTraining():
            action = random.choice(validActions)
        else:
            action = self.computeActionFromQValues(self.env.getCurrentState())
        
        self.observeTransition(action)
        return action
    
    def computeActionFromQValues(self, state):
        bestValue, bestAction = float("-inf"), None
        for action in state.getValidActions():
            qVal = self.getQValue(state, action)
            if qVal > bestValue:
                bestValue, bestAction = qVal, action
        return bestAction
    
    def startEpisode(self, snake, env):
        self.episodeRewards = 0.0
        self.snake = snake
        self.env = env
    
    def stopEpisode(self):
        if self.training:
            self.accumTrainRewards += self.episodeRewards
        else:
            self.accumTestRewards += self.episodeRewards
            self.epsilon = 0.0    # no exploration
            self.alpha = 0.0      # no learning
        self.episodesSoFar += 1
        self.snake = None
        self.env = None
        
    def computeValueFromQValues(self, state):
        # if len(self.getLegalActions(state)) == 0:
        #     return 0.0
        bestValue = float("-inf")
        for action in state.getValidActions():
            qVal = self.getQValue(state, action)
            if qVal > bestValue:
                bestValue = qVal
        return bestValue
    
    def getValue(self, state):
        return self.computeValueFromQValues(state)
    
    # Implement approximative Q-learning
    def getQValue(self, state, action):
        feats = self.featExtractor.getFeatures(state, action)
        q = self.weights * feats
        #print('q',q)
        return q
    
    def observeTransition(self, action):
        reward = self.env.getReward(action)
        self.episodeRewards += reward
        if self.training:
           self.update(self.env.getCurrentState(), action, self.env.getNextState(action), reward)
    
    def update(self, state, action, nextState, reward):
        feats = self.featExtractor.getFeatures(state, action)
        diff = (reward + self.discount*self.getValue(nextState)) - self.getQValue(state,action)
        for i in feats:
            self.weights[i] += (self.alpha*diff)*feats[i]
            #print('weights',self.weights)
    
