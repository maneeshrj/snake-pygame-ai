import random
from util import Counter
import featureExtractors as feat

### EXACT Q LEARNING AGENT
class QLearningAgent:
    def __init__(self, numTraining=100, epsilon=0.5, gamma=0.8, alpha=0.2):
        self.numTraining = numTraining
        self.epsilon = epsilon
        self.discount = gamma
        self.alpha = alpha
        self.qValues = Counter()
        self.episodesSoFar = 0
        self.accumTrainRewards = 0.0
        self.accumTestRewards = 0.0
        self.training = True
    
    def __str__(self):
        return "QLearningAgent"
    
    def getQValue(self, state, action):
        """
            Returns Q(state,action)
            Should return 0.0 if we have never seen a state
            or the Q node value otherwise
        """
        return self.qValues[(state, action)]# if (state, action) in self.qValues else 0.0
    
    def startEpisode(self, gameState):
        self.gameState = gameState
        self.lastAction = None
        self.episodeRewards = 0.0
    
    def stopEpisode(self):
        if self.episodesSoFar < self.numTraining:
            self.accumTrainRewards += self.episodeRewards
        else:
            self.accumTestRewards += self.episodeRewards
        self.episodesSoFar += 1
        if self.episodesSoFar >= self.numTraining:
            self.epsilon = 0.0
            self.alpha = 0.0

    def computeActionFromQValues(self, state):
        """
            Compute the best action to take in a state.  Note that if there
            are no legal actions, which is the case at the terminal state,
            you should return None.
        """
        legalActions = state.getValidActions()
        if len(legalActions) == 0:
            return None
        else:
            maxQ = float("-inf")
            bestActions = []
            for action in legalActions:
                q = self.getQValue(state, action)
                if q > maxQ:
                  maxQ = q
                  bestActions = [action]
                elif q == maxQ:
                  bestActions.append(action)
            chosenAction = random.choice(bestActions)
            # print(f"Picking a policy action {chosenAction} with a Q value of {maxQ}")
            return chosenAction
    
    def computeValueFromQValues(self, state):
        """
            Compute the best value to take for a state.  Note that if there
            are no legal actions, which is the case at the terminal state,
            you should return a value of 0.0.
        """
        legalActions = state.getValidActions()
        if len(legalActions) == 0:
            return 0.0
        else:
            maxQ = float("-inf")
            for action in legalActions:
                q = self.getQValue(state, action)
                if q > maxQ:
                  maxQ = q
            return maxQ
    
    def getNextAction(self, state):
        """
            Compute the action to take in the current state.  With
            probability self.epsilon, we should take a random action and
            take the best policy action otherwise.  Note that if there are
            no legal actions, which is the case at the terminal state, you
            should choose None as the action.
        """
        legalActions = state.getValidActions()
        action = None
        if len(legalActions) == 0:
            return None
        else:
            r = random.random()
            if r < self.epsilon and self.training:
                action = random.choice(legalActions)
                # print(f"Picking a random action {action}")
            else:
                action = self.getPolicy(state)
        self.observeTransition(state, action, state.getSuccessor(action), state.getReward(action))
        return action

    def update(self, state, action, nextState, reward):
        """
            The parent class calls this to observe a
            state = action => nextState and reward transition.
            You should do your Q-Value update here
            NOTE: You should never call this function,
            it will be called on your behalf
        """
        sample = reward + self.discount*self.getValue(nextState)
        self.qValues[(state, action)] = self.getQValue(state, action) + self.alpha*(sample - self.getQValue(state, action))

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)
    
    def getValue(self, state):
        return self.computeValueFromQValues(state)

    def observeTransition(self, state,action,nextState,deltaReward):
        """
            Called by environment to inform agent that a transition has
            been observed. This will result in a call to self.update
            on the same arguments
        """
        self.episodeRewards += deltaReward
        self.update(state,action,nextState,deltaReward)

    def stopTraining(self):
        self.training = False
     
        
     
        
### APPROXIMATE Q-LEARNING AGENT
# class ApproxQAgent: 
#     def __init__(self, epsilon=1.0, gamma=0.99, alpha=0.25):
#         self.gameState = None
#         self.env = None
#         self.weights = Counter()
#         self.featExtractor = feat.SimpleExtractor()
#         self.epsilon = epsilon
#         self.discount = gamma
#         self.alpha = alpha
#         self.training = True
#         self.episodesSoFar = 0
#         self.accumTrainRewards = 0.0
#         self.accumTestRewards = 0.0
    
#     def __str__(self):
#         return "ApproxQAgent"

#     def getWeights(self):
#         return self.weights
    
#     def setWeights(self, weights):
#         self.weights = weights
    
#     def isInTraining(self):
#         return self.training

#     def stopTraining(self):
#         self.training = False
    
#     def getNextAction(self):
#         if(self.env == None): return None
#         validActions = self.gameState.getValidActions()
#         action = None
#         r = random.random()
#         if (r < self.epsilon) and self.isInTraining():
#             action = random.choice(validActions)
#         else:
#             action = self.computeActionFromQValues(self.env.getCurrentState())
        
#         self.observeTransition(action)
#         return action
    
#     def computeActionFromQValues(self, state):
#         bestValue, bestAction = float("-inf"), None
#         print('\nPicking from options:')
#         for action in state.getValidActions():
#             qVal = self.getQValue(state, action)
#             if qVal > bestValue:
#                 bestValue, bestAction = qVal, action
#         print('Picked', bestAction)
#         return bestAction
    
#     def startEpisode(self, gameState, env):
#         self.episodeRewards = 0.0
#         self.gameState = gameState
#         self.env = env
    
#     def stopEpisode(self):
#         if self.training:
#             self.accumTrainRewards += self.episodeRewards
#         else:
#             self.accumTestRewards += self.episodeRewards
#             self.epsilon = 0.0    # no exploration
#             self.alpha = 0.0      # no learning
#         self.episodesSoFar += 1
#         self.gameState = None
#         self.env = None
        
#     def computeValueFromQValues(self, state):
#         # if len(self.getValidActions(state)) == 0:
#         #     return 0.0
#         bestValue = float("-inf")
#         for action in state.getValidActions():
#             qVal = self.getQValue(state, action)
#             if qVal > bestValue:
#                 bestValue = qVal
#         return bestValue
    
#     def getValue(self, state):
#         return self.computeValueFromQValues(state)
    
#     # Implement approximative Q-learning
#     def getQValue(self, state, action):
#         feats = self.featExtractor.getFeatures(state, action)
#         q = self.weights * feats # dot product of W and features, weighted linear function
#         if not self.training:
#             print('Action:', action, ', q:', q)
#         return q
    
#     def observeTransition(self, action):
#         reward = self.env.getRewardAlive(action)
#         self.episodeRewards += reward
#         if self.training:
#            self.update(self.env.getCurrentState(), action, self.env.getNextState(action), reward)
    
#     def update(self, state, action, nextState, reward):
#         feats = self.featExtractor.getFeatures(state, action)
#         diff = (reward + self.discount*self.getValue(nextState)) - self.getQValue(state,action)
#         for i in feats:
#             self.weights[i] += (self.alpha*diff)*feats[i]
#             #print('weights',self.weights)
