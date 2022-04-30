import sys, random
import pickle
from util import Counter, updatePosition, manhattanDistance, distance
import numpy as np
# import featureExtractors as feat


### EXACT Q LEARNING AGENT
class QLearningAgent:
    def __init__(self):
        self.qValues = Counter()
        self.episodesSoFar = 0
        self.accumTrainRewards = 0.0
        self.accumTestRewards = 0.0
        self.training = False
        self.step = 0

    def __str__(self):
        return "QLearningAgent"

    def getQValue(self, state, action):
        """
            Returns Q(state,action)
            Should return 0.0 if we have never seen a state
            or the Q node value otherwise
        """
        return self.qValues[(state, action)]  # if (state, action) in self.qValues else 0.0

    def startEpisode(self, gameState):
        self.gameState = gameState
        self.lastAction = None
        self.episodeRewards = 0.0
        self.step = 0

    def stopEpisode(self):
        self.step = 0
        if self.training:
            self.accumTrainRewards += self.episodeRewards
            self.epsilon -= (self.initEpsilon)*(1.0/(self.numTraining+1))
            if self.epsilon < 0: self.epsilon = 0
        else:
            self.accumTestRewards += self.episodeRewards
            self.epsilon = 0.0
            self.alpha = 0.0
        self.episodesSoFar += 1
            

    def computeActionFromQValues(self, state):
        """
            Compute the best action to take in a state.
        """
        legalActions = state.getValidActions()
        if len(legalActions) == 0:
            return None
        else:
            maxQ = float("-inf")
            bestActions = []
            for action in legalActions:
                # print('Calling getQValue from computeActionFromQValues')
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
            Compute the best value to take for a state. If there
            are no legal actions, returns a value of 0.0.
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

    def getNextAction(self):
        """
            Compute the action to take in the current state.  With
            probability self.epsilon, we should take a random action and
            take the best policy action otherwise.  Note that if there are
            no legal actions, which is the case at the terminal state, you
            should choose None as the action.
        """
        state = self.gameState
        legalActions = state.getValidActions()
        
        # print('In getNextAction, LEGAL ACTIONS=',legalActions, 'w/ current direction=', state.direction)
        action = None
        if len(legalActions) == 0:
            return None
        else:
            if not self.training:
                action = self.getPolicy(state)
            else:
                if random.random() < self.epsilon:
                    action = random.choice(legalActions)
                    # print(f"In getNextAction, picking a random action {action}")
                else:
                    action = self.getPolicy(state)
                    # print(f"In getNextAction, picking a policy action {action}")
        self.step += 1
        self.observeTransition(state, action, state.getSuccessor(action), state.getReward(action, self.step))
        return action

    def update(self, state, action, nextState, reward):
        """
            The parent class calls this to observe a
            state = action => nextState and reward transition.
            You should do your Q-Value update here
            NOTE: You should never call this function,
            it will be called on your behalf
        """
        sample = reward + self.discount * self.getValue(nextState)
        self.qValues[(state, action)] = self.getQValue(state, action) + self.alpha * (
                    sample - self.getQValue(state, action))

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)

    def observeTransition(self, state, action, nextState, deltaReward):
        """
            Called by environment to inform agent that a transition has
            been observed. This will result in a call to self.update
            on the same arguments
        """
        self.episodeRewards += deltaReward
        if self.training:
            self.update(state, action, nextState, deltaReward)

    def stopTraining(self):
        self.training = False

    def loadCheckpoint(self, fname='qvalues.pkl'):
        """
        Loads the q values into the agents qValues table from
        the given pickle file.
        """
        with open(fname, 'rb') as f:
            d = pickle.load(f)
            self.qValues.loadDict(d)
            
    def saveCheckpoint(self, fname='qvalues.pkl'):
        qValDict = self.qValues.asDict()
        print('Q Value dictionary size:', sys.getsizeof(qValDict), 'bytes')
        print('Number of Q-states explored:', len(qValDict))
        with open(fname, 'wb') as f:
            pickle.dump(qValDict, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(fname, 'rb') as f:
            counts = pickle.load(f)
            print('Length of qval dict saved:', len(counts))
    
    def startTraining(self, alpha=0.3, gamma=0.8, epsilon=0.8, numTraining=100):
        self.training = True
        self.discount = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.initEpsilon = epsilon
        self.numTraining = numTraining


### APPROX Q LEARNING FEATURE EXTRACTORS
def getFeatures(state, action, extractor=1):
    features = Counter()
    features["bias"] = 1.0

    # all features bases on next state
    curPos = state.getSnakePos()
    curHead = curPos[0]
    nextHead = updatePosition(curHead, action)
    foodPos = state.getFoodPos()
    
    frameX, frameY = (state.frameX//10), (state.frameY//10)
    
    # get distance to food as a number between 0 and 1
    nextFoodDist = distance(nextHead, foodPos) / (frameX * frameY)
    currFoodDist = distance(curHead, foodPos) / (frameX * frameY)
    # features['foodDist'] = nextFoodDist - currFoodDist
    features['foodDist'] = nextFoodDist

    if extractor == 1:
        # Get the next state as a matrix
        nextState = state.getSuccessor(action)
        nextMat = nextState.getAsMatrix()
        nextX, nextY = nextHead[0]//10, nextHead[1]//10
        direction =  nextState.direction
        
        minDistToBody = float("inf")
        for bodyElem in curPos[1:]:
            distToBody = distance(nextHead, bodyElem)
            if distToBody < minDistToBody:
                minDistToBody = distToBody
        
        features['minDistToBody'] = (minDistToBody / (frameX * frameY))
        
        foodX, foodY = foodPos[0]//10, foodPos[1]//10
        features['bodyObstacle'] = 0.
        features['bodyObstacle2'] = 0.
        if foodX == nextX:
            if direction == 'UP':# and foodY < nextY:
                for i in range(0,nextY):
                    if nextMat[i, nextX] > 0:
                        features['bodyObstacle'] = 1.0
                        
            if direction == 'DOWN':# and foodY > nextY:
                for i in range(nextY+1, frameY):
                    if nextMat[i, nextX] > 0:
                        features['bodyObstacle'] = 1.0
        elif foodY == nextY:
            if direction == 'LEFT':# and foodX < nextX:
                for i in range(0,nextX):
                    if nextMat[nextY, i] > 0:
                        features['bodyObstacle'] = 1.0
            if direction == 'RIGHT':# and foodX > nextX:
                for i in range(nextX+1, frameX):
                    if nextMat[nextY, i] > 0:
                        features['bodyObstacle'] = 1.0
        
        # centerX, centerY = (frameX)/2, (frameY)/2
        # features['distToCenter'] = distance(nextHead,[centerX, centerY]) / (frameX*frameY)
        
        # features['distToWall'] = 0
        # if(direction == 'RIGHT'):
        #     features['distToWall']  = abs(frameX - nextX)/(frameX*frameY)
        # if(direction == 'LEFT'):
        #     features['distToWall']  = (nextX+1)/(frameX*frameY)
        # if(direction == 'UP'):
        #     features['distToWall']  = abs(frameY - nextY)/(frameX*frameY)
        # if(direction == 'DOWN'):
        #     features['distToWall']  = (nextY+1)/(frameX*frameY)
        
        # print("*" * 40)
        # print('Trying action', action)
        # print("NEXT MATRIX:\n", nextMat)
        # print("NEXT STATE: ", nextState)
        # print("*" * 40)

        # 
        # matrix = state.getAsMatrix()
        # # print(headX, headY)
        # # the snake looks in the direction of the action 
        # nextDirection = action
        # if action == 'CONTINUE':
        #     nextDirection = state.direction
            
        # if nextX < 0: features["outOfBoundsL"] = 1.
        # else: features["outOfBoundsL"] = 0.
        
        # if nextY < 0: features["outOfBoundsU"] = 1.
        # else: features["outOfBoundsU"] = 0.
        
        # if nextX >= matrix.shape[0]: features["outOfBoundsR"] = 1.
        # else: features["outOfBoundsR"] = 0.
        
        # if nextY >= matrix.shape[1]: features["outOfBoundsD"] = 1.
        # else: features["outOfBoundsD"] = 0.

        # the min distance to an obstacle in the direction of the action
        # the min distance to an obstacle to the left of the action
        # the min distance to an obstacle to the right of the action
        # an obstacle is the snakes body or the wall
        # TODO: Implement this!

    features.divideAll(10.0)
    return features

class ApproxQAgent(QLearningAgent):
    def __init__(self, **args):
        QLearningAgent.__init__(self, **args)
        self.weights = Counter()
    
    def __str__(self):
        return 'ApproxQAgent'
    
    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        # Given a state, action pair, return the Q value
        # Use the weights to compute the Q value
        # print("Calling getQValue on", state,'+',action)
        features = getFeatures(state, action)
        q_value = 0
        for feature in features:
          q_value += self.weights[feature] * features[feature]

        return q_value
    
    def update(self, state, action, nextState, reward):
        difference = ((reward) + (self.discount * self.getValue(nextState))) - (self.getQValue(state, action))
        # print("Called Inside Update for", state, '+', action)
        for feature, value in getFeatures(state, action).items():
          self.weights[feature] = (self.weights[feature]) + (self.alpha * difference * value)
    
    def loadCheckpoint(self, fname='approxq_weights.pkl'):
        """
        Loads the weights into the agent's weights table from
        the given pickle file.
        """
        with open(fname, 'rb') as f:
            d = pickle.load(f)
            self.weights.loadDict(d)
            
    def saveCheckpoint(self, fname='approxq_weights.pkl'):
        weightsDict = self.weights.asDict()
        # print('Weights dictionary size:', sys.getsizeof(weightsDict), 'bytes')
        # print('Number of weights saved:', len(weightsDict))
        with open(fname, 'wb') as f:
            print('Saving weights: ', weightsDict)
            pickle.dump(weightsDict, f, protocol=pickle.HIGHEST_PROTOCOL)
        # with open(fname, 'rb') as f:
        #     counts = pickle.load(f)
        #     print('Length of qval dict saved:', len(counts))
