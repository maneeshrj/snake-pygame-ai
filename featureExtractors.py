from util import Counter, manhattanDistance, updatePosition

class FeatureExtractor:
    def getFeatures(self, state, action):
        """
          Returns a dict from features to counts
          Usually, the count will just be 1.0 for
          indicator functions.
        """
        
##############################################################################
class IdentityExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = Counter()
        feats[(state, action)] = 1.0
        return feats
    
    
##############################################################################
class SimpleExtractor(FeatureExtractor):
    """
    Returns simple features
    """
    def getFeatures(state, action):
        features = Counter()
        features["bias"] = 1.0
    
        # all features bases on next state
        curPos = state.getSnakePos()
        curHead = curPos[0]
        nextPos = updatePosition(curHead, action)
        foodPos = state.getFoodPos()
        
        # get distance to food as a number between 0 and 1
        features["foodDist"] = manhattanDistance(nextPos, foodPos) / ((state.frameX // 10) * (state.frameY // 10))

        features.divideAll(10.0)
        return features
    
class DodgeExtractor(FeatureExtractor):
    """
    Returns simple features
    """
    
    def getFeatures(state, action, extractor=0):
        features = Counter()
        features["bias"] = 1.0
    
        # all features bases on next state
        curPos = state.getSnakePos()
        curHead = curPos[0]
        nextPos = updatePosition(curHead, action)
        foodPos = state.getFoodPos()
        
        # get distance to food as a number between 0 and 1
        features["foodDist"] = manhattanDistance(nextPos, foodPos) / ((state.frameX // 10) * (state.frameY // 10))
        
        mat = state.getAsMatrix()
        # the snake looks in the direction of the action and has three features
        # the min distance to an obstacle in the direction of the action
        # the min distance to an obstacle to the left of the action
        # the min distance to an obstacle to the right of the action
        # an obstacle is the snakes body or the wall
        # TODO: Implement this!
    
        features.divideAll(10.0)
        return features