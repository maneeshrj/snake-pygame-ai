import util
class FeatureExtractor:
    def getFeatures(self, state, action):
        """
          Returns a dict from features to counts
          Usually, the count will just be 1.0 for
          indicator functions.
        """
        

class IdentityExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[(state,action)] = 1.0
        return feats

# def closestFood(pos, food, walls):
#     """
#     closestFood -- this is similar to the function that we have
#     worked on in the search project; here its all in one place
#     """
#     fringe = [(pos[0], pos[1], 0)]
#     expanded = set()
#     while fringe:
#         pos_x, pos_y, dist = fringe.pop(0)
#         if (pos_x, pos_y) in expanded:
#             continue
#         expanded.add((pos_x, pos_y))
#         # if we find a food at this location then exit
#         if food[pos_x][pos_y]:
#             return dist
#         # otherwise spread out from the location to its neighbours
#         nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
#         for nbr_x, nbr_y in nbrs:
#             fringe.append((nbr_x, nbr_y, dist+1))
#     # no food found
#     return None


class SimpleExtractor(FeatureExtractor):
    """
    Returns simple features
    """
    def getFeatures(self, state, action):
        features = util.Counter()
        # DO NOT DECREASE OR REMOVE THIS
        # Will diverge
        features["bias"] = 1000.0 
        
        # Position of snake head
        pos = state.getSnakePosition()
        head = pos[0]
        #print('head',head)
        # features["headX"] = head[0]
        # features["headY"] = head[1]
        
        """features["frameXDist"] = state.frameX - head[0]
        features["frameYDist"] = state.frameY - head[1]"""
        
        foodPos = state.getFoodPosition()
        # features["foodX"] = foodPos[0]
        # features["foodY"] = foodPos[1]
        # features["foodDistX"] = foodPos[0] - head[0]
        # features["foodDistY"] = foodPos[1] - head[1]
        
        # Position of snake head next time step
        nextPos = util.updatePosition(head, action)
        #print(nextPos)
        features["nextX"] = nextPos[0]
        features["nextY"] = nextPos[1]
        
        if(nextPos[0] < 0) or (nextPos[0] > state.frameX):
            features["nextXOut"] = 1.0
        else:
            features["nextXOut"] = 0.0
            
        if(nextPos[1] < 0) or (nextPos[1] > state.frameY):
            features["nextYOut"] = 1.0
        else:
            features["nextYOut"] = 0.0
        
        features.divideAll(1000.0)
        return features

# from Snake import GameState, Snake
# if __name__ == "__main__":
#     snake = Snake([[0,0]])
#     gameState = GameState(snake, 0, [10,0])
#     ex = SimpleExtractor()
#     action = 'RIGHT'
#     print(ex.getFeatures(gameState, action))