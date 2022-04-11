from util import updatePosition, distance, manhattanDistance

# REFLEX AGENT
# At each time step, takes the action that most decreases the distance
# between the head of the snake and the food.
class ReflexAgent:
    def __init__(self, gameState, env):
        self.gameState = gameState
        self.env = env
    
    def __str__(self):
        return "ReflexAgent"
    
    def getNextAction(self):
        validActions = self.gameState.getValidActions()
        currentPos = self.gameState.pos
        foodPos = self.env.foodPos
        newPos = None
        bestDist, bestAction = float('inf'), None
        
        for action in validActions:
            newHeadPos = updatePosition(self.gameState.pos[0], action)
            dist = distance(newHeadPos, foodPos)
            if dist < bestDist:
                bestDist, bestAction = dist, action
        return bestAction
