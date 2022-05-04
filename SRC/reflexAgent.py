# Imports
from util import updatePosition, distance

class ReflexAgent:
    '''
    A Reflex agent chooses the action that minimizes manhatten distance to the food. 
    '''
    def __init__(self):
        self.accumulatedRewards = 0
        self.gameState = None

    def __str__(self):
        return "ReflexAgent"

    def startEpisode(self, gameState):
        self.gameState = gameState

    def stopEpisode(self):
        self.accumulatedRewards += self.gameState.score
        self.gameState = None

    # Find the action that minimizes manhatten distance to the food
    def getNextAction(self):
        validActions = self.gameState.getValidActions()
        foodPos = self.gameState.foodPos
        bestDist, bestAction = float('inf'), None

        for action in validActions:
            newHeadPos = updatePosition(self.gameState.pos[0], action)
            dist = distance(newHeadPos, foodPos)
            if dist < bestDist:
                bestDist, bestAction = dist, action
        return bestAction
