from util import distance

# REFLEX AGENT
# At each time step, takes the action that most decreases the distance
# between the head of the snake and the food.
class ReflexAgent:
    def __init__(self, snake, env):
        self.snake = snake
        self.env = env
    
    def getNextAction(self):
        validActions = self.snake.get_valid_actions()
        currentPos = self.snake.pos
        foodPos = self.env.food_pos
        newPos = None
        bestDist, bestAction = float('inf'), None
        for action in validActions:
            newPos = [self.snake.pos[0][0], self.snake.pos[0][1]] 
            if action == 'UP':
                newPos = [self.snake.pos[0][0], self.snake.pos[0][1] - 10] 
            if action == 'DOWN':
                newPos = [self.snake.pos[0][0], self.snake.pos[0][1] + 10] 
            if action == 'LEFT':
                newPos = [self.snake.pos[0][0] - 10, self.snake.pos[0][1]] 
            if action == 'RIGHT':
                newPos = [self.snake.pos[0][0] + 10, self.snake.pos[0][1]] 
            if action == 'CONTINUE':
                newPos = self.snake.pos[0]

            if distance(newPos, foodPos) < bestDist:
                bestDist = distance(newPos, foodPos)
                bestAction = action                
        return bestAction
