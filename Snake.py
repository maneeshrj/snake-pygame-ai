import pygame, sys, time, random, copy, util
import numpy as np

# Defalut colors for graphics
COLORS = {
    "black": pygame.Color(0, 0, 0),
    "white": pygame.Color(255, 255, 255),
    "red": pygame.Color(255, 0, 0),
    "green": pygame.Color(0, 255, 0),
    "darkgreen": pygame.Color(0, 200, 0)
}

class GameState:
    """
    A GameState contains all the information necessary to define
    a unique state of the game.
    """
    
    def __init__(self, pos=[[30, 20], [20, 20], [10, 20]], direction='RIGHT', score=0, foodPos=[0, 0], frameSizeX=100,
                 frameSizeY=100):
        # NOTE: Head of the snake is the first element of the list
        self.pos = pos
        self.direction = direction
        self.foodPos = foodPos
        self.score = score
        self.frameX, self.frameY = frameSizeX, frameSizeY
        self.foodSpawned = True
        self.timeout = False

    def getValidActions(self):
        """
        Calculates valid actions give the current direction
        """
        # Gets the valid actions for the snake
        if self.direction == 'UP' or self.direction == 'DOWN':
            return ['LEFT', 'RIGHT', 'CONTINUE']
        elif self.direction == 'LEFT' or self.direction == 'RIGHT':
            return ['UP', 'DOWN', 'CONTINUE']
        return ['CONTINUE']

    def isValidAction(self, action):
        return (action in self.getValidActions())

    def moveSnake(self, action):
        """
        Given a direction, updates the position of the snake
        """
        self.pos.insert(0, list(self.pos[0]))  # duplicate head

        # NOTE: If the action is valid, snake's current direction will be updated
        # If the action is invalid, snake's direction will remain the same as the
        # last timestep (i.e. the action will default to CONTINUE)
        if (action != 'CONTINUE') and (self.isValidAction(action)):
            self.direction = action
        if self.direction == 'UP':
            self.pos[0][1] -= 10
        if self.direction == 'DOWN':
            self.pos[0][1] += 10
        if self.direction == 'LEFT':
            self.pos[0][0] -= 10
        if self.direction == 'RIGHT':
            self.pos[0][0] += 10

        if self.pos[0] == self.foodPos:
            self.score += 1
            self.foodSpawned = False
        else:
            self.pos.pop()  # pop tail
            self.foodSpawned = True

    def getSnakePos(self):
        """
        Returns list of positions of the snake body, ordered from head to tail
        """
        return self.pos

    def getSnakeDir(self):
        """
        Returns current snake direction
        """
        return self.direction

    def getFoodPos(self):
        """
        Returns current food location
        """
        return self.foodPos

    def reachedFood(self):
        """
        Checks if snake head is in the same position as food
        """
        return self.pos[0] == self.foodPos

    def isGameOver(self):
        """
        Returns true if this game state is game over, false otherwise
        """
        if self.pos[0][0] < 0 or self.pos[0][0] > (self.frameX - 10):
            return True
        if self.pos[0][1] < 0 or self.pos[0][1] > (self.frameY - 10):
            return True

        for block in self.pos[1:]:
            if self.pos[0] == block:
                return True
        return self.timeout

    def getSuccessor(self, action):
        """
        Returns the next state given this current state and a possible action
        """
        nextState = copy.deepcopy(self)
        nextState.moveSnake(action)
        return nextState

    def getReward(self, action, step=0):
        """
        Returns a numerical reward for taking an action from the current state
        """
        nextState = self.getSuccessor(action)
		# positive reward for eating food
        if nextState.reachedFood():
            return 25.0
		# negative reward for dying
        if nextState.isGameOver():
            return -10.0
		# otherwise default reward
        if step == 1000:
            self.timeout = True	# gameover if stuck in a loop
        return -0.1

    def __hash__(self):
        """
        Returns a hash of the game state object.
        Uses only snake position, food position, and direction for hashing.
        """
        # TODO: Hash the attributes of the state for qlearning dictionary lookup
        tup = (str(self.pos), tuple(self.foodPos), self.direction)
        # print(tup)
        return hash(tup)

    def __eq__(self, other):
        """
        Checks for equality between two game states.
        Compares only snake position, food position, and direction.
        """
        if isinstance(other, self.__class__):
            return (self.pos == other.pos) and (self.foodPos == other.foodPos) and (self.direction == other.direction)
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)
    
    def getAsMatrix(self):
        """
        Returns the state as a matrix where the matrix represents the window
        frame. The snake position is represented by 1's, empty spaces are 0s,
        and food position is represented by -1.
        """
        matrix = np.zeros((self.frameX // 10, self.frameY // 10))
        for i, snakePos in enumerate(self.pos):
            snakeX = snakePos[0] // 10
            snakeY = snakePos[1] // 10
            # print(snakeX, snakeY)
            if snakeX >= 0 and snakeX < matrix.shape[0]:
                if snakeY >= 0 and snakeY < matrix.shape[1]:
                    if i == 0:
                        matrix[snakeY, snakeX] = 2
                    else:
                        matrix[snakeY, snakeX] = 1
        foodX = self.foodPos[0] // 10
        foodY = self.foodPos[1] // 10
        matrix[foodY, foodX] = 3
        return matrix
    
    def __str__(self):
        """
        Returns a string representation of the game state
        """
        return '{Snake: ' + str(self.pos) + '\nFood: ' + str(self.foodPos) + '\nDirection: ' + self.direction + '\nScore: ' + str(self.score) + '}'
    
class Trial:
    """
    A Trial is a collection of games and holds values that should stay consistent
    from game to game.
      
    """
    def __init__(self):
        self.gameHistory = []
        self.foodPosList = []
        self.currentGame = None

    def setCurrentGame(self, game):
        # Set the current game
        self.currentGame = game
        
        # If this is the first game in the trial, set the foodPosList that should be
        # used for the trial
        if len(self.gameHistory) == 0 and len(self.foodPosList) == 0:
            print('Setting foodPosList')
            self.foodPosList = game.foodPosList.copy()

        # Add the game to the game history
        self.gameHistory.append(game)

    def setFoodPosList(self, foodPosList):
        self.foodPosList = foodPosList.copy()
        
    def getFoodPosList(self):
        return self.foodPosList.copy()

    def __str__(self):
        return 'Current Game: ' + str(self.currentGame) + '\n' + 'Game History: ' + str(self.gameHistory) + '\n' + 'Food Pos List: ' + str(self.foodPosList)



class Game:
    '''
    A Game defines the structure of a snake game.
    '''
    def __init__(self, gameState=GameState(), graphics=False, framerate=10, plain=False, foodPosList=[], randomFood=False):
        self.graphics = graphics
        self.gameState = gameState
        # Window size
        self.frameX = gameState.frameX
        self.frameY = gameState.frameY
        self.fps_controller = pygame.time.Clock()
        self.framerate = framerate
        self.first_step = True
        self.plain = plain
        self.foodPosList = foodPosList

        # Generate a random food position from a set food list
        if randomFood:
            rng = np.random.RandomState()
            for i in range((self.frameX//10)):
                for j in range((self.frameY//10)):
                    self.foodPosList.append([i*10, j*10])
            rng.shuffle(self.foodPosList)
        else:
            rng = np.random.RandomState(2022)
            if len(self.foodPosList) == 0:
                for i in range((self.frameX//10)):
                    for j in range((self.frameY//10)):
                        self.foodPosList.append([i*10, j*10])
                rng.shuffle(self.foodPosList)
        
        # Have to set food pos outside of init otherwise we pop the first element
        # before we have a chance to set the foodPosList in the trial
        # self.setFoodPos()

        # Check for errors
        if (self.graphics):
            check_errors = pygame.init()
            if check_errors[1] > 0:
                print(f'[!] Had {check_errors[1]} errors when initialising game, exiting...')
                sys.exit(-1)
            else:
                print('[+] Game successfully initialised')

            pygame.display.set_caption('Snake Eater')
            self.game_window = pygame.display.set_mode((self.frameX, self.frameY))
            self.fps_controller = pygame.time.Clock()

    def setFoodPos(self):
        # Pop from a predetermined food spawn list
        if len(self.foodPosList) > 0:
            self.foodPos = self.foodPosList.pop()
        else:
            self.foodPos = [random.randrange(1, (self.frameX // 10)) * 10,
                            random.randrange(1, (self.frameY // 10)) * 10]
        # Make sure the food pos is not on the snake
        if self.foodPos in self.gameState.pos:
            self.setFoodPos()

        self.gameState.foodPos = self.foodPos
        self.gameState.foodSpawned = True
        # print(self.gameState.getAsMatrix())
        
    # For an action the snake takes, update the game state
    def playStep(self, action):
        self.gameState.moveSnake(action)
        events = None

        if not self.gameState.foodSpawned:
            self.setFoodPos()

        # Draw the updated window
        if (self.graphics):
            self.drawWindow()
            if not self.plain:
                self.showScore(1, COLORS["white"], 'consolas', 20)

            pygame.display.update()
            events = pygame.event.get()
            # Refresh rate
            self.fps_controller.tick(self.framerate)

        return self.gameState.isGameOver(), self.gameState.score

    # Create the game window
    def drawWindow(self):
        self.game_window.fill(COLORS["black"])
        pygame.draw.rect(self.game_window, COLORS["darkgreen"], pygame.Rect(self.gameState.pos[0][0], self.gameState.pos[0][1], 10, 10))
        for pos in self.gameState.pos[1:]:
            pygame.draw.rect(self.game_window, COLORS["green"], pygame.Rect(pos[0], pos[1], 10, 10))

        # Draw snake food
        pygame.draw.rect(self.game_window, COLORS["white"], pygame.Rect(self.foodPos[0], self.foodPos[1], 10, 10))

    # Display game over screen
    def gameOver(self):
        if (self.graphics):
            my_font = pygame.font.SysFont('times new roman', self.frameX // 5)
            gameOver_surface = my_font.render('YOU DIED', True, COLORS["red"])
            gameOver_rect = gameOver_surface.get_rect()
            gameOver_rect.midtop = (self.frameX / 2, self.frameY / 4)
            self.game_window.fill(COLORS["black"])

            if not self.plain:
                self.game_window.blit(gameOver_surface, gameOver_rect)
                self.showScore(0, COLORS["red"], 'times', 20)

            pygame.display.flip()
            pygame.quit()

    # Show the score
    def showScore(self, choice, color, font, size):
        score_font = pygame.font.SysFont(font, size)
        score_surface = score_font.render(str(self.gameState.score), True, color)
        score_rect = score_surface.get_rect()
        if choice == 1:
            score_rect.midtop = (self.frameX / 10, 15)
        else:
            score_rect.midtop = (self.frameX / 2, self.frameY / 1.25)
        self.game_window.blit(score_surface, score_rect)

    # Helper functions for feature extraction
    def getCurrentState(self):
        return self.gameState

    def getNextState(self, action):
        return self.gameState.getSuccessor(action)

    def getReward(self, action, step=0):
        return self.gameState.getReward(action, step)

    def getScreenAsNumpy(self):
        # Get the current pygame window as a numpy array
        if self.graphics:
            screen = pygame.display.get_surface()
            screen_array = pygame.surfarray.array3d(screen)
            return np.asarray(screen_array)




if __name__ == '__main__':
    gameState = GameState()
    nextState = gameState.getSuccessor('RIGHT')
    print(gameState.pos)
    print(nextState.pos)
