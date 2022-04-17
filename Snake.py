# Imports
import pygame, sys, time, random, copy, util
import numpy as np

# Default colors
COLORS = {
    "black": pygame.Color(0, 0, 0),
    "white": pygame.Color(255, 255, 255),
    "red": pygame.Color(255, 0, 0),
    "green": pygame.Color(0, 255, 0),
}

class GameState:
    '''
    A GameState defines the attributes of the snake
    '''
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

    # Calculate valid actions give the current direction
    def getValidActions(self):
        if self.direction == 'UP' or self.direction == 'DOWN':
            return ['LEFT', 'RIGHT', 'CONTINUE']
        elif self.direction == 'LEFT' or self.direction == 'RIGHT':
            return ['UP', 'DOWN', 'CONTINUE']
        return ['CONTINUE']

    # Given a direction, update the position of the snake
    def moveSnake(self, direction):
        self.pos.insert(0, list(self.pos[0]))  # duplicate head

        if direction != 'CONTINUE':
            self.direction = direction
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

    # Return current snake body position
    def getSnakePos(self):
        return self.pos

    # Return current snake direction
    def getSnakeDir(self):
        return self.direction

    # Return current food location
    def getFoodPos(self):
        return self.foodPos

    # Check if food has been consumed
    def reachedFood(self):
        return self.pos[0] == self.foodPos

    # Check the game over conditions
    def isGameOver(self):

        # Check if snake has hit the wall
        if self.pos[0][0] < 0 or self.pos[0][0] > self.frameX - 10:
            return True
        if self.pos[0][1] < 0 or self.pos[0][1] > self.frameY - 10:
            return True

        # Check if snake has hit itself
        for block in self.pos[1:]:
            if self.pos[0] == block:
                return True
        return self.timeout

    # Get the next state of the snake. Sends a call to move the snake
    def getSuccessor(self, action):
        nextState = copy.deepcopy(self)
        nextState.moveSnake(action)
        return nextState

    # Defined rewards for training
    def getReward(self, action, step=0):
        if step == 10000:
            self.timeout = True
            return -10.0
        nextState = self.getSuccessor(action)
        if (nextState.reachedFood()):
            return 10.0
        if (nextState.isGameOver()):
            return -5.0
        return -0.0001

    # Hash function for the state
    def __hash__(self):
        # TODO: Hash the attributes of the state for qlearning dictionary lookup
        tup = (str(self.pos), tuple(self.foodPos), self.direction)
        return hash(tup)

    # Override the equality operator to check for equal snake states
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (self.pos == other.pos) and (self.foodPos == other.foodPos) and (self.direction == other.direction)
        else:
            return False

    # Not equal operator
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def getAsMatrix(self):
        """
        Returns the state as a matrix where the matrix is the frame size in 
        increments of 10 and the snake position is represented by 1's and the 
        food position is represented by 2's
        """
        matrix = np.zeros((self.frameX // 10, self.frameY // 10))
        for snakePos in self.pos:
            # FIXME gets an index out of bounds error. Perhaps subtracting one from the x and y values would work?
            snakeX = snakePos[0] // 10 - 1
            snakeY = snakePos[1] // 10 - 1
            print(snakeX, snakeY)
            matrix[snakeY][snakeX] = 1
        foodX = self.foodPos[0] // 10
        foodY = self.foodPos[1] // 10
        matrix[foodY][foodX] = 2
        return matrix
    
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
        
        # If this is the first game in the trial, set the foodPosList that should be used for the trial
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
    def __init__(self, gameState=GameState(), graphics=False, framerate=10, plain=False, foodPosList=[]):
        self.graphics = graphics
        self.gameState = gameState
        self.frameX = gameState.frameX
        self.frameY = gameState.frameY
        self.fps_controller = pygame.time.Clock()
        self.framerate = framerate
        self.first_step = True
        self.plain = plain
        self.foodPosList = foodPosList

        # FIXME: Setting the random seed causes the snake to be deterministic which 
        # prevents the snake from taking different path and learning the q table.
        #random.seed(69)
        # generate 100 random food positions as a generator to call next()
        """for i in range((self.frameX // 10) ** 2):
            self.foodPosList.append(
                [random.randrange(1, (self.frameX // 10)) * 10, random.randrange(1, (self.frameY // 10)) * 10])"""
        # loop through all positions in the frame in 10x10 increments

        # if len(self.foodPosList) == 0:
        #     for i in range((self.frameX//10)):
        #         for j in range((self.frameY//10)):
        #             self.foodPosList.append([i*10, j*10])
        #     random.shuffle(self.foodPosList)

        rng = np.random.RandomState(69)
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
        for pos in self.gameState.pos:
            pygame.draw.rect(self.game_window, COLORS["green"], pygame.Rect(pos[0], pos[1], 10, 10))

        # Draw the food
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

    # Get the current pygame window as a numpy array
    def getScreenAsNumpy(self):
        if self.graphics:
            screen = pygame.display.get_surface()
            screen_array = pygame.surfarray.array3d(screen)
            return np.asarray(screen_array)

if __name__ == '__main__':
    gameState = GameState()
    nextState = gameState.getSuccessor('RIGHT')
