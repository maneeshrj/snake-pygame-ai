import pygame, sys, time, random, copy, util
import numpy as np
COLORS = {
    "black": pygame.Color(0, 0, 0),
    "white": pygame.Color(255, 255, 255),
    "red": pygame.Color(255, 0, 0),
    "green": pygame.Color(0, 255, 0),
}

class GameState:
    def __init__(self, pos=[[30, 20], [20, 20], [10, 20]], direction='RIGHT', score=0, foodPos=[0,0], frameSizeX=100, frameSizeY=100):
        # Head of the snake is the first element of the list
        self.pos = pos
        self.direction = direction
        self.foodPos = foodPos
        self.score = score
        self.frameX, self.frameY = frameSizeX, frameSizeY
        self.foodSpawned = True

    def getValidActions(self):
        # Gets the valid actions for the snake
        if self.direction == 'UP' or self.direction == 'DOWN':
            return ['LEFT', 'RIGHT', 'CONTINUE']
        elif self.direction == 'LEFT'or self.direction == 'RIGHT':
            return ['UP', 'DOWN', 'CONTINUE']
        return ['CONTINUE']
        
    def moveSnake(self, direction):
        self.pos.insert(0, list(self.pos[0])) # duplicate head

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
            self.pos.pop() # pop tail
            self.foodSpawned = True
        
    def getSnakePos(self):
        return self.pos
    
    def getSnakeDir(self):
        return self.direction
    
    def getFoodPos(self):
        return self.foodPos

    def reachedFood(self):
        return self.pos[0] == self.foodPos
        
    def isGameOver(self):
        if self.pos[0][0] < 0 or self.pos[0][0] > self.frameX-10:
            return True
        if self.pos[0][1] < 0 or self.pos[0][1] > self.frameY-10:
            return True

        for block in self.pos[1:]:
            if self.pos[0] == block:
                return True
        return False
    
    def getSuccessor(self, action):
        nextState = copy.deepcopy(self)
        nextState.moveSnake(action)
        return nextState

    def __hash__(self):
        # TODO: Hash the attributes of the state for qlearning dictionary lookup
        pass


class Game:
    def __init__(self, gameState=GameState(), graphics=False, framerate=10, plain=False):
        self.graphics = graphics
        self.gameState = gameState
        # Window size
        self.frameX = gameState.frameX
        self.frameY = gameState.frameY
        self.fps_controller = pygame.time.Clock()
        self.framerate = framerate
        self.first_step = True
        self.plain = plain
        # generate 100 random food positions as a generator to call next()
        self.foodPosList = []
        for i in range((self.frameX//10)**2):
            self.foodPosList.append([random.randrange(1, (self.frameX//10)) * 10, random.randrange(1, (self.frameY//10)) * 10])
        # loop through all positions in the frame in 10x10 increments
        """for i in range((self.frameX//10) - 1):
            for j in range((self.frameY//10) - 1):
                self.foodPosList.append([i*10, j*10])"""
        self.setFoodPos()

        # Check for errors
        if(self.graphics):
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
            self.foodPos = [random.randrange(1, (self.frameX//10)) * 10, random.randrange(1, (self.frameY//10)) * 10]
        # Make sure the food pos is not on the snake
        if self.foodPos in self.gameState.pos:
            self.setFoodPos()
        
        self.gameState.foodPos = self.foodPos
        self.gameState.foodSpawned = True
        
    def playStep(self, action):
        self.gameState.moveSnake(action)
        events = None

        if not self.gameState.foodSpawned:
            self.setFoodPos()

        # Draw the updated window
        if(self.graphics):
            self.drawWindow()
            if not self.plain:
                self.showScore(1, COLORS["white"], 'consolas', 20)

            pygame.display.update()
            events = pygame.event.get()
            # Refresh rate
            self.fps_controller.tick(self.framerate)
            
        return self.gameState.isGameOver(), self.gameState.score
    
    def drawWindow(self):
        self.game_window.fill(COLORS["black"])
        #print("-" * 20)
        for pos in self.gameState.pos:
            # Snake body
            # .draw.rect(play_surface, color, xy-coordinate)
            # xy-coordinate -> .Rect(x, y, size_x, size_y)
            #print(pos)
            pygame.draw.rect(self.game_window, COLORS["green"], pygame.Rect(pos[0], pos[1], 10, 10))

        # Snake food
        pygame.draw.rect(self.game_window, COLORS["white"], pygame.Rect(self.foodPos[0], self.foodPos[1], 10, 10))

    def gameOver(self):
        if(self.graphics):
            my_font = pygame.font.SysFont('times new roman', self.frameX // 5)
            gameOver_surface = my_font.render('YOU DIED', True, COLORS["red"])
            gameOver_rect = gameOver_surface.get_rect()
            gameOver_rect.midtop = (self.frameX/2, self.frameY/4)
            self.game_window.fill(COLORS["black"])
            
            if not self.plain:
                self.game_window.blit(gameOver_surface, gameOver_rect)
                self.showScore(0, COLORS["red"], 'times', 20)
            
            pygame.display.flip()
            pygame.quit()
        #sys.exit()
    
    # Score
    def showScore(self, choice, color, font, size):
        score_font = pygame.font.SysFont(font, size)
        score_surface = score_font.render(str(self.gameState.score), True, color)
        score_rect = score_surface.get_rect()
        if choice == 1:
            score_rect.midtop = (self.frameX/10, 15)
        else:
            score_rect.midtop = (self.frameX/2, self.frameY/1.25)
        self.game_window.blit(score_surface, score_rect)
    
    # Helper functions for feature extraction
    def getCurrentState(self):
        return self.gameState
    
    def getNextState(self, action):
        return self.gameState.getSuccessor(action)
    
    def getReward(self, action):
        nextState = self.gameState.getSuccessor(action)
        if(nextState.reachedFood()):
            #print('reward 1')
            return 1.0
        if(nextState.isGameOver()):
            #print('reward -1')
            return -1.0
        #print('no reward\n')
        return 0.0

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
    

