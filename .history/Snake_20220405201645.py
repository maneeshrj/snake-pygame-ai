import pygame, sys, time, random, copy, util
COLORS = {
    "black": pygame.Color(0, 0, 0),
    "white": pygame.Color(255, 255, 255),
    "red": pygame.Color(255, 0, 0),
    "green": pygame.Color(0, 255, 0),
}

class Snake:
    def __init__(self, pos=[[100, 50], [100-10, 50], [100-20, 50]], direction='RIGHT'):
        # Head of the snake is the first element of the list
        self.pos = pos
        self.direction = direction

    def get_valid_actions(self):
        # Gets the valid actions for the snake
        if self.direction == 'UP' or self.direction == 'DOWN':
            return ['LEFT', 'RIGHT', 'CONTINUE']
        elif self.direction == 'LEFT'or self.direction == 'RIGHT':
            return ['UP', 'DOWN', 'CONTINUE']
        return ['CONTINUE']
        
    def move_snake(self, direction):
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

class GameState:
    def __init__(self, snake, score, foodPos, frameX=720, frameY=480):
        self.snakeCopy = Snake(pos=copy.deepcopy(snake.pos), direction=snake.direction)
        self.score = score
        self.foodPos = foodPos
        self.frameX = frameX
        self.frameY = frameY
    
    def getSnakePosition(self):
        return self.snakeCopy.pos
    
    def getSnakeDirection(self):
        return self.snakeCopy.direction
    
    def getFoodPosition(self):
        return self.foodPos
    
    def getValidActions(self):
        return self.snakeCopy.get_valid_actions()
    
    def reachedFood(self):
        return self.snakeCopy.pos[0] == self.foodPos
    
    def takeAction(self, action):
        self.snakeCopy.move_snake(action)
        
    def isGameOver(self):
        if self.snakeCopy.pos[0][0] < 0 or self.snakeCopy.pos[0][0] > self.frameX-10:
            return True
        if self.snakeCopy.pos[0][1] < 0 or self.snakeCopy.pos[0][1] > self.frameY-10:
            return True
        for block in self.snakeCopy.pos[1:]:
            if self.snakeCopy.pos[0] == block:
                return True
        return False

class Game:
    def __init__(self, snake, graphics=False, frame_size_x=480, frame_size_y=480):
        self.graphics = graphics
        self.snake = snake
        self.score = 0
        # Window size
        self.frame_size_x = frame_size_x
        self.frame_size_y = frame_size_y
        self.fps_controller = pygame.time.Clock()
        self.framerate = 10
        self.first_step = True
        self.set_food_pos()
        
        
        # Check for errors
        if(self.graphics):
            check_errors = pygame.init()
            if check_errors[1] > 0:
                print(f'[!] Had {check_errors[1]} errors when initialising game, exiting...')
                sys.exit(-1)
            else:
                print('[+] Game successfully initialised')

            pygame.display.set_caption('Snake Eater')
            self.game_window = pygame.display.set_mode((self.frame_size_x, self.frame_size_y))
            self.fps_controller = pygame.time.Clock()
        
    def set_food_pos(self):
        # Make sure the food pos is not on the snake
        self.food_pos = [random.randrange(1, (self.frame_size_x//10)) * 10, random.randrange(1, (self.frame_size_y//10)) * 10]
        if self.food_pos in self.snake.pos:
            self.set_food_pos()
        self.food_spawn = True
        
    def play_step(self, action):
        # Returns tuple(is_over, score)
        is_over = False

        # duplicate head
        self.snake.pos.insert(0, list(self.snake.pos[0]))
        # update head
        self.snake.move_snake(action)
        
        # Check if snake has eaten the food
        if self.snake.pos[0] == self.food_pos:
            self.score += 1
            self.food_spawn = False
        else:
            self.snake.pos.pop()
        if not self.food_spawn:
            self.set_food_pos()
        self.food_spawn = True

        # Draw the updated window
        if(self.graphics):
            self.draw_window()

        # Game Over conditions
        # Getting out of bounds
        if self.snake.pos[0][0] < 0 or self.snake.pos[0][0] > self.frame_size_x-10:
            is_over = True
        if self.snake.pos[0][1] < 0 or self.snake.pos[0][1] > self.frame_size_y-10:
            is_over = True

        # Touching the snake body
        for block in self.snake.pos[1:]:
            if self.snake.pos[0] == block:
                is_over = True

        if(self.graphics):
            self.show_score(1, COLORS["white"], 'consolas', 20)
        
        # Refresh game screen
        if(self.graphics):
            pygame.display.update()
            # Refresh rate
            self.fps_controller.tick(self.framerate)
            
        return is_over, self.score
    
    def draw_window(self):
        self.game_window.fill(COLORS["black"])
        #print("-" * 20)
        for pos in self.snake.pos:
            # Snake body
            # .draw.rect(play_surface, color, xy-coordinate)
            # xy-coordinate -> .Rect(x, y, size_x, size_y)
            #print(pos)
            pygame.draw.rect(self.game_window, COLORS["green"], pygame.Rect(pos[0], pos[1], 10, 10))

        # Snake food
        pygame.draw.rect(self.game_window, COLORS["white"], pygame.Rect(self.food_pos[0], self.food_pos[1], 10, 10))

    def game_over(self):
        if(self.graphics):
            my_font = pygame.font.SysFont('times new roman', 90)
            game_over_surface = my_font.render('YOU DIED', True, COLORS["red"])
            game_over_rect = game_over_surface.get_rect()
            game_over_rect.midtop = (self.frame_size_x/2, self.frame_size_y/4)
            self.game_window.fill(COLORS["black"])
            self.game_window.blit(game_over_surface, game_over_rect)
            self.show_score(0, COLORS["red"], 'times', 20)
            pygame.display.flip()
            time.sleep(3)
            pygame.quit()
        #sys.exit()
    
    # Score
    def show_score(self, choice, color, font, size):
        score_font = pygame.font.SysFont(font, size)
        score_surface = score_font.render('Score : ' + str(self.score), True, color)
        score_rect = score_surface.get_rect()
        if choice == 1:
            score_rect.midtop = (self.frame_size_x/10, 15)
        else:
            score_rect.midtop = (self.frame_size_x/2, self.frame_size_y/1.25)
        self.game_window.blit(score_surface, score_rect)
    
    # Helper functions for feature extraction
    def getCurrentState(self):
        return GameState(self.snake, self.score, self.food_pos, self.frame_size_x, self.frame_size_y)
    
    def getNextState(self, action):
        currState = self.getCurrentState()
        currState.takeAction(action)
        return currState
    
    def getReward(self, action):
        nextState = self.getNextState(action)
        if(nextState.reachedFood()):
            #print('reward 1')
            return 1.0
        if(nextState.isGameOver()):
            #print('reward -1')
            return -1.0
        #print('no reward\n')
        return 0.0

if __name__ == '__main__':
    snake = Snake([[-1,1],[0,1]])
    gameState = GameState(snake, 0, [0,1])
    print(gameState.isGameOver())
    

