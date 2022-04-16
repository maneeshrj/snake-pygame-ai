'''
This file contains the manual Snake game.
Credit for this code goes to Rajat Biswas, the original author of this file.
The code was modified to fit the needs of this project.
The original code is available at: https://github.com/rajatdiptabiswas/snake-pygame 
'''

# For each run, rerun this file and scores will be logged

# Imports
import argparse
import pygame, sys, time, random
import json

# Map size options
WINDOW_SIZE_MAP = {
    'small': (100, 100),
    'medium': (250, 250),
    'large': (500, 500)
}

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--screen_size", help="Size of the window", type=str, default="small",
                        choices=["small", "medium", "large"])
parser.add_argument("-f", "--framerate", help="Set game speed", type=int, default=7)
parser.add_argument("-r", "--reset", help="Reset game log", type=bool, default=False)

# Set argument values
args = parser.parse_args()
framerate = args.framerate
(frameSizeX, frameSizeY) = WINDOW_SIZE_MAP[args.screen_size]
reset = args.reset

# Window size
wsize = 100

# Checks for errors encountered
checkError = pygame.init()

if checkError[1] > 0:
    print(f'[!] Had {checkError[1]} errors when initialising game, exiting...')
    sys.exit(-1)
else:
    print('[+] Game successfully initialised')

# Initialize game window
pygame.display.set_caption('Snake Game')
gameWindow = pygame.display.set_mode((frameSizeX, frameSizeY))

# Colors (R, G, B)
black = pygame.Color(0, 0, 0)
white = pygame.Color(255, 255, 255)
red = pygame.Color(255, 0, 0)
green = pygame.Color(0, 255, 0)
blue = pygame.Color(0, 0, 255)

# FPS controller
fpsController = pygame.time.Clock()

# Game variables
snakePos = [30, 20]
snakeBody = [[30, 20], [20, 20], [10, 20]]

foodPos = [random.randrange(1, (frameSizeX // 10)) * 10, random.randrange(1, (frameSizeY // 10)) * 10]
foodSpawn = True

direction = 'RIGHT'
changeTo = direction

score = 0


# Game over
def game_over():
    myFont = pygame.font.SysFont('times new roman', wsize // 5)
    gameOverSurface = myFont.render('YOU DIED', True, red)
    gameOverRect = gameOverSurface.get_rect()
    gameOverRect.midtop = (frameSizeX / 2, frameSizeY / 4)
    gameWindow.fill(black)
    gameWindow.blit(gameOverSurface, gameOverRect)
    show_score(0, red, 'times', 20)
    pygame.display.flip()
    time.sleep(3)
    pygame.quit()

    # Keep running average of manually played scores in a json file
    scores = None
    with open('manualGameScores.json', "r") as f:
        scores = json.load(f)
    if reset:
        scores = []
    scores.append(score)
    with open('manualGameScores.json', 'w') as f:
        json.dump(scores, f)

    print('Scores:', scores)
    print('Avg score:', sum(scores) / len(scores))
    sys.exit()


# Scoring
def show_score(choice, color, font, size):
    scoreFont = pygame.font.SysFont(font, size)
    scoreSurface = scoreFont.render('Score : ' + str(score), True, color)
    scoreRect = scoreSurface.get_rect()
    if choice == 1:
        scoreRect.midtop = (frameSizeX / 10, 15)
    else:
        scoreRect.midtop = (frameSizeX / 2, frameSizeY / 1.25)
    gameWindow.blit(scoreSurface, scoreRect)
    # pygame.display.flip()

# Main logic
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        # Logic for whenever a key is pressed down
        elif event.type == pygame.KEYDOWN:

            if event.key == pygame.K_UP or event.key == ord('w'):
                changeTo = 'UP'
            if event.key == pygame.K_DOWN or event.key == ord('s'):
                changeTo = 'DOWN'
            if event.key == pygame.K_LEFT or event.key == ord('a'):
                changeTo = 'LEFT'
            if event.key == pygame.K_RIGHT or event.key == ord('d'):
                changeTo = 'RIGHT'
            if event.key == pygame.K_ESCAPE:
                pygame.event.post(pygame.event.Event(pygame.QUIT))

    # Making sure the snake cannot move in the opposite direction instantaneously
    if changeTo == 'UP' and direction != 'DOWN':
        direction = 'UP'
    if changeTo == 'DOWN' and direction != 'UP':
        direction = 'DOWN'
    if changeTo == 'LEFT' and direction != 'RIGHT':
        direction = 'LEFT'
    if changeTo == 'RIGHT' and direction != 'LEFT':
        direction = 'RIGHT'

    # Moving the snake
    if direction == 'UP':
        snakePos[1] -= 10
    if direction == 'DOWN':
        snakePos[1] += 10
    if direction == 'LEFT':
        snakePos[0] -= 10
    if direction == 'RIGHT':
        snakePos[0] += 10

    # Snake body growing mechanism
    snakeBody.insert(0, list(snakePos))
    if snakePos[0] == foodPos[0] and snakePos[1] == foodPos[1]:
        score += 1
        foodSpawn = False
    else:
        snakeBody.pop()

    # Spawning food on the screen
    if not foodSpawn:
        while True:
            foodPos = [random.randrange(1, (frameSizeX // 10)) * 10, random.randrange(1, (frameSizeY // 10)) * 10]
            if foodPos not in snakeBody:
                break
    foodSpawn = True

    # Graphics
    gameWindow.fill(black)
    for pos in snakeBody:
        pygame.draw.rect(gameWindow, green, pygame.Rect(pos[0], pos[1], 10, 10))

    pygame.draw.rect(gameWindow, white, pygame.Rect(foodPos[0], foodPos[1], 10, 10))

    # Game Over conditions:

    # Getting out of bounds
    if snakePos[0] < 0 or snakePos[0] > frameSizeX - 10:
        game_over()
    if snakePos[1] < 0 or snakePos[1] > frameSizeY - 10:
        game_over()

    # Touching the snake body
    for block in snakeBody[1:]:
        if snakePos[0] == block[0] and snakePos[1] == block[1]:
            game_over()

    # Show score after death
    show_score(1, white, 'consolas', 20)

    # Refresh game screen
    pygame.display.update()

    # Refresh rate
    fpsController.tick(framerate)