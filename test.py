import json
import argparse
import time
import pygame, sys

import numpy as np
import matplotlib.pyplot as plt
import torch

from Snake import Game, GameState, Trial
from agents import AGENT_MAP, getAgentName
from qLearningAgent import QLearningAgent
from dqnAgent import DQNAgent
from dqn import DQN

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WINDOW_SIZE_MAP = {
    'small': (100, 100),
    'medium': (250, 250),
    'large': (500, 500)
}

if __name__ == "__main__":

    # Add command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--agents", help="Agents to use", nargs='+', type=str,
                        default=["random", "reflex", "exactq", "approxq", "dqn"],
                        choices=["random", "reflex", "exactq", "approxq", "dqn"])
    parser.add_argument("-n", "--num_runs", help="Number of runs", type=int, default=1)
    parser.add_argument("-v", "--verbose", help="Verbose output", action="store_true", default=False)
    parser.add_argument("-g", "--graphics", help="Use graphics", action="store_true", default=False)
    parser.add_argument("-s", "--screen_size", help="Size of the window", type=str, default="small",
                        choices=["small", "medium", "large"])
    parser.add_argument("-p", "--plain", help="No score or words in graphics. Use for training CNN",
                        action="store_true", default=False)
    parser.add_argument("-j", "--json", help="Read from json file", action="store_true", default=False)
    parser.add_argument("-f", "--framerate", help="Set game speed", type=int, default=10)
    parser.add_argument("-l", "--load", help="Load model", type=str, default=None)
    parser.add_argument("-r", "--random_food", help="Random food spawn", action="store_true", default=False)

    # Parse arguments and assign to variables
    args = parser.parse_args()
    agentNames = args.agents
    agents = [AGENT_MAP[agent] for agent in agentNames]
    testRuns = args.num_runs
    verbose = args.verbose
    useGraphics = args.graphics
    plain = args.plain
    (frameSizeX, frameSizeY) = WINDOW_SIZE_MAP[args.screen_size]
    readFromJson = args.json
    framerate = args.framerate
    checkpoints = dict()
    isDQN = False
    grid_height = frameSizeY // 10
    grid_width = frameSizeX // 10
    randomFood = args.random_food

    if readFromJson:
        with open('testSettings.json', "r") as settingsf:
            settings = json.load(settingsf)
            useGraphics = settings['useGraphics']
            plain = settings['plain']
            testRuns = settings['testRuns']
            verbose = settings['verbose']
            (frameSizeX, frameSizeY) = WINDOW_SIZE_MAP[settings['screenSize']]
            agentNames = settings['agents']
            agents = [AGENT_MAP[agent] for agent in agentNames]
            checkpoints = settings['checkpoints']
            randomFood = settings['randomFood']
            # print(randomFood)

    avgGameLengths, avgGameScores = [], []

    # Test each supplied agent
    for i, agentType in enumerate(agents):
        print()
        print('=' * 40)
        print('Testing', getAgentName(agentType))
        gameLengths, gameScores = [], []
        startTime = time.time()
        # screenNp, screenMat, screenNpStacked = None, None, None
        agent = agentType()
        if isinstance(agent, DQNAgent):
            isDQN = True
            net = DQN((grid_height, grid_width, 1), 5)
            if readFromJson:
                model_path = checkpoints['dqn']
            else:
                model_path = args.load
            net.load_state_dict(torch.load(model_path, map_location=device))
            net.to(device)
            agent.loadNetwork(net)

        if isinstance(agent, QLearningAgent):
            # print(agentType)
            # print(agentNames[i])
            # print(checkpoints)
            checkpoint = checkpoints[agentNames[i]]
            if (checkpoint != None) and len(checkpoint) > 0:
                agent.loadCheckpoint(checkpoint)
                print('Loading checkpoint from', checkpoint)
            else:
                agent.loadCheckpoint()
                print('Loading default checkpoint file')
            agent.stopTraining()  # unnecessary, just to be safe
        
        timeouts = 0
        for i in range(testRuns):
            gameState = GameState(pos=[[30, 20], [20, 20], [10, 20]], direction='RIGHT', frameSizeX=frameSizeX,
                                  frameSizeY=frameSizeY)
            env = Game(gameState, graphics=useGraphics, plain=plain, framerate=framerate, randomFood=randomFood)
            env.setFoodPos()
            
            agent.startEpisode(gameState)
            step = 0
            if verbose: print("Starting test " + str(i + 1) + ":")
            action = 'CONTINUE'
            gameOver = False
            while not gameOver:
            
                step += 1
                action = agent.getNextAction()
                
                #print(action)
                    
                gameOver, score = env.playStep(action)
                
                if step >= 1000:
                    if verbose: print("timeout reached")
                    timeouts += 1
                    gameOver = True

                if useGraphics:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit()

                    """if step == 1 or step == 2:
                        if screenNp is None:
                            screenNp = np.mean(env.getScreenAsNumpy(), axis=2)
                            screenMat = env.gameState.getAsMatrix()
                        else:
                            screenNpStacked = np.dstack((screenNp, np.mean(env.getScreenAsNumpy(), axis=2)))"""
                # print(is_over)
                if gameOver:
                    if verbose:
                        print("\tGame over in", step, "steps")
                        print("\tScore: ", score)
                    gameLengths.append(step)
                    gameScores.append(score)
                    env.gameOver()

        elapsedTime = round((time.time() - startTime) / 60, 2)
        avgGameLengths.append(round(np.mean(gameLengths), 3))
        avgGameScores.append(round(np.mean(gameScores), 3))

        print()

        print('-' * 40)
        print(testRuns, "test runs completed in", elapsedTime, "mins")
        print("Average game len:\t", avgGameLengths[-1], "timesteps")
        print("Min/Max game len:\t", min(gameLengths), '/', max(gameLengths), "timesteps")
        print("Average score:\t\t", avgGameScores[-1])
        print("Min/Max score:\t\t", min(gameScores), '/', max(gameScores))
        print("Number of games timed out:", timeouts)
        print('=' * 40)

        agent.stopEpisode()
    print()
    
    # if useGraphics:
        ##matplotlib display image as greyscale
        # print(screenNpStacked.shape)
        # print('Shape', screen_np.shape, ' min/max', np.min(screen_np), ' / ', np.max(screen_np))
        # fig, ax = plt.subplots(1,2)
        # ax[0].imshow(screenNpStacked[...,0], cmap='gray')
        # ax[1].imshow(screenNpStacked[...,1], cmap='gray')
        # print(screenMat)
        # plt.show()
