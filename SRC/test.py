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
                        default=["random", "reflex"],
                        choices=["random", "reflex", "exactq", "approxq", "dqn"])
    parser.add_argument("-n", "--num_runs", help="Number of runs", type=int, default=12)
    parser.add_argument("-v", "--verbose", help="Verbose output", action="store_true", default=False)
    parser.add_argument("-g", "--graphics", help="Use graphics", action="store_true", default=False)
    parser.add_argument("-s", "--screen_size", help="Size of the window", type=str, default="small",
                        choices=["small", "medium", "large"])
    parser.add_argument("-p", "--plain", help="No score or words in graphics. Use for training CNN",
                        action="store_true", default=False)
    parser.add_argument("-j", "--json", help="Read from json file", action="store_true", default=False)
    parser.add_argument("-f", "--framerate", help="Set game speed", type=int, default=10)
    parser.add_argument("--load_dqn", help="Load dqn model", type=str, default='models/DQN_50000_random.pth')
    parser.add_argument("--load_exactq", help="Load exactq model", type=str, default='exactq_values.pkl')
    parser.add_argument("--load_approxq", help="Load approxq model", type=str, default='approxq_weights.pkl')
    parser.add_argument("-ff", "--fixed_food", help="Fixed random seed for food spawn", action="store_true", default=False)

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
    randomFood = not(args.fixed_food)
    timeoutLength = (10*grid_height*grid_width)
    checkpoints['dqn'], checkpoints['exactq'], checkpoints['approxq'] = \
        args.load_dqn, args.load_exactq, args.load_approxq

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

    avgGameLengths, avgGameScores = [], []

    # Test each supplied agent
    for i, agentType in enumerate(agents):
        print()
        print('=' * 40)
        print('Testing', getAgentName(agentType))
        
        gameLengths, gameScores = [], []
        startTime = time.time()
        agent = agentType()
        if isinstance(agent, DQNAgent):
            isDQN = True
            net = DQN((grid_height, grid_width, 1), 5)
            model_path = checkpoints['dqn']
            net.load_state_dict(torch.load(model_path, map_location=device))
            net.to(device)
            agent.loadNetwork(net)

        if isinstance(agent, QLearningAgent):
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
            if testRuns > 100 and i % (testRuns // 100) == 0:
                print(f"Completed {i} tests.")
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
                
                gameOver, score = env.playStep(action)
                
                if step >= timeoutLength:
                    if verbose: print("timeout reached")
                    timeouts += 1
                    gameOver = True

                if useGraphics:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit()

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