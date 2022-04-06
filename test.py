from ctypes.wintypes import CHAR
import json
import argparse
import time

import numpy as np

from Snake import Snake, Game, GameState
import reinforcement as rl

AGENT_MAP = {
    'qlearning': rl.approxQAgent,
    'reflex': rl.reflexAgent,
    'random': rl.randomAgent
}

WINDOW_SIZE_MAP = {
    'small': (100, 100),
    'medium': (250, 250),
    'large': (500, 500)
}

if __name__ == "__main__":

    # Add command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--agents", help="Agents to use", nargs='+', type=str, default=["random", "reflex", "qlearning"], choices=["random", "reflex", "qlearning"])
    parser.add_argument("-n", "--num_runs", help="Number of runs", type=int, default=1)
    parser.add_argument("-v", "--verbose", help="Verbose output", action="store_true", default=False)
    parser.add_argument("-g", "--graphics", help="Use graphics", action="store_true", default=False)
    parser.add_argument("-s", "--screen_size", help="Size of the window", type=str, default="small", choices=["small", "medium", "large"])
    parser.add_argument("-j", "--json", help="Read from json file", action="store_true", default=False)

    # Parse arguments and assign to variables
    args = parser.parse_args()
    agents = [AGENT_MAP[agent] for agent in args.agents]
    testRuns = args.num_runs
    verbose = args.verbose
    useGraphics = args.graphics
    (frame_size_x, frame_size_y) = WINDOW_SIZE_MAP[args.screen_size]
    readFromJson = args.json
    
    if readFromJson:
        with open('testSettings.json', "r") as settingsf:
            settings = json.load(settingsf)
            useGraphics = settings['useGraphics']
            testRuns = settings['testRuns']
            verbose = settings['displayEachRun']
            agents = [rl.randomAgent, rl.reflexAgent, rl.ApproxQAgent]

    avgGameLengths, avgGameScores = [], []
    
    # Test each supplied agent
    for agentType in agents:
        print()
        print('='*40)
        print('Testing', rl.getAgentName(agentType))
        gameLengths, gameScores = [], []
        startTime=time.time()
        for i in range(testRuns):
            snake = Snake(pos=[[30, 20], [20, 20], [10, 20]], direction='RIGHT')
            env = Game(snake, graphics=useGraphics, frame_size_x=frame_size_x, frame_size_y=frame_size_y)
            agent = agentType(snake, env)
            step = 0
            game_over = False
            if verbose: print("Starting test "+str(i+1)+":")
            while not game_over:
                step += 1
                action = agent.getNextAction()
                game_over, score = env.play_step(action)
                # print(is_over)
                if game_over:
                    if verbose:
                        print("\tGame over in", step, "steps")
                        print("\tScore: ", score)
                    gameLengths.append(step)
                    gameScores.append(score)
                    env.game_over()

        elapsedTime = round((time.time() - startTime) / 60,2)
        avgGameLengths.append(round(np.mean(gameLengths), 3))
        avgGameScores.append(round(np.mean(gameScores), 3))
        
        print('-'*40)
        print(testRuns, "test runs completed in", elapsedTime, "mins")
        print("Average game:\t\t", avgGameLengths[-1], "timesteps")
        print("Average score:\t\t", avgGameScores[-1])
        print("Min/Max score:\t\t", min(gameScores),'/',max(gameScores))
        print('='*40)
    print()
    