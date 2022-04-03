from Snake import Snake, Game
import reinforcement as rl
import random
import sys
import time
import numpy as np
import json

if __name__ == "__main__":
    readFromJson = True
    useGraphics = False
    testRuns = 1
    verbose = False
    agents = [rl.randomAgent, rl.reflexAgent]

    # print(f"Arguments count: {len(sys.argv)}")
    for i, arg in enumerate(sys.argv):
        #print(f"Argument {i:>6}: {arg}")
        if arg == '-json':
            readFromJson = True
            break
        if arg == '-g':
            useGraphics = True
        if arg == '-v':
            verbose = True
        if arg.startswith('-n='):
            testRuns = int(arg[3:])
    
    if readFromJson:
        with open('testSettings.json', "r") as settingsf:
            settings = json.load(settingsf)
            useGraphics = settings['useGraphics']
            testRuns = settings['testRuns']
            verbose = settings['displayEachRun']

    for agentType in agents:
        print()
        print('='*40) 
        print('Testing', rl.getAgentName(agentType))
        gameLengths, gameScores = [], []
        startTime=time.time()
        for i in range(1,testRuns+1):
            snake = Snake(pos=[[100, 50], [100-10, 50], [100-20, 50]], direction='RIGHT')
            env = Game(snake, graphics=useGraphics)
            agent = agentType(snake, env)
            step = 0
            game_over = False
            if verbose: print("Starting test "+str(i)+":")
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
        print('-'*40)
        print("Testing complete:")
        print(testRuns, "test runs completed in", elapsedTime, "mins")
        print("Average game:\t\t", np.mean(gameLengths), "timesteps")
        print("Average score:\t\t", np.mean(gameScores))
        print("Min/Max score:\t\t", min(gameScores),'/',max(gameScores))
        print('='*40)
    print()
    