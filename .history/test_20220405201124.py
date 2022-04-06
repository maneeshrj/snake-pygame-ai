from Snake import Snake, Game, GameState
import reinforcement as rl
import random
import sys, time
import numpy as np
import json

if __name__ == "__main__":
    readFromJson = False
    useGraphics = False
    testRuns = 1
    verbose = False
    agents = []
    frame_size_x = 480
    frame_size_y = 480

    # print(f"Arguments count: {len(sys.argv)}")
    for i, arg in enumerate(sys.argv):
        #print(f"Argument {i:>6}: {arg}")
        if arg == '-json':
            readFromJson = True
            break
        if arg == '-nojson':
            readFromJson = False
        if arg == '-g':
            useGraphics = True
        if arg == '-v':
            verbose = True
        if arg.startswith('-n='):
            testRuns = int(arg[3:])
        if arg == '-ra':
            agents.append(rl.randomAgent)
        if arg == '-rf':
            agents.append(rl.reflexAgent)
        if arg == '-qa':
            agents.append(rl.ApproxQAgent)
        if arg == '-small':
            frame_size_x = 100
            frame_size_y = 100
        if arg == '-medium':
            frame_size_x = 250
            frame_size_y = 250
        if arg == '-large':
            frame_size_x = 500
            frame_size_y = 500
    
    if readFromJson:
        with open('testSettings.json', "r") as settingsf:
            settings = json.load(settingsf)
            useGraphics = settings['useGraphics']
            testRuns = settings['testRuns']
            verbose = settings['displayEachRun']
            agents = [rl.randomAgent, rl.reflexAgent, rl.ApproxQAgent]

    avgGameLengths, avgGameScores = [], []
    
    for agentType in agents:
        print()
        print('='*40) 
        print('Testing', rl.getAgentName(agentType))
        gameLengths, gameScores = [], []
        startTime=time.time()
        for i in range(1,testRuns+1):
            snake = Snake(pos=[[100, 50], [100-10, 50], [100-20, 50]], direction='RIGHT', )
            env = Game(snake, graphics=useGraphics, frame_size_x=frame_size_x, frame_size_y=frame_size_y)
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
        avgGameLengths.append(round(np.mean(gameLengths), 3))
        avgGameScores.append(round(np.mean(gameScores), 3))
        
        print('-'*40)
        print(testRuns, "test runs completed in", elapsedTime, "mins")
        print("Average game:\t\t", avgGameLengths[-1], "timesteps")
        print("Average score:\t\t", avgGameScores[-1])
        print("Min/Max score:\t\t", min(gameScores),'/',max(gameScores))
        print('='*40)
    print()
    