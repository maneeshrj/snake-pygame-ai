from Snake import Game, GameState, Trial
from qLearningAgent import QLearningAgent, ApproxQAgent, getFeatures
import numpy as np
import random
import argparse, sys, time
import json, pickle
import time

class Trainer:
    def __init__(self, agent, trainRandom=False, testRandom=False, saveFile='checkpoint.pkl'):
        self.agent = agent
        self.trainingTrial = None
        self.testingTrial = None
        self.totalTrainRewards = 0.0
        self.trainRandom = trainRandom
        self.testRandom = testRandom
        self.saveFile = saveFile

    def train(self, trainingEpisodes=1000, verbose=False, saveWeights=None):
        """
        Train the agent for the specified number of episodes.
        Each episode is a complete game.
        """
        self.totalTrainRewards = 0.0
        print('\n'+"="*40+"\nTraining",str(self.agent),"for", trainingEpisodes, "episodes.\n"+"-"*40)
        self.agent.startTraining(numTraining=trainingEpisodes)

        startTime = time.time()
        trainingTrial = Trial()
        self.trainingTrial = trainingTrial

        for episode in range(1,trainingEpisodes+1):
            gameState = GameState(pos=[[30, 20], [20, 20], [10, 20]], direction='RIGHT', frameSizeX=100,
                                  frameSizeY=100)
            trainGraphics = False
            if verbose and episode % (trainingEpisodes // 5) == 0:
                trainGraphics = True
            game = Game(gameState=gameState, graphics=trainGraphics, plain=True,
                        foodPosList=trainingTrial.getFoodPosList())
            trainingTrial.setCurrentGame(game)
            game.setFoodPos()

            self.agent.startEpisode(gameState)
            gameOver = False
            i = 0
            # print("Start State:", game.gameState)
            # print("Start Matrix:\n", game.gameState.getAsMatrix())
            while not gameOver:
                
                i += 1
                # print("Determining Action...")
                action = self.agent.getNextAction()
                # reward = game.getReward(action)
                # nextState = game.getNextState(action)
                gameOver, score = game.playStep(action)
                # if i == 1:
                #     # print("Current State:", game.gameState)
                #     # print(game.gameState.getAsMatrix())
                #     while(True):
                #         time.sleep(1000)

            game.gameOver()
            self.agent.stopEpisode()

            if episode % (trainingEpisodes // 5) == 0:
                print(f"\nFinished episode {episode} of {trainingEpisodes}.")
                if verbose:
                    self.totalTrainRewards = self.agent.accumTrainRewards - self.totalTrainRewards
                    print('Accumulated rewards at 25% training interval:', self.totalTrainRewards)
                    self.totalTrainRewards = self.agent.accumTrainRewards
                if saveWeights:
                    print('Saving checkpoint to', self.saveFile)
                    self.agent.saveCheckpoint(self.saveFile)
        
        self.agent.stopTraining()
        if saveWeights:
            self.agent.saveCheckpoint(self.saveFile)
        elapsedTime = round((time.time() - startTime) / 60, 2)
        print('\nTraining completed in', elapsedTime, 'mins.')
        print('Average rewards per training episode:', (self.agent.accumTrainRewards/trainingEpisodes))

    def test(self, testRuns=10, graphics=True, verbose=False):
        """
        Test the agent for the specified number of runs.
        Each run is a complete game.
        """
        print('\n'+"="*40+"\nTesting agent for", testRuns, "runs.\n"+'-'*40)
        gameLengths, gameScores = [], []
        testingTrial = Trial()
        if not self.testRandom:
            testingTrial.setFoodPosList(self.trainingTrial.getFoodPosList())
        self.testingTrial = testingTrial

        for i in range(testRuns):
            gameState = GameState(pos=[[30, 20], [20, 20], [10, 20]], direction='RIGHT',
                                  frameSizeX=100, frameSizeY=100)
            game = Game(gameState=gameState, graphics=graphics, plain=True, framerate=10,
                        foodPosList=testingTrial.getFoodPosList(), randomFood=self.testRandom)
            testingTrial.setCurrentGame(game)
            game.setFoodPos()
            self.agent.startEpisode(gameState)
            gameOver = False
            step = 0
            while not gameOver:
                step += 1
                action = self.agent.getNextAction()
                
                # time.sleep(10000)
                gameOver, score = game.playStep(action)
            gameLengths.append(step)
            gameScores.append(score)
            self.agent.stopEpisode()
            game.gameOver()

        print("Average game:\t\t", np.mean(gameLengths), "timesteps")
        print("Min/Max game length:\t", min(gameLengths), '/', max(gameLengths), "timesteps")
        print("Average score:\t\t", np.mean(gameScores))
        print("Min/Max score:\t\t", min(gameScores), '/', max(gameScores))
        if verbose:
            print('Scores:', gameScores)
            print('Game lengths:', gameLengths)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or test the agent')
    parser.add_argument("-a", "--agent", help="Agent to use", type=str, default="q", choices=["q", "approxq"])
    parser.add_argument("-n", "--num_episodes", help="Number of training episodes", type=int, default=4000)
    parser.add_argument("-t", "--test_runs", help="Number of test runs", type=int, default=10)
    parser.add_argument("-v", "--verbose", help="Verbose output", action="store_true")
    parser.add_argument("-l", "--load", help="Load qvalues from pickle file", action="store_true")
    parser.add_argument("-s", "--save_weights", help="Save trained weights", action="store_true")
    parser.add_argument("-r", "--test_random", help="Random food spawn during testing", action="store_true")

    args = parser.parse_args()
    agentType = args.agent
    numEpisodes = args.num_episodes
    testRuns = args.test_runs
    verbose = args.verbose
    loadQValues = args.load
    saveWeights= args.save_weights
    testRandom = args.test_random
    trainRandom = True

    if agentType == "q":
        agent = QLearningAgent()
        trainRandom = False
        saveFilename = 'qvalues.pkl'
    elif agentType == "approxq":
        agent = ApproxQAgent()
        saveFilename = 'approxq_weights_2.pkl'
    
    if loadQValues:
        agent.loadCheckpoint(saveFilename)
            
    if agentType == 'approxq':
        print('\nInitial weights: ', agent.weights)
            
    trainer = Trainer(agent, trainRandom=trainRandom, testRandom=testRandom, saveFile=saveFilename)
    trainer.train(trainingEpisodes=numEpisodes, verbose=verbose, saveWeights=saveWeights)
    trainer.test(testRuns=testRuns, graphics=True, verbose=verbose)
    
    if agentType == 'approxq':
        print('\nFinal weights: ', agent.weights)
