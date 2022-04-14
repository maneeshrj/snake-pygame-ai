from Snake import Game, GameState, Trial
from qLearningAgent import QLearningAgent
import numpy as np
import random
import argparse, sys
import json, pickle

class Trainer:
    def __init__(self, agent):
        self.agent = agent
        self.trainingTrial = None
        self.testingTrial = None
        self.totalTrainRewards = 0.0

    def train(self, trainingEpisodes=1000, verbose=False, saveWeights=True):
        """
        Train the agent for the specified number of episodes.
        Each episode is a complete game.
        """
        #random.seed(42)
        self.totalTrainRewards = 0.0
        print("Training agent for", trainingEpisodes, "episodes.")
        print("=" * 40)
        self.agent.startTraining(numTraining=trainingEpisodes)

        trainingTrial = Trial()
        self.trainingTrial = trainingTrial

        for episode in range(trainingEpisodes):
            gameState = GameState(pos=[[30, 20], [20, 20], [10, 20]], direction='RIGHT', frameSizeX=100,
                                  frameSizeY=100)
            if verbose and episode % (trainingEpisodes // 5) == 0:
                trainGraphics = True
            else:
                trainGraphics = False
            game = Game(gameState=gameState, graphics=trainGraphics, plain=True,
                        foodPosList=trainingTrial.getFoodPosList())
            trainingTrial.setCurrentGame(game)
            game.setFoodPos()

            self.agent.startEpisode(gameState)
            gameOver = False
            while not gameOver:
                action = self.agent.getNextAction()

                reward = game.getReward(action)
                nextState = game.getNextState(action)
                gameOver, score = game.playStep(action)

                # I don't think this needs to be called here if it is called in getNextAction?
                #self.agent.observeTransition(state, action, nextState, reward)

            game.gameOver()
            # if verbose and episode % (trainingEpisodes // 10) == 0:
            #     print(f"Finished episode {episode} of {trainingEpisodes}.")
            #     print('Accumulated rewards', self.agent.episodeRewards)
            
            self.agent.stopEpisode()

            if episode % (trainingEpisodes // 4) == 0 and verbose:
                print(f"Finished episode {episode} of {trainingEpisodes}.")
                self.totalTrainRewards = self.agent.accumTrainRewards - self.totalTrainRewards
                print('Accumulated rewards at 25% training interval:', self.totalTrainRewards)
                self.totalTrainRewards = self.agent.accumTrainRewards
                qValDict = self.agent.getQValuesAsDictionary()
                print('Q Value dictionary size:', sys.getsizeof(qValDict), 'bytes')
                print('Number of Q-states explored:', len(qValDict))
                if saveWeights:
                    with open('qvalues.pkl', 'wb') as f:
                        pickle.dump(qValDict, f, protocol=pickle.HIGHEST_PROTOCOL)
                    with open('qvalues.pkl', 'rb') as f:
                        counts = pickle.load(f)
                        print('Length of qval dict saved:', len(counts))

        self.agent.stopTraining()
        print('Average rewards per training episode:', (self.agent.accumTrainRewards/trainingEpisodes))

    def test(self, testRuns=10, graphics=False, verbose=False):
        """
        Test the agent for the specified number of runs.
        Each run is a complete game.
        """
        print("Testing agent for", testRuns, "runs.")
        gameLengths, gameScores = [], []
        print(testRuns)
        testingTrial = Trial()
        testingTrial.setFoodPosList(self.trainingTrial.getFoodPosList())
        self.testingTrial = testingTrial

        for i in range(testRuns):
            gameState = GameState(pos=[[30, 20], [20, 20], [10, 20]], direction='RIGHT',
                                  frameSizeX=100, frameSizeY=100)
            game = Game(gameState=gameState, graphics=True, plain=True,
                        foodPosList=testingTrial.getFoodPosList())
            testingTrial.setCurrentGame(game)
            game.setFoodPos()
            self.agent.startEpisode(gameState)
            gameOver = False
            step = 0
            while not gameOver:
                step += 1
                action = self.agent.getNextAction()
                gameOver, score = game.playStep(action)
            gameLengths.append(step)
            gameScores.append(score)
            self.agent.stopEpisode()
            game.gameOver()

        
        print("Average game:\t\t", np.mean(gameLengths), "timesteps")
        print("Min/Max length:\t", min(gameLengths), '/', max(gameLengths), "timesteps")
        print("Average score:\t\t", np.mean(gameScores))
        print("Min/Max score:\t\t", min(gameScores), '/', max(gameScores))
        print(gameScores, ' ... ', gameLengths)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or test the agent')
    parser.add_argument("-a", "--agent", help="Agent to use", type=str, default="q", choices=["q", "approxq"])
    parser.add_argument("-n", "--num_episodes", help="Number of training episodes", type=int, default=4000)
    parser.add_argument("-t", "--test_runs", help="Number of test runs", type=int, default=10)
    parser.add_argument("-v", "--verbose", help="Verbose output", action="store_true")
    parser.add_argument("-l", "--load", help="Load qvalues from pickle file", action="store_true")
    parser.add_argument("-s", "--save_weights", help="Save trained weights", action="store_true")

    args = parser.parse_args()
    agentType = args.agent
    numEpisodes = args.num_episodes
    testRuns = args.test_runs
    verbose = args.verbose
    loadQValues = args.load
    saveWeights= args.save_weights

    if agentType == "q":
        agent = QLearningAgent()
        if loadQValues:
            agent.loadQValues()
        trainer = Trainer(agent)
        trainer.train(trainingEpisodes=numEpisodes, verbose=verbose, saveWeights=saveWeights)
        trainer.test(testRuns=testRuns, verbose=verbose)
    """elif agentType == "approxq":
        agent = ApproximateQLearningAgent()
        if loadQValues:
            agent.loadQValues()
        trainer = Trainer(agent)
        trainer.train(trainingEpisodes=numEpisodes, verbose=verbose, saveWeights=saveWeights)
        trainer.test(testRuns=testRuns, verbose=verbose)"""


# class ApproxQTrainer:
#     def __init__(self, episodes=10):
#         self.episodes = episodes

#     def startTraining(self, verbose=False, graphics=False):
#         gameLengths, gameScores = [], []
#         agent = ApproxQAgent()
#         for ep in range(self.episodes+1):
#             if (self.episodes < 10) or (ep % (self.episodes//10) == 0): print('Starting training episode', ep)

#             # gameState = GameState(pos=[[30, 20], [20, 20], [10, 20]], direction='RIGHT')
#             gameState = GameState()
#             env = Game(graphics=graphics, frameSizeX=100, frameSizeY=100)
#             agent.startEpisode(gameState, env)
#             step = 0
#             gameOver = False
#             print("")
#             while not gameOver:
#                 print(f"Food pos step {step}: {env.foodPos}")
#                 step += 1
#                 action = agent.getNextAction()
#                 gameOver, score = env.playStep(action)
#                 if gameOver:
#                     if verbose:
#                         print("\tGame over in", step, "steps")
#                         print("\tScore: ", score)
#                     gameLengths.append(step)
#                     gameScores.append(score)
#                     #print('weights:\t',agent.getWeights())
#                     env.gameOver()
#             agent.stopEpisode()
#         print('Final weights:\n\t',agent.getWeights())
#         print('Avg game length:', round(np.mean(gameLengths), 3))
#         print('Min/max game length:', min(gameLengths), ' / ', max(gameLengths))
#         print('Avg score:', round(np.mean(gameScores), 3))
#         print('Min/max score:', min(gameScores), ' / ', max(gameScores))
#         self.agent = agent

#     def testAgent(self, testRuns=10, verbose=True, graphics=False):
#         print("--- Starting", testRuns, "test runs --")
#         self. agent.stopTraining()
#         gameLengths, gameScores = [], []
#         for ep in range(testRuns):
#             if verbose: print('Starting testing episode', ep)
#             # gameState = GameState(pos=[[30, 20], [20, 20], [10, 20]], direction='RIGHT')
#             gameState = GameState()
#             env = Game(gameState, graphics=graphics, frameSizeX=100, frameSizeY=100)
#             self.agent.startEpisode(gameState, env)
#             step = 0
#             gameOver = False
#             while not gameOver:
#                 step += 1
#                 action = self.agent.getNextAction()
#                 gameOver, score = env.playStep(action)
#                 if gameOver:
#                     if verbose:
#                         print("\tTest run",ep,"over in", step, "steps")
#                         print("\tScore: ", score)
#                     gameLengths.append(step)
#                     gameScores.append(score)
#                     #print('weights:\t',agent.getWeights())
#                     env.gameOver()
#             self.agent.stopEpisode()
#         print('Final weights:\n\t', self.agent.getWeights())
#         print('Avg game length:', round(np.mean(gameLengths), 3))
#         print('Min/max game length:', min(gameLengths), ' / ', max(gameLengths))
#         print('Avg score:', round(np.mean(gameScores), 3))
#         print('Min/max score:', min(gameScores), ' / ', max(gameScores))
