from Snake import Snake, Game, GameState
from qLearningAgent import ApproxQAgent
import numpy as np

class ApproxQTrainer:
    def __init__(self, episodes=10):
        self.episodes = episodes
    
    def startTraining(self, verbose=False, graphics=False):
        gameLengths, gameScores = [], []
        agent = ApproxQAgent()
        for ep in range(self.episodes+1):
            if (self.episodes < 10) or (ep % (self.episodes//10) == 0): print('Starting training episode', ep)
            snake = Snake(pos=[[30, 20], [20, 20], [10, 20]], direction='RIGHT')
            env = Game(snake, graphics=graphics, frame_size_x=100, frame_size_y=100)
            agent.startEpisode(snake, env)
            step = 0
            game_over = False
            while not game_over:
                step += 1
                action = agent.getNextAction()
                game_over, score = env.play_step(action)
                if game_over:
                    if verbose:
                        print("\tGame over in", step, "steps")
                        print("\tScore: ", score)
                    gameLengths.append(step)
                    gameScores.append(score)
                    #print('weights:\t',agent.getWeights())
                    env.game_over()
            agent.stopEpisode()
        print('Final weights:\n\t',agent.getWeights())
        print('Avg game length:', round(np.mean(gameLengths), 3))
        print('Min/max game length:', min(gameLengths), ' / ', max(gameLengths))
        print('Avg score:', round(np.mean(gameScores), 3))
        print('Min/max score:', min(gameScores), ' / ', max(gameScores))
        self.agent = agent
    
    def testAgent(self, testRuns=10, verbose=True, graphics=False):
        print("--- Starting", testRuns, "test runs --")
        self. agent.stopTraining()
        gameLengths, gameScores = [], []
        for ep in range(testRuns):
            if verbose: print('Starting testing episode', ep)
            snake = Snake(pos=[[30, 20], [20, 20], [10, 20]], direction='RIGHT')
            env = Game(snake, graphics=graphics, frame_size_x=100, frame_size_y=100)
            self.agent.startEpisode(snake, env)
            step = 0
            game_over = False
            while not game_over:
                step += 1
                action = self.agent.getNextAction()
                game_over, score = env.play_step(action)
                if game_over:
                    if verbose:
                        print("\tTest run",ep,"over in", step, "steps")
                        print("\tScore: ", score)
                    gameLengths.append(step)
                    gameScores.append(score)
                    #print('weights:\t',agent.getWeights())
                    env.game_over()
            self.agent.stopEpisode()
        print('Final weights:\n\t', self.agent.getWeights())
        print('Avg game length:', round(np.mean(gameLengths), 3))
        print('Min/max game length:', min(gameLengths), ' / ', max(gameLengths))
        print('Avg score:', round(np.mean(gameScores), 3))
        print('Min/max score:', min(gameScores), ' / ', max(gameScores))
        

trainer = ApproxQTrainer()
trainer.startTraining()
trainer.testAgent(2, False, True)
            