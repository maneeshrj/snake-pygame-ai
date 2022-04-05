from Snake import Snake, Game, GameState
from qLearningAgent import ApproxQAgent
import numpy as np

class ApproxQTrainer:
    def __init__(self, episodes=10):
        self.episodes = 3000
    
    def startTraining(self, verbose=False, graphics=False):
        gameLengths, gameScores = [], []
        agent = ApproxQAgent()
        for ep in range(self.episodes):
            if (ep % (self.episodes//10) == 0): print('Starting training episode', ep)
            snake = Snake(pos=[[100, 50], [100-10, 50], [100-20, 50]], direction='RIGHT')
            if ep+1==self.episodes:
                env = Game(snake, graphics=True)
            else:
                env = Game(snake, graphics=graphics)
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
        

trainer = ApproxQTrainer()
trainer.startTraining()
            