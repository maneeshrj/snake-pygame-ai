from Snake import Snake, Game, GameState
from qLearningAgent import ApproxQAgent

class ApproxQTrainer:
    def __init__(self, episodes=10):
        self.episodes = 100
    
    def startTraining(self, verbose=True):
        gameLengths, gameScores = [], []
        agent = ApproxQAgent()
        for ep in range(self.episodes):
            print('Starting training episode', ep)
            snake = Snake(pos=[[100, 50], [100-10, 50], [100-20, 50]], direction='RIGHT')
            env = Game(snake, graphics=False)
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
                    env.game_over()
            agent.stopEpisode()

trainer = ApproxQTrainer()
trainer.startTraining()
            