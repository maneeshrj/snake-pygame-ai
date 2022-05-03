import gym
import numpy as np
from gym import spaces
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import time
import argparse

class CustomSnakeEnv(gym.Env):
    """
    A custom gym environment for snake
    """

    metadata = {'render.modes': ['console']}

    LEFT = 0
    RIGHT = 1
    CONTINUE = 2
    TIMEOUT_LIM = 200
    
    def __init__(self, width=10, height=10, n_food=1):
        super(CustomSnakeEnv, self).__init__()
        # Define the action and observation space
        # They must be gym.spaces objects

        self.grid_size = (width, height)
        self.snake_full_pos = [[2, 4], [2, 3], [2, 2]]
        self.snake_head = self.snake_full_pos[0].copy()
        self.snake_direction = "EAST"
        self.food_pos = [[4, 4]]
        self.set_grid()
        
        self.LEFT = 0
        self.RIGHT = 1
        self.CONTINUE = 2

        self.timed_out = False
        self.debug_mode = False
        self.score = 0
        self.steps = 0
        self.num_timeouts = 0
        self.num_episodes = 0

        # The snake can contnue moving in the same direction or turn left or right
        n_actions = 3
        self.action_space = spaces.Discrete(n_actions)
        
        # The snake lives in a grid of size width x height with food on the grid
        # possible values are 0 (empty cell) and 1 (snake) and 2 (food)
        self.observation_space = spaces.Box(low=0, high=2, shape=(width*height, ), dtype=np.float32)

    def set_grid(self):
        self.grid = np.zeros(self.grid_size)
        for pos in self.snake_full_pos:
            self.grid[pos[0], pos[1]] = 1
        for pos in self.food_pos:
            self.grid[pos[0], pos[1]] = 2
    
    def add_food(self):
        """
        Add Food to the grid
        """
        food_loc = [np.random.randint(0, self.grid_size[0]), np.random.randint(0, self.grid_size[1])]

        # Make sure the food is not in the snake
        while food_loc in self.snake_full_pos:
            food_loc = [np.random.randint(0, self.grid_size[0]), np.random.randint(0, self.grid_size[1])]
        
        self.food_pos.append(food_loc)

    def reset(self):
        """
        Reset the environment to an initial state
        Returns
        -------
        observation: np.array
            the initial observation of the space.
        """
        # Initial grid is width x height filled with zeros
        self.snake_full_pos = [[2, 4], [2, 3], [2, 2]]
        self.snake_head = self.snake_full_pos[0].copy()
        self.snake_direction = "EAST"
        self.food_pos = [[4, 4]]
        self.score = 0
        self.steps = 0
        self.num_episodes += 1
        self.timed_out = False
        self.set_grid()
        return np.array(self.grid).astype(np.float32).flatten()

    def get_snake_direction(self):
        """
        Get's the snakes direction based on its head position and the location of he neck (second element in the snake)
        """
        # Check if moving east
        if self.snake_head[1] - 1 == self.snake_full_pos[1][1]:
            return "EAST"
        # Check if moving west
        elif self.snake_head[1] + 1 == self.snake_full_pos[1][1]:
            return "WEST"
        # Check if moving south
        elif self.snake_head[0] - 1 == self.snake_full_pos[1][0]:
            return "SOUTH"
        # Check if moving north
        elif self.snake_head[0] + 1 == self.snake_full_pos[1][0]:
            return "NORTH"
        else:
            return "ERROR"

    def move_snake(self, action):
        """
        Move the snake on the grid
        """

        if action == self.CONTINUE:
            if self.snake_direction == "EAST":
                self.snake_head[1] += 1
            elif self.snake_direction == "WEST":
                self.snake_head[1] -= 1
            elif self.snake_direction == "SOUTH":
                self.snake_head[0] += 1
            elif self.snake_direction == "NORTH":
                self.snake_head[0] -= 1
        elif action == self.LEFT:
            if self.snake_direction == "EAST":
                self.snake_head[0] -= 1
            elif self.snake_direction == "WEST":
                self.snake_head[0] += 1
            elif self.snake_direction == "SOUTH":
                self.snake_head[1] += 1
            elif self.snake_direction == "NORTH":
                self.snake_head[1] -= 1
        elif action == self.RIGHT:
            if self.snake_direction == "EAST":
                self.snake_head[0] += 1
            elif self.snake_direction == "WEST":
                self.snake_head[0] -= 1
            elif self.snake_direction == "SOUTH":
                self.snake_head[1] -= 1
            elif self.snake_direction == "NORTH":
                self.snake_head[1] += 1
        else:
            raise ValueError("Received invalid action={} which is not part of the action space".format(action))

        # Duplicate the head of the snake and add it to the front of the snake
        self.snake_full_pos.insert(0, list(self.snake_head))

        reached_food = False
        # Check if the snake has eaten the food
        if self.snake_head in self.food_pos:
            self.food_pos.remove(self.snake_head)
            # Add a new food to the grid
            self.add_food()
            reached_food = True
            self.score += 1
        else:
            # Remove the tail of the snake
            self.snake_full_pos.pop()

        # Check if the snake has hit the wall
        dead = False
        if self.snake_head[0] < 0 or self.snake_head[0] >= self.grid_size[0]:
            dead = True
        if self.snake_head[1] < 0 or self.snake_head[1] >= self.grid_size[1]:
            dead = True
        # Check if the snake hit itself
        if self.snake_head in self.snake_full_pos[1:]:
            dead = True
        
        return reached_food, dead

    def step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the environment
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        # Move the snake on the grid
        self.steps += 1
        reached_food, done = self.move_snake(action)
        reward = -1
        if done:
            reward = -15.0
        else:
            if reached_food:
                reward = 10.0
            self.set_grid()

        # normalize the reward to be between -1 and 1
        
        # Update the snakes direciton
        self.snake_direction = self.get_snake_direction()

        if self.steps > self.TIMEOUT_LIM:
            self.num_timeouts += 1
            self.timed_out = True
            done = True

        # Return the observation, reward, done, and info
        info = {"score": self.score,
                "num_timeouts": self.num_timeouts,
                "num_episodes": self.num_episodes}
        if self.timed_out:
            info["TimeLimit.truncated"] = True
        return np.array(self.grid).astype(np.float32).flatten(), reward, done, info

    def set_debug_mode(self, debug_mode):
        self.debug_mode = debug_mode

    def render(self, mode='console'):
        """
        Render the environment in human or machine readable format
        """
        if mode == 'console':
            print(self.grid)
        elif mode == 'human':
            pass
        else:
            print("ERROR: Invalid render mode")

    def close(self):
        """
        Clean up the environment
        """
        pass

def action_num_to_str(action):
    if action == 0:
        return "LEFT"
    elif action == 1:
        return "RIGHT"
    elif action == 2:
        return "CONTINUE"
    else:
        return "ERROR"
        
if __name__ == "__main__":

    # Parse the command line arguments
    parser = argparse.ArgumentParser(description='Snake Gym Environment')
    parser.add_argument('-r', '--random', action='store_true', help='Run the environment in random mode')
    parser.add_argument('-n', '--num_episodes', type=int, default=1, help='Number of episodes to run')
    parser.add_argument('-s', '--size', type=int, default=10, help='Size of the grid')
    parser.add_argument('-dqn', '--dqn', action='store_true', help='Run the environment in DQN mode')
    parser.add_argument('-ppo', '--ppo', action='store_true', help='Run the environment in PPO mode')
    parser.add_argument('-a2c', '--a2c', action='store_true', help='Run the environment in A2C mode')

    args = parser.parse_args()
    random_mode = args.random
    num_episodes = args.num_episodes
    grid_size = args.size
    use_dqn = args.dqn
    use_ppo = args.ppo
    use_a2c = args.a2c
    if [use_dqn, random_mode, use_ppo, use_a2c].count(True) > 1:
        raise ValueError("Cannot run in multiple modes at the same time")

    env = CustomSnakeEnv(height=grid_size, width=grid_size, n_food=1)
    # Add a time limit to the environment

    env = make_vec_env(lambda: env, n_envs=1)
    obs = env.reset()
    if random_mode:
        env.render("console")
        for _ in range(num_episodes):
            action = env.action_space.sample()
            print("Action: ", action_num_to_str(action))
            obs, reward, done, info = env.step([action])
            env.render("console")
            time.sleep(0.5)
            if done:
                print("Score: ", info[0]["score"])
                obs = env.reset()
    elif use_dqn:
        model = DQN(policy="MlpPolicy", env=env)
        model.learn(total_timesteps=int(num_episodes))
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
        print("Mean Reward:", mean_reward, "Std Reward:", std_reward)
    elif use_ppo:
        model = PPO(policy="MlpPolicy", env=env)
        model.learn(total_timesteps=int(num_episodes))
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
        print("Mean Reward:", mean_reward, "Std Reward:", std_reward)
    elif use_a2c:
        model = A2C(policy="MlpPolicy", env=env)
        model.learn(total_timesteps=int(num_episodes))
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
        print("Mean Reward:", mean_reward, "Std Reward:", std_reward)
    
    obs = env.reset()

    eval_runs = 100
    scores = []
    steps = []

    for i in range(eval_runs):
        render = False
        done = False
        step = 0
        # if i multiple of 1/10th the number of eval runs
        """if i % (eval_runs/10) == 0:
            render = True
            print("-" * 40)"""
        while not done:
            step += 1
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, [done], info = env.step(action)
            score = info[0]["score"]
            if render:
                env.render("console")
        scores.append(score)
        steps.append(step)
    
    print("Average Eval Score:", np.mean(scores))
    print("Average Eval Steps:", np.mean(steps))
    print(f"Total Timeout Percentage: {round(info[0]['num_timeouts']*100/(info[0]['num_episodes']), 3)}%")











