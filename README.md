# Snake
A snake game written in Python using the Pygame library, with implementations for different intelligent agents to play the game. Agents include a random agent, reflex agent, Q-learning agent, approximate Q-learning agent, and deep Q-learning agent.

# Dependencies
Use Python 3.7 or above

matplotlib==3.5.1

numpy==1.22.3

pygame==2.1.2

torch==1.11.0

torchvision==0.12.0

# How to run
## To play the game manually
In order to play the game as a human player, run `python3 userAgent.py`.

## To run testing (located in `test.py`)

### Flags:

`-a {String}`, `--agent {String}` (options: "random", "reflex", "exactq", "approxq", "dqn")

`-n {Integer}`, `--num_runs {Integer}`

`-v`, `--verbose` (Extra detail per test run)

`-g`, `--graphics` (Turn on graphics for testing)

`-p`, `--plain` (Show no score on graphics)

`-j`, `--json` (run settings from _testSettings.json_)

`-f {Integer}`, `--framerate {Integer}` (change the framerate of the game)

`-ff`, `--fixed_food` (Turn off random food spawning)

`--load_dqn {Filename}` (For dqn, specify which model to load)

`--load_exactq {Filename}` (For exact q agent, specify which model to load)

`--load_approxq {Filename}` (For appro q agent, specify which model to load)

### Example Commands:

**Test all agents with graphics for 5 runs each (random food spawning):**

`python3 test.py -a random reflex exactq approxq -n 5 -g`

Note: DQNAgent is not included in the above list because it has a small chance of getting stuck in an infinite loop.
The game will timeout after many steps, but this is boring to watch with graphics turned on.

**Test Exact-Q & DQN agent with graphics for 5 runs each (fixed food spawning):**

`python3 test.py -a exactq dqn -n 5 -g -ff --load_dqn models/DQN_10000_fixed.pth`

Observation: Note how differently these agents act with fixed food spawning.

**Test DQN agent with graphics for 1 run (random food spawn)**

`python3 test.py -a dqn -n 1 -g`

Note: This agent may get stuck in a loop, and if it does, it will run until the game times out.

**Test Random Agent for 1000 runs:**

`python3 test.py -a random -n 1000`

## To run training for a Q-learning agent (located in `qLearningTrain.py`)

### Flags:

`-a {String}` (Agent to use)

`-n {Integer}` (Number of training episodes)

`-t {Integer}` (Number of test runs)

`-g` (Use graphics)

`-v` (Verbose output)

`-l` (Load checkpoint from pickle file)

`-s` (Save trained weights or qvalues)

`-r` (Random food spawn during testing)

`--checkpoint {Filename}` (Filename of checkpoint to load)

`--save_filename {Filename}` (Filename for saving weights)

### Example Commands:

**Train Exact-Q Agent for 5000 runs:**

`python3 qLearningTrain.py -a q -n 5000 -t 20 -s --save_filename exactq.pkl`

Note: Weights will be saved to `SRC/`

**Train Approximate-Q Agent for 5000 runs:**

`python3 qLearningTrain.py -a approxq -n 5000 -t 20 -s --save_filename approxq.pkl`

Note: Weights will be saved to `SRC/`

## To run training for a DQN agent (located in `dqnTrain.py`)

### Flags:

`-e {Integer}`, `--episodes {Integer}` (number of episodes to train for)

`-g`, `--graphics` (show graphics)

`-s`, `--save` (save model)

`-r`, `--random_food` (spawn food randomly)

`-d`, `--record_data` (record run data as csv file)

`-l`, `--load` (load saved model)

### Example Commands:

**Train DQN Agent for 5000 runs with fixed food spawning:**

`python3 dqnTrain.py -e 5000 -s`

Note: Model & stats files will be saved in `models/`

