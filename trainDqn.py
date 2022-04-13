from Snake import Game, Snake

from PIL import Image
import random
import math
import torch
import torch.optim as optim
import torchvision.transforms as T
import torch.nn.functional as F
from itertools import count
from collections import namedtuple

from dqn import DQN, ReplayMemory

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
steps_done = 0

screen_height = 40
screen_width = 40

# Get number of actions from gym action space
# Maybe change actions to left, right, continue so that there are never illegal moves and only 3 actions
n_actions = 4

policy_net = DQN(screen_height, screen_width, n_actions).to(DEVICE)
target_net = DQN(screen_height, screen_width, n_actions).to(DEVICE)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=DEVICE, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(BATCH_SIZE, device=DEVICE)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(2)]], device=DEVICE, dtype=torch.long)


if __name__ == "__main__":

    num_episodes = 50
    resize = T.Compose([T.ToPILImage(),
                        T.Resize(40, interpolation=T.InterpolationMode.BICUBIC),
                        T.ToTensor()])
    episode_durations = []

    for i_episode in range(num_episodes):
        # Initialize the environment and state
        # env.reset()
        print(f"Starting Episode {i_episode}")
        snake = Snake([[30, 20], [20, 20], [10, 20]], 'RIGHT')
        game = Game(snake, graphics=True, frame_size_x=100, frame_size_y=100, plain=True)
        # Converts to PIL image, resize height to 40, convert to tensor
        state = resize(game.get_window_as_np())

        step = 0
        game_over = False
        while not game_over:

            action = select_action(state)  # make sure it only selects valid actions
            game_over, score = game.play_step(action)  # Change score to a reward?
            score = torch.tensor([score], device=DEVICE)

            new_screen = resize(game.get_window_as_np())
            if not game_over:
                next_state = new_screen
            else:
                next_state = None

            memory.push(state, action, next_state, score)
            state = next_state

            optimize_model()
            if game_over:
                episode_durations.append(step + 1)
                # plot_durations()
            step += 1
        # Update the target network
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
