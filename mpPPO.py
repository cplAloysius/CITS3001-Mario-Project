import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import time
import torch.multiprocessing as mp
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# Maximum time per episode in seconds
MAX_TIME_PER_EPISODE = 60

# Model Definition for Mario with both Policy and Value heads.
class MarioNet(nn.Module):
    def __init__(self, output_dim):
        super(MarioNet, self).__init__()

        # Convolutional layers for processing the game screen.
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Fully connected layers for decision making.
        self.fc1 = nn.Linear(64 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, output_dim)

        # Value function for estimating the value of a state.
        self.value_head = nn.Linear(512, 1)

    def forward(self, x):
        # Process screen using convolutional layers.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        # Process the flattened tensor using fully connected layers.
        x = F.relu(self.fc1(x))
        
        # Get action probabilities and state value estimates.
        action_probs = x.softmax(dim=-1)
        value = self.value_head(x)
        return action_probs, value

# Transformations to preprocess the game screen.
resize = T.Compose([T.ToPILImage(),
                    T.Resize((84, 84), interpolation=Image.CUBIC),
                    T.ToTensor()])

# Function to process a game frame to the desired format.
def get_screen(frame):
    return resize(frame).unsqueeze(0).permute(0, 3, 1, 2)

# Utility function to save model weights to disk.
def save_model(model, filename):
    torch.save(model.state_dict(), filename)

# Function to compute advantages using Generalized Advantage Estimation.
def compute_advantages(rewards, values, gamma=0.99, tau=0.95):
    advantages = torch.zeros_like(rewards)
    gae = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * tau * gae
        advantages[t] = gae
    return advantages

# The worker function that each process will run for training.
def worker(process_id, global_model, optimizer, lock, action_dim, episodes_per_save=10):
    # Local model for each worker.
    local_model = MarioNet(action_dim)
    local_model.load_state_dict(global_model.state_dict())

    # Setup the environment for Super Mario Bros.
    env = gym_super_mario_bros.make('SuperMarioBros-v3')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env.seed(process_id)

    # Main loop for training episodes.
    for episode in range(episodes_per_save):
        state = env.reset()
        state = get_screen(state)
        done = False
        start_time = time.time()
        episode_reward = 0
        
        # Lists to store results for each step.
        saved_log_probs = []
        rewards = []
        values = []

        # Loop for steps inside an episode.
        while not done and time.time() - start_time < MAX_TIME_PER_EPISODE:
            # Get action probabilities and value estimate for current state.
            action_prob, value = local_model(state)
            m = torch.distributions.Categorical(action_prob)
            action = m.sample().item()

            # Store log probabilities and value estimates.
            saved_log_probs.append(m.log_prob(torch.tensor(action)))
            values.append(value)

            # Take the chosen action in the environment.
            next_state, reward, done, info = env.step(action)
            next_state = get_screen(next_state)

            # Store rewards.
            rewards.append(reward)
            state = next_state

        # Compute advantages using the rewards and value estimates.
        advantages = compute_advantages(rewards, values)
        old_log_probs = torch.stack(saved_log_probs)
        old_action_probs = torch.exp(old_log_probs)

        # Compute the new policy after the episode.
        new_action_probs, new_values = local_model(torch.cat(states))
        new_log_probs = new_action_probs.log()

        # PPO objective and loss calculations.
        ratio = new_action_probs / old_action_probs
        clip_epsilon = 0.2
        obj = ratio * advantages
        obj_clipped = torch.clamp(ratio, 1.
