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

# Model Definition
class MarioNet(nn.Module):
    # Initializing the model with the expected output dimension
    def __init__(self, output_dim):
        super(MarioNet, self).__init__()
        # Defining Convolutional Layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # Defining Fully Connected Layers
        self.fc1 = nn.Linear(64 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, output_dim)

    # Forward propagation
    def forward(self, x):
        x = F.relu(self.conv1(x))  # Applying convolution and ReLU activation
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flattening for fully connected layer
        x = F.relu(self.fc1(x))    # Applying FC layer and ReLU activation
        x = self.fc2(x)            # Final output layer
        return x.softmax(dim=-1)   # Applying softmax activation for action probs

# Preprocessing - Transformation to apply to the screen image
resize = T.Compose([T.ToPILImage(),
                    T.Resize((84, 84), interpolation=Image.CUBIC),
                    T.ToTensor()])

# Function to process the screen (frame from the environment)
def get_screen(frame):
    # Resizing, adding batch dimension, and moving channel to the correct place
    return resize(frame).unsqueeze(0).permute(0, 3, 1, 2)

# Function to save the model parameters
def save_model(model, filename):
    torch.save(model.state_dict(), filename)

# Worker Function
def worker(process_id, global_model, optimizer, lock, action_dim, episodes_per_save=10):
    # Local model is initiated and syncs with the global model
    local_model = MarioNet(action_dim)
    local_model.load_state_dict(global_model.state_dict())
    # Setting up the environment
    env = gym_super_mario_bros.make('SuperMarioBros-v3')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env.seed(process_id)  # Setting a seed based on process id for diversity
    
    # Training Loop
    for episode in range(episodes_per_save):
        state = env.reset()  # Getting initial state
        state = get_screen(state)  # Processing the screen image
        done = False  # Boolean flag indicating the end of an episode
        start_time = time.time()  # Time at the episode start
        episode_reward = 0  # Cumulative reward for the episode
        
        # Episode Loop
        while not done and time.time() - start_time < MAX_TIME_PER_EPISODE:
            action_prob = local_model(state)  # Getting action probabilities
            action = torch.argmax(action_prob).item()  # Choosing an action
            # Taking a step in the environment
            next_state, reward, done, info = env.step(action)
            next_state = get_screen(next_state)  # Processing next state
            
            # Calculating loss
            optimizer.zero_grad()
            loss = -torch.log(action_prob[0, action]) * reward  # Log prob * reward
            loss.backward()  # Backpropagation
            optimizer.step()  # Updating weights
            episode_reward += reward  # Updating cumulative reward
            state = next_state  # Moving to the next state
        
        print(f"Process {process_id}, Episode {episode}: Episode complete with reward {episode_reward}")
        
        # Locking to perform a safe update to global model
        lock.acquire()
        try:
            # Transferring local gradients to global model
            for global_param, local_param in zip(global_model.parameters(), local_model.parameters()):
                global_param._grad = local_param.grad
            optimizer.step()  # Updating the global model
        finally:
            lock.release()  # Releasing lock
        
        # Syncing local model with updated global model
        local_model.load_state_dict(global_model.state_dict())
        
        # Saving model
        if episode % episodes_per_save == 0:
            save_model(global_model, f"mario_model.pt")
            print(f"Process {process_id}: Model saved!")
    env.close()  # Closing the environment

# Hyperparameters
MAX_TIME_PER_EPISODE = 60

if __name__ == "__main__":
    mp.set_start_method('spawn')  # Ensuring safe multiprocessing
    action_dim = len(SIMPLE_MOVEMENT)  # Getting action space size
    global_model = MarioNet(action_dim).share_memory()  # Defining and sharing global model
    optimizer = optim.Adam(global_model.parameters(), lr=0.001)  # Defining optimizer
    lock = mp.Lock()  # Lock for safe updates to global model

    num_processes = mp.cpu_count()  # Number of available CPUs
    processes = []

    # Spawning worker processes
    for i in range(num_processes):
        p = mp.Process(target=worker, args=(i, global_model, optimizer, lock, action_dim))
        p.start()
        processes.append(p)

    # Waiting for all processes to finish
    for p in processes:
        p.join()