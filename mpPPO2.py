'''
    follows https://lit.labml.ai/github/vpj/rl_samples/tree/master/ppo.py
    might need to account for other vars from the info dictionary such as "flag"
'''
import multiprocessing
import multiprocessing.connection
from typing import Dict, List

import labml
from labml import lab
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.distributions import Categorical
from torchrl.objectives import PPOLoss, ClipPPOLoss
from torchrl.objectives.value.functional import generalized_advantage_estimate
from torchrl.envs.libs.gym import _get_envs                                                         #https://pytorch.org/rl/tutorials/torchrl_envs.html
from labml.utils.pytorch import get_modules

from PIL import Image
import torch.multiprocessing as mp
from labml import monit, tracker, logger, experiment

# import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
# from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack

import warnings

warnings.filterwarnings("ignore")

config_dict = {
    'check_repo_dirty': False,
    'data_path': 'data',
    'experiments_path': 'logs',
    'analytics_path': 'analytics',
    'web_api': None,
    'web_api_frequency': None,
    'web_api_verify_connection': False,
    'web_api_open_browser': False,
    'indicators': [
        {
            'class_name': 'Scalar',
            'is_print': True,
            'name': '*'
        }
    ]
}

lab.configure(configurations=config_dict)

if torch.cuda.is_available():
    device = torch.device("cuda:1")
else:
    device = torch.device("cpu")

LEARNING_RATE = 0.000001

class Round:
    def __init__(self):
        self.env = gym_super_mario_bros.make('SuperMarioBros-v3', apply_api_compatibility=True, render_mode='human')
        self.env = JoypadSpace(self.env, SIMPLE_MOVEMENT)
        # self.env = GrayScaleObservation(self.env, keep_dim=True)
        # self.env = ResizeObservation(self.env, shape=(84, 84))
        self.obs_4 = np.zeros((4, 84, 84))
        self.rewards = []
        self.lives = 2 # Should be 3 but 2 seems to be the default?

    def fourSteps(self, action):
        totalRewards = 0
        done = False

        for i in range(4):
            obs, reward, terminated, truncated, info = self.env.step(action)
            totalRewards += reward
            livesLeft = info["life"]
            if livesLeft < self.lives or terminated or truncated:
                done = True
                break

        obs = self.process_obs(obs)
        self.rewards.append(reward)

        if done:
            episode_info =  {"reward": sum(self.rewards), "length": len(self.rewards)}
            self.reset()
        else: 
            episode_info = None

        self.obs_4 = np.roll(self.obs_4, shift=-1, axis=0)
        self.obs_4[-1] = obs
        
        return self.obs_4, reward, done, episode_info
    
    def reset(self):
        obs, x = self.env.reset()
        obs = self.process_obs(obs)
        for i in range(4): 
            self.obs_4[i] = obs
        self.rewards = []
        self.lives = 2
        return self.obs_4

    def process_obs(self, obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
        return obs


def worker_process(remote: multiprocessing.connection.Connection):
    round = Round()

    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            remote.send(round.fourSteps(data))  
        elif cmd == "reset":
            remote.send(round.reset())
        elif cmd == "close":
            remote.close()
            break
        else:
            raise NotImplementedError

class Worker:
    def __init__(self):
        self.child, parent = multiprocessing.Pipe()
        self.process = multiprocessing.Process(target=worker_process, args=(parent,))
        self.process.start()

class MarioModel(nn.Module):
    '''
    Outputs the parameters for a policy distribution and a value estimate,
    Policy π: Used to select actions while interacting with the environment. Actions are typically sampled from π, driving exploration and exploitation.
    Value Function: Used to estimate future returns and thereby helps in making more informed policy updates by understanding the value of different states.
    '''
    def __init__(self): # define layers - maybe try torchrl.modules.ConvNet
        super().__init__()

        # convolutional layers to process input images
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4) # produces a 20x20 frame from 84x84 frame
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2) # produces a 9x9 frame from 20x20 frame
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1) # produces a 7x7 frame from 9x9 frame


        self.lin = nn.Linear(in_features=7 * 7 * 64, out_features=512) # takes flattened rame from conv3 and outputs 512 features
        self.pi_logits = nn.Linear(in_features=512, out_features=7) # a fully connected layer to get logits for pi                      <- check if aligns with action space
        self.value = nn.Linear(in_features=512, out_features=1) # estimates expected future reward from the current state


    def forward(self, obs: torch.Tensor): # apply layers - defines how input data from obs is processed through layers to produce outputs

        h = F.relu(self.conv1(obs)) # pass obs thru the conv layers and apply relu activation function to introduce non-linearities
        h = F.relu(self.conv2(h))  
        h = F.relu(self.conv3(h))

        h = h.reshape((-1, 7 * 7 * 64)) # flatten tensor to make it compatible for processing 

        h = F.relu(self.lin(h)) # learn and extract features from data

        pi = Categorical(logits=self.pi_logits(h)) # derive policy to dictate the probability of selecting various actions
        value = self.value(h).reshape(-1) # expected return value

        return pi, value
        
def obs_to_torch(obs):
    # note: torch.from_numpy shares the same memory but torch.tensor creates a copy
    # o = torch.from_numpy(obs)
    output = torch.tensor(obs, dtype=torch.float32, device=device) / 255
    return output

class Main:
    def __init__(self): 
        # maybe change to match stable baselines trained model to compare
        self.gamma = 0.99
        self.lamda = 0.95
        self.updates = 10000
        self.epochs = 4
        self.learning_rate = 2.5e-4
        self.n_workers = 4 # changed from 8
        self.worker_steps = 128
        self.n_mini_batch = 4
        self.batch_size = self.n_workers * self.worker_steps
        self.mini_batch_size = self.batch_size // self.n_mini_batch
        assert (self.batch_size % self.n_mini_batch == 0)

        self.workers = [Worker() for i in range(4)]

        # initialise tensors for observations
        self.obs = np.zeros((self.n_workers, 4, 84, 84), dtype=np.uint8) # {self.obs.shape} is (4, 4, 84, 84)

        for worker in self.workers:
            worker.child.send(("reset", None))
        for i, worker in enumerate(self.workers):
            self.obs[i] = worker.child.recv()

        self.model = MarioModel().to(device) # laptop = CPU, PC = GPU

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate) # updates parameters of model during training

    def sample(self) -> (Dict[str, np.ndarray], List):
        # store data for every step taken by each worker in parallel
        rewards = np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)
        actions = np.zeros((self.n_workers, self.worker_steps), dtype=np.int32)
        done = np.zeros((self.n_workers, self.worker_steps), dtype=bool)
        obs = np.zeros((self.n_workers, self.worker_steps, 4, 84, 84), dtype=np.uint8)
        log_pis = np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)
        values = np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)


        # for each step use the model to decide the next action based on the current observation
        # perform the action and record the result
        for t in range(self.worker_steps):
            with torch.no_grad(): # don't compute gradients 
                obs[:, t] = self.obs # tracks observation from each worker for the model to sample

                # sample actions and store data
                pi, v = self.model(obs_to_torch(self.obs))
                values[:, t] = v.cpu().numpy()
                a = pi.sample()
                actions[:, t] = a.cpu().numpy()
                log_pis[:, t] = pi.log_prob(a).cpu().numpy()

            # perform sampled actions on each worker
            for w, worker in enumerate(self.workers):
                worker.child.send(("step", actions[w, t]))
            for w, worker in enumerate(self.workers):
                self.obs[w], rewards[w, t], done[w, t], info = worker.child.recv()
                if info is not None:
                    tracker.add('reward', info['reward'])
                    print("info not none")
                else:
                    print("info is none")                                                                           

            advantages = self._calc_advantages(done, rewards, values)
            samples = {
                'obs': obs,
                'actions': actions,
                'values': values,
                'log_pis': log_pis,
                'advantages': advantages
            }

            samples_flat = {}
            for k, v in samples.items():
                v = v.reshape(v.shape[0] * v.shape[1], *v.shape[2:])
                if k == 'obs':
                    samples_flat[k] = obs_to_torch(v)
                else:
                    samples_flat[k] = torch.tensor(v, device=device)

            return samples_flat
            
    def _calc_advantages(self, done: np.ndarray, rewards: np.ndarray, values: np.ndarray) -> np.ndarray:
        advantages = np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)
        last_advantage = 0

        _, last_value = self.model(obs_to_torch(self.obs))
        last_value = last_value.cpu().data.numpy()

        for t in reversed(range(self.worker_steps)):
            mask = 1.0 - done[:, t]
            last_value = last_value * mask
            last_advantage = last_advantage * mask
            delta = rewards[:, t] + self.gamma * last_value - values[:, t]

            last_advantage = delta + self.gamma * self.lamda * last_advantage
            advantages[:, t] = last_advantage

            last_value = values[:, t]

        return advantages
    
    def train(self, samples: Dict[str, torch.Tensor], learning_rate: float, clip_range: float):
        
        for i in range(self.epochs):
            # trains with a small random subset of the data
            indexes = torch.randperm(self.batch_size)
            for start in range(0, self.batch_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                mini_batch_indexes = indexes[start: end]
                mini_batch = {}
                for k, v in samples.items():
                    mini_batch[k] = v[mini_batch_indexes]
                
                # get loss - how difference between model's predictions and desired outcomes
                loss = self._calc_loss(clip_range=clip_range, samples=mini_batch)

                # calculate gradients - rate of change of loss with respect to model parameters
                for pg in self.optimizer.param_groups:
                    pg['lr'] = learning_rate
                self.optimizer.zero_grad()
                loss.backward()

                # clip gradient
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                # update model parameters
                self.optimizer.step()
                print("train")

    @staticmethod
    def _normalize(adv: torch.Tensor):
        # adv is a tensor containing advantage values
        # used to manage scale of advantages and stabilise training
        return (adv - adv.mean()) / (adv.std() + 1e-8)
    
    def _calc_loss(self, samples: Dict[str, torch.Tensor], clip_range: float) -> torch.Tensor: #returns a tensor representing loss

        # calculate expected reward approximated by the sum of the est. value and advantage
        sampled_return = samples['values'] + samples['advantages']

        # normalise to stabilise
        # sampled_normalized_advantage = self._normalize(samples['advantages'])
        sampled_normalized_advantage = self._normalize(samples['advantages'])

        # pass obs thru model to get policy and value function
        # policy (pi) defines probability distribution over actions
        # value function estimates expected future rewards
        pi, value = self.model(samples['obs'])

        '''POLICY LOSS'''
        # compute probabilities of taking actions under the policy
        log_pi = pi.log_prob(samples['actions'])

        # ratio - how much the policy has changed since the data was sampled in terms of the probability of taking certain actions in given states
        # ratio - difference between the new policy's log probability and the old policy's log probability
        ratio = torch.exp(log_pi - samples['log_pis'])

        # ensures that the policy does not change too drastically by clipping the ratio
        # clip range defines how much change is allowed
        clipped_ratio = ratio.clamp(min=1.0 - clip_range, max=1.0 + clip_range)

        # takes minimum of the OG policy gradient update and 
        #  the clipped policy gradient update to compute policy reward
        policy_reward = torch.min(ratio * sampled_normalized_advantage,
                                clipped_ratio * sampled_normalized_advantage)
        policy_reward = policy_reward.mean()


        '''ENTROPY BONUS'''
        # encourages exploration by adding a bonus for more random policies
        entropy_bonus = pi.entropy()
        # calculated as the entropy of the policy and averaged over all data points
        entropy_bonus = entropy_bonus.mean()


        '''VALUE LOSS'''
        # make value estimate close to estimated return
        # clipped to avoid large updates
        clipped_value = samples['values'] + (value - samples['values']).clamp(min=-clip_range, max=clip_range)
        vf_loss = torch.max((value - sampled_return) ** 2, (clipped_value - sampled_return) ** 2)
        vf_loss = 0.5 * vf_loss.mean()


        '''TOTAL LOSS'''
        # scale down value loss and entropy bonus so they don't dominate policy reward
        loss = -(policy_reward - 0.5 * vf_loss + 0.01 * entropy_bonus)

        # approximate KL divergence and the fraction of the policy ratio that is clipped - for training
        approx_kl_divergence = .5 * ((samples['log_pis'] - log_pi) ** 2).mean()
        clip_fraction = (abs((ratio - 1.0)) > clip_range).to(torch.float).mean()

        tracker.add({'policy_reward': policy_reward, 'vf_loss': vf_loss, 'entropy_bonus': entropy_bonus, 'kl_div': approx_kl_divergence, 'clip_fraction': clip_fraction})

        return loss
    
    def run_training_loop(self):
        """
        ### Run training loop
        """

        # last 100 episode information
        tracker.set_queue('reward', 100, True)
        tracker.set_queue('length', 100, True)

        for update in monit.loop(self.updates):
            progress = update / self.updates

            # decreasing `learning_rate` and `clip_range` $\epsilon$
            learning_rate = 2.5e-4 * (1 - progress)
            clip_range = 0.1 * (1 - progress)

            # sample with current policy
            samples = self.sample()

            # train the model
            self.train(samples, learning_rate, clip_range)

            # write summary info to the writer, and log to the screen
            tracker.save()
            if (update + 1) % 1_000 == 0:
                logger.log()

    def destroy(self):
        """
        ### Destroy
        Stop the workers
        """
        for worker in self.workers:
            worker.child.send(("close", None))

# ## Run it
if __name__ == "__main__":
    experiment.create()
    m = Main()
    experiment.start()
    m.run_training_loop()
    m.destroy()





