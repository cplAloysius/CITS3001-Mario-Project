'''
    follows https://lit.labml.ai/github/vpj/rl_samples/tree/master/ppo.py
'''

import multiprocessing
import multiprocessing.connection
from typing import Dict, List

import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.distributions import Categorical
from PIL import Image
import torch.multiprocessing as mp
# from labml import monit, tracker, logger, experiment

import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack


if torch.cuda.is_available():
    device = torch.device("cuda:1")
else:
    device = torch.device("cpu")

class Round:
    def __init__(self):
        self.env = gym_super_mario_bros.make('SuperMarioBros-v3', apply_api_compatibility=True, render_mode='human')
        self.env = JoypadSpace(self.env, SIMPLE_MOVEMENT)
        # self.env = GrayScaleObservation(self.env, keep_dim=True)
        # self.env = ResizeObservation(self.env, shape=(84, 84))
        self.obs_4 = np.zeros((4, 84, 84))
        self.rewards = []
        self.lives = 3

    def fourSteps(self, action):
        print("in fourSteps")
        totalRewards = 0
        done = False

        for i in range(4):
            obs, reward, terminated, truncated, info = self.env.step(action)
            print("reward:", str(reward))
            totalRewards += reward
            livesLeft = info[0]["life"]
            if livesLeft < self.lives or terminated or truncated:
                if livesLeft < self.lives:
                    print("livesLeft: ", livesLeft, "self.lives", self.lives)
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
        self.lives = 3

    def process_obs(self, obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
        return obs


def worker_process(remote: multiprocessing.connection.Connection):
    round = Round()

    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            remote.send(round.fourSteps(data))  # assuming you're using the fourSteps method in the Round class
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
    def __init__(self):
        super().__init__()



r = Round()
r.__init__()
r.reset()
action = r.env.action_space.sample()
obs, reward, done, info = r.fourSteps(action)