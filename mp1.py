import multiprocessing
if multiprocessing.get_start_method(allow_none=True) != 'spawn':
    multiprocessing.set_start_method('spawn')

import os
import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation
from gymnasium.spaces import MultiDiscrete, Box
import sys
sys.modules["gym"] = gym

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import  CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.env_checker import check_env, _check_spaces


CHECKPOINT_DIR = "SubprocVE_train"
LOG_DIR = "SubprocVE_logs"
TIMESTEPS = 1000
LEARNING_RATE = 0.000001
N_STEPS = 512
LOG_NAME = "PPO-" + str(LEARNING_RATE) + "-" + str(N_STEPS)
BEST_MODEL_PATH = "SubprocVE_best_model"


class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(ObservationWrapper, self).__init__(env)
        self.observation_space = Box(low=0, high=1, shape=(240, 256, 1), dtype=np.float32)

    def observation(self, observation):
        return observation / 255.0


class JoypadToMultiDiscrete(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = MultiDiscrete([2, 2, 2, 2])  # [right, left, A, B]

    def action(self, multi_discrete_action):
        # Convert MultiDiscrete action to SIMPLE_MOVEMENT action
        actions = []
        if multi_discrete_action[0] == 1:
            actions.append('right')
        if multi_discrete_action[1] == 1:
            actions.append('left')
        if multi_discrete_action[2] == 1:
            actions.append('A')
        if multi_discrete_action[3] == 1:
            actions.append('B')
        
        if actions in SIMPLE_MOVEMENT:
            return SIMPLE_MOVEMENT.index(actions)
        else:
            # Default to 'NOOP' if no matching action is found.
            return 0


def make_env(id, rank, seed=0):
    def _init():
        # env = StepAPICompatibility(gym_super_mario_bros.make(id, apply_api_compatibility=True, render_mode='human'))
        # env = gym_super_mario_bros.make('SuperMarioBros-v0', apply_api_compatibility=True)
        env = gym_super_mario_bros.make(id, apply_api_compatibility=True, render_mode='human')
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env = JoypadToMultiDiscrete(env)
        
        print(check_env(env, True))
        
        # if isinstance(env.observation_space, (gym.spaces.Box, gym.spaces.Dict)):
        #     print("isinstance")
        # print("OBS BEFORE WRAPPER: ", env.observation_space)
        env = ObservationWrapper(env)
        print(check_env(env, True))

        # if isinstance(env.observation_space, (gym.spaces.Box, gym.spaces.Dict)):
        #     print("isinstance")
        # print("OBS AFTER WRAPPER: ", env.observation_space)
    
        # env = VecFrameStack(env, 4, channels_order='last')
        JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)  
        # env = MaxAndSkipEnv(env, 4)
        print(check_env(env, True))
        env.reset(seed=seed+rank)
        print(check_env(env, True))
        return env
    
    set_random_seed(seed)
    return _init


os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_PATH, exist_ok=True)


if __name__ == "__main__":
    
    id = "SuperMarioBros-v0"
    num_processes = 4
    # env = VecMonitor(SubprocVecEnv([make_env(id, i) for i in range(num_processes)]), "SubprocVE_logs/TestMonitor")
    envs = VecFrameStack(VecMonitor(SubprocVecEnv([make_env(id, i) for i in range(num_processes)])))
    print(check_env(envs, True))

    checkpoint_callback = CheckpointCallback(save_freq=100, save_path=LOG_DIR) #https://araffin.github.io/post/sb3/

    model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=LEARNING_RATE, n_steps=N_STEPS)
    model.learn(total_timesteps=TIMESTEPS, tb_log_name=LOG_NAME, callback=[checkpoint_callback])
    model.save("Multiprocessing_model_0.000001_512")
