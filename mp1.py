import os
import gymnasium as gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gymnasium.wrappers import StepAPICompatibility
import sys
sys.modules["gym"] = gym


import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import  CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from matplotlib import pyplot as plt

class DiscretizedActionWrapper(ActionWrapper):
    """ Discretizes the action space of an `env` using
        `transform.discretize()`.
        The `reverse_action` method is currently not implemented.
    """
    def __init__(self, env, steps):
        super(DiscretizedActionWrapper, self).__init__(env)
        trafo = discretize(env.action_space, steps)
        self.action_space = trafo.target
        self.action = trafo.convert_from

class MarioActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(MarioActionWrapper, self).__init__(env)
        # Change the action space to Discrete which is supported
        self.action_space = gym.spaces.Discrete(len(env.unwrapped.get_action_meanings()))

    def action(self, action):
        # Convert the discrete action back to the original JoypadSpace action
        return self.env.unwrapped._action_map[action]


def make_env(id, rank, seed=0):

    def _init():

        # env = StepAPICompatibility(gym_super_mario_bros.make(id, apply_api_compatibility=True, render_mode='human'))
        env = gym_super_mario_bros.make(id, apply_api_compatibility=True, render_mode='human')
        # env = gym_super_mario_bros.make('SuperMarioBros-v0', apply_api_compatibility=True)
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)
        env = MarioActionWrapper(env)
        # env = ActionConverterWrapper(env)
        # env = GrayScaleObservation(env, keep_dim=True)
        # env = gym.make(id)    
        env = MaxAndSkipEnv(env, 4)
        env.reset(seed=seed+rank)
        return env
    
    set_random_seed(seed)
    return _init


CHECKPOINT_DIR = "SubprocVE_train"
LOG_DIR = "SubprocVE_logs"
TIMESTEPS = 1000
LEARNING_RATE = 0.000001
N_STEPS = 512
LOG_NAME = "PPO-" + str(LEARNING_RATE) + "-" + str(N_STEPS)
BEST_MODEL_PATH = "SubprocVE_best_model"


os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_PATH, exist_ok=True)


if __name__ == "__main__":
    
    id = "SuperMarioBros-v0"
    num_processes = 4
    for i in range(num_processes):
        env = VecMonitor(SubprocVecEnv([make_env(id, i)]))
        env.close()
    # env = VecMonitor(SubprocVecEnv([make_env(id, i) for i in range(num_processes)]))
    checkpoint_callback = CheckpointCallback(save_freq=100, save_path=LOG_DIR) #https://araffin.github.io/post/sb3/

    model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=LEARNING_RATE, n_steps=N_STEPS)
    model.learn(total_timesteps=TIMESTEPS, tb_log_name=LOG_NAME, callback=[checkpoint_callback])
    model.save("Multiprocessing_model_0.000001_512")







'''
multiprocessing refs: 
https://colab.research.google.com/github/araffin/rl-tutorial-jnrr19/blob/sb3/3_multiprocessing.ipynb#scrollTo=AvO5BGrVv2Rk
https://subscription.packtpub.com/book/programming/9781839210686/16/ch16lvl1sec38/vectorized-environments
https://www.youtube.com/watch?v=PxoG0A2QoFs&t=778s&ab_channel=ClarityCoders


callbacks:
https://colab.research.google.com/github/araffin/rl-tutorial-jnrr19/blob/master/4_callbacks_hyperparameter_tuning.ipynb
https://www.youtube.com/watch?v=dLP-2Y6yu70&t=12s&ab_channel=sentdex
https://www.youtube.com/watch?v=dLP-2Y6yu70&t=12s&ab_channel=sentdex
    16:26

CALLBACK FUNCTION FROM: https://colab.research.google.com/github/araffin/rl-tutorial-jnrr19/blob/master/4_callbacks_hyperparameter_tuning.ipynb#scrollTo=nzMHj7r3h78m&line=1&uniqifier=1


https://grid2op.readthedocs.io/en/latest/gym.html#default-action-space

https://www.gymlibrary.dev/content/vectorising/

https://pytorch.org/rl/tutorials/multiagent_ppo.html
'''