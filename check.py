from typing import Sequence
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gymnasium
from gymnasium import spaces, ActionWrapper
import numpy as np
import numpy.typing as npt
from stable_baselines3.common.env_checker import check_env

class MarioWrapper(ActionWrapper):
    


gymnasium.register(
    id="SuperMarioBros-v0",
    entry_point='gym_super_mario_bros:SuperMarioBrosEnv',
    max_episode_steps=9999999,
    reward_threshold=9999999,
    nondeterministic=True,
)

env = gymnasium.make("SuperMarioBros-v0", apply_api_compatibility=True)
env.action_space = spaces.Discrete(7)
# env = gym_super_mario_bros.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
# env = JoypadSpace(env, SIMPLE_MOVEMENT)
print(check_env(env=env, warn=True))
# print(env.action_space.dtype)


'''
AssertionError: action space does not inherit from `gymnasium.spaces.Space`, actual type: <class 'gym.spaces.discrete.Discrete'>
'''