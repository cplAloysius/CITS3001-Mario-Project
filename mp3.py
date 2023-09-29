'''https://www.gymlibrary.dev/content/vectorising/#custom-spaces'''
import os
import numpy as np
import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
from gym.vector import AsyncVectorEnv
from stable_baselines3.common.callbacks import  CheckpointCallback


NUM_ENVS = 4
ENV_ID = "SuperMarioBros-v0"
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


class CustomDiscrete(gym.Space):
    def __init__(self, n):
        super().__init__(shape=(), dtype=np.int)
        self.n = n

    def sample(self):
        return self.np_random.randint(self.n)

    def contains(self, x):
        return 0 <= x < self.n


class JoypadToCustomDiscrete(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = CustomDiscrete(JoypadSpace.action_space)

    def action(self, action):
        # Convert the CustomDiscrete action into a JoypadSpace action
        return JoypadSpace.action_space
    

def make_env(env_id):
    def _init():
        env = gym_super_mario_bros.make(env_id, apply_api_compatibility=True)
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env = JoypadToCustomDiscrete(env)
        JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)
        return env
    return _init


if __name__ == "__main__":
    env = AsyncVectorEnv([make_env(ENV_ID) for i in range(NUM_ENVS)], shared_memory=False)
    print(env.is_vector_env) #True
    # checkpoint_callback = CheckpointCallback(save_freq=100, save_path=LOG_DIR) #https://araffin.github.io/post/sb3/

    # model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=LEARNING_RATE, n_steps=N_STEPS)
    # model.learn(total_timesteps=TIMESTEPS, tb_log_name=LOG_NAME, callback=[checkpoint_callback])
    # model.save("Multiprocessing_model_0.000001_512")

    env.close()

