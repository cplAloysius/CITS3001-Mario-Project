import time
import numpy as np
import gym_super_mario_bros
import gymnasium as gym
import sys
sys.modules["gym"] = gym
from gymnasium.wrappers import StepAPICompatibility
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import  CheckpointCallback


def make_env(env_id, rank, seed=0):
    def _init():
        # env = gym.make(env_id)
        env = StepAPICompatibility(gym_super_mario_bros.make(id, apply_api_compatibility=True, render_mode='human'))
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)
        env.reset(seed=seed + rank)
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


env_id = "SuperMarioBros-v0"
# The different number of processes that will be used
PROCESSES_TO_TEST = [1, 2, 4, 8, 16]
NUM_EXPERIMENTS = 3  # RL algorithms can often be unstable, so we run several experiments (see https://arxiv.org/abs/1709.06560)
TRAIN_STEPS = 5000
# Number of episodes for evaluation
EVAL_EPS = 20
ALGO = PPO

# We will create one environment to evaluate the agent on
eval_env = gym.make(env_id)

reward_averages = []
reward_std = []
training_times = []
total_procs = 0
for n_procs in PROCESSES_TO_TEST:
    total_procs += n_procs
    print(f"Running for n_procs = {n_procs}")
    if n_procs == 1:
        # if there is only one process, there is no need to use multiprocessing
        train_env = DummyVecEnv([lambda: gym.make(env_id)])
    else:
        # Here we use the "fork" method for launching the processes, more information is available in the doc
        # This is equivalent to make_vec_env(env_id, n_envs=n_procs, vec_env_cls=SubprocVecEnv, vec_env_kwargs=dict(start_method='fork'))
        train_env = SubprocVecEnv(
            [make_env(env_id, i + total_procs) for i in range(n_procs)],
            start_method="fork",
        )

    rewards = []
    times = []

    for experiment in range(NUM_EXPERIMENTS):
        # it is recommended to run several experiments due to variability in results
        train_env.reset()
        model = PPO('CnnPolicy', train_env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=LEARNING_RATE, n_steps=N_STEPS)
        checkpoint_callback = CheckpointCallback(save_freq=100, save_path=LOG_DIR) #https://araffin.github.io/post/sb3/
        start = time.time()
        model.learn(total_timesteps=TRAIN_STEPS)
        model.save("Multiprocessing_model_0.000001_512")
        times.append(time.time() - start)
        mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=EVAL_EPS)
        rewards.append(mean_reward)
    # Important: when using subprocesses, don't forget to close them
    # otherwise, you may have memory issues when running a lot of experiments
    train_env.close()
    reward_averages.append(np.mean(rewards))
    reward_std.append(np.std(rewards))
    training_times.append(np.mean(times))

    '''
    https://stackoverflow.com/questions/73694119/how-to-register-custom-environment-with-openais-gym-package-to-use-make-vec-env
    https://github.com/DLR-RM/stable-baselines3/issues/993
    '''