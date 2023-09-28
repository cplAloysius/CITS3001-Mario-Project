import os
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from matplotlib import pyplot as plt

# CALLBACK FUNCTION FROM: https://colab.research.google.com/github/araffin/rl-tutorial-jnrr19/blob/master/4_callbacks_hyperparameter_tuning.ipynb#scrollTo=nzMHj7r3h78m&line=1&uniqifier=1
class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq, log_dir, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model at {} timesteps".format(x[-1]))
                    print("Saving new best model to {}.zip".format(self.save_path))
                  self.model.save(self.save_path)

        return True


class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(
                self.save_path, 'model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True


def make_env(id, rank, seed=0):

    def _init():
        env = gym_super_mario_bros.make(id)
        # env = gym_super_mario_bros.make('SuperMarioBros-v0', apply_api_compatibility=True)
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        # env = GrayScaleObservation(env, keep_dim=True)
        env = MaxAndSkipEnv(env, 4)
        env.reset(seed=seed+rank )
        return env
    
    set_random_seed(seed)
    return _init


CHECKPOINT_DIR = "SubprocVE_train"
LOG_DIR = "SubprocVE_logs"
TIMESTEPS = 100000
LEARNING_RATE = 0.000001
N_STEPS = 512
LOG_NAME = "PPO-" + str(LEARNING_RATE) + "-" + str(N_STEPS)
BEST_MODEL_PATH = ""



os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


if __name__ == "__main__":
    
    id = "SuperMarioBros-v0"
    num_processes = 4
    env = VecMonitor(SubprocVecEnv([make_env(id, i) for i in range(num_processes)]), start_method="fork")
    callback = EvalCallback(env, )

    model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=LEARNING_RATE, n_steps=N_STEPS)
    model.learn(total_timesteps=TIMESTEPS, tb_log_name=LOG_NAME)








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
'''