import os
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from matplotlib import pyplot as plt


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


def main():
    env = gym_super_mario_bros.make(
        'SuperMarioBros-v0', apply_api_compatibility=True)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = GrayScaleObservation(env, keep_dim=True)
    env = DummyVecEnv([lambda:env])
    env = VecFrameStack(env, 4, channels_order='last')

    # to fix the seed error
    JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

    # state = env.reset()
    # state, reward, done, info = env.step([5])

    # plt.figure(figsize=(10, 10))
    # for idx in range(state.shape[3]):
    #     plt.subplot(1, 4, idx+1)
    #     plt.imshow(state[0][:, :, idx])
    # plt.show()

    CHECKPOINT_DIR = 'train'
    LOG_DIR = 'logs'

    callback = TrainAndLoggingCallback(
        check_freq=100000, save_path=CHECKPOINT_DIR)

    #train from scratch
    model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR,
                learning_rate=0.000001, n_steps=512)
    
    #load a previously trained model for further training
    # MODEL_DIR = 'train/model_1000000'
    # MODEL_LOG_DIR = 'logs/PPO_2'
    # model = PPO.load(MODEL_DIR, tensorboard_log=MODEL_LOG_DIR)
    # model.set_env(env)

    # Train the agent
    model.learn(total_timesteps=1000000, callback=callback)

    # model.save('mario_model2')


if __name__ == '__main__':
    main()
