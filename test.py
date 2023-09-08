import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

def main():
    env = gym_super_mario_bros.make(
        'SuperMarioBros-v0', apply_api_compatibility=True, render_mode='human')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = GrayScaleObservation(env, keep_dim=True)
    env = DummyVecEnv([lambda:env])
    env = VecFrameStack(env, 4, channels_order='last')

    JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

    model = PPO.load('train/model_1000000')

    state = env.reset()

    while True:
        action, _ = model.predict(state)
        state, reward, done, info = env.step(action)
        env.render()


if __name__ == '__main__':
    main()
