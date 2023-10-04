from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack


env = gym_super_mario_bros.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode='human')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
print("---BEFORE GREYSCALE---")
print("OBS TYPE: ", type(obs))
print("OBS TYPE: ", env.observation_space.dtype)
print("OBS SHAPE: ", env.observation_space.shape)

env = GrayScaleObservation(env, keep_dim=True)
env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
print("---AFTER GREYSCALE---")
print("OBS TYPE: ", type(obs))
print("OBS TYPE: ", env.observation_space.dtype)
print("OBS SHAPE: ", env.observation_space.shape)

done = True
print("breakpoint didn't work")
for step in range(5):
    if done:
        state = env.reset()
    # state, reward, terminated, truncated, info = env.step(env.action_space.sample())
    state, reward, done, info = env.step(env.action_space.sample())
    print(info)
    env.render()

env.close()