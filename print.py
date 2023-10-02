from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

done = True
for step in range(5):
    if done:
        state = env.reset()
    # state, reward, terminated, truncated, info = env.step(env.action_space.sample())
    state, reward, done, info = env.step(env.action_space.sample())
    print(info)
    env.render()

env.close()