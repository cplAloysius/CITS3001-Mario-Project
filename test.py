from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import gym
env = gym.make('SuperMarioBros-v0',
               apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, COMPLEX_MOVEMENT)
done = True
env.reset()
for step in range(5000):

    action = 3
    obs, reward, terminated, truncated, info = env.step(action)

    print(reward)
    print(info['x_pos'])
    done = terminated or truncated
    if done:
        state = env.reset()
env.close()
