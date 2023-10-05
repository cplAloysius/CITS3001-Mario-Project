# from nes_py.wrappers import JoypadSpace
# import gym_super_mario_bros
# from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
# from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack


# env = gym_super_mario_bros.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode='human')
# env = JoypadSpace(env, SIMPLE_MOVEMENT)

# inital = env.reset()

# # obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
# # print("---BEFORE GREYSCALE---")
# # print("OBS TYPE: ", type(obs))
# # print("OBS TYPE: ", env.observation_space.dtype)
# # print("OBS SHAPE: ", env.observation_space.shape)

# # env = GrayScaleObservation(env, keep_dim=True)
# # env.reset()
# # obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
# # print("---AFTER GREYSCALE---")
# # print("OBS TYPE: ", type(obs))
# # print("OBS TYPE: ", env.observation_space.dtype)
# # print("OBS SHAPE: ", env.observation_space.shape)

# done = True
# for step in range(5):
#     if done:
#         state = env.reset()
#     state, reward, terminated, truncated, info = env.step(env.action_space.sample())
#     # state, reward, done, info = env.step(env.action_space.sample())
#     print(info)
#     env.render()

# env.close()

'''
python -m pip install labml
pip3 install torchrl  
python -m pip install git+https://github.com/pytorch/rl@v0.1.0
LABML WARNING: https://github.com/labmlai/labml/blob/master/guides/labml_yaml_file.md
'''
import labml
print(labml.__file__)
print(labml.__package__)

# /Users/oliviamorrison/anaconda3/envs/mario/lib/python3.8/site-packages/labml/__init__.py