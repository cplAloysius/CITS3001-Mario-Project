import csv
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

    model = PPO.load('mario_model_1e6_512/mario_model_1e6_512_8mstep.zip', env=env)

    # time, actions, different levels
    
    flag_count = 0
    all_info = []
    world = 1
    stage = 1

    while flag_count < 2:
        state = env.reset()
        actions = 0
        total_reward = 0
        flag = False
        
        while True:

            action, _ = model.predict(state)
            state, reward, done, info = env.step(action)

            actions += 1
            total_reward += reward[0]
            flag = info[0]["flag_get"]

            if flag:
                print("GOT FLAG")
                flag_count += 1
                i = {
                    "time_left" : info[0]["time"],
                    "num_actions" : actions, 
                    "world" : world,
                    "stage" : stage,
                    "lives_left" : info[0]["life"],
                    "reward" : total_reward,
                    "score" : info[0]["score"]
                }
                all_info.append(i)
                break

            if done[0]:
                break

            env.render()
    
    print(f"--COMPLETE---")
    for d in all_info:
        print(d)


    filename = f"time_flag_results2.csv"

    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["time_left", "num_actions", "world", "stage", "lives_left", "reward", "score"])
        writer.writeheader()
        writer.writerows(all_info)


if __name__ == '__main__':
    main()
