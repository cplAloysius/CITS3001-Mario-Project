import csv
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv


def main():
    env = gym_super_mario_bros.make(
        'SuperMarioBros-4-1-v0', apply_api_compatibility=True, render_mode='human')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = GrayScaleObservation(env, keep_dim=True)
    env = DummyVecEnv([lambda:env])
    env = VecFrameStack(env, 4, channels_order='last')

    JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

    model = PPO.load('mario_model_1e6_512/mario_model_1e6_512_8mstep.zip', env=env)

    # time, actions, different levels
    
    rounds = 5
    all_info = []
    world = 4
    stage = 1


    for r in range(1, rounds+1):
        state = env.reset()
        actions = 0
        total_reward = 0
        flag = False
        flag_count = 0
        
        while True:

            action, _ = model.predict(state)
            state, reward, done, info = env.step(action)

            actions += 1
            total_reward += reward[0]
            flag = info[0]["flag_get"]
            if flag:
                flag_count += 1
                print("GOT FLAG")

            if done[0]:
                i = {
                    "time_left" : info[0]["time"],
                    "num_actions" : actions, 
                    "world" : world,
                    "stage" : stage,
                    "reward" : total_reward,
                    "score" : info[0]["score"],
                    "x_pos" : info[0]["x_pos"]
                }
                all_info.append(i)
                break

            env.render()
    
    print(f"--COMPLETE---")
    for d in all_info:
        print(d)


    filename = f"compare_world_{world}_results.csv"

    with open(filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["time_left", "num_actions", "world", "stage", "reward", "score", "x_pos"])
        # writer.writeheader()
        writer.writerows(all_info)


if __name__ == '__main__':
    main()
