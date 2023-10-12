import csv
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv


def main():
    env = gym_super_mario_bros.make(
        'SuperMarioBros-8-4-v0', apply_api_compatibility=True, render_mode='human')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = GrayScaleObservation(env, keep_dim=True)
    env = DummyVecEnv([lambda:env])
    env = VecFrameStack(env, 4, channels_order='last')

    JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

    model = PPO.load('mario_model_1e6_512/mario_model_1e6_512_8mstep.zip', env=env)

    # time, actions, different levels
    
    rounds = 5
    world = 8
    stage = 4

    avg_time = 0
    avg_num_actions = 0
    avg_reward = 0
    avg_score = 0
    avg_x_pos = 0
    avg_flag_count = 0


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
                avg_time += info[0]["time"]
                avg_num_actions += actions
                avg_reward += total_reward
                avg_score += info[0]["score"]
                avg_x_pos += info[0]["x_pos"]
                avg_flag_count += flag_count
                break

            env.render()
    
    results = {
        "world" : world,
        "stage" : stage,
        "avg_time" : avg_time/rounds,
        "avg_num_actions" : avg_num_actions/rounds,
        "avg_reward" : avg_reward/rounds,
        "avg_score" : avg_score/rounds,
        "avg_x_pos" : avg_x_pos/rounds,
        "avg_flag_count" : avg_flag_count/rounds
    }


    print(f"--COMPLETE---")
    print(results)

    filename = f"compare_world_{world}_results.csv"

    with open(filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["world", "stage", "avg_time", "avg_num_actions", "avg_reward", "avg_score", "avg_x_pos", "avg_flag_count",])
        # writer.writeheader()
        writer.writerow(results)

if __name__ == '__main__':
    main()
