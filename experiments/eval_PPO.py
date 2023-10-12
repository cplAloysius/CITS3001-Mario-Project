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

    model = PPO.load('train/best_model_1000000 (1).zip', env=env)

    # time, actions, different levels
    
    rewards = []
    num_rounds = 2

    for round in range(1, num_rounds+1):
        state = env.reset()
        steps = 0
        total_rewards = 0
        flag = False
        x_pos_end = 0
        while True:

            action, _ = model.predict(state)
            state, reward, done, info = env.step(action)

            steps += 1
            total_rewards += reward
            flag = info[0]["flag_get"]
            if flag:
                print("GOT FLAG")

            # if reward == -15 or done[0] or flag:
            if done[0] or flag: 
                avg_reward = total_rewards/steps
                r = {
                    "round" : round,
                    "steps" : steps,
                    "total_rewards" : total_rewards[0],
                    "avg_reward_per_step" : avg_reward[0],  
                    "flag" : flag,
                    "time" : info[0]["time"],
                    "x_pos" : info[0]["x_pos"],
                }
                rewards.append(r)
                break

            env.render()
    
    print(f"--{num_rounds} ROUNDS COMPLETE---")
    for d in rewards:
        print(d)

    lr = "1e6"
    model_steps = "1m"

    filename = f"{lr}_{model_steps}_results.csv"

    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["round", "steps", "total_rewards", "avg_reward_per_step", "flag", "time", "x_pos"])
        writer.writeheader()
        writer.writerows(rewards)


if __name__ == '__main__':
    main()
