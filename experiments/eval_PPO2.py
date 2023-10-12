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

    model = PPO.load("1e4_models/1e4/1e4_12m.zip", env=env)

    lr = "1e-4"
    model_steps = "12m"
    num_rounds = 20

    highest_reward = 0
    avg_reward = 0
    highest_score = 0
    avg_score = 0
    flag_count = 0

    for round in range(1, num_rounds+1):
        state = env.reset()
        total_round_rewards = 0
        flag = False
        # highest_x_pos = 0

        while True:

            action, _ = model.predict(state)
            state, reward, done, info = env.step(action)

            total_round_rewards += reward[0]
            flag = info[0]["flag_get"]
            if flag:
                flag_count += 1
                print("GOT FLAG")
            
            # if info[0]["x_pos"] > highest_x_pos:
            #     highest_x_pos = info[0]["x_pos"]

            if done[0]: 

                avg_reward += total_round_rewards
                if total_round_rewards > highest_reward:
                    print("new highest reward: ", total_round_rewards)
                    highest_reward = total_round_rewards

                avg_score += info[0]["score"]
                if info[0]["score"] > highest_score:
                    print("new highest score: ", info[0]["score"])
                    highest_score = info[0]["score"]

                break

            env.render()
    
    avg_reward = avg_reward/num_rounds
    avg_score = avg_score/num_rounds

    results = {
        "learning_rate" : lr,
        "n_steps" : model_steps, 
        "highest_reward" : highest_reward,
        "avg_reward" : avg_reward,
        "highest_score" : highest_score,
        "avg_score" : avg_score,
        "total_flags" : flag_count
    }

    print(f"--{num_rounds} ROUNDS COMPLETE---")
    print(results)

    filename = f"{lr}_PPO_results.csv"

    with open(filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["learning_rate", "n_steps", "highest_reward", "avg_reward", "highest_score", "avg_score", "total_flags"])
        # writer.writeheader()
        writer.writerow(results)

    print("---COMPLETE---")


if __name__ == '__main__':
    main()


'''
models = [
        "1e4_models/1e4/1e4_1m.zip", 
        "1e4_models/1e4/1e4_4m.zip",
        "1e4_models/1e4/1e4_8m.zip",
        "1e4_models/1e4/1e4_12m.zip",
        "1e4_models/1e4/1e4_16m.zip",
        "train/best_model_1000000 (1).zip",
        "mario_model_1e6_512/mario_model_1e6_512_4mstep.zip",
        "mario_model_1e6_512/mario_model_1e6_512_8mstep.zip",
        "mario_model_1e6_512/mario_model_1e6_512_12mstep.zip",
        "mario_model_1e6_512/mario_model_1e6_512_16mstep.zip"
    ]
'''