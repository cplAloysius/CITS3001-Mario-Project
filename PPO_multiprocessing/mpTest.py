# import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import multiprocessing
import time
import torch

# Maximum time per episode in seconds
MAX_TIME_PER_EPISODE = 60  # for instance, one minute

def worker(process_id):
    env = gym_super_mario_bros.make(
        'SuperMarioBros-v3', apply_api_compatibility=True, render_mode='human')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    # env = GrayScaleObservation(env, keep_dim=True)
    # env = DummyVecEnv([lambda:env])
    # env.seed(process_id)
    JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

   
    state = env.reset()
    done = False
   
    # Get the start time of the episode
    start_time = time.time()
   
    while not done:
        # Check if the max time has been exceeded
        # if time.time() - start_time > MAX_TIME_PER_EPISODE:
        #     print(f"Process {process_id}: Episode terminated due to reaching max time.")


        #     break  # End the loop and thus end the episode
       
        # Choose a random action (replace this with your model's policy in practice)
        action = env.action_space.sample()
        output = env.step(action)
       
        # Step in the environment
        next_state, reward, terminated, truncated, info = env.step(action)
        # obs, reward, terminated, info = env.step(action)
        state = next_state
   
    # Close the environment when done
    env.close()

if __name__ == "__main__":
    num_processes = multiprocessing.cpu_count()
    # num_processes = 12
   
    processes = []
    for i in range(num_processes):
        p = multiprocessing.Process(target=worker, args=(i,))
        p.start()
        processes.append(p)
   
    for p in processes:
        p.join()