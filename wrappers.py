class VectorizedEnvWrapper(gym.Wrapper):
    def __init__(self, make_env, num_envs=1):
        super().__init__(make_env())
        self.num_envs = num_envs
        self.envs = [make_env() for env_index in range(num_envs)]
    
    def reset(self):
        return np.asarray([env.reset() for env in self.envs])
    
    def reset_at(self, env_index):
        return self.envs[env_index].reset()
    
    def step(self, actions):
        next_states, rewards, dones, infos = [], [], [], []
        for env, action in zip(self.envs, actions):
            next_state, reward, done, info = env.step(action)
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        return np.asarray(next_states), np.asarray(rewards), \
            np.asarray(dones), np.asarray(infos)
    

    https://alexandervandekleut.github.io/gym-wrappers/c


# class DiscretizedActionWrapper(ActionWrapper):
#     """ Discretizes the action space of an `env` using
#         `transform.discretize()`.
#         The `reverse_action` method is currently not implemented.
#     """
#     def __init__(self, env, steps):
#         super(DiscretizedActionWrapper, self).__init__(env)
#         trafo = discretize(env.action_space, steps)
#         self.action_space = trafo.target
#         self.action = trafo.convert_from

# class MarioActionWrapper(gym.ActionWrapper):
#     def __init__(self, env):
#         super(MarioActionWrapper, self).__init__(env)
#         # Change the action space to Discrete which is supported
#         self.action_space = gym.spaces.Discrete(len(env.unwrapped.get_action_meanings()))

#     def action(self, action):
#         # Convert the discrete action back to the original JoypadSpace action
#         return self.env.unwrapped._action_map[action]