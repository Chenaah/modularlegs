

import pdb
import time
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from modular_legs.envs.gym.rendering import RecordVideo
import gymnasium as gym

class VecReal(VecEnv):

    def __init__(self, env, max_episode_steps=None):

        self.env = env

        try:
            self.num_envs = env.num_envs
        except AttributeError:
            # TODO: Handle the case when env does not have num_envs attribute
            self.num_envs = env.env.env.num_envs
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.max_episode_steps = max_episode_steps

        self.t = 0
        self.t_start = time.time()

    def reset(self):
        self.t = 0
        self.rewards = []
        observation, _ = self.env.reset()
        return observation

    def step(self, action):
        # Perform the action in the environment
        observation, reward, done, t, information = self.env.step(action)
        # print("MOTOR:")
        # print(observation[:,-3])

        self.t += 1
        # Check if the episode is done
        if self.max_episode_steps is not None and self.t >= self.max_episode_steps:
            done[:] = True
            self.t = 0

        self.rewards.append(reward)
        if np.any(done):
            for i in range(self.num_envs):
                ep_rew = np.sum(np.array(self.rewards)[:,i])
                ep_len = len(np.array(self.rewards)[:,i])
                information[i]["episode"] = {"r": round(ep_rew, 6), "l": ep_len, "t": round(time.time() - self.t_start, 6)}

            observation = self.reset() # Automatically reset the environment according to the Stable Baselines3 API

        return observation, reward, done, information
    
    def step_async(self, actions):
        # Not used in this implementation
        pass

    def step_wait(self):
        # Not used in this implementation
        pass

    def close(self):
        # Close the environment
        self.env.close()

    # def render(self, mode='human'):
    #     # Render the environment
    #     return self.env.render(mode=mode)
    def env_is_wrapped(self, wrapper_class):
        # Check if the environment is wrapped with a specific wrapper class
        return isinstance(self.env, wrapper_class)
    
    def env_method(self, method_name, *args, **kwargs):
        # Call a method on the environment
        return getattr(self.env, method_name)(*args, **kwargs)
    
    def get_attr(self, attr_name, indices = None):
        return super().get_attr(attr_name, indices)
    
    def set_attr(self, attr_name, value, indices = None):
        return super().set_attr(attr_name, value, indices)
