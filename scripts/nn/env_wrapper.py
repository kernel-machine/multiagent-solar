import gymnasium as gym
from gymnasium.spaces import Box, Discrete, Dict
import numpy as np

class EnvWrapper(gym.Env):
    def __init__(self, env, agent_id):
        self.env = env
        
        self.observation_space = env._observation_spaces[agent_id]
        
        # mixed radix representation !!!
        self.action_space = Discrete((env._processing_rate + 1) * 3 * env._num_agents * (env._processing_rate + 1))
        
        
    def reset(self):
        pass
    
    def step(self, actions):
        pass
    
    def render(self):
        pass
    
    def close(self):
        pass