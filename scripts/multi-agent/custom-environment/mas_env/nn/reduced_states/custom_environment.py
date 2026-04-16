import os
import sys

from pettingzoo import ParallelEnv
from gymnasium import spaces
import matplotlib.pyplot as plt
from copy import copy

import functools
import numpy as np
import random
import math

import interpol as ip
import jit_reward_function as jrf
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from base_envirionment import BaseEnvironment
class CustomEnvironment(BaseEnvironment):

    def __init__(self, num_agents, irradiance_datapaths, delta_time, proc_interval, proc_rate, arr_rate, batteries, panel_surfaces, power_idle, power_max, w, seed):
        super().__init__(num_agents, irradiance_datapaths, delta_time, proc_interval, proc_rate, arr_rate, batteries, panel_surfaces, power_idle, power_max, w, seed)

        self._action_spaces = { #(Local framerate, Offloading mode [No, Send, Receive], Offloading target, Offloading framerate)])
            agent: spaces.MultiDiscrete([self._processing_rate + 1, 3, self._num_agents, self._processing_rate + 1]) for agent in self.possible_agents
        }
        
        self._observation_spaces = {
            # (Battery level, Backlog level, Timestep, Average battery level of other agents, Average backlog level of other agents)
            # Backlog level is discretized in 4 levels: 0 (empty), 1 (low), 2 (medium), 3 (high)
            agent: spaces.Box(
                low=np.array([0.0, 0, 0.0, 0.0, 0], dtype=np.float32),
                high=np.array([1.0, 3, 1.0, 1.0, 3], dtype=np.float32),
                dtype=np.float32
            ) 
            for agent in self.possible_agents
        }       

    def gen_obs(self):
        observations = {}
        
        for agent in range(0, self._num_agents):
            obs = self.states[agent]
            if len(obs) < self._observation_spaces[agent].shape[0]:
                obs = np.pad(obs, (0, self._observation_spaces[agent].shape[0] - len(obs)), 'constant')
            
            avg_battery = 0.0

            other_agents = []
            for elem in range(0, self._num_agents):
                if(elem != agent):
                    other_agents.append(elem)

            for i in other_agents:
                avg_battery += self.states[i][0]
                
            avg_battery /= math.ceil(len(other_agents))
            
            obs[3] = avg_battery
            avg_backlog = 0.0

            for i in other_agents:
                avg_backlog += self.states[i][1]
                
            avg_backlog = math.ceil(avg_backlog / len(other_agents))
            
            obs[4] = avg_backlog

            observations[agent] = np.array(obs, np.float32)
            self.states[agent] = np.array(obs, np.float32)

        
        return observations


