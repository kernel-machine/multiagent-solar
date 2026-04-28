import os
import sys

from gymnasium import spaces
from copy import copy

import numpy as np
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from base_envirionment import BaseEnvironment

class CustomEnvironment(BaseEnvironment):

    def __init__(self, num_agents, irradiance_datapaths, delta_time, proc_interval, proc_rate, arr_rate, batteries, panel_surfaces, power_idle, power_max, w, seed):
        super().__init__(num_agents, irradiance_datapaths, delta_time, proc_interval, proc_rate, arr_rate, batteries, panel_surfaces, power_idle, power_max, w, seed)   

        # self._action_spaces = {
        #     agent: spaces.MultiDiscrete([self._processing_rate + 1, 3, self._num_agents, self._processing_rate + 1]) for agent in self.possible_agents
        # }
        self._action_spaces = {
            agent: spaces.MultiDiscrete([self._processing_rate + 1]) for agent in self.possible_agents
        }
        self._observation_spaces = {
            agent: spaces.Box(
                # Batt, Backlog, Time, Min Batt, Avg Batt, Max Batt, Min Backlog, Avg Backlog, Max Backlog
                low=np.array([0.0, 0, 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32
            ) 
            for agent in self.possible_agents
        }
        
    def gen_obs(self):
        # build space for each agent, including aggregated states
        observations = {}
        for agent in range(self._num_agents):
            obs = []

            obs.append(self.battery_energies[agent] / self.battery_capacities[agent])
            obs.append(self.backlogs[agent] / self.max_storage)
            obs.append(self.timestep / self.max_steps)

            # aggregated batteries
            # print(self.states)
            # min_battery = min(self.states[key][0] for key in range(0, self._num_agents))
            # max_battery = max(self.states[key][0] for key in range(0, self._num_agents))
            
            # avg_battery = 0.0

            # other_agents = []
            # for elem in range(0, self._num_agents):
            #     if(elem != agent):
            #         other_agents.append(elem)

            # for i in other_agents:
            #     avg_battery += self.states[i][0]
                
            # avg_battery /= math.ceil(len(other_agents))
            
            # obs.append(min_battery)
            # obs.append(avg_battery)
            # obs.append(max_battery)
        
            # # aggregated backlogs
            # min_backlog = min(self.states[key][1] for key in range(0, self._num_agents))
            # max_backlog = max(self.states[key][1] for key in range(0, self._num_agents))
            
            # avg_backlog = 0.0

            # other_agents = []
            # for elem in range(0, self._num_agents):
            #     if(elem != agent):
            #         other_agents.append(elem)

            # for i in other_agents:
            #     avg_backlog += self.states[i][1]
                
            # avg_backlog = math.ceil(avg_backlog / len(other_agents))
            
            # obs.append(min_backlog)
            # obs.append(avg_backlog)
            # obs.append(max_backlog)        
            
            observations[agent] = np.array(obs, dtype=np.float32)
            # print(observations[agent])
            
        return observations
