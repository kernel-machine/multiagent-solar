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

class CustomEnvironment(ParallelEnv):
    metadata = {
        "name": "custom_environment_v0",
    }

    def __init__(self, num_agents, irradiance_datapaths, delta_time, proc_interval, proc_rate, arr_rate, batteries, panel_surfaces, power_idle, power_max, w, seed):
        super().__init__()
        
        self.agents = []
        self.possible_agents = [i for i in range(0, num_agents)]
        
        self._num_agents = num_agents
        self._processing_rate = proc_rate
        self._arrival_rate = arr_rate
        self._proc_interval = proc_interval
        
        self.p_idle = power_idle
        self.p_max = power_max
        
        self.max_irrad = 1000.0
        self.panel_efficiency = 0.2
        
        self.irradiance_data = []
        self.irradiance_arrays = []
        self.seed = seed
        
        for filepath in irradiance_datapaths:
            # print(filepath, delta_time, proc_interval)
            df = ip.interpolate(filepath, delta_time, proc_interval)
            
            self.irradiance_data.append(df)
            
            self.irradiance_arrays.append(df['ghi'].values)
        
        self.irradiance_level = [0.0 for i in range(0, self._num_agents)]
        
        self.battery_capacities = [(battery*3600) for battery in batteries]
        self.battery_energies = [0.0 for i in range(0, self._num_agents)]
        self.panel_surfaces = panel_surfaces
        
        self.e_idle = power_idle * self._proc_interval
        self.e_frame = (0.8 * (power_max - power_idle) * 1) / proc_rate
        self.e_tx_rx = (0.2 * (power_max - power_idle) * 1) / proc_rate
        
        self.backlogs = [0 for i in range(0, self._num_agents)]
        
        # state_i = (battery_level_i, daily_completion_i)
        self.states = [[0.0, 0, 0.0] for i in range(0, self._num_agents)]
        self.actions = [[0.0, 0, 0.0, 0.0] for i in range(0, self._num_agents)]
        self.rewards = [0 for i in range(0, self._num_agents)]
        
        # internal counters for episode compeltion 
        self.timestep = 0
        self.max_steps = (3600 / proc_interval) * 5
        self.episode = 355
        self._w = w
        
        try:
            self.max_steps = int(24 * 60 * 60 / proc_interval)
        except:
            self.max_steps = 1

        self._action_spaces = {
            agent: spaces.MultiDiscrete([self._processing_rate + 1, 3, self._num_agents, self._processing_rate + 1]) for agent in self.possible_agents
        }
        
        self._observation_spaces = {
            agent: spaces.Box(
                low=np.array([0.0, 0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0], dtype=np.float32),
                high=np.array([1.0, 3, 1.0, 1.0, 1.0, 1.0, 3, 3, 3], dtype=np.float32),
                dtype=np.float32
            ) 
            for agent in self.possible_agents
        
        }
        
        self.fs = [0 for i in range(0, self._num_agents)]
        self.hs = [0 for i in range(0, self._num_agents)]
        self.hs_counter = [0 for i in range(0, self.num_agents)]
        
    # function for retrieving level of backlog
    def calculate_backlog_level(self, agent_id):
        qty = self.backlogs[agent_id]
        max_storage = self._processing_rate * self._proc_interval * 10
        
        if(qty == 0):
            return 0
        elif(qty > 0 and qty < int(max_storage / 3)):
            return 1
        elif(qty >= int(max_storage / 3) and qty < int((2/3) * max_storage)):
            return 2
        else:
            return 3

    def calculate_reward_locally(self, agent_id):
        fti = self.actions[agent_id][0]
        xti = self.actions[agent_id][1]
        gti = self.actions[agent_id][2]
        hti = self.actions[agent_id][3]
                
        ft_gti = self.actions[gti][0]
        xt_gti = self.actions[gti][1]
        gt_gti = self.actions[gti][2]
        ht_gti = self.actions[gti][3]
        
        idx = (self.episode * self.max_steps) + self.timestep
        # print(idx)
        irradiance = self.irradiance_arrays[agent_id][idx]
        self.irradiance_level[agent_id] = self.irradiance_arrays[agent_id][idx] / self.max_irrad
        
        fti = self.actions[agent_id][0]
    
        idx = (self.episode * self.max_steps) + self.timestep
        irradiance = self.irradiance_arrays[agent_id][idx]
        
        panel_energy = irradiance * self.panel_surfaces[agent_id] * self.panel_efficiency * self._proc_interval
        actual_battery = self.battery_energies[agent_id] + panel_energy
        backlog = self.backlogs[agent_id]
        
        processable = max(min(backlog, int((actual_battery - self.e_idle) / self.e_frame), self._processing_rate * self._proc_interval), 0)
        needed_energy = (fti * self._proc_interval * self.e_frame) + self.e_idle
        
        return jrf.jit_calculate_reward_local(fti,
                                   xti,
                                   gti,
                                   hti,
                                   ft_gti,
                                   xt_gti,
                                   gt_gti,
                                   ht_gti,
                                   irradiance,
                                   self.panel_surfaces[agent_id],
                                   self.panel_efficiency,
                                   self.backlogs[agent_id],
                                   self.backlogs[gti],
                                   self.e_idle,
                                   self.e_frame,
                                   self.e_tx_rx,
                                   self.battery_energies[agent_id],
                                   self.battery_capacities[agent_id],
                                   self._processing_rate,
                                   self._proc_interval,
                                   agent_id
                                   )
    
    def calculate_reward_offloading(self, agent_id):
        fti = self.actions[agent_id][0]
        xti = self.actions[agent_id][1]
        gti = self.actions[agent_id][2]
        hti = self.actions[agent_id][3]
                
        ft_gti = self.actions[gti][0]
        xt_gti = self.actions[gti][1]
        gt_gti = self.actions[gti][2]
        ht_gti = self.actions[gti][3]
        
        idx = (self.episode * self.max_steps) + self.timestep
        # print(idx)
        irradiance = self.irradiance_arrays[agent_id][idx]
        self.irradiance_level[agent_id] = self.irradiance_arrays[agent_id][idx] / self.max_irrad
        
        return jrf.jit_calculate_reward_offloading(fti,
                                   xti,
                                   gti,
                                   hti,
                                   ft_gti,
                                   xt_gti,
                                   gt_gti,
                                   ht_gti,
                                   irradiance,
                                   self.panel_surfaces[agent_id],
                                   self.panel_efficiency,
                                   self.backlogs[agent_id],
                                   self.backlogs[gti],
                                   self.e_idle,
                                   self.e_frame,
                                   self.e_tx_rx,
                                   self.battery_energies[agent_id],
                                   self.battery_capacities[agent_id],
                                   self._processing_rate,
                                   self._proc_interval,
                                   self._arrival_rate,
                                   agent_id,
                                   self._w
                                   )
    
    
    
    def update_state(self, agent_id, panel_energy, needed_energy, processed):
        '''
        state: [0 -> battery, 1-> backlog, 2-> timestep]
        '''
        
        # battery update
        self.battery_energies[agent_id] += (panel_energy - needed_energy)
        if(self.battery_energies[agent_id] < 0.0):
            self.battery_energies[agent_id] = 0.0
        
        # backlog upate
        self.backlogs[agent_id] -= processed
        if(self.backlogs[agent_id] < 0):
            self.backlogs[agent_id] = 0
        
        self.states[agent_id][0] = round(float(self.battery_energies[agent_id] / self.battery_capacities[agent_id]), 2)
        self.states[agent_id][1] = self.calculate_backlog_level(agent_id)
        self.states[agent_id][2] = round(float(self.timestep / self.max_steps), 4)
        
    def reset(self, seed=None, options=None):
        self.agents = copy(self.possible_agents)
        
        # setting to 0 all training variables
        self.timestep = 0
        self.states = [[0.5, 0, 0.0] for i in range(0, self._num_agents)]
        self.actions = [[0.0, 0, 0.0, 0.0] for i in range(0, self._num_agents)]
        self.battery_energies = [(self.battery_capacities[i] * self.states[i][0]) for i in range(0, self._num_agents)]
        self.backlogs = [0 for i in range(0, self._num_agents)]

        self.fs = [0 for i in range(0, self._num_agents)]
        self.hs = [0 for i in range(0, self._num_agents)]
        self.hs_counter = [0 for i in range(0, self._num_agents)]

        observations = {}
        
        for agent in range(0, self._num_agents):
            obs = self.states[agent]
            
            # aggregated batteries
            # print(self.states)
            min_battery = min(self.states[key][0] for key in range(0, self._num_agents))
            max_battery = max(self.states[key][0] for key in range(0, self._num_agents))
            
            avg_battery = 0.0

            other_agents = []
            for elem in range(0, self._num_agents):
                if(elem != agent):
                    other_agents.append(elem)

            for i in other_agents:
                avg_battery += self.states[i][0]
                
            avg_battery /= math.ceil(len(other_agents))
            
            obs.append(min_battery)
            obs.append(avg_battery)
            obs.append(max_battery)
        
            # aggregated backlogs
            min_backlog = min(self.states[key][1] for key in range(0, self._num_agents))
            max_backlog = max(self.states[key][1] for key in range(0, self._num_agents))
            
            avg_backlog = 0.0

            other_agents = []
            for elem in range(0, self._num_agents):
                if(elem != agent):
                    other_agents.append(elem)

            for i in other_agents:
                avg_backlog += self.states[i][1]
                
            avg_backlog = math.ceil(avg_backlog / len(other_agents))
            
            obs.append(min_backlog)
            obs.append(avg_backlog)
            obs.append(max_backlog)        
            
            observations[agent] = np.array(obs, dtype=np.float32)
            # print(observations[agent])

        infos = {a: {} for a in self.agents}
        
        if(self.seed == "fixed_winter"):
            self.episode = 355
        elif(self.seed == "fixed_summer"):
            self.episode = 172
        elif(self.seed == "linear"):
            self.episode = (self.episode + 1) % 365
            
        return observations, infos
        
    def update_states_locally(self):
        local_states = []
        
        for agent_id in range(0, self._num_agents):
            fti = self.actions[agent_id][0]
            
            idx = (self.episode * self.max_steps) + self.timestep
            self.irradiance_level[agent_id] = self.irradiance_arrays[agent_id][idx] / self.max_irrad
            panel_energy = self.irradiance_level[agent_id] * self.max_irrad * self.panel_surfaces[agent_id] * self._proc_interval * self.panel_efficiency
            
            actual_battery = self.battery_energies[agent_id] + panel_energy
            backlog = self.backlogs[agent_id]
            
            # LOCAL PROCESSING
            processable = max(min(backlog, int((actual_battery - self.e_idle) / self.e_frame), self._processing_rate * self._proc_interval), 0)
            needed_energy = (fti * self._proc_interval * self.e_frame) + self.e_idle
            
            local_processing = 0
            
            if actual_battery > needed_energy and processable > 0:
                processed = min(processable, fti * self._proc_interval)
                backlog = max(backlog - processed, 0)
                actual_battery = max(actual_battery - needed_energy, 0)
                local_processing = processed / self._proc_interval
            else:
                actual_battery = max(actual_battery - self.e_idle, 0)
            
            local_states.append({
                'battery': actual_battery,
                'backlog': backlog,
                'local_processing': local_processing
            })
            
            self.fs[agent_id] += local_processing
        
        for agent_id in range(self._num_agents):
            self.battery_energies[agent_id] = local_states[agent_id]['battery']
            self.backlogs[agent_id] = local_states[agent_id]['backlog']
        
    def update_states_offloading(self):
        
        for agent_id in range(0, self._num_agents):
            fti = self.actions[agent_id][0]
            xti = self.actions[agent_id][1]
            gti = self.actions[agent_id][2]
            hti = self.actions[agent_id][3]
            
            ft_gti = self.actions[gti][0]
            xt_gti = self.actions[gti][1]
            gt_gti = self.actions[gti][2]
            ht_gti = self.actions[gti][3]
            
            if(fti + hti) > self._processing_rate:
                hti = self._processing_rate - fti
                
            if(ft_gti + ht_gti) > self._processing_rate:
                ht_gti = self._processing_rate - ft_gti
            
            actual_battery = self.battery_energies[agent_id]
            remaining_framerate = self._processing_rate - fti
            
            offloading_processing = 0
            
            if remaining_framerate > 0 and xti != 0:
                if(xti == 1 and gti != agent_id and hti > 0 and xt_gti == 2 and gt_gti == agent_id and ht_gti > 0 and self.backlogs[gti] <= (self._arrival_rate * self._proc_interval)):
                    ht = min(hti, ht_gti)
                    backlog = self.backlogs[agent_id]
                    
                    processable = max(min(backlog, int((actual_battery - self.e_idle) / self.e_tx_rx), remaining_framerate * self._proc_interval), 0)
                    processed = min(ht * self._proc_interval, processable)
                    needed_energy = ht * self.e_tx_rx * self._proc_interval
                    
                    if(needed_energy <= actual_battery and processable > 0):
                        self.backlogs[agent_id] = max(backlog - processed, 0)
                        actual_battery = max(actual_battery - needed_energy, 0)
                        offloading_processing = processed / self._proc_interval
                        self.hs_counter[agent_id] += 1
                
                if(xti == 2 and gti != agent_id and hti > 0 and xt_gti == 1 and gt_gti == agent_id and ht_gti > 0 and self.backlogs[agent_id] <= (self._arrival_rate * self._proc_interval)):
                    ht = min(hti, ht_gti)
                    backlog = self.backlogs[gti]
                    
                    processable = max(min(backlog, int((actual_battery - self.e_idle) / (self.e_tx_rx + self.e_frame)), remaining_framerate * self._proc_interval), 0)
                    processed = min(ht * self._proc_interval, processable)
                    needed_energy = ht * (self.e_tx_rx + self.e_frame) * self._proc_interval
                    
                    if(needed_energy <= actual_battery and processable > 0):
                        actual_battery = max(actual_battery - needed_energy, 0)
                        offloading_processing = processed / self._proc_interval
                        self.hs_counter[agent_id] += 1
            
            self.hs[agent_id] += offloading_processing
            self.battery_energies[agent_id] = min(actual_battery, self.battery_capacities[agent_id])
        
        for agent_id in range(self._num_agents):
            self.states[agent_id][0] = round(self.battery_energies[agent_id] / self.battery_capacities[agent_id], 2)
            self.states[agent_id][1] = self.calculate_backlog_level(agent_id)
            self.states[agent_id][2] = round(self.timestep / self.max_steps, 4)    

    def step(self, actions):
        # manual copy of actions inside internal actions variable
        for i in range(0, self._num_agents):
            for j in range(0, len(actions[i])):
                self.actions[i][j] = actions[i][j]

        # updating backlogs with arriving frames for each agent
        for agent_id in range(0, self._num_agents):
            frames_arrived = self._arrival_rate * self._proc_interval
            self.backlogs[agent_id] += frames_arrived

        # for each agent is returned the reward according the reward function defined a priori        
        rewards = {a: self.calculate_reward_locally(a) for a in self.agents}
        self.update_states_locally()

        # input(rewards)
               
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        
        if(self.timestep == (self.max_steps - 1)):
            # self.episode = 355
            truncations = {a: True for a in self.agents}
        
        self.timestep += 1
        
        # update of states after receiving all actions
        rewards_offloading = {a: self.calculate_reward_offloading(a) for a in self.agents}
        self.update_states_offloading()
        
        # input(rewards_offloading)
        
        for elem in rewards_offloading:
            rewards[elem] += rewards_offloading[elem]
        
        # observations structure is a dictionary with keys the indeces of agents
        observations = {}
        
        for agent in range(0, self._num_agents):
            obs = self.states[agent]
            
            # aggregated batteries
            min_battery = min(self.states[key][0] for key in range(0, self._num_agents))
            max_battery = max(self.states[key][0] for key in range(0, self._num_agents))
            
            avg_battery = 0.0

            other_agents = []
            for elem in range(0, self._num_agents):
                if(elem != agent):
                    other_agents.append(elem)

            for i in other_agents:
                avg_battery += self.states[i][0]
                
            avg_battery /= math.ceil(len(other_agents))
            
            obs[3] = min_battery
            obs[4] = avg_battery
            obs[5] = max_battery
            
            # aggregated backlogs
            min_backlog = min(self.states[key][1] for key in range(0, self._num_agents))
            max_backlog = max(self.states[key][1] for key in range(0, self._num_agents))
            
            avg_backlog = 0.0

            for i in other_agents:
                avg_backlog += self.states[i][1]
                
            avg_backlog = math.ceil(avg_backlog / len(other_agents))
            
            obs[6] = min_backlog
            obs[7] = avg_backlog
            obs[8] = max_backlog        
            
            observations[agent] = np.array(obs, dtype=np.float32)
            self.states[agent] = np.array(obs, dtype=np.float32)

        infos = {}
        for a in self.agents:
            panel_energy = self.irradiance_level[a] * self.max_irrad * self.panel_surfaces[a] * self._proc_interval * self.panel_efficiency
            panel_energy /= self.max_irrad * self.panel_surfaces[a] * self.panel_efficiency * self._proc_interval
            infos[a] = {
                "panel_energy": panel_energy,
                #"backlog": self.backlogs[a]/(self._processing_rate * self._proc_interval * 10),
            }
        
        if any(terminations.values()) or all(truncations.values()):
            self.agents = []
        
        return observations, rewards, terminations, truncations, infos

    def render(self):
        pass

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._action_spaces[agent]
    
    def observe(self, agent):
        return np.array(self.observations[agent])
