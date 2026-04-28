from abc import abstractmethod
import functools

from pettingzoo import ParallelEnv
from gymnasium import spaces
import numpy as np
import jit_reward_function as jrf
import interpol as ip
from copy import copy


class BaseEnvironment(ParallelEnv):
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
        self.max_storage = self._processing_rate * self._proc_interval * 30
        
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
        # internal counters for episode compeltion 
        self.timestep = 0
        self.max_steps = (3600 / proc_interval) * 5
        self.episode = 1
        self._w = w
        self.total_frames_processed = 0

        try:
            self.max_steps = int(24 * 60 * 60 / proc_interval)
        except:
            self.max_steps = 1
        
        self.fs = [0 for i in range(0, self._num_agents)]
        self.hs = [0 for i in range(0, self._num_agents)]
        self.hs_counter = [0 for i in range(0, self._num_agents)] # Message exchange counter for each agent, used for logging purposes
        
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
                    # Sender mode
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
                    # Receiver mode
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
            if self.battery_energies[agent_id] <= 0:
                self.backlogs[agent_id] = 0
        
        for agent_id in range(self._num_agents):
            self.states[agent_id][0] = self.battery_energies[agent_id] / self.battery_capacities[agent_id]
            self.states[agent_id][1] = self.calculate_backlog_level(agent_id)
            self.states[agent_id][2] = self.timestep / self.max_steps    

    def step(self, actions):
        # manual copy of actions inside internal actions variable
        # for i in range(0, self._num_agents):
        #     agent_action = actions[i]

        #     if np.isscalar(agent_action):
        #         self.actions[i][0] = int(agent_action)
        #         self.actions[i][1] = 0
        #         self.actions[i][2] = 0
        #         self.actions[i][3] = 0
        #         continue

        #     agent_action = np.asarray(agent_action).flatten()
        #     self.actions[i][0] = int(agent_action[0]) if len(agent_action) > 0 else 0
        #     self.actions[i][1] = int(agent_action[1]) if len(agent_action) > 1 else 0
        #     self.actions[i][2] = int(agent_action[2]) if len(agent_action) > 2 else 0
        #     self.actions[i][3] = int(agent_action[3]) if len(agent_action) > 3 else 0

        # updating backlogs with arriving frames for each agent
        for agent_id in range(0, self._num_agents):
            if self.battery_energies[agent_id] > 0:
                frames_arrived = self._arrival_rate * self._proc_interval
                self.backlogs[agent_id] += frames_arrived

        rewards = {}

        for agent_id in range(0, self._num_agents): # Penalize buffer overflow
            if self.backlogs[agent_id] > self.max_storage:
                difference = self.backlogs[agent_id] - self.max_storage
                self.backlogs[agent_id] = self.max_storage
                rewards[agent_id] = -difference

        # Local state update (inlined from update_states_locally)
        terminations = {}
        for agent_id in range(0, self._num_agents):
            fti = actions[agent_id][0]
            idx = (self.episode * self.max_steps) + self.timestep
            self.irradiance_level[agent_id] = self.irradiance_arrays[agent_id][idx] / self.max_irrad
            panel_energy = self.irradiance_level[agent_id] * self.max_irrad * self.panel_surfaces[agent_id] * self._proc_interval * self.panel_efficiency

            actual_battery = self.battery_energies[agent_id] + panel_energy
            backlog = self.backlogs[agent_id]

            #processable = max(min(backlog, int((actual_battery - self.e_idle) / self.e_frame), self._processing_rate * self._proc_interval), 0)
            processed_images = fti * self._proc_interval
            processed_images = min(processed_images, backlog)
            needed_energy = (processed_images * self.e_frame) + self.e_idle

            local_processing = 0
            if actual_battery > needed_energy:
                self.total_frames_processed += processed_images
                backlog = max(backlog - processed_images, 0)
                local_processing = processed_images / self._proc_interval
                
                rewards[agent_id] =  processed_images
                terminations[agent_id] = False
            else:
                rewards[agent_id] = -processed_images - backlog
                terminations[agent_id] = True

            actual_battery = max(actual_battery - needed_energy, 0)


            if backlog > self.max_storage:
                backlog_difference = backlog - self.max_storage
                rewards[agent_id] -= backlog_difference
                backlog = self.max_storage

            #print(f"Agent {agent_id} - IDX: {idx} - rewards: {rewards[agent_id]} - Batt: {actual_battery} - Proc: {processable} - Actions: {actions[agent_id]}")
            rewards[agent_id] /= (self._processing_rate * self._proc_interval)
            self.battery_energies[agent_id] = min(actual_battery, self.battery_capacities[agent_id])
            self.backlogs[agent_id] = backlog
            self.fs[agent_id] += local_processing

        # update_states_offloading calculate offloading processing and energy penalty
        #self.update_states_offloading()
        
        truncations = {a: False for a in self.agents}
                
        if(self.timestep == (self.max_steps - 1)):
            truncations = {a: True for a in self.agents}

        self.timestep += 1
        obs = self.gen_obs()
        
        infos = {}
        for a in self.possible_agents:
            idx = (self.episode * self.max_steps) + self.timestep
            panel_energy = self.irradiance_level[a] * self.max_irrad * self.panel_surfaces[a] * self._proc_interval * self.panel_efficiency
            panel_energy /= self.max_irrad * self.panel_surfaces[a] * self.panel_efficiency * self._proc_interval
            infos[a] = {
                "panel_energy": panel_energy,
                "processed_frames": self.fs[a]/(self._processing_rate * self.max_steps),
                "tx_frames_step": self.hs[a],
                "rx_frames_step": self.hs_counter[a]
            }



        return obs, rewards, terminations, truncations, infos

    def reset(self, seed=None, options=None):
        self.agents = copy(self.possible_agents)
        self.timestep = 0
        self.battery_energies = [(self.battery_capacities[i] * 0.5) for i in range(0, self._num_agents)]
        self.backlogs = [0 for i in range(0, self._num_agents)]
        self.total_frames_processed = 0

        self.fs = [0 for i in range(0, self._num_agents)]
        self.hs = [0 for i in range(0, self._num_agents)]
        self.hs_counter = [0 for i in range(0, self._num_agents)]
        observations = self.gen_obs()

        if(self.seed == "fixed_winter"):
            self.episode = 355
        elif(self.seed == "fixed_summer"):
            self.episode = 172
        elif(self.seed == "linear"):
            self.episode = (self.episode + 1) % 365
        
        return observations, {a: {} for a in self.agents}

    @abstractmethod
    def gen_obs(self) -> dict:
        ...
    
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