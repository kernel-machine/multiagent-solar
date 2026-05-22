from abc import abstractmethod
import functools
import os

from pettingzoo import ParallelEnv
from gymnasium import spaces
import numpy as np
import pandas as pd
import interpol as ip
from copy import copy
import torch

import sys
from prediction_module.prediction import GHIPredictorLSTM


class BaseEnvironment(ParallelEnv):
    metadata = {
        "name": "custom_environment_v0",
    }

    def __init__(self, num_agents, irradiance_datapaths, delta_time, proc_interval, proc_rate, arr_rate, batteries, panel_surfaces, power_idle, power_max, w, seed, use_gossip=False, gossip_interval=5, gossip_targets=2, gossip_state_nodes=3, battery_hard_threshold=0.0, use_random_battery=False, use_lstm_prediction=False, use_lstm_prediction_demo=False, disable_offloading=False):
        super().__init__()
        
        self.disable_offloading = disable_offloading
        
        self.use_gossip = use_gossip
        self.gossip_interval = gossip_interval
        self.gossip_targets = gossip_targets
        self.gossip_state_nodes = gossip_state_nodes
        self.battery_hard_threshold = battery_hard_threshold
        self.use_random_battery = use_random_battery
        self.use_lstm_prediction = use_lstm_prediction
        self.use_lstm_prediction_demo = use_lstm_prediction_demo
        # Convenience flag: either mode needs raw data + temporal features
        self._lstm_features_enabled = use_lstm_prediction or use_lstm_prediction_demo
        
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
        
        # Raw 15-min resolution data for LSTM prediction
        self.irradiance_raw_arrays = []
        self.lstm_lookback = 96   # 24h at 15-min resolution
        self.lstm_horizon = 24    # 6h at 15-min resolution
        # Ratio between raw (15-min) and env (proc_interval) resolution
        self.raw_to_env_ratio = int(delta_time / proc_interval)
        self.delta_time = delta_time
        
        for filepath in irradiance_datapaths:
            # print(filepath, delta_time, proc_interval)
            df = ip.interpolate(filepath, delta_time, proc_interval)
            
            self.irradiance_data.append(df)
            
            self.irradiance_arrays.append(df['ghi'].values)
            
            # Load raw CSV at original 15-min resolution for LSTM / demo input
            if self._lstm_features_enabled:
                raw_df = pd.read_csv(filepath)
                raw_ghi = raw_df['ghi'].values.astype(np.float32)
                self.irradiance_raw_arrays.append(raw_ghi)
        
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
        self.total_transferred_frames = 0

        try:
            self.max_steps = int(24 * 60 * 60 / proc_interval)
        except:
            self.max_steps = 1
        
        # ── LSTM Prediction Module ────────────────────────────────────────────
        if self._lstm_features_enabled:
            # Compute per-agent min/max for MinMax scaling (needed by both modes)
            self._lstm_min = []
            self._lstm_max = []
            for raw_ghi in self.irradiance_raw_arrays:
                self._lstm_min.append(float(np.min(raw_ghi)))
                self._lstm_max.append(float(np.max(raw_ghi)))
        
        if self.use_lstm_prediction and not self.use_lstm_prediction_demo:
            # Load the frozen LSTM model (only needed in real prediction mode)
            self._lstm_device = torch.device('cpu')
            self._lstm_model = GHIPredictorLSTM(
                input_size=1, hidden_size=64, num_layers=3, output_size=self.lstm_horizon
            )
            lstm_weights_path = os.path.join(
                os.path.dirname(__file__), 'prediction_module', 'ghi_predictor_lstm.pth'
            )
            self._lstm_model.load_state_dict(
                torch.load(lstm_weights_path, map_location=self._lstm_device)
            )
            self._lstm_model.eval()
            for param in self._lstm_model.parameters():
                param.requires_grad = False
        
        self.fs = [0 for i in range(0, self._num_agents)]
        self.hs = [0 for i in range(0, self._num_agents)]
        self.hs_counter = [0 for i in range(0, self._num_agents)] # Message exchange counter for each agent, used for logging purposes

        
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

            
            if remaining_framerate > 0 and xti == 0 and hti > 0:
                # 2-state semantics: xti is a receive flag only.
                # The sender stays in non-receive mode and the target must be in receive mode.
                if(gti != agent_id and xt_gti == 1 and gt_gti == agent_id and self.backlogs[gti] <= (self._arrival_rate * self._proc_interval)):
                    ht = hti
                    backlog = self.backlogs[agent_id]

                    processable = max(min(backlog, int((actual_battery - self.e_idle) / self.e_tx_rx), remaining_framerate * self._proc_interval), 0)
                    processed = min(ht * self._proc_interval, processable)
                    needed_energy = ht * self.e_tx_rx * self._proc_interval

                    if(needed_energy <= actual_battery and processable > 0):
                        self.backlogs[agent_id] = max(backlog - processed, 0)
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
            seconds_into_day = (self.timestep * self._proc_interval) % (24 * 3600)
            hour = seconds_into_day / 3600.0
            self.states[agent_id][2] = np.sin(hour / 23.0)
            self.states[agent_id][3] = np.cos(hour / 23.0)

    def step(self, actions):

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

        # Local state update
        # Dead-agent penalty per step: all frames that arrived but cannot be processed.
        _dead_penalty = self._arrival_rate * self._proc_interval  # unnormalized, same scale as rewards
        terminations = {a: False for a in self.agents}
        for agent_id in range(0, self._num_agents):
            hard_threshold_energy = self.battery_capacities[agent_id] * self.battery_hard_threshold
            if self.battery_hard_threshold > 0 and self.battery_energies[agent_id] <= hard_threshold_energy:
                actions[agent_id] = [0, 0, agent_id, 0]
                rewards[agent_id] = rewards.get(agent_id, 0) - _dead_penalty

            fti = actions[agent_id][0]
            idx = ((self.episode * self.max_steps) + self.timestep) % len(self.irradiance_arrays[agent_id])
            self.irradiance_level[agent_id] = self.irradiance_arrays[agent_id][idx] / self.max_irrad
            panel_energy = self.irradiance_level[agent_id] * self.max_irrad * self.panel_surfaces[agent_id] * self._proc_interval * self.panel_efficiency

            actual_battery = self.battery_energies[agent_id] + panel_energy

            #processable = max(min(backlog, int((actual_battery - self.e_idle) / self.e_frame), self._processing_rate * self._proc_interval), 0)
            processed_images = fti * self._proc_interval
            processed_images = min(processed_images, self.backlogs[agent_id])
            needed_energy = (processed_images * self.e_frame) + self.e_idle

            local_processing = 0
            if actual_battery > needed_energy:
                self.total_frames_processed += processed_images
                self.backlogs[agent_id] = max(self.backlogs[agent_id] - processed_images, 0)
                local_processing = processed_images / self._proc_interval
                
                rewards[agent_id] = rewards.get(agent_id, 0) + processed_images
            else:
                rewards[agent_id] = rewards.get(agent_id, 0) - processed_images - self.backlogs[agent_id]

            actual_battery = max(actual_battery - needed_energy, 0)

            self.battery_energies[agent_id] = min(actual_battery, self.battery_capacities[agent_id])
            self.fs[agent_id] += local_processing

            off_rate = actions[agent_id][3]          # index 3 = off_rate
            target   = int(actions[agent_id][2])     # index 2 = target agent
            
            can_offload = True
            if self.disable_offloading:
                can_offload = False
            elif self.use_gossip and len(self.gossip_memory[agent_id]) < self.gossip_state_nodes:
                can_offload = False
            
            offloaded_images = off_rate * self._proc_interval
            if can_offload and off_rate > 0 and target != agent_id and self.backlogs[agent_id] > 0:
                # 2-state semantics: 0 = not receiving, 1 = receiving.
                # The sender must be in non-receive mode, the target must be in receive mode.
                if actions[agent_id][1] == 0 and actions[target][1] == 1: #Rewards handshaking
                    rewards[agent_id] += 0.2*offloaded_images
                    needed_energy = offloaded_images * self.e_tx_rx# * self._proc_interval
                    if self.battery_energies[agent_id] > needed_energy:
                        self.backlogs[agent_id] = max(self.backlogs[agent_id] - offloaded_images, 0)
                        self.backlogs[target] += offloaded_images
                        if self.backlogs[target] > self.max_storage:
                            diff = self.backlogs[target] - self.max_storage
                            real_images = offloaded_images - diff
                            self.backlogs[target] = self.max_storage
                            rewards[agent_id] += 0.5*real_images
                        else:
                            rewards[agent_id] += 0.5*offloaded_images
                        self.total_transferred_frames += offloaded_images
                    else: # Not enough energy to transmit
                        rewards[agent_id] -= 0.5*offloaded_images
                    self.battery_energies[agent_id] = max(self.battery_energies[agent_id] - needed_energy, 0)
                else: # Wrong target or sender still in receive mode
                    rewards[agent_id] -= 0.5*offloaded_images
        
        # Penalize buffer overflow
        for agent_id in range(0, self._num_agents):
            if self.backlogs[agent_id] > self.max_storage:
                backlog_difference = self.backlogs[agent_id] - self.max_storage
                rewards[agent_id] -= backlog_difference
                self.backlogs[agent_id] = self.max_storage

        # Normalize rewards
        rewards = {a: rewards[a] / (self._processing_rate * self._proc_interval) for a in self.agents}
        
        truncations = {a: False for a in self.agents}
        terminations = {a: self.battery_energies[a] <= 0 for a in self.agents}
                
        if(self.timestep == (self.max_steps - 1)):
            truncations = {a: True for a in self.agents}

        self.timestep += 1
        
        # Gossip communication
        if self.use_gossip and (self.timestep % self.gossip_interval == 0):
            for agent_id in self.agents:
                other_agents = [j for j in self.agents if j != agent_id]
                targets = np.random.choice(other_agents, min(self.gossip_targets, len(other_agents)), replace=False)
                info = {
                    'battery': self.battery_energies[agent_id] / self.battery_capacities[agent_id],
                    'backlog': self.backlogs[agent_id] / self.max_storage,
                    'timestamp': self.timestep
                }
                for target_agent in targets:
                    self.gossip_memory[target_agent][agent_id] = info

        obs = self.gen_obs()
        
        infos = {}
        for a in self.possible_agents:
            idx = ((self.episode * self.max_steps) + self.timestep) % len(self.irradiance_arrays[a])
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
        # Check if we're in evaluation mode (always use 50% battery)
        is_evaluation = options is not None and options.get('evaluate', False)
        if is_evaluation or not self.use_random_battery:
            # Fixed 50% for evaluation or when use_random_battery is disabled
            self.battery_energies = [(self.battery_capacities[i] * 0.5) for i in range(0, self._num_agents)]
        else:
            # Random battery between 40% and 60% for training
            self.battery_energies = [(self.battery_capacities[i] * np.random.uniform(0.4, 0.6)) for i in range(0, self._num_agents)]
        self.backlogs = [0 for i in range(0, self._num_agents)]
        self.total_frames_processed = 0
        self.total_transferred_frames = 0
        self.gossip_memory = {a: {} for a in self.agents}

        self.fs = [0 for i in range(0, self._num_agents)]
        self.hs = [0 for i in range(0, self._num_agents)]
        self.hs_counter = [0 for i in range(0, self._num_agents)]
        observations = self.gen_obs()

        if is_evaluation:
            self.episode = 0
        elif(self.seed == "fixed_winter"):
            self.episode = 0
        elif(self.seed == "fixed_summer"):
            self.episode = 172
        elif(self.seed == "linear"):
            self.episode = (self.episode + 1) % 365
        elif(self.seed == "random"):
            self.episode = np.random.randint(0, 365)
        
        return observations, {a: {} for a in self.agents}

    def get_lstm_prediction_features(self, agent_id):
        """
        Compute predictions for the next 6 hours (24 steps at 15-min)
        and return a flat array of 24×4 = 96 values.
        Each of the 24 prediction steps has:
          (predicted_value, sin(t/23), cos(t/23), t/23)
        where t is the hour of day (0-23) for that future step.

        In demo mode (use_lstm_prediction_demo) the real future GHI values
        are used instead of LSTM predictions, providing an oracle baseline.
        """
        # Current position in raw 15-min array
        env_idx = ((self.episode * self.max_steps) + self.timestep) % len(self.irradiance_arrays[agent_id])
        raw_idx = env_idx // self.raw_to_env_ratio
        
        raw_ghi = self.irradiance_raw_arrays[agent_id]
        n_raw = len(raw_ghi)
        mn = self._lstm_min[agent_id]
        mx = self._lstm_max[agent_id]
        
        if self.use_lstm_prediction_demo:
            # ── Demo / Oracle mode: use real future values ────────────────
            future_indices = [(raw_idx + 1 + i) % n_raw
                             for i in range(self.lstm_horizon)]
            future_values = raw_ghi[future_indices].astype(np.float32)
            # Apply the same MinMax normalisation so values are in [0, 1]
            if mx > mn:
                preds_np = (future_values - mn) / (mx - mn)
            else:
                preds_np = future_values
        else:
            # ── Real LSTM prediction mode ─────────────────────────────────
            # Build lookback window (96 values at 15-min resolution)
            lookback_indices = [(raw_idx - self.lstm_lookback + 1 + i) % n_raw
                               for i in range(self.lstm_lookback)]
            lookback_values = raw_ghi[lookback_indices].astype(np.float32)
            
            # Apply the same MinMax scaling used during LSTM training
            if mx > mn:
                lookback_scaled = (lookback_values - mn) / (mx - mn)
            else:
                lookback_scaled = lookback_values
            
            # LSTM forward pass (no grad, frozen)
            x = torch.from_numpy(lookback_scaled).reshape(1, self.lstm_lookback, 1)
            with torch.no_grad():
                preds = self._lstm_model(x)  # shape (1, 24)
            preds_np = preds.squeeze(0).numpy()  # (24,)
        
        # Compute the current hour of day
        # Each env timestep = proc_interval seconds
        seconds_into_day = (self.timestep * self._proc_interval) % (24 * 3600)
        current_hour = seconds_into_day / 3600.0  # fractional hour
        
        # Build (value, sin(t/23), cos(t/23), t/23) for each of the 24 predictions
        features = []
        for i in range(self.lstm_horizon):
            # Each prediction step is 15 min = 0.25 hours into the future
            future_hour = (current_hour + (i + 1) * (self.delta_time / 3600.0)) % 24.0
            t_norm = future_hour / 23.0
            features.extend([
                float(preds_np[i]),
                np.sin(t_norm),
                np.cos(t_norm),
                t_norm
            ])
        
        return np.array(features, dtype=np.float32)

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