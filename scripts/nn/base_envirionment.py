from collections import abc
from operator import is_
from pandas import options
from abc import abstractmethod
import functools
import os
from math import log, ceil

from pettingzoo import ParallelEnv
from gymnasium import spaces
import numpy as np
import pandas as pd
import interpol as ip
from copy import copy
import torch
import random

import sys
from prediction_module.prediction import GHIPredictorLSTM

MAX_EPISODE_STEPS = 288

class BaseEnvironment(ParallelEnv):
    metadata = {
        "name": "custom_environment_v0",
    }

    def __init__(self, num_agents, irradiance_datapaths, delta_time, proc_interval, proc_rate, arr_rate, batteries, panel_surfaces, power_idle, power_max, w, seed, use_gossip=False, gossip_interval=5, gossip_peers=2, gossip_state_nodes=3, battery_hard_threshold=0.0, use_random_battery=False, use_lstm_prediction=False, use_lstm_prediction_demo=False, disable_offloading=False, handshaking_weight=0.4, offloading_weight=0.5, overflow_weight=0.2, processed_images_weight=1.0, unprocessed_images_weight=1.0, backlog_loss_weight=1.0, survival_bonus=0.0, variable_arrival_rate=False, battery_reward_weight=1.0, max_buffer_steps=10):
        super().__init__()
        
        self.disable_offloading = disable_offloading
        self.handshaking_weight = handshaking_weight
        self.offloading_weight = offloading_weight
        self.overflow_weight = overflow_weight
        self.processed_images_weight = processed_images_weight
        self.backlog_loss_weight = backlog_loss_weight
        self.survival_bonus = survival_bonus
        self.battery_reward_weight = battery_reward_weight
        self.unprocessed_images_weight = unprocessed_images_weight
        
        self.use_gossip = use_gossip
        self.gossip_interval = gossip_interval
        # Number of fixed peers each node selects at reset for gossip sending
        self.gossip_peers = gossip_peers
        self.gossip_state_nodes = gossip_state_nodes
        self.battery_hard_threshold = battery_hard_threshold
        self.use_random_battery = use_random_battery
        self.use_lstm_prediction = use_lstm_prediction
        self.use_lstm_prediction_demo = use_lstm_prediction_demo
        # Convenience flag: either mode needs raw data + temporal features
        self._lstm_features_enabled = use_lstm_prediction or use_lstm_prediction_demo
        self.variable_arrival_rate = variable_arrival_rate
        # Period of 2 hours expressed in timesteps
        self._arrival_period_steps = int(2 * 3600 / proc_interval)
        
        self.agents = [i for i in range(0, num_agents)]
        
        self._num_agents = num_agents
        self._processing_rate = proc_rate
        self._arrival_rate = arr_rate
        self._proc_interval = proc_interval
        
        self.p_idle = power_idle
        self.p_max = power_max
        self.is_evaluation = False
        
        self.max_irrad = 1000.0
        self.panel_efficiency = 0.12
        self.max_storage = self._arrival_rate * self._proc_interval * max_buffer_steps
        
        self.irradiance_data = []
        self.irradiance_arrays = []
        self.seed = seed
        self.np_random = np.random.RandomState(self._seed_to_int(seed))
        
        # Raw 15-min resolution data for LSTM prediction
        self.irradiance_raw_arrays = []
        self.lstm_lookback = 96   # 24h at 15-min resolution
        self.lstm_horizon = 16    # 4h at 15-min resolution
        # Ratio between raw (15-min) and env (proc_interval) resolution
        self.raw_to_env_ratio = int(delta_time / proc_interval)
        self.delta_time = delta_time
        
        for filepath in irradiance_datapaths:
            # print(filepath, delta_time, proc_interval)
            df = ip.interpolate(filepath, delta_time, proc_interval)
            
            self.irradiance_data.append(df)
            
            self.irradiance_arrays.append(df['gti'].values)
            
            # Load raw CSV at original 15-min resolution for LSTM / demo input
            if self._lstm_features_enabled:
                raw_df = pd.read_csv(filepath)
                raw_ghi = raw_df['gti'].values.astype(np.float32)
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
        self.daily_timestamp = 0
        self.day = 0
        self.max_day_steps = int(24 * 60 * 60 / proc_interval)
        self.episode_steps = 0
        
        self.total_frames_processed = 0
        self.total_transferred_frames = 0
        self.tx_energy_consumed = 0.0
        self.offloaded_images_counter = [[0 for i in range(0, self._num_agents)] for j in range(0, self._num_agents)]
        
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

    def _seed_to_int(self, seed):
        if isinstance(seed, (int, np.integer)):
            return int(seed)
        if seed is None:
            return 0
        import hashlib
        seed_text = str(seed).encode("utf-8")
        return int.from_bytes(hashlib.sha256(seed_text).digest()[:4], byteorder="little", signed=False)

        

    def step(self, actions):

        # updating backlogs with arriving frames for each agent
        frames_arrived_list = []
        for agent_id in range(0, self._num_agents):
            if self.battery_energies[agent_id] > 0:
                frames_arrived = self.get_frames_arrived(agent_id)
                self.backlogs[agent_id] += frames_arrived
                frames_arrived_list.append(frames_arrived)
            else:
                frames_arrived_list.append(0.0)

        is_day_changed = False
        processing_reward = {}
        offloading_reward = {}
        overflow_reward = {}
        battery_reward = {}
        threshold_reward = {}
        real_processed_images = {}
        for agent_id in range(0, self._num_agents):
            processing_reward[agent_id] = 0
            offloading_reward[agent_id] = 0
            overflow_reward[agent_id] = 0
            battery_reward[agent_id] = 0
            threshold_reward[agent_id] = 0

        # truncations = {a: False for a in self.agents}
        # terminations = {a: False for a in self.agents}

        # for agent_id in range(0, self._num_agents): # Penalize buffer overflow
        #     if self.backlogs[agent_id] > self.max_storage:
        #         difference = self.backlogs[agent_id] - self.max_storage
        #         self.backlogs[agent_id] = self.max_storage
        #         rewards[agent_id] = -difference

        # Local state update
        # Dead-agent penalty per step: all frames that arrived but cannot be processed.

        #_dead_penalty = self._arrival_rate * self._proc_interval  # unnormalized, same scale as rewards
        offloaded_images_dict = {i: 0 for i in range(0, self._num_agents)}
        received_images_dict = {i: 0 for i in range(0, self._num_agents)}
        for agent_id in range(0, self._num_agents):
            fti = actions[agent_id][0]
            hard_threshold_energy = self.battery_capacities[agent_id] * self.battery_hard_threshold
            if self.battery_hard_threshold > 0 and self.battery_energies[agent_id] <= hard_threshold_energy:
                #actions[agent_id] = [0, 0, agent_id, 0]
                actual_battery_percentage = self.battery_energies[agent_id] / self.battery_capacities[agent_id]
                a = actual_battery_percentage / self.battery_hard_threshold
                #battery_reward[agent_id] = (-1 / (actual_battery_percentage + 1e-6))*self.battery_reward_weight
                battery_reward[agent_id] = self.battery_reward_weight*(-1 / ((actual_battery_percentage / self.battery_hard_threshold) + 1e-6))
                fti = 0
            else:
                battery_reward[agent_id] = 0
            
            irradiance_level = self.get_irradiance_level(self.day, self.daily_timestamp, agent_id)
            panel_energy = irradiance_level * self.max_irrad * self.panel_surfaces[agent_id] * self._proc_interval * self.panel_efficiency
            #print("Panel energy for agent", agent_id, ":", panel_energy)
            #assert panel_energy >= self.e_idle, f"Panel energy {panel_energy} is less than idle energy {self.e_idle} on node {agent_id} at day {self.day} timestamp {self.daily_timestamp}"
            if panel_energy == 0:
                is_day_changed = self.scroll_untill_next_day()
            
            actual_battery = self.battery_energies[agent_id] + panel_energy

            #processable = max(min(backlog, int((actual_battery - self.e_idle) / self.e_frame), self._processing_rate * self._proc_interval), 0)
            processed_images = fti * self._proc_interval
            max_processed_images = self._processing_rate * self._proc_interval
            processed_images = min(processed_images, self.backlogs[agent_id])
            real_processed_images[agent_id] = processed_images / max_processed_images
            needed_energy = (processed_images * self.e_frame) + self.e_idle

            backlog_loss_weight = self.backlog_loss_weight
            local_processing = 0
            battery_capacity = self.battery_capacities[agent_id]
            if actual_battery > needed_energy:
                self.total_frames_processed += processed_images
                self.backlogs[agent_id] = max(self.backlogs[agent_id] - processed_images, 0)
                local_processing = processed_images / self._proc_interval
                unprocessed_images = max_processed_images - processed_images
                processing_reward[agent_id] = processed_images 
                #battery_reward[agent_id] = self.survival_bonus * max(actual_battery - needed_energy, 0) / (battery_capacity + 1e-6)
            else:
                processing_reward[agent_id] = -processed_images - self.backlogs[agent_id]
                
            #battery_reward[agent_id] = -self.survival_bonus * max(needed_energy - actual_battery, 0) / (battery_capacity + 1e-6)

            processing_reward[agent_id] /= (self._processing_rate)
            processing_reward[agent_id] *= self.processed_images_weight
            actual_battery = max(actual_battery - needed_energy, 0)

            self.battery_energies[agent_id] = min(actual_battery, self.battery_capacities[agent_id])
            self.fs[agent_id] += local_processing

            off_rate = actions[agent_id][2]          # index 3 = off_rate
            target   = int(actions[agent_id][1])     # index 2 = target agent

            
            can_offload = True
            if self.disable_offloading:
                can_offload = False
            elif self.use_gossip and len(self.gossip_memory[agent_id]) < self.gossip_state_nodes:
                can_offload = False
            
            # Under gossip mode, allow offloading only to nodes that have sent gossip info to this agent
            if can_offload and self.use_gossip:
                allowed_senders = list(self.gossip_memory[agent_id].keys())
                if target not in allowed_senders:
                    can_offload = False

            if can_offload and off_rate > 0 and target != agent_id:
                offloaded_images = min(off_rate * self._proc_interval, self.backlogs[agent_id], self.max_storage - self.backlogs[target])
                offloaded_images = max(offloaded_images, 0)
                offloaded_images_dict[agent_id] = offloaded_images
                received_images_dict[target] += offloaded_images
                needed_energy = offloaded_images * self.e_tx_rx # * self._proc_interval
                self.tx_energy_consumed += needed_energy
                offload_step_reward = 0
                if self.battery_energies[agent_id] > needed_energy and self.battery_energies[target] > needed_energy:
                    self.backlogs[agent_id] = max(self.backlogs[agent_id] - offloaded_images, 0)
                    self.backlogs[target] += offloaded_images
                    offload_step_reward = offloaded_images
                    self.offloaded_images_counter[agent_id][target] += offloaded_images
                else: # Not enough energy to transmit
                    offload_step_reward = -offloaded_images
                self.battery_energies[agent_id] = max(self.battery_energies[agent_id] - needed_energy, 0)
                self.battery_energies[target] = max(self.battery_energies[target] - needed_energy, 0)

                # self.offload_weight < processing_weight to prioritize local processing over offloading, and vice versa
                offloading_reward[agent_id] += self.offloading_weight * offload_step_reward / (self._processing_rate)
                target_quality = self.gossip_memory[agent_id][target]['battery'] * (1- self.gossip_memory[agent_id][target]['backlog'])
                local_quality = (self.battery_energies[agent_id]/self.battery_capacities[agent_id]) * (1 - (self.backlogs[agent_id] / (self.max_storage)))
                offloading_reward[agent_id] *= (target_quality - local_quality - 0.1) #tsp 60
                #battery_quality = self.gossip_memory[agent_id][target]['battery'] - (self.battery_energies[agent_id] / self.battery_capacities[agent_id])
                #backlog_quality = self.gossip_memory[agent_id][target]['backlog'] - (self.backlogs[agent_id] / self.max_storage)
                #offloading_reward[agent_id] *= (0.5*battery_quality + 0.5*backlog_quality)
                #offloading_reward[agent_id] -= (self.offloading_weight/2)*self.tx_energy_consumed / (self.e_tx_rx * self._processing_rate * self._num_agents)
        
        #Penalize buffer overflow
        for agent_id in range(0, self._num_agents):
            #buffer_fill = self.backlogs[agent_id] / (self.max_storage + 1e-6)
            #overflow_reward[agent_id] = ((self.overflow_weight**buffer_fill) - 1)/(self._processing_rate)
            backlog_ratio = self.backlogs[agent_id] / self.max_storage
            #overflow_reward[agent_id] = 1/((1-backlog_ratio)+1e-6)
            #if self.backlogs[agent_id] > self.max_storage:
            if False:
                if backlog_ratio > 0.8:
                    overflow_reward[agent_id] = -1/((1-backlog_ratio)+1e-6)
                else:
                    overflow_reward[agent_id] = 0
                self.backlogs[agent_id] = min(self.backlogs[agent_id], self.max_storage)
            else:
                if self.backlogs[agent_id] > self.max_storage:
                    backlog_difference = self.backlogs[agent_id] - self.max_storage
                    overflow_reward[agent_id] = -backlog_difference*self.overflow_weight / (self._processing_rate * self._proc_interval)
                    self.backlogs[agent_id] = self.max_storage
                else:
                    overflow_reward[agent_id] = 0

            # battery_capacity = self.battery_capacities[agent_id]
            # battery_ratio = self.battery_energies[agent_id] / (battery_capacity + 1e-6)
            # threshold_scale = self.survival_bonus if self.survival_bonus != 0 else 1.0
            # threshold_center = self.battery_hard_threshold if self.battery_hard_threshold > 0 else 0.5
            # threshold_temperature = 0.15 if self.battery_hard_threshold > 0 else 0.25

            # threshold_reward[agent_id] = threshold_scale * np.tanh(
            #     (battery_ratio - threshold_center) / max(1e-6, threshold_temperature)
            # )
            # battery_reward[agent_id] = threshold_reward[agent_id]

            #overflow_reward[agent_id] = 0.2* (-self.backlogs[agent_id] / self.max_storage)
            #processing_reward[agent_id] = 0.5*actions[agent_id][0] 

        # Normalize rewards
        #rewards = {a: rewards[a] / (self._processing_rate * self._proc_interval) for a in self.agents}
        terminations = {}
        truncations = {}
        for a in self.agents:
            terminations[a] = self.battery_energies[a] <= 0
        
        truncations = {a: self.episode_steps > MAX_EPISODE_STEPS  for a in self.agents}

        self.daily_timestamp += 1
        self.episode_steps += 1
        
        # Gossip communication: send only to the persistent peers selected at reset
        if self.use_gossip and (self.daily_timestamp % self.gossip_interval == 0):
            for agent_id in self.agents:
                # send to the persistent peers selected at reset (no random fallback)
                if getattr(self, 'peers', None) is None:
                    # if peers not initialized, select up to gossip_peers now
                    k = min(self.gossip_peers, self._num_agents - 1)
                    other_agents = [j for j in self.agents if j != agent_id]
                    if k <= 0:
                        targets = []
                    else:
                        targets = list(self.np_random.choice(other_agents, k, replace=False))
                else:
                    targets = list(self.peers[agent_id])

                info = {
                    'battery': self.battery_energies[agent_id] / self.battery_capacities[agent_id],
                    'backlog': self.backlogs[agent_id] / self.max_storage,
                    'timestamp': self.daily_timestamp
                }
                for target_agent in targets:
                    self.gossip_memory[target_agent][agent_id] = info

        obs = self.gen_obs()
        
        infos = {}
        max_frames_arrived = 2.0 * self._arrival_rate * self._proc_interval if self.variable_arrival_rate else self._arrival_rate * self._proc_interval
        for a in self.agents:
            panel_energy = self.get_irradiance_level(self.day, self.daily_timestamp, a) * self.max_irrad * self.panel_surfaces[a] * self._proc_interval * self.panel_efficiency
            panel_energy /= self.max_irrad * self.panel_surfaces[a] * self.panel_efficiency * self._proc_interval
            # Normalize frames_arrived to [0, 1]
            frames_arrived_norm = frames_arrived_list[a] / max_frames_arrived if max_frames_arrived > 0 else 0.0
            infos[a] = {
                "panel_energy": panel_energy,
                "processed_frames": self.fs[a]/(self._processing_rate * self.max_day_steps),
                "offloaded_images": offloaded_images_dict[a]/(self._processing_rate * self._proc_interval),
                "received_images": received_images_dict[a]/(self._processing_rate * self._proc_interval),
                "processing_reward": processing_reward[a],
                "offloading_reward": offloading_reward[a],
                "overflow_reward": overflow_reward[a],
                "battery_reward": battery_reward[a],
                "threshold_reward": threshold_reward[a],
                "is_day_changed": is_day_changed,
                "frames_arrived_norm": frames_arrived_norm,
                "real_processed_images": real_processed_images[a],
            }
            #print(f"Reward for agent {a}: processing {processing_reward[a]:.4f}, offloading {offloading_reward[a]:.4f}, overflow {overflow_reward[a]:.4f}, battery {battery_reward[a]:.4f}")


        #print("Battery rewards:", battery_reward)
        rewards = {}
        for agent_id in range(0, self._num_agents):
            rewards[agent_id] = processing_reward[agent_id] + offloading_reward[agent_id] + overflow_reward[agent_id] + battery_reward[agent_id]
        
        return obs, rewards, terminations, truncations, infos

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.RandomState(self._seed_to_int(seed))

        is_evaluation = options is not None and options.get('evaluate', False)
        reset_fields = options is not None and options.get('reset_fields', False)
        self.is_evaluation = is_evaluation
        self.tx_energy_consumed = 0.0
        self.offloaded_images_counter = [[0 for i in range(0, self._num_agents)] for j in range(0, self._num_agents)]

        # First episode always starts from a clean state.
        # During evaluation we keep the previous day's battery/backlog.
        # During training we randomize them at each reset after the first episode.
        if reset_fields: 
            if is_evaluation:
                self.battery_energies = [(self.battery_capacities[i] * 0.5) for i in range(0, self._num_agents)]
                self.backlogs = [0 for i in range(0, self._num_agents)]
                self.day = 0
                self.total_frames_processed = 0
                self.total_transferred_frames = 0
            else:
                self.battery_energies = [(self.battery_capacities[i] * self.np_random.uniform(0.1,0.5)) for i in range(0, self._num_agents)]
                self.backlogs = [random.randint(0, self.max_storage) for i in range(0, self._num_agents)]
        

        self.gossip_memory = {a: {} for a in self.agents}

        # Persistent peer selection: each node selects a fixed set of peers
        # Number of peers defaults to self.gossip_peers (set from constructor)
        if self.gossip_peers == 0:
            k = ceil(2 * log(self._num_agents) + 3)
        else:
            k = min(self.gossip_peers, self._num_agents - 1)
        self.peers = {}
        for a in self.agents:
            others = [j for j in self.agents if j != a]
            if k <= 0:
                self.peers[a] = []
            else:
                self.peers[a] = list(self.np_random.choice(others, k, replace=False))

        self.fs = [0 for i in range(0, self._num_agents)]
        self.hs = [0 for i in range(0, self._num_agents)]
        self.hs_counter = [0 for i in range(0, self._num_agents)]

        # Preserve `self.day` across evaluation resets so consecutive
        # evaluation episodes continue from the previous day. Only
        # update day during training / non-evaluation resets.
        if self.seed == "linear" or is_evaluation:
            self.day = (self.day + 1) % 365
        elif (self.seed == "fixed_winter"):
            self.day = 0
        elif(self.seed == "fixed_summer"):
            self.day = 172
        elif(self.seed == "random"):
            self.day = random.randint(0, 365)

        self.episode_steps = 0
        self.daily_timestamp = 0
        self.scroll_untill_next_day()
        if not is_evaluation:
            self.daily_timestamp += random.randint(0, 4*15)
        # Go head untill the sun is out
        #self.scroll_untill_next_day()
        observations = self.gen_obs()

        return observations, {a: {} for a in self.agents}
    
    def scroll_untill_next_day(self) -> bool:
        day_changed = False
        agent_id = 0
        while self.get_irradiance_level(self.day, self.daily_timestamp, agent_id) == 0:
            self.daily_timestamp += 1
            if self.daily_timestamp >= self.max_day_steps:
                self.daily_timestamp = 0
                self.day = (self.day + 1) % 365
                day_changed = True
        return day_changed
    
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
        env_idx = ((self.day * self.max_day_steps) + self.daily_timestamp) % len(self.irradiance_arrays[agent_id])
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
        seconds_into_day = (self.daily_timestamp * self._proc_interval) % (24 * 3600)
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

    def get_frames_arrived(self, agent_id: int) -> float:
        """Return the number of frames arriving at this timestep.
        When variable_arrival_rate is enabled the arrival rate follows a
        sinusoidal pattern with a 2-hour period, oscillating between 0 and
        2 * base_arrival_rate.  Otherwise, it returns the constant rate."""
        if not self.variable_arrival_rate:
            return self._arrival_rate * self._proc_interval
        # Sinusoidal modulation: (1 + sin(2π·t / period)) maps to [0, 2]
        phase = (2.0 * np.pi * self.daily_timestamp / self._arrival_period_steps)
        phase += (agent_id * 2.0 * np.pi / self._num_agents)  # phase shift per agent for diversity
        modulation = 1.0 + np.sin(phase)  # range [0, 2]
        return modulation * self._arrival_rate * self._proc_interval

    def get_frames_arrived_norm(self, agent_id: int) -> float:
        """Return the normalized frames_arrived in [0, 1]."""
        fa = self.get_frames_arrived(agent_id)
        if self.variable_arrival_rate:
            max_fa = 2.0 * self._arrival_rate * self._proc_interval
        else:
            max_fa = self._arrival_rate * self._proc_interval
        return fa / max_fa if max_fa > 0 else 0.0

    def get_irradiance_level(self, day:int , dayly_timestamp:int , agent_id:int) -> float:
        idx = ((day * self.max_day_steps) + dayly_timestamp) % len(self.irradiance_arrays[agent_id])
        return self.irradiance_arrays[agent_id][idx] / self.max_irrad

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
