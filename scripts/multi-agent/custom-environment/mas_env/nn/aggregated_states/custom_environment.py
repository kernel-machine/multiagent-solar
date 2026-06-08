import os
import sys

from gymnasium import spaces
from copy import copy

import numpy as np
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from base_envirionment import BaseEnvironment

class CustomEnvironment(BaseEnvironment):

    def __init__(self, num_agents, irradiance_datapaths, delta_time, proc_interval, proc_rate, arr_rate, batteries, panel_surfaces, power_idle, power_max, w, seed, use_cross_attention=False, use_deepsets=False, use_deepsets_spatial=False, max_agents=None, random_nodes=0, use_gossip=False, gossip_interval=5, gossip_peers=None, gossip_state_nodes=3, gossip_order=None, termination_mode="early", battery_hard_threshold=0.0, use_random_battery=False, use_lstm_prediction=False, use_lstm_prediction_demo=False, disable_offloading=False, handshaking_weight=0.4, offloading_weight=0.5, overflow_weight=0.2, processed_images_weight=1.0, unprocessed_images_weight=1.0, backlog_loss_weight=1.0, survival_bonus=0.0):
        super().__init__(num_agents, irradiance_datapaths, delta_time, proc_interval, proc_rate, arr_rate, batteries, panel_surfaces, power_idle, power_max, w, seed, use_gossip, gossip_interval, gossip_peers=gossip_peers, gossip_state_nodes=gossip_state_nodes, battery_hard_threshold=battery_hard_threshold, use_random_battery=use_random_battery, use_lstm_prediction=use_lstm_prediction, use_lstm_prediction_demo=use_lstm_prediction_demo, disable_offloading=disable_offloading, handshaking_weight=handshaking_weight, offloading_weight=offloading_weight, overflow_weight=overflow_weight, processed_images_weight=processed_images_weight, unprocessed_images_weight=unprocessed_images_weight, backlog_loss_weight=backlog_loss_weight, survival_bonus=survival_bonus)
        self.termination_mode = termination_mode
        self.gossip_order = gossip_order

        self.use_cross_attention = use_cross_attention
        self.use_deepsets = use_deepsets
        self.use_deepsets_spatial = use_deepsets_spatial
        self.random_nodes = random_nodes
        # max_agents defines the padded observation size; defaults to num_agents.
        self.max_agents = max_agents if max_agents is not None else num_agents
        assert self.max_agents >= num_agents, "max_agents must be >= num_agents"
        if self.random_nodes > 0:
            assert self.random_nodes < self._num_agents, "random_nodes must be less than the total number of agents"

        # LSTM prediction adds 16×4 = 64 extra observation values
        self._lstm_obs_dim = 64 if self._lstm_features_enabled else 0

        self._action_spaces = {
            agent: spaces.MultiDiscrete([self._processing_rate + 1, self._num_agents, self._processing_rate + 1]) for agent in self.agents
        }

        # Base own observation size: solar irradiance, battery, backlog buckets..., sin_h, cos_h
        own_obs_dim = 5
        if self.use_deepsets_spatial:
            n_others = self.max_agents - 1
            obs_dim  = own_obs_dim + 4 * n_others + self._lstm_obs_dim
            self._observation_spaces = {
                agent: spaces.Box(
                    low=np.zeros(obs_dim, dtype=np.float32),
                    high=np.ones(obs_dim, dtype=np.float32) * 2.0,
                    dtype=np.float32
                )
                for agent in self.agents
            }
        elif self.use_cross_attention or self.use_deepsets:
            # ── Cross-attention / Deep Sets observation space ────────────────
            # [own (4)] + [others_flat  2*(max_agents-1)] + [mask (max_agents-1)]
            # Total: 4 + 3*(max_agents-1)
            n_others = self.max_agents - 1
            obs_dim  = own_obs_dim + 3 * n_others + self._lstm_obs_dim
            self._observation_spaces = {
                agent: spaces.Box(
                    low=np.zeros(obs_dim, dtype=np.float32),
                    high=np.ones(obs_dim, dtype=np.float32) * 2.0,
                    dtype=np.float32
                )
                for agent in self.agents
            }
        elif self.random_nodes > 0:
            # ── Random nodes observation space ──────────────────────────────────
            # [own (4)] + [random nodes (2 * random_nodes)]
            # Total: 4 + 2 * random_nodes
            obs_dim = own_obs_dim + 2 * self.random_nodes + self._lstm_obs_dim
            self._observation_spaces = {
                agent: spaces.Box(
                    low=np.zeros(obs_dim, dtype=np.float32),
                    high=np.ones(obs_dim, dtype=np.float32) * 2.0,
                    dtype=np.float32
                )
                for agent in self.agents
            }
        elif self.use_gossip:
            # ── Gossip observation space ─────────────────────────────────────────
            # [own (5)] + [gossip nodes (4 * gossip_state_nodes)]
            # Values per gossip node: normalized_id, battery, backlog, age
            # Total: 5 + 4 * gossip_state_nodes
            obs_dim = own_obs_dim + 4 * self.gossip_state_nodes + self._lstm_obs_dim
            self._observation_spaces = {
                agent: spaces.Box(
                    low=np.zeros(obs_dim, dtype=np.float32),
                    high=np.ones(obs_dim, dtype=np.float32) * 2.0,
                    dtype=np.float32
                )
                for agent in self.agents
            }

        else:
            # ── Original aggregated observation space ────────────────────────
            # [battery_i, backlog_i, sin_hour, cos_hour,
            #  min_batt, avg_batt, max_batt,
            #  min_back, avg_back, max_back]  →  10 values
            obs_dim = own_obs_dim + 6 + self._lstm_obs_dim
            self._observation_spaces = {
                agent: spaces.Box(
                    low=np.zeros(obs_dim, dtype=np.float32),
                    high=np.ones(obs_dim, dtype=np.float32) * 2.0,
                    dtype=np.float32
                )
                for agent in self.agents
            }

    def gen_obs(self):
        observations = {}
        for agent in range(self._num_agents):
            batt_i  = self.battery_energies[agent] / self.battery_capacities[agent]
            backlog = self.backlogs[agent] / self.max_storage
            # Cyclic hour-of-day encoding (invariant to episode length)
            seconds_into_day = (self.daily_timestamp * self._proc_interval) % (24 * 3600)
            hour = seconds_into_day / 3600.0  # 0.0 – 23.99
            sin_h = np.sin(hour / 23.0)
            cos_h = np.cos(hour / 23.0)
            solar_irradiance = self.get_irradiance_level(self.day, self.daily_timestamp, agent)  # Normalize by panel surface to get a per-unit value
            own     = [solar_irradiance, batt_i, backlog, sin_h, cos_h]

            other_agents = [j for j in range(self._num_agents) if j != agent]

            if self.use_deepsets_spatial:
                n_others  = self.max_agents - 1
                # others_flat: 3 * n_others values (battery_j, backlog_j, pos_index)
                others_flat = [0.0] * (3 * n_others)
                mask        = [0.0] * n_others

                for slot, j in enumerate(other_agents):
                    others_flat[3 * slot]     = self.battery_energies[j] / self.battery_capacities[j]
                    others_flat[3 * slot + 1] = sum(self.backlogs[j]) / self.max_storage
                    others_flat[3 * slot + 2] = j / max(1, self.max_agents - 1)  # Normalized index
                    mask[slot]                = 1.0
                
                obs = own + others_flat + mask
                
            elif self.use_cross_attention or self.use_deepsets:
                # ── Cross-attention or Deep Sets format ──────────────────────
                # Slot order: other_agents first (sorted), then padding.
                n_others  = self.max_agents - 1
                # others_flat: 2 * n_others values (battery_j, backlog_j)
                others_flat = [0.0] * (2 * n_others)
                mask        = [0.0] * n_others

                for slot, j in enumerate(other_agents):
                    others_flat[2 * slot]     = self.battery_energies[j] / self.battery_capacities[j]
                    others_flat[2 * slot + 1] = sum(self.backlogs[j]) / self.max_storage
                    mask[slot]                = 1.0
                # Remaining slots stay 0.0 (padding, mask=0)

                obs = own + others_flat + mask
            
            elif self.random_nodes > 0:
                # ── Random nodes format ──────────────────────────────────────
                sampled_others = self.np_random.choice(other_agents, self.random_nodes, replace=False)
                others_flat = []
                for j in sampled_others:
                    others_flat.append(self.battery_energies[j] / self.battery_capacities[j])
                    others_flat.append(sum(self.backlogs[j]) / self.max_storage)
                
                obs = own + others_flat

            elif self.use_gossip:
                # ── Gossip nodes format (Bounded size) ───────────────────────
                gossip_flat = []
                # Restrict to self.gossip_state_nodes and pad if necessary
                mem_nodes = list(self.gossip_memory[agent].items())
                
                if self.gossip_order == "priority":
                    # Ordina per "priorità" (Decrescente): batteria alta, backlog basso, età bassa
                    mem_nodes.sort(
                        key=lambda x: x[1]['battery'] - x[1]['backlog'] - ((self.daily_timestamp - x[1]['timestamp']) / self.max_day_steps),
                        reverse=True
                    )
                elif self.gossip_order == "timestamp":
                    # Preferisci i nodi con gossip piu' recente, cosi' i piu' aggiornati salgono in cima
                    mem_nodes.sort(key=lambda x: x[1]['timestamp'], reverse=True)
                else:
                    # Ordina per ID di default per consistenza della rete
                    mem_nodes.sort(key=lambda x: x[0])
                
                for i in range(self.gossip_state_nodes):
                    if i < len(mem_nodes):
                        node_id, info = mem_nodes[i]
                        age = (self.daily_timestamp - info['timestamp']) / self.max_day_steps
                        normalized_id = node_id / max(1, self.max_agents - 1)
                        gossip_flat.extend([normalized_id, info['battery'], info['backlog'], age])
                    else:
                        # Padding for empty slots
                        gossip_flat.extend([0.0, 0.0, 0.0, 1.0])
                
                obs = own + gossip_flat


            else:
                # ── Aggregated stats (original) ──────────────────────────────
                batts = [self.battery_energies[j] / self.battery_capacities[j] for j in other_agents]
                backs = [sum(self.backlogs[j]) / self.max_storage                    for j in other_agents]

                obs = own + [
                    min(batts), sum(batts) / len(batts), max(batts),
                    min(backs), sum(backs) / len(backs), max(backs),
                ]

            observations[agent] = np.array(obs, dtype=np.float32)

            # Append LSTM prediction features if enabled (real LSTM or demo oracle)
            if self._lstm_features_enabled:
                lstm_features = self.get_lstm_prediction_features(agent)
                observations[agent] = np.concatenate([observations[agent], lstm_features])

        return observations
