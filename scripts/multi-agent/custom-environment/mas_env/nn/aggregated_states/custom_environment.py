import os
import sys

from gymnasium import spaces
from copy import copy

import numpy as np
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from base_envirionment import BaseEnvironment

class CustomEnvironment(BaseEnvironment):

    def __init__(self, num_agents, irradiance_datapaths, delta_time, proc_interval, proc_rate, arr_rate, batteries, panel_surfaces, power_idle, power_max, w, seed, use_cross_attention=False, use_deepsets=False, use_deepsets_spatial=False, max_agents=None, random_nodes=0, use_gossip=False, gossip_interval=5, gossip_targets=2, gossip_state_nodes=3, termination_mode="early", battery_hard_threshold=0.0, use_random_battery=False, use_lstm_prediction=False, use_lstm_prediction_demo=False, disable_offloading=False):
        super().__init__(num_agents, irradiance_datapaths, delta_time, proc_interval, proc_rate, arr_rate, batteries, panel_surfaces, power_idle, power_max, w, seed, use_gossip, gossip_interval, gossip_targets, gossip_state_nodes, battery_hard_threshold=battery_hard_threshold, use_random_battery=use_random_battery, use_lstm_prediction=use_lstm_prediction, use_lstm_prediction_demo=use_lstm_prediction_demo, disable_offloading=disable_offloading)
        self.termination_mode = termination_mode

        self.use_cross_attention = use_cross_attention
        self.use_deepsets = use_deepsets
        self.use_deepsets_spatial = use_deepsets_spatial
        self.random_nodes = random_nodes
        # max_agents defines the padded observation size; defaults to num_agents.
        self.max_agents = max_agents if max_agents is not None else num_agents
        assert self.max_agents >= num_agents, "max_agents must be >= num_agents"
        if self.random_nodes > 0:
            assert self.random_nodes < self._num_agents, "random_nodes must be less than the total number of agents"

        # LSTM prediction adds 24×4 = 96 extra observation values
        self._lstm_obs_dim = 96 if self._lstm_features_enabled else 0

        self._action_spaces = {
            agent: spaces.MultiDiscrete([self._processing_rate + 1, 2, self._num_agents, self._processing_rate + 1]) for agent in self.possible_agents
        }

        if self.use_deepsets_spatial:
            n_others = self.max_agents - 1
            obs_dim  = 4 + 4 * n_others + self._lstm_obs_dim
            self._observation_spaces = {
                agent: spaces.Box(
                    low=np.zeros(obs_dim, dtype=np.float32),
                    high=np.ones(obs_dim, dtype=np.float32) * 2.0,
                    dtype=np.float32
                )
                for agent in self.possible_agents
            }
        elif self.use_cross_attention or self.use_deepsets:
            # ── Cross-attention / Deep Sets observation space ────────────────
            # [own (4)] + [others_flat  2*(max_agents-1)] + [mask (max_agents-1)]
            # Total: 4 + 3*(max_agents-1)
            n_others = self.max_agents - 1
            obs_dim  = 4 + 3 * n_others + self._lstm_obs_dim
            self._observation_spaces = {
                agent: spaces.Box(
                    low=np.zeros(obs_dim, dtype=np.float32),
                    high=np.ones(obs_dim, dtype=np.float32) * 2.0,
                    dtype=np.float32
                )
                for agent in self.possible_agents
            }
        elif self.random_nodes > 0:
            # ── Random nodes observation space ──────────────────────────────────
            # [own (4)] + [random nodes (2 * random_nodes)]
            # Total: 4 + 2 * random_nodes
            obs_dim = 4 + 2 * self.random_nodes + self._lstm_obs_dim
            self._observation_spaces = {
                agent: spaces.Box(
                    low=np.zeros(obs_dim, dtype=np.float32),
                    high=np.ones(obs_dim, dtype=np.float32) * 2.0,
                    dtype=np.float32
                )
                for agent in self.possible_agents
            }
        elif self.use_gossip:
            # ── Gossip observation space ─────────────────────────────────────────
            # [own (4)] + [gossip nodes (3 * max_agents)]
            # Values per gossip node: battery, backlog, age
            # Total: 4 + 3 * max_agents
            obs_dim = 4 + 3 * self.max_agents + self._lstm_obs_dim
            self._observation_spaces = {
                agent: spaces.Box(
                    low=np.zeros(obs_dim, dtype=np.float32),
                    high=np.ones(obs_dim, dtype=np.float32) * 2.0,
                    dtype=np.float32
                )
                for agent in self.possible_agents
            }

        else:
            # ── Original aggregated observation space ────────────────────────
            # [battery_i, backlog_i, sin_hour, cos_hour,
            #  min_batt, avg_batt, max_batt,
            #  min_back, avg_back, max_back]  →  10 values
            obs_dim = 10 + self._lstm_obs_dim
            self._observation_spaces = {
                agent: spaces.Box(
                    low=np.zeros(obs_dim, dtype=np.float32),
                    high=np.ones(obs_dim, dtype=np.float32) * 2.0,
                    dtype=np.float32
                )
                for agent in self.possible_agents
            }

    def gen_obs(self):
        observations = {}
        for agent in range(self._num_agents):
            batt_i  = self.battery_energies[agent] / self.battery_capacities[agent]
            back_i  = self.backlogs[agent] / self.max_storage
            # Cyclic hour-of-day encoding (invariant to episode length)
            seconds_into_day = (self.timestep * self._proc_interval) % (24 * 3600)
            hour = seconds_into_day / 3600.0  # 0.0 – 23.99
            sin_h = np.sin(hour / 23.0)
            cos_h = np.cos(hour / 23.0)
            own     = [batt_i, back_i, sin_h, cos_h]

            other_agents = [j for j in range(self._num_agents) if j != agent]

            if self.use_deepsets_spatial:
                n_others  = self.max_agents - 1
                # others_flat: 3 * n_others values (battery_j, backlog_j, pos_index)
                others_flat = [0.0] * (3 * n_others)
                mask        = [0.0] * n_others

                for slot, j in enumerate(other_agents):
                    others_flat[3 * slot]     = self.battery_energies[j] / self.battery_capacities[j]
                    others_flat[3 * slot + 1] = self.backlogs[j] / self.max_storage
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
                    others_flat[2 * slot + 1] = self.backlogs[j] / self.max_storage
                    mask[slot]                = 1.0
                # Remaining slots stay 0.0 (padding, mask=0)

                obs = own + others_flat + mask
            
            elif self.random_nodes > 0:
                # ── Random nodes format ──────────────────────────────────────
                sampled_others = np.random.choice(other_agents, self.random_nodes, replace=False)
                others_flat = []
                for j in sampled_others:
                    others_flat.append(self.battery_energies[j] / self.battery_capacities[j])
                    others_flat.append(self.backlogs[j] / self.max_storage)
                
                obs = own + others_flat

            elif self.use_gossip:
                # ── Gossip nodes format (Global Masked State) ────────────────
                gossip_flat = []
                # Allocate exactly max_agents slots. Slot j corresponds to Target j.
                for j in range(self.max_agents):
                    if j == agent:
                        # Own node slot: we can pad it out to ignore it
                        gossip_flat.extend([0.0, 0.0, 1.0])
                    elif j in self.gossip_memory[agent]:
                        info = self.gossip_memory[agent][j]
                        age = (self.timestep - info['timestamp']) / self.max_steps
                        gossip_flat.extend([info['battery'], info['backlog'], age])
                    else:
                        # Node not in memory (or virtual padding node)
                        gossip_flat.extend([0.0, 0.0, 1.0])
                
                obs = own + gossip_flat


            else:
                # ── Aggregated stats (original) ──────────────────────────────
                batts = [self.battery_energies[j] / self.battery_capacities[j] for j in other_agents]
                backs = [self.backlogs[j] / self.max_storage                    for j in other_agents]

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
