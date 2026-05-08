import os
import sys

from gymnasium import spaces
from copy import copy

import numpy as np
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from base_envirionment import BaseEnvironment

class CustomEnvironment(BaseEnvironment):

    def __init__(self, num_agents, irradiance_datapaths, delta_time, proc_interval, proc_rate, arr_rate, batteries, panel_surfaces, power_idle, power_max, w, seed, use_cross_attention=False, max_agents=None):
        super().__init__(num_agents, irradiance_datapaths, delta_time, proc_interval, proc_rate, arr_rate, batteries, panel_surfaces, power_idle, power_max, w, seed)

        self.use_cross_attention = use_cross_attention
        # max_agents defines the padded observation size; defaults to num_agents.
        self.max_agents = max_agents if max_agents is not None else num_agents
        assert self.max_agents >= num_agents, "max_agents must be >= num_agents"

        self._action_spaces = {
            agent: spaces.MultiDiscrete([self._processing_rate + 1, 2, self._num_agents, self._processing_rate + 1]) for agent in self.possible_agents
        }

        if not use_cross_attention:
            # ── Original aggregated observation space ────────────────────────
            # [battery_i, backlog_i, timestep,
            #  min_batt, avg_batt, max_batt,
            #  min_back, avg_back, max_back]  →  9 values
            self._observation_spaces = {
                agent: spaces.Box(
                    low=np.zeros(9, dtype=np.float32),
                    high=np.ones(9, dtype=np.float32),
                    dtype=np.float32
                )
                for agent in self.possible_agents
            }
        else:
            # ── Cross-attention observation space ────────────────────────────
            # [own (3)] + [others_flat  2*(max_agents-1)] + [mask (max_agents-1)]
            # Total: 3 + 3*(max_agents-1)
            n_others = self.max_agents - 1
            obs_dim  = 3 + 3 * n_others
            self._observation_spaces = {
                agent: spaces.Box(
                    low=np.zeros(obs_dim, dtype=np.float32),
                    high=np.ones(obs_dim, dtype=np.float32),
                    dtype=np.float32
                )
                for agent in self.possible_agents
            }

    def gen_obs(self):
        observations = {}
        for agent in range(self._num_agents):
            batt_i  = self.battery_energies[agent] / self.battery_capacities[agent]
            back_i  = self.backlogs[agent] / self.max_storage
            time_i  = self.timestep / self.max_steps
            own     = [batt_i, back_i, time_i]

            other_agents = [j for j in range(self._num_agents) if j != agent]

            if not self.use_cross_attention:
                # ── Aggregated stats (original) ──────────────────────────────
                batts = [self.battery_energies[j] / self.battery_capacities[j] for j in other_agents]
                backs = [self.backlogs[j] / self.max_storage                    for j in other_agents]

                obs = own + [
                    min(batts), sum(batts) / len(batts), max(batts),
                    min(backs), sum(backs) / len(backs), max(backs),
                ]
            else:
                # ── Cross-attention format ───────────────────────────────────
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

            observations[agent] = np.array(obs, dtype=np.float32)

        return observations
