import numpy as np
import pytest

from custom_environment import CustomEnvironment


class TestEnv:
    def setup_method(self):
        irradiance_datapaths = [
            '../../../../../dataset/csv_41.89109712745386_12.503566993103867_fixed_23_180_PT15M_2024.csv',
            '../../../../../dataset/csv_41.89109712745386_12.503566993103867_fixed_23_180_PT15M_2024.csv',
            '../../../../../dataset/csv_41.89109712745386_12.503566993103867_fixed_23_180_PT15M_2024.csv',
            '../../../../../dataset/csv_41.89109712745386_12.503566993103867_fixed_23_180_PT15M_2024.csv',
            '../../../../../dataset/csv_41.89109712745386_12.503566993103867_fixed_23_180_PT15M_2024.csv',
            '../../../../../dataset/csv_41.89109712745386_12.503566993103867_fixed_23_180_PT15M_2024.csv',
            '../../../../../dataset/csv_41.89109712745386_12.503566993103867_fixed_23_180_PT15M_2024.csv',
            '../../../../../dataset/csv_41.89109712745386_12.503566993103867_fixed_23_180_PT15M_2024.csv',
            '../../../../../dataset/csv_41.89109712745386_12.503566993103867_fixed_23_180_PT15M_2024.csv',
            '../../../../../dataset/csv_41.89109712745386_12.503566993103867_fixed_23_180_PT15M_2024.csv'
            ]

        delta_time = 15 * 60
        proc_interval = 5 * 60
        proc_rate = 20
        arrival_rate = 15

        num_episodes = 4001

        eps_init = 1.0
        eps_fin = 0.05
        eps_dec = 0.9985

        num_agents = 5
        battery_capacities = [25, 100, 50, 37, 65]
        panel_surfaces = [1.0, 0.5, 0.75, 0.85, 0.65]

        power_idle = 2.6
        power_max = 6.0

        w = 1.0
        seed = "linbear" 

        self.env = CustomEnvironment(
        num_agents,
        irradiance_datapaths,
        delta_time,
        proc_interval,
        proc_rate,
        arrival_rate,
        battery_capacities,
        panel_surfaces,
        power_idle,
        power_max,
        w,
        seed)
        self.env.reset()

    def _idle_actions(self):
        return [[0.0, 0, int((i + 1) % self.env._num_agents), 0.0] for i in range(0, self.env._num_agents)]

    def test_reset_initializes_agents_and_state(self):
        observations, infos = self.env.reset()

        assert len(self.env.agents) == self.env._num_agents
        assert len(observations) == self.env._num_agents
        assert len(infos) == self.env._num_agents

        for agent_id in range(self.env._num_agents):
            assert self.env.states[agent_id][0] == 0.5
            assert self.env.states[agent_id][1] == 0
            assert self.env.states[agent_id][2] == 0.0
            assert len(observations[agent_id]) == 9

    def test_step_increments_timestep(self):
        timestep_before = self.env.timestep
        self.env.step(self._idle_actions())
        assert self.env.timestep == timestep_before + 1

    def test_step(self):
        # Do nothing: no local processing and no offloading.
        actions = self._idle_actions()
        obs, rewards, terminations, truncations, infos = self.env.step(actions)

        assert len(obs) == self.env._num_agents
        assert len(rewards) == self.env._num_agents
        assert len(terminations) == self.env._num_agents
        assert len(truncations) == self.env._num_agents
        assert len(infos) == self.env._num_agents

        for i in range(self.env._num_agents):
            assert 0 <= self.env.battery_energies[i] <= self.env.battery_capacities[i], "Battery energy out of bounds"
            assert len(obs[i]) == 9

    def test_step_truncates_at_max_steps(self):
        self.env.timestep = self.env.max_steps - 1

        _, _, _, truncations, _ = self.env.step(self._idle_actions())

        assert all(truncations.values())
        assert self.env.agents == []

    def test_step_local_processing_updates_battery_and_backlog(self):
        for i in range(self.env._num_agents):
            self.env.irradiance_arrays[i] = np.zeros_like(self.env.irradiance_arrays[i])

        self.env._arrival_rate = 0

        self.env.battery_energies = [0.0 for _ in range(self.env._num_agents)]
        self.env.backlogs = [0 for _ in range(self.env._num_agents)]

        self.env.battery_energies[0] = 10000.0
        self.env.backlogs[0] = 1000

        actions = self._idle_actions()
        actions[0] = [2.0, 0, 1, 0.0]

        self.env.step(actions)

        expected_processed = min(1000, int(2.0 * self.env._proc_interval))
        expected_energy = 10000.0 - (self.env.e_idle + (2.0 * self.env._proc_interval * self.env.e_frame))

        assert self.env.backlogs[0] == 1000 - expected_processed
        assert self.env.battery_energies[0] == pytest.approx(expected_energy)

    def test_step_reward_matches_local_reward_formula(self):
        for i in range(self.env._num_agents):
            self.env.irradiance_arrays[i] = np.zeros_like(self.env.irradiance_arrays[i])

        self.env._arrival_rate = 0

        self.env.battery_energies = [0.0 for _ in range(self.env._num_agents)]
        self.env.backlogs = [0 for _ in range(self.env._num_agents)]

        self.env.battery_energies[0] = 10000.0
        self.env.backlogs[0] = 1000

        actions = self._idle_actions()
        actions[0] = [2.0, 0, 1, 0.0]

        battery_capacity = self.env.battery_capacities[0]
        backlog = self.env.backlogs[0]
        actual_battery = self.env.battery_energies[0]
        processable = max(
            min(
                backlog,
                int((actual_battery - self.env.e_idle) / self.env.e_frame),
                self.env._processing_rate * self.env._proc_interval,
            ),
            0,
        )
        processed = min(actions[0][0] * self.env._proc_interval, processable)
        expected_reward = (processed / processable) + (actual_battery / battery_capacity) + (processed / backlog)

        _, rewards, _, _, _ = self.env.step(actions)

        assert rewards[0] == pytest.approx(expected_reward)

    def test_step_offloading_updates_sender_battery_and_backlog(self):
        for i in range(self.env._num_agents):
            self.env.irradiance_arrays[i] = np.zeros_like(self.env.irradiance_arrays[i])

        self.env._arrival_rate = 0

        self.env.battery_energies = [0.0 for _ in range(self.env._num_agents)]
        self.env.backlogs = [0 for _ in range(self.env._num_agents)]

        self.env.battery_energies[0] = 10000.0
        self.env.battery_energies[1] = 10000.0
        self.env.backlogs[0] = 100
        self.env.backlogs[1] = 0

        actions = self._idle_actions()
        actions[0] = [0.0, 1, 1, 2.0]
        actions[1] = [0.0, 2, 0, 2.0]

        self.env.step(actions)

        expected_sender_after_local = 10000.0 - self.env.e_idle
        expected_receiver_after_local = 10000.0 - self.env.e_idle
        expected_sender_offloading_energy = 2.0 * self.env.e_tx_rx * self.env._proc_interval

        assert self.env.backlogs[0] == 0
        assert self.env.battery_energies[0] == pytest.approx(expected_sender_after_local - expected_sender_offloading_energy)
        assert self.env.battery_energies[1] == pytest.approx(expected_receiver_after_local)
