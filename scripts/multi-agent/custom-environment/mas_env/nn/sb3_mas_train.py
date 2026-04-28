import os
import time
import numpy as np
import pandas as pd
import torch as th
import matplotlib.pyplot as plt
from pathlib import Path
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback

import gymnasium as gym
import multiprocessing as mp
import copy

def _worker(remote, parent_remote, env):
    parent_remote.close()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                obs, reward, term, trunc, info = env.step(data)
                remote.send((obs, reward, term, trunc, info))
            elif cmd == 'reset':
                obs, info = env.reset()
                remote.send((obs, info))
            elif cmd == 'close':
                remote.close()
                break
        except EOFError:
            break

class AsyncMultiAgentEnvs:
    def __init__(self, env, num_envs):
        self.num_envs = num_envs
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(num_envs)])
        self.processes = []
        for work_remote, remote in zip(self.work_remotes, self.remotes):
            process = mp.Process(target=_worker, args=(work_remote, remote, copy.deepcopy(env)))
            process.daemon = True
            process.start()
            self.processes.append(process)
            work_remote.close()
        self.max_steps = env.max_steps
        self.episode = env.episode
        self.total_frames_processed = env.total_frames_processed
        self.irradiance_arrays = env.irradiance_arrays
        
    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        obs_batch, infos_batch = zip(*results)
        return list(obs_batch), list(infos_batch)

    def step(self, actions_list):
        for remote, action in zip(self.remotes, actions_list):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        obs_batch, rewards_batch, terminations_batch, truncations_batch, infos_batch = zip(*results)
        return list(obs_batch), list(rewards_batch), list(terminations_batch), list(truncations_batch), list(infos_batch)

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for process in self.processes:
            process.join()

class EnvWrapper(gym.Env):
    """
    A wrapper to make the multi-agent environment look like a single-agent Gymnasium environment.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, env, agent_id):
        super().__init__()
        self.env = env
        self.agent_id = agent_id
        self.observation_space = env.observation_space(agent_id)
        self.action_space = env.action_space(agent_id)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs, info = self.env.reset(seed=seed, options=options)
        return obs[self.agent_id], info

    def render(self):
        pass

    def step(self, action):
        raise NotImplementedError("Use the trainer's step logic for multi-agent interaction.")

class SB3_MAS_Train:
    def __init__(self, num_agents, num_episodes, irradiance_datapaths, delta_time, proc_interval, proc_rate, arrival_rate, eps_init, eps_fin, eps_dec, battery_capacities, panel_surfaces, power_idle, power_max, train_freq, w, mode, batch_size, seed, env, save_path, num_envs=4, algo="PPO", ppo_initial_lr=3e-4, ppo_final_lr=3e-5):
        self.num_agents = num_agents
        self.num_episodes = num_episodes
        self.max_steps = env.max_steps
        self.eval_env = env
        self.num_envs = num_envs if algo.upper() == "PPO" else 1
        
        if self.num_envs > 1:
            self.env = AsyncMultiAgentEnvs(env, self.num_envs)
        else:
            self.env = env
            
        self.save_path = save_path
        self.algo = algo.upper()
        self.batch_size = batch_size
        self.random_seed = seed
        self.seed = seed
        self.proc_interval = proc_interval
        self.proc_rate = proc_rate
        self.w = w
        self.train_all_mode = 1
        self.battery_capacities = battery_capacities
        self.panel_surfaces = panel_surfaces
        self.train_freq = train_freq
        self.ppo_initial_lr = ppo_initial_lr
        self.ppo_final_lr = ppo_final_lr
        
        # Hyperparameters
        self.eps_init = eps_init
        self.eps_fin = eps_fin
        self.eps_dec = eps_dec
        self.eps = eps_init
        
        os.makedirs(self.save_path, exist_ok=True)
        
        device = "cuda" if th.cuda.is_available() else "cpu"
        
        if self.algo == "DQN":
            self.models = {i: DQN("MlpPolicy", EnvWrapper(self.eval_env, i), learning_rate=1e-4, buffer_size=100000, learning_starts=1000, batch_size=batch_size, train_freq=train_freq, target_update_interval=1000, exploration_fraction=0.5, verbose=0, tensorboard_log=None, device=device) for i in range(num_agents)}
        else:
            from stable_baselines3.common.vec_env import DummyVecEnv
            lr_schedule = self._linear_lr_schedule(self.ppo_initial_lr, self.ppo_final_lr)
            self.models = {}
            for i in range(num_agents):
                if self.num_envs > 1:
                    # Each sub-env gets its own deepcopy so DummyVecEnv's
                    # internal reset() calls don't corrupt eval_env's state
                    # (episode counter, irradiance index, etc.)
                    sub_envs = [copy.deepcopy(self.eval_env) for _ in range(self.num_envs)]
                    venv = DummyVecEnv([(lambda e=sub_envs[j], ag=i: EnvWrapper(e, ag))
                                        for j in range(self.num_envs)])
                else:
                    venv = EnvWrapper(self.eval_env, i)
                
                self.models[i] = PPO(
                    "MlpPolicy",
                    venv,
                    learning_rate=lr_schedule,
                    n_steps=512,
                    batch_size=batch_size,
                    n_epochs=10,
                    verbose=0,
                    device=device,
                )

        for i in range(num_agents):
            self.models[i].set_logger(configure(None, ["stdout", "csv"]))

    def decode(self, action):
        # Handle potential array/tensor shapes
        if hasattr(action, 'flatten'):
            action = action.flatten()[0]
        elif isinstance(action, (list, np.ndarray)):
            action = action[0]
        
        action = int(action)
        fti = action % 21
        rem = action // 21
        oti = rem % 3
        rem = rem // 3
        target = rem % self.num_agents
        off_rate = rem // self.num_agents
        return [float(fti), int(oti), int(target), float(off_rate)]

    @staticmethod
    def _linear_lr_schedule(initial_lr, final_lr):
        def schedule(progress_remaining):
            # progress_remaining goes from 1.0 to 0.0 in SB3.
            return final_lr + (initial_lr - final_lr) * progress_remaining

        return schedule

    def train(self):
        print(f"Starting {self.algo} Training for {self.num_episodes} episodes...")
        
        rewards_history = {i: [] for i in range(self.num_agents)}
        steps_history = []
        
        for ep in range(self.num_episodes):
            obs, info = self.env.reset()
            done = False
            ep_reward = {i: 0 for i in range(self.num_agents)}
            step_count = 0
            
            while not done:
                actions_encoded = {}
                actions = {}
                
                if self.num_envs == 1:
                    for i in range(self.num_agents):
                        if self.algo == "DQN":
                            # Manually handle epsilon-greedy for DQN in multi-agent loop
                            if np.random.rand() < self.eps:
                                act = self.models[i].action_space.sample()
                            else:
                                act, _ = self.models[i].predict(obs[i], deterministic=False)
                        else:
                            # PPO uses its internal distribution
                            obs_tensor = th.as_tensor(obs[i]).to(self.models[i].device).unsqueeze(0)
                            with th.no_grad():
                                act, value, log_prob = self.models[i].policy(obs_tensor)
                            act = act.cpu().numpy()[0]
                            # Store these for the rollout buffer
                            self.models[i]._last_value = value
                            self.models[i]._last_log_prob = log_prob
                            self.models[i]._last_obs = obs[i]

                        actions_encoded[i] = act
                        actions[i] = self.decode(act)
                    
                    next_obs, rewards, terminations, truncations, infos = self.eval_env.step(actions)
                else:
                    # Parallel environments for PPO
                    actions_list = [{} for _ in range(self.num_envs)]
                    for i in range(self.num_agents):
                        obs_i = np.array([o[i] for o in obs])
                        obs_tensor = th.as_tensor(obs_i).to(self.models[i].device)
                        with th.no_grad():
                            act, value, log_prob = self.models[i].policy(obs_tensor)
                        
                        act = act.cpu().numpy()
                        self.models[i]._last_value = value
                        self.models[i]._last_log_prob = log_prob
                        self.models[i]._last_obs = obs_i
                        
                        actions_encoded[i] = act
                        for env_idx in range(self.num_envs):
                            actions_list[env_idx][i] = self.decode(act[env_idx])
                            
                    next_obs, rewards, terminations, truncations, infos = self.env.step(actions_list)
                
                for i in range(self.num_agents):
                    if self.num_envs == 1:
                        is_done = terminations[i] or truncations[i]
                        ep_reward[i] += rewards[i]
                        
                        if self.algo == "DQN":
                            self.models[i].replay_buffer.add(obs[i], next_obs[i], np.array([actions_encoded[i]]), rewards[i], is_done, [{}])
                            if self.models[i].num_timesteps > 1000 and self.models[i].num_timesteps % self.train_freq == 0:
                                self.models[i].train(gradient_steps=1, batch_size=self.batch_size)
                        else:
                            # PPO add to buffer
                            self.models[i].rollout_buffer.add(
                                self.models[i]._last_obs, 
                                np.array([actions_encoded[i]]), 
                                rewards[i], 
                                np.array([step_count == 0]), # episode start
                                self.models[i]._last_value, 
                                self.models[i]._last_log_prob
                            )
                            if self.models[i].rollout_buffer.full:
                                last_obs_tensor = th.as_tensor(next_obs[i]).to(self.models[i].device).unsqueeze(0)
                                with th.no_grad():
                                    last_val = self.models[i].policy.predict_values(last_obs_tensor)
                                self.models[i].rollout_buffer.compute_returns_and_advantage(last_val, np.array([is_done]))
                                
                                # Update progress for LR schedule
                                progress = 1.0 - (ep / self.num_episodes)
                                self.models[i]._current_progress_remaining = progress
                                
                                self.models[i].train()
                                self.models[i].rollout_buffer.reset()
                        
                        self.models[i].num_timesteps += 1
                    else:
                        is_done_arr = np.array([terminations[env_idx][i] or truncations[env_idx][i] for env_idx in range(self.num_envs)])
                        rewards_arr = np.array([rewards[env_idx][i] for env_idx in range(self.num_envs)])
                        ep_reward[i] += rewards_arr[0]
                        
                        action_shape = actions_encoded[i].reshape(self.num_envs, 1) if len(actions_encoded[i].shape) == 1 else actions_encoded[i]
                        
                        self.models[i].rollout_buffer.add(
                            self.models[i]._last_obs, 
                            action_shape, 
                            rewards_arr, 
                            np.array([step_count == 0] * self.num_envs), 
                            self.models[i]._last_value, 
                            self.models[i]._last_log_prob
                        )
                        
                        self.models[i].num_timesteps += self.num_envs
                        
                        if self.models[i].rollout_buffer.full:
                            next_obs_i = np.array([n_o[i] for n_o in next_obs])
                            last_obs_tensor = th.as_tensor(next_obs_i).to(self.models[i].device)
                            with th.no_grad():
                                last_val = self.models[i].policy.predict_values(last_obs_tensor).flatten()
                            self.models[i].rollout_buffer.compute_returns_and_advantage(last_val, is_done_arr)
                            
                            progress = 1.0 - (ep / self.num_episodes)
                            self.models[i]._current_progress_remaining = progress
                            
                            self.models[i].train()
                            self.models[i].rollout_buffer.reset()
                
                obs = next_obs
                step_count += 1
                
                if self.num_envs == 1:
                    done = any(terminations.values()) or all(truncations.values())
                else:
                    done = any(terminations[0].values()) or all(truncations[0].values())

            # Update epsilon for DQN
            self.eps = max(self.eps_fin, self.eps_init - ep * (self.eps_init - self.eps_fin) / (self.num_episodes * self.eps_dec))
            
            steps_history.append(step_count)
            for i in range(self.num_agents):
                rewards_history[i].append(ep_reward[i])
            
            if ep % 20 == 0:
                if self.num_envs == 1:
                    print(f"Ep {ep}/{self.num_episodes} - Steps: {step_count} - Rewards: {ep_reward} - Obs: {[obs[i].tolist() for i in range(self.num_agents)]}")
                else:
                    print(f"Ep {ep}/{self.num_episodes} - Steps: {step_count} - Rewards: {ep_reward} - Obs: {[obs[0][i].tolist() for i in range(self.num_agents)]}")
            
            if ep % 500 == 0:
                self.save_models()
                
        if self.num_envs > 1:
            self.env.close()
            
        self.save_models()

    def save_models(self):
        for i in range(self.num_agents):
            self.models[i].save(os.path.join(self.save_path, f"agent_{i}"))

    def _load_latest_model_for_agent(self, agent_id):
        model_dir = Path(self.save_path)
        pattern = f"agent_{agent_id}*.zip"
        file_py = list(model_dir.glob(pattern))

        if not file_py:
            return False

        most_recent = max(file_py, key=lambda f: f.stat().st_mtime)
        print(f"Loading agent {agent_id}: {most_recent}")

        wrapped_env = EnvWrapper(self.eval_env, agent_id)

        if self.algo == "DQN":
            model = DQN.load(str(most_recent), env=wrapped_env)
        else:
            # Use env= directly in load() to avoid n_envs mismatch
            # (model was trained with num_envs>1 but evaluation uses 1 env)
            model = PPO.load(str(most_recent), env=wrapped_env)

        self.models[agent_id] = model
        return True

    def evaluate(self, model_paths: str = None):
        if model_paths:
            self.save_path = model_paths

        for i in range(self.num_agents):
            loaded = self._load_latest_model_for_agent(i)
            if not loaded:
                batt = int(self.battery_capacities[i])
                print(f"No saved model found for agent {i} ({batt}Wh), keeping fresh model")
        
        obs = self.eval_env.reset(self.random_seed)[0]
        agents_logs = {agent_id: {"battery": [], "processing": [], "panel_energy": [], "backlog": [], "state": [], "processed_frames": [], "hs_counter": [], "offloading": [], "reward": []} for agent_id in range(self.num_agents)}
        terminate = False
        total_rewards = {i: 0 for i in range(self.num_agents)}
        while not terminate:
            actions = {}
            for agent_id in range(self.num_agents):
                agent_obs = obs[agent_id]
                action, _ = self.models[agent_id].predict(agent_obs, deterministic=True)
                actions[agent_id] = self.decode(action)

            next_obs, rewards, terminations, truncations, infos = self.eval_env.step(actions)
            for agent_id in range(self.num_agents):

                total_rewards[agent_id] += rewards[agent_id]

                print(f"Agent {agent_id} - State: {obs[agent_id][2]})")
                agents_logs[agent_id]["battery"].append(obs[agent_id][0])
                agents_logs[agent_id]["panel_energy"].append(infos[agent_id]["panel_energy"])
                agents_logs[agent_id]["processing"].append(actions[agent_id][0]/20)
                agents_logs[agent_id]["backlog"].append(obs[agent_id][1])
                agents_logs[agent_id]["reward"].append(rewards[agent_id])

                done = terminations[agent_id] or truncations[agent_id]
                if done:
                    terminate = True

            obs = next_obs
        
        window_size = 50  # Più è alto, più appiattisce
        
        # Plot battery levels and processing decisions for all agents
        plt.figure(figsize=(12, 24))
        for agent_id in range(self.num_agents):
            # processing_smoth = pd.Series(agents_logs[agent_id]['processing']).rolling(window=window_size, center=True).mean()
            #backlog_smoth = agents_logs[agent_id]['backlog'] #pd.Series(agents_logs[agent_id]['backlog']).rolling(window=window_size, center=True).mean()

            plt.subplot(self.num_agents, 1, agent_id + 1)
            plt.plot(agents_logs[agent_id]['battery'], label='Battery Level')
            plt.plot(agents_logs[agent_id]['processing'], label='Processing Decision')
            plt.plot(agents_logs[agent_id]['panel_energy'], label='Panel Energy')
            plt.plot(agents_logs[agent_id]['backlog'], label='Backlog')
            #plt.plot(agents_logs[agent_id]['reward'], label='Reward')

            # Color area when battery is 0
            threshold = 0
            plt.fill_between(range(len(agents_logs[agent_id]['battery'])),
                            0,
                            1,
                            where=(np.array(agents_logs[agent_id]['battery']) <= threshold), 
                            color='red', alpha=0.2, label='Battery Depleted')
            

            plt.title(f'Agent {agent_id} Evaluation')
            plt.xlabel('Timestep')
            plt.ylabel('Value')
            plt.legend()
            plt.grid()
        plt.tight_layout()
        file_name = f"evaluation_{self.num_episodes-1}_{self.eval_env.episode}_{self.proc_interval}_{self.w}_{self.num_agents}agents_{self.train_all_mode}.png"
        file_name = os.path.join(self.save_path, file_name)
        print("Saved in ", file_name)
        plt.savefig(file_name)
        plt.close() 

        print("Total processed frames during evaluation:", self.eval_env.total_frames_processed)
        print("Total rewards during evaluation:", total_rewards)
        
        print("\n--- Total Solar Energy Accumulated ---")
        episode = self.eval_env.episode
        max_steps = int(self.eval_env.max_steps)
        for agent_id in range(self.num_agents):
            total_solar_joules = 0.0
            for t in range(max_steps):
                idx = (episode * max_steps) + t
                if idx < len(self.eval_env.irradiance_arrays[agent_id]):
                    irradiance = self.eval_env.irradiance_arrays[agent_id][idx]
                    total_solar_joules += irradiance * self.panel_surfaces[agent_id] * 0.2 * self.proc_interval
            print(f"Agent {agent_id}: {total_solar_joules:.2f} Joules ({total_solar_joules/3600:.2f} Wh)")
        print("--------------------------------------")


    def plot_results(self, rewards_history, steps_history):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        for i in range(self.num_agents):
            plt.plot(rewards_history[i], label=f'Agent {i}')
        plt.title('Rewards')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(steps_history)
        plt.title('Steps Survival')
        plt.savefig(os.path.join(self.save_path, 'training_results.png'))
        plt.close()