from stable_baselines3 import PPO

from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from custom_environment import CustomEnvironment
from env_wrapper import EnvWrapper, make_parallel_single_agent_env
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
from torch.utils.tensorboard import SummaryWriter

import time
import csv
import os
import glob

from pathlib import Path
from datetime import datetime
import gc


class SB3_MAS_Train_PPO:
    def __init__(self, 
                 num_agents,
                 num_episodes,
                 irradiance_datapaths,
                 delta_time,
                 proc_interval,
                 proc_rate,
                 arrival_rate,
                 battery_capacities,
                 panel_surfaces,
                 power_idle,
                 power_max,
                 train_freq,
                 w,
                 mode,
                 batch_size,
                 smart_node,
                 seed,
                 parallel_envs=4,
                 train_all_mode=1,
                 num_rounds=1
                ):
        
        self.num_agents = num_agents
        self.num_episodes = num_episodes
        self.num_rounds = num_rounds
        
        self.delta_time = delta_time
        self.proc_interval = proc_interval
        self.proc_rate = proc_rate
        
        self.battery_capacities = battery_capacities
        self.panel_surfaces = panel_surfaces
        
        self.power_idle = power_idle
        self.power_max = power_max
        
        self.seed = seed
        self.w = w
        self.mode = mode
        
        self.train_freq = train_freq
        self.grad_steps = 1
        self.batch_size = batch_size
        self.parallel_envs = max(1, int(parallel_envs))
        self.parallel_envs_round_mode = 1
        
        self.smart_node = smart_node
        self.train_all_mode = train_all_mode
        print(smart_node)

        if self.train_all_mode == 2 and self.parallel_envs > self.parallel_envs_round_mode:
            print(
                f"Mode 2 RAM guard: forcing {self.parallel_envs_round_mode} env per turn "
                f"instead of {self.parallel_envs}"
            )
            self.parallel_envs = self.parallel_envs_round_mode

        self.random_seed = self._normalize_seed(seed)
        
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
            seed,
            rng_seed=self.random_seed,
        )
        
        self.max_steps = self.env.max_steps
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"CUDA Available: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            print(f"CUDA NOT Available - using CPU")

        run_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        
        self.env_kwargs = dict(
            num_agents=num_agents,
            irradiance_datapaths=irradiance_datapaths,
            delta_time=delta_time,
            proc_interval=proc_interval,
            proc_rate=proc_rate,
            arr_rate=arrival_rate,
            batteries=battery_capacities,
            panel_surfaces=panel_surfaces,
            power_idle=power_idle,
            power_max=power_max,
            w=w,
            seed=seed,
        )

        self.models = {i : PPO(
                policy="MlpPolicy",
            env=self._build_vec_env(i),
                learning_rate=0.0003,
                n_steps=512,
                batch_size=self.batch_size,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.001,
                vf_coef=0.5,
                max_grad_norm=0.5,
                                tensorboard_log=f"runs/ppo_{run_name}/agent_{i}",
                policy_kwargs=dict(net_arch=[128, 128]),
                                verbose=1,
                device=mode,
                _init_setup_model=True
            )
          for i in range(0, num_agents)}
        
        for i in range(num_agents):
            self.models[i].set_logger(configure(None, ["stdout"]))
            print(f"Agent {i} device: {self.models[i].device}")

        self.models_folder = Path("./models/once_a_time_ppo")

    def _normalize_seed(self, seed_value):
        if isinstance(seed_value, int):
            return seed_value
        if isinstance(seed_value, str):
            return sum((idx + 1) * ord(ch) for idx, ch in enumerate(seed_value)) % (2 ** 31)
        return 42

    def _make_env(self, agent_id, env_rank):
        return self._make_env_with_teammates(agent_id, env_rank, teammate_model_paths=None)

    def _make_env_with_teammates(self, agent_id, env_rank, teammate_model_paths):
        env_seed = self.random_seed + (agent_id * 1000) + env_rank
        return partial(
            make_parallel_single_agent_env,
            env_kwargs={**self.env_kwargs, "rng_seed": env_seed},
            agent_id=agent_id,
            rng_seed=env_seed,
            teammate_model_paths=teammate_model_paths,
            teammate_device="cpu",
        )

    def _build_vec_env(self, agent_id, teammate_model_paths=None, num_envs=None):
        env_count = self.parallel_envs if num_envs is None else max(1, int(num_envs))
        env_factories = [
            self._make_env_with_teammates(agent_id, env_rank, teammate_model_paths)
            for env_rank in range(env_count)
        ]

        if env_count == 1:
            vec_env = DummyVecEnv(env_factories)
        else:
            vec_env = SubprocVecEnv(env_factories)

        return VecMonitor(vec_env)

    def _latest_model_path_for_agent(self, agent_id):
        pattern = f"PPO_agent{agent_id}_*.zip"
        candidates = list(self.models_folder.glob(pattern))
        if not candidates:
            return None
        return str(max(candidates, key=lambda f: f.stat().st_mtime))

    def _teammate_model_paths_for(self, target_agent_id):
        paths = {}
        for agent_id in range(self.num_agents):
            if agent_id == target_agent_id:
                continue
            paths[agent_id] = self._latest_model_path_for_agent(agent_id)
        return paths


    def _load_latest_model_for_agent(self, agent_id):
        pattern = f"PPO_agent{agent_id}_*.zip"
        file_py = list(self.models_folder.glob(pattern))

        if not file_py:
            return False

        most_recent = max(file_py, key=lambda f: f.stat().st_mtime)
        print(f"Loading agent {agent_id}: {most_recent}")

        venv = self._build_vec_env(agent_id)
        model = PPO.load(str(most_recent), env=venv, device=self.mode)
        self.models[agent_id] = model
        return True


    def _save_model_for_agent(self, agent_id):
        os.makedirs(self.models_folder, exist_ok=True)
        model_filename = (
            f"PPO_agent{agent_id}_"
            f"{self.battery_capacities[agent_id]}Wh_"
            f"{self.num_episodes - 1}_{self.env.episode}_{self.proc_interval}_{self.num_agents}agents_{self.mode}"
        )
        self.models[agent_id].save(str(self.models_folder / model_filename))
        print(f"saved model: {self.models_folder / model_filename}.zip")
        
    
    def plot_rewards(self, folder_path, rewards):
    
    # print(rewards)
    
        window = 10
        plt.suptitle("Multi-agent : rewards")
        plt.title(f"P_i = {self.power_idle}, P_f = {self.power_max}, fps = {self.proc_rate}, interval: {self.proc_interval}s")
        
        plt.xlabel("Episodes")
        plt.ylabel("Rewards")
        
        for i in range(0, self.num_agents):
            # print(rewards[i])
            plt.plot(range(window - 1, len(rewards[i])), np.convolve(rewards[i], np.ones(window)/window, mode='valid'), label = f"smooth {self.battery_capacities[i]}Wh", alpha = 1.0)
            # plt.plot(rewards[i], label = f"raw {self.battery_capacities[i]}Wh", alpha = 0.3)
        
        plt.grid()
        # plt.ylim(-10, 500)
        plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
        plt.tight_layout()
        plt.savefig(f"./{folder_path}/rewards_plot_{self.num_episodes - 1}_{self.env.episode}_{self.proc_interval}_{self.w}_{self.num_agents}agents_{self.mode}.pdf")
        plt.close()


    def _smooth_for_plot(self, values, window):
        values = np.asarray(values)
        if values.size == 0:
            return None, None

        smooth_values = np.convolve(values, np.ones(window) / window, mode='valid')
        x_values = range(len(smooth_values))
        return x_values, smooth_values

    
    def plot_battery_levels(self, folder_path, levels):
        window = 10
        plt.suptitle("Multi-agent : battery levels")
        plt.title(f"P_i = {self.power_idle}, P_f = {self.power_max}, fps = {self.proc_rate}, interval: {self.proc_interval}s")
        
        plt.xlabel("Episodes")
        plt.ylabel("Battery")
        
        for i in range(0, self.env._num_agents):    
            # print(rewards[i])
            x_values, smooth_values = self._smooth_for_plot(levels[i], window)
            if x_values is None:
                continue
            plt.plot(x_values, smooth_values, label = f"smooth {self.battery_capacities[i]}Wh", alpha = 1.0)
            # plt.plot(levels[i], label = f"raw {self.battery_capacities[i]}Wh", alpha = 0.3)
        
        plt.grid()
        plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
        plt.tight_layout()
        plt.savefig(f"./{folder_path}/avg_battery_plot_{self.num_episodes-1}_{self.env.episode}_{self.proc_interval}_{self.w}_{self.num_agents}agents_{self.mode}.pdf")
        plt.close()


    def plot_battery_levels_individual(self, folder_path, levels):
        window = 10
        # plt.suptitle("Multi-agent : battery levels")
        # plt.title(f"P_i = {power_idle}, P_f = {power_max}, fps = {proc_rate}, interval: {proc_interval}s")
        
        # plt.xlabel("Episodes")
        # plt.ylabel("Battery")
        
        for i in range(0, self.env._num_agents):
            plt.suptitle("Multi-agent : battery levels")
            plt.title(f"P_i = {self.power_idle}, P_f = {self.power_max}, fps = {self.proc_rate}, interval: {self.proc_interval}s")
            
            plt.xlabel("Episodes")
            plt.ylabel("Battery")
        
            # print(rewards[i])
            x_values, smooth_values = self._smooth_for_plot(levels[i], window)
            if x_values is None:
                continue
            plt.plot(x_values, smooth_values, label = f"smooth {self.battery_capacities[i]}Wh", alpha = 1.0)
            # plt.plot(levels[i], label = f"raw {self.battery_capacities[i]}Wh", alpha = 0.3)
        
            plt.grid()
            plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
            plt.tight_layout()
            plt.savefig(f"./{folder_path}/avg_battery_plot_{self.battery_capacities[i]}Wh_{self.num_episodes-1}_{self.env.episode}_{self.proc_interval}_{self.w}_{self.num_agents}agents_{self.mode}.pdf")
            plt.close()

    
    def plot_backlogs(self, folder_path, backlogs):
        window = 10
        plt.suptitle("Multi-agent : average backlog")
        plt.title(f"P_i = {self.power_idle}, P_f = {self.power_max}, fps = {self.proc_rate}, interval: {self.proc_interval}s")
        
        plt.xlabel("Timesteps")
        plt.ylabel("Backlog")
        
        for i in range(0, self.env._num_agents):
            # print(rewards[i])
            x_values, smooth_values = self._smooth_for_plot(backlogs[i], window)
            if x_values is None:
                continue
            plt.plot(x_values, smooth_values, label = f"smooth {self.battery_capacities[i]}Wh", alpha = 1.0)
            # plt.plot(backlogs[i], label = f"raw {self.battery_capacities[i]}Wh", alpha = 0.3)
        
        plt.grid()
        # plt.legend()
        plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
        plt.tight_layout()
        plt.savefig(f"./{folder_path}/backlog_plot_{self.num_episodes-1}_{self.env.episode}_{self.proc_interval}_{self.w}_{self.num_agents}agents_{self.mode}.pdf")
        plt.close()

    
    def plot_battery_daily(self, folder_path, data):    
        for elem in range(0, self.env._num_agents):
                    
            window = 40
            plt.suptitle("Multi-agent : daily battery")
            plt.title(f"B: {self.env.battery_capacities[elem] / 3600} - P_i = {self.power_idle}, P_f = {self.power_max}, fps = {self.proc_rate}, interval: {self.proc_interval}s")
            
            plt.xlabel("Timesteps")
            plt.ylabel("Battery")
            for i in range(0, len(data[elem])):
                # print(rewards[i])
                x_values, smooth_values = self._smooth_for_plot(data[elem][i], window)
                if x_values is None:
                    continue
                plt.plot(x_values, smooth_values, label = f"{i * (int((self.num_episodes-1) / 10))}-th episode", alpha = 1.0)
            
            plt.grid()
            plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
            plt.savefig(f"./{folder_path}/battery_{int(self.env.battery_capacities[elem] / 3600)}Wh_{self.num_episodes-1}_{self.env.episode}_{self.proc_interval}_{self.w}_{self.num_agents}agents_{self.mode}.pdf")
            plt.close()
            
    
    def plot_backlog_daily(self, folder_path, data):
        
        for elem in range(0, self.env._num_agents):
            
            window = 40
            plt.suptitle("Multi-agent : daily backlog")
            plt.title(f"B: {self.env.battery_capacities[elem] / 3600} - P_i = {self.power_idle}, P_f = {self.power_max}, fps = {self.proc_rate}, interval: {self.proc_interval}s")
            
            plt.xlabel("Timesteps")
            plt.ylabel("Backlog")
            for i in range(0, len(data[elem])):
                # print(rewards[i])
                x_values, smooth_values = self._smooth_for_plot(data[elem][i], window)
                if x_values is None:
                    continue
                plt.plot(x_values, smooth_values, label = f"{i* (int((self.num_episodes-1) / 10))}-th episode", alpha = 1.0)
            
            plt.grid()
            # plt.legend()
            plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
            plt.savefig(f"./{folder_path}/backlog_{int(self.env.battery_capacities[elem] / 3600)}Wh_{self.num_episodes-1}_{self.env.episode}_{self.proc_interval}_{self.w}_{self.num_agents}agents_{self.mode}.pdf")
            plt.close()        

    
    def plot_framerate(self, folder_path, fs):
        window = 10
        plt.suptitle("Multi-agent : average framerate")
        plt.title(f"P_i = {self.power_idle}, P_f = {self.power_max}, fps = {self.proc_rate}, interval: {self.proc_interval}s")
        
        plt.xlabel("Episodes")
        plt.ylabel("Framerate")
        
        for i in range(0, self.env._num_agents):
            # print(rewards[i])
            x_values, smooth_values = self._smooth_for_plot(fs[i], window)
            if x_values is None:
                continue
            plt.plot(x_values, smooth_values, label = f"smooth {self.battery_capacities[i]}Wh", alpha = 1.0)
            # plt.plot(fs[i], label = f"raw {self.battery_capacities[i]}Wh", alpha = 0.3)
        
        plt.grid()
        plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
        plt.tight_layout()
        plt.savefig(f"./{folder_path}/framerate_plot_{self.num_episodes-1}_{self.env.episode}_{self.proc_interval}_{self.w}_{self.env._num_agents}agents_{self.mode}.pdf")
        plt.close()
        
    
    def plot_local_framerate(self, folder_path, fs):
        window = 10
        plt.suptitle("Multi-agent : local average framerate")
        plt.title(f"P_i = {self.power_idle}, P_f = {self.power_max}, fps = {self.proc_rate}, interval: {self.proc_interval}s")
        
        plt.xlabel("Episodes")
        plt.ylabel("Backlog")
        
        for i in range(0, self.env._num_agents):
            # print(rewards[i])
            x_values, smooth_values = self._smooth_for_plot(fs[i], window)
            if x_values is None:
                continue
            plt.plot(x_values, smooth_values, label = f"smooth {self.battery_capacities[i]}Wh", alpha = 1.0)
            # plt.plot(fs[i], label = f"raw {self.battery_capacities[i]}Wh", alpha = 0.3)
        
        plt.grid()
        plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
        plt.tight_layout()
        plt.savefig(f"./{folder_path}/local_framerate_plot_{self.num_episodes-1}_{self.env.episode}_{self.proc_interval}_{self.w}_{self.env._num_agents}agents_{self.mode}.pdf")
        plt.close()
      
      
    def plot_offloading_framerate(self, folder_path, fs):
        window = 10
        plt.suptitle("Multi-agent : offloading average framerate")
        plt.title(f"P_i = {self.power_idle}, P_f = {self.power_max}, fps = {self.proc_rate}, interval: {self.proc_interval}s")
        
        plt.xlabel("Episodes")
        plt.ylabel("Backlog")
        
        for i in range(0, self.env._num_agents):
            # print(rewards[i])
            x_values, smooth_values = self._smooth_for_plot(fs[i], window)
            if x_values is None:
                continue
            plt.plot(x_values, smooth_values, label = f"smooth {self.battery_capacities[i]}Wh", alpha = 1.0)
            # plt.plot(fs[i], label = f"raw {self.battery_capacities[i]}Wh", alpha = 0.3)
        
        plt.grid()
        plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
        plt.tight_layout()
        plt.savefig(f"./{folder_path}/offloading_framerate_plot_{self.num_episodes-1}_{self.env.episode}_{self.proc_interval}_{self.w}_{self.env._num_agents}agents_{self.mode}.pdf")
        plt.close()
        
    def plot_offloading_matchings(self, folder_path, fs):
        window = int(self.num_episodes / 100)
        plt.suptitle("Multi-agent : average offloading matchings")
        plt.title(f"P_i = {self.power_idle}, P_f = {self.power_max}, fps = {self.proc_rate}, interval: {self.proc_interval}s")
        
        plt.xlabel("Episodes")
        plt.ylabel("Matchings")
        plt.ylim(0, 10)
        
        for i in range(0, self.env._num_agents):
            # print(rewards[i])
            x_values, smooth_values = self._smooth_for_plot(fs[i], window)
            if x_values is None:
                continue
            plt.plot(x_values, smooth_values, label = f"smooth {self.battery_capacities[i]}Wh", alpha = 1.0)
            # plt.plot(fs[i], label = f"raw {self.battery_capacities[i]}Wh", alpha = 0.3)
        
        plt.grid()
        plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
        plt.tight_layout()
        plt.savefig(f"./{folder_path}/offloading_matchings_plot_{self.num_episodes-1}_{self.env.episode}_{self.proc_interval}_{self.w}_{self.env._num_agents}agents_{self.mode}.pdf")
        plt.close()

    def save_battery_csv(self, folder_path, battery_daily):
        
        for agent_id in range(self.num_agents):
            filename = f"./{folder_path}/csvs/csvs_batch_{self.batch_size}/battery_{self.battery_capacities[agent_id]}_{self.num_episodes-1}_{self.env.episode}_{self.proc_interval}_{self.num_agents}agents_{self.mode}.csv"
            
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                
                if len(battery_daily[agent_id]) > 0 and len(battery_daily[agent_id][0]) > 0:
                    n_timesteps = len(battery_daily[agent_id][0])
                    header = [f't{i}' for i in range(n_timesteps)]
                    writer.writerow(header)
                    
                    for episode_data in battery_daily[agent_id]:
                        writer.writerow(episode_data)
            
            print(f"saved: {filename}")
    
    def save_backlog_csv(self, folder_path, backlog_daily):
        
        for agent_id in range(self.num_agents):
            filename = f"./{folder_path}/csvs/csvs_batch_{self.batch_size}/backlog_{self.battery_capacities[agent_id]}_{self.num_episodes-1}_{self.env.episode}_{self.proc_interval}_{self.num_agents}agents_{self.mode}.csv"
            
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                
                if len(backlog_daily[agent_id]) > 0 and len(backlog_daily[agent_id][0]) > 0:
                    n_timesteps = len(backlog_daily[agent_id][0])
                    header = [f't{i}' for i in range(n_timesteps)]
                    writer.writerow(header)
                    
                    for episode_data in backlog_daily[agent_id]:
                        writer.writerow(episode_data)
            
            print(f"saved: {filename}")

    def save_rewards_csv(self, folder_path, rewards):
        os.makedirs('./csvs/{folder_path}/csvs_batch_{self.batch_size}', exist_ok=True)
        
        for agent_id in range(self.num_agents):
            filename = f"./{folder_path}/csvs/csvs_batch_{self.batch_size}/rewards_agent_{self.battery_capacities[agent_id]}_{self.num_episodes-1}_{self.env.episode}_{self.proc_interval}_{self.num_agents}agents_{self.mode}.csv"
            
            with open(filename, "w") as file:
                for elem in rewards[agent_id]:
                    file.write(str(float(elem)) + "\n")
                
        print(f"saved: {filename}")
        
    def save_time_csv(self, folder_path, times):
        os.makedirs(f'./{folder_path}/csvs/csvs_batch_{self.batch_size}', exist_ok=True)
        filename = f"./{folder_path}/csvs/csvs_batch_{self.batch_size}/time_{self.num_episodes-1}_{self.env.episode}_{self.proc_interval}_{self.num_agents}agents_{self.mode}.csv"

        with open(filename, "w") as file:
            for elem in times:
                file.write(str(float(elem)) + "\n")
                
        print(f"saved: {filename}")

    def save_framerate_csv(self, folder_path, fs, hs, framerates):
        os.makedirs(f'./{folder_path}/csvs/csvs_batch_{self.batch_size}', exist_ok=True)
        
        for agent_id in range(self.num_agents):
            filename = f"./{folder_path}/csvs/csvs_batch_{self.batch_size}/local_framerate_{self.battery_capacities[agent_id]}_{self.num_episodes-1}_{self.env.episode}_{self.proc_interval}_{self.num_agents}agents_{self.mode}.csv"
            with open(filename, "w") as file:
                for elem in fs[agent_id]:
                    file.write(str(float(elem)) + "\n")

                print(f"saved: {filename}")

            
            filename = f"./{folder_path}/csvs/csvs_batch_{self.batch_size}/offloading_framerate_{self.battery_capacities[agent_id]}_{self.num_episodes-1}_{self.env.episode}_{self.proc_interval}_{self.num_agents}agents_{self.mode}.csv"
            with open(filename, "w") as file:
                for elem in hs[agent_id]:
                    file.write(str(float(elem)) + "\n")
                print(f"saved: {filename}")

            
            filename = f"./{folder_path}/csvs/csvs_batch_{self.batch_size}/total_framerate_{self.battery_capacities[agent_id]}_{self.num_episodes-1}_{self.env.episode}_{self.proc_interval}_{self.num_agents}agents_{self.mode}.csv"
            with open(filename, "w") as file:
                for elem in framerates[agent_id]:
                    file.write(str(float(elem)) + "\n")
                print(f"saved: {filename}")
        
    def save_offloading_matchings_csv(self, folder_path, hs_matchings):
        os.makedirs(f'./{folder_path}/csvs/csvs_batch_{self.batch_size}', exist_ok=True)
        
        for agent_id in range(self.num_agents):
            filename = f"./{folder_path}/csvs/csvs_batch_{self.batch_size}/matchings_{self.battery_capacities[agent_id]}_{self.num_episodes-1}_{self.env.episode}_{self.proc_interval}_{self.num_agents}agents_{self.mode}.csv"
            with open(filename, "w") as file:
                for elem in hs_matchings[agent_id]:
                    file.write(str(int(elem)) + "\n")
                print(f"saved: {filename}")


    
    
    def get_action(self, agent_id, obs, actions_encoded, actions, ppo_data):
        obs_tensor = torch.as_tensor(obs).float().to(self.models[agent_id].device).unsqueeze(0)
        with torch.no_grad():
            actions_tensor, values, log_probs = self.models[agent_id].policy(obs_tensor)
        action = actions_tensor.cpu().numpy()[0]
        
        ppo_data[agent_id] = {'value': values, 'log_prob': log_probs}
        actions_encoded[agent_id] = action
        actions[agent_id] = action
        
    
    def train(self):
        if not self.models_folder.exists():
            self.models_folder.mkdir(parents=True, exist_ok=True)

        print("Fresh training enabled: skipping pretrained checkpoint loading for all agents")

        # In SB3, total_timesteps already counts transitions across all parallel envs.
        total_timesteps = self.num_episodes * self.max_steps
        rollout_size = self.models[0].n_steps * self.parallel_envs
        expected_updates = max(1, total_timesteps // rollout_size)

        if self.train_all_mode == 2:
            round_timesteps = max(1, total_timesteps // max(1, self.num_rounds))
            print(
                f"Round-robin training enabled: {self.num_rounds} rounds, "
                f"{round_timesteps} timesteps per agent per round"
            )

            for round_idx in range(self.num_rounds):
                print(f"Starting round {round_idx + 1}/{self.num_rounds}")
                for agent_id in range(self.num_agents):
                    teammate_model_paths = self._teammate_model_paths_for(agent_id)
                    previous_env = self.models[agent_id].get_env()
                    if previous_env is not None:
                        previous_env.close()

                    new_env = self._build_vec_env(
                        agent_id,
                        teammate_model_paths=teammate_model_paths,
                        num_envs=self.parallel_envs_round_mode,
                    )
                    self.models[agent_id].set_env(new_env)
                    gc.collect()
                    print(
                        f"Training agent {agent_id} with teammate policies "
                        f"for {round_timesteps} timesteps"
                    )
                    self.models[agent_id].learn(
                        total_timesteps=round_timesteps,
                        reset_num_timesteps=False,
                        progress_bar=False,
                        log_interval=10,
                    )
                    self._save_model_for_agent(agent_id)
        else:
            for agent_id in range(self.num_agents):
                print(
                    f"Training agent {agent_id} with {self.parallel_envs} parallel envs "
                    f"for {total_timesteps} timesteps (~{expected_updates} PPO updates)"
                )
                self.models[agent_id].learn(
                    total_timesteps=total_timesteps,
                    reset_num_timesteps=False,
                    progress_bar=False,
                    log_interval=10,
                )
                self._save_model_for_agent(agent_id)

        print(f"Training completed with {self.parallel_envs} parallel envs per agent")

        print(f"Training mode {self.train_all_mode} completed for all nodes")
        
    
    def evaluate(self):
        for i in range(self.num_agents):
            loaded = self._load_latest_model_for_agent(i)
            if not loaded:
                batt = int(self.battery_capacities[i])
                print(f"No saved model found for agent {i} ({batt}Wh), keeping fresh model")
        
        obs = self.env.reset(self.seed)[0]
        agents_logs = {agent_id: {"battery": [], "processing": [], "panel_energy": [], "backlog": [], "state": [], "processed_frames": [], "hs_counter": [], "offloading": [], "tx_frames": [], "rx_frames": []} for agent_id in range(self.num_agents)}
        terminate = False
        while not terminate:
            actions = {}
            for agent_id in range(self.num_agents):
                agent_obs = obs[agent_id]
                action, _ = self.models[agent_id].predict(agent_obs, deterministic=True)
                actions[agent_id] = action
                agents_logs[agent_id]['processing'].append(actions[agent_id][0]/self.proc_rate)
                agents_logs[agent_id]['offloading'].append(actions[agent_id][1]/2)
            next_obs, rewards, terminations, truncations, infos = self.env.step(actions)
            
            for agent_id in range(self.num_agents):
                print(f"Agent {agent_id} - State: {obs[agent_id][2]})")
                #agents_logs[agent_id]["sun"].append(obs[agent_id][2])
                agents_logs[agent_id]["battery"].append(obs[agent_id][0])
                agents_logs[agent_id]["state"].append(obs[agent_id][1]/3)
                agents_logs[agent_id]["panel_energy"].append(infos[agent_id]["panel_energy"])
                agents_logs[agent_id]["backlog"].append(obs[agent_id][1]/3)
                agents_logs[agent_id]["processed_frames"].append(infos[agent_id]["processed_frames"])
                agents_logs[agent_id]["hs_counter"].append(self.env.hs[agent_id])
                agents_logs[agent_id]["tx_frames"].append(infos[agent_id]["tx_frames"])
                agents_logs[agent_id]["rx_frames"].append(infos[agent_id]["rx_frames"])
                #agents_logs[agent_id]["processing"].append(obs[agent_id][4])
                done = terminations[agent_id] or truncations[agent_id]
                if done:
                    terminate = True

            obs = next_obs
        
        window_size = 50  # Più è alto, più appiattisce
        
        # Plot battery levels and processing decisions for all agents
        plt.figure(figsize=(12, 24))
        for agent_id in range(self.num_agents):
            processing_smoth = pd.Series(agents_logs[agent_id]['processing']).rolling(window=window_size, center=True).mean()
            #backlog_smoth = agents_logs[agent_id]['backlog'] #pd.Series(agents_logs[agent_id]['backlog']).rolling(window=window_size, center=True).mean()

            plt.subplot(self.num_agents, 1, agent_id + 1)
            plt.plot(agents_logs[agent_id]['battery'], label='Battery Level')
            #plt.plot(processing_smoth, label='Smoothed Processing Decision')
            plt.plot(agents_logs[agent_id]['panel_energy'], label='Panel Energy')
            plt.plot(agents_logs[agent_id]['backlog'], label='Backlog')
            plt.plot(agents_logs[agent_id]['processed_frames'], label='Processed Frames')  # Aggiunta della linea dei frame processati
            #plt.plot(agents_logs[agent_id]['hs_counter'], label='Processing from offloading')  # Aggiunta della linea del contatore dei messaggi
            plt.plot(agents_logs[agent_id]['offloading'], label='Offloading Decision')  # Aggiunta della linea della decisione di offloading

            # TX/RX traffic normalized by max framerate
            max_frames = self.proc_rate * self.proc_interval
            tx_norm = np.array(agents_logs[agent_id]['tx_frames']) / max_frames
            rx_norm = np.array(agents_logs[agent_id]['rx_frames']) / max_frames
            plt.plot(tx_norm, label='Frames TX', color='tab:green')
            plt.plot(rx_norm, label='Frames RX', color='tab:cyan')

            #plt.plot(agents_logs[agent_id]['state'], label='Smoothed State')  # Aggiunta della linea dello stato (con trasparenza)

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
        plt.savefig(f"evaluation_ppo.png")
        plt.close() 

        print("Total processed frames during evaluation:", self.env.total_frames_processed)
            

