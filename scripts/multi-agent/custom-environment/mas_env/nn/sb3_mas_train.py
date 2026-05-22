import os
import hashlib
import time
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from datetime import datetime
import gymnasium as gym

class DeepSetsExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, max_agents: int = 5, features_dim: int = 128, use_spatial_index: bool = False):
        super(DeepSetsExtractor, self).__init__(observation_space, features_dim)
        self.use_spatial_index = use_spatial_index
        self.max_agents = max_agents
        self.n_others = max_agents - 1
        
        self.own_dim = 3
        # If use_spatial_index is True, each other agent has 3 features (battery, backlog, index)
        # Otherwise, 2 features (battery, backlog)
        self.other_node_dim = 3 if use_spatial_index else 2
        
        # Shared MLP for processing each other node
        self.node_mlp = nn.Sequential(
            nn.Linear(self.other_node_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # MLP to process own state mapping it to the same dimension before concat
        self.own_mlp = nn.Sequential(
            nn.Linear(self.own_dim, 64),
            nn.ReLU()
        )
        
        # We concatenate process(own_state) [64] + max_pool(process(other_nodes)) [64] = 128
        self._features_dim = 128
        
    def forward(self, observations: th.Tensor) -> th.Tensor:
        # observations shape: (batch_size, obs_dim)
        # Split into own, others_flat, mask
        own_state = observations[:, :self.own_dim]
        
        others_flat_end = self.own_dim + self.other_node_dim * self.n_others
        others_flat = observations[:, self.own_dim:others_flat_end]
        mask = observations[:, others_flat_end:]
        
        batch_size = observations.shape[0]
        
        # Reshape others_flat to (batch_size, n_others, other_node_dim)
        others_reshaped = others_flat.view(batch_size, self.n_others, self.other_node_dim)
        
        # Apply shared MLP to each other node
        # others_reshaped shape: (batch_size, n_others, node_mlp_out_dim)
        node_embeddings = self.node_mlp(others_reshaped)
        
        # Apply mask to node_embeddings before pooling. Mask shape is (batch_size, n_others)
        # We add a dimension to match node_embeddings: (batch_size, n_others, 1)
        mask_expanded = mask.unsqueeze(-1)
        
        # For max pooling, we want masked items to be very small negatively so they don't affect max
        masked_embeddings = node_embeddings.masked_fill(mask_expanded == 0, -1e9)
        
        # Max pooling over the n_others dimension
        # global_features shape: (batch_size, 64)
        global_features, _ = th.max(masked_embeddings, dim=1)
        
        # If all nodes were masked out, max_pool will return -1e9, we fix it by making it 0
        global_features = th.where(global_features == -1e9, th.zeros_like(global_features), global_features)
        
        # Process own state
        own_features = self.own_mlp(own_state)
        
        # Concatenate
        return th.cat([own_features, global_features], dim=1)


class LSTMAttentionExtractor(BaseFeaturesExtractor):
    """
    Feature extractor that applies learnable attention pooling over the
    24 LSTM prediction tokens (each with 4 features: value, sin, cos, t_norm)
    appended at the end of the observation vector.

    The base observation (battery, backlog, timestep, ...) is processed
    through a small MLP.  The attention-pooled prediction vector (32-dim)
    is concatenated with it to produce the final feature vector.
    """

    LSTM_TOKEN_DIM = 4      # (value, sin, cos, t_norm)
    LSTM_NUM_TOKENS = 24    # 24 prediction steps
    LSTM_FLAT_DIM = LSTM_TOKEN_DIM * LSTM_NUM_TOKENS  # 96

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 96,       # 64 (base MLP) + 32 (attention)
        attn_output_dim: int = 32,
        attn_hidden_dim: int = 32,
    ):
        super(LSTMAttentionExtractor, self).__init__(observation_space, features_dim)

        total_obs_dim = int(np.prod(observation_space.shape))
        self.base_obs_dim = total_obs_dim - self.LSTM_FLAT_DIM

        # ── Base-state MLP ──────────────────────────────────────────────────
        base_mlp_out = features_dim - attn_output_dim  # e.g. 64
        self.base_mlp = nn.Sequential(
            nn.Linear(self.base_obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, base_mlp_out),
            nn.ReLU(),
        )

        # ── Attention Pooling over 24 LSTM tokens ──────────────────────────
        # Project each token to hidden_dim, then use a learnable query vector
        # to compute attention weights.  The weighted sum is projected to
        # attn_output_dim (32).
        self.token_proj = nn.Linear(self.LSTM_TOKEN_DIM, attn_hidden_dim)
        self.attn_query = nn.Parameter(th.randn(attn_hidden_dim))
        self.attn_out_proj = nn.Linear(attn_hidden_dim, attn_output_dim)

        self._features_dim = features_dim

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # Split observation into base state and LSTM prediction features
        base_obs = observations[:, :self.base_obs_dim]
        lstm_flat = observations[:, self.base_obs_dim:]

        # ── Process base state ──────────────────────────────────────────────
        base_features = self.base_mlp(base_obs)

        # ── Attention pooling ───────────────────────────────────────────────
        batch_size = lstm_flat.shape[0]
        # Reshape flat (batch, 96) → (batch, 24, 4)
        tokens = lstm_flat.view(batch_size, self.LSTM_NUM_TOKENS, self.LSTM_TOKEN_DIM)
        # Project tokens: (batch, 24, hidden_dim)
        projected = th.relu(self.token_proj(tokens))
        # Attention scores: (batch, 24)
        scores = th.matmul(projected, self.attn_query)
        weights = th.softmax(scores, dim=-1)
        # Weighted sum: (batch, hidden_dim)
        context = th.sum(projected * weights.unsqueeze(-1), dim=1)
        # Output projection: (batch, attn_output_dim=32)
        attn_features = self.attn_out_proj(context)

        return th.cat([base_features, attn_features], dim=1)

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
                obs, info = env.reset(seed=data)
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
        
    def reset(self, seed=None):
        for idx, remote in enumerate(self.remotes):
            if isinstance(seed, (int, np.integer)):
                remote_seed = int(seed) + idx
            else:
                remote_seed = seed
            remote.send(('reset', remote_seed))
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
    def __init__(self, num_agents, num_episodes, irradiance_datapaths, delta_time, proc_interval, proc_rate, arrival_rate, eps_init, eps_fin, eps_dec, battery_capacities, panel_surfaces, power_idle, power_max, train_freq, w, mode, batch_size, seed, env, save_path, num_envs=4, algo="PPO", ppo_initial_lr=3e-4, ppo_final_lr=3e-5, train_all_mode=1, rotation_episodes=1, total_steps=None, max_agents=None, attn_d_model=16, ppo_n_steps=512, termination_mode="early", use_deepsets=False, use_deepsets_spatial=False, use_cross_attention=False, evaluation_enabled=True, use_lstm_prediction=False, net_arch=None, eval_termination_mode=None):
        self.num_agents = num_agents
        self.num_episodes = num_episodes
        self.max_steps = env.max_steps
        self.eval_env = env
        self.algo = algo.upper()
        # For mode=2 with PPO we allow parallel envs; DQN always uses 1.
        self.num_envs = (num_envs if self.algo == "PPO" else 1)
        
        if self.num_envs > 1:
            self.env = AsyncMultiAgentEnvs(env, self.num_envs)
        else:
            self.env = env
        # Keep a plain single env for mode=2 when num_envs==1
        self.rotating_env = env
            
        self.save_path = save_path
        self.max_agents  = max_agents if max_agents is not None else num_agents
        self.attn_d_model = attn_d_model
        self.use_deepsets = use_deepsets
        self.use_deepsets_spatial = use_deepsets_spatial
        self.use_cross_attention = use_cross_attention
        self.use_lstm_prediction = use_lstm_prediction
        self.batch_size = batch_size
        self.random_seed = seed
        self.seed = seed
        self.rng_seed = self._seed_to_int(seed)
        self.proc_interval = proc_interval
        self.proc_rate = proc_rate
        self.w = w
        self.train_all_mode = train_all_mode
        self.rotation_episodes = rotation_episodes
        self.total_steps = total_steps
        self.total_steps_done = 0
        self.battery_capacities = battery_capacities
        self.panel_surfaces = panel_surfaces
        self.train_freq = train_freq
        self.ppo_initial_lr = ppo_initial_lr
        self.ppo_final_lr = ppo_final_lr
        self.ppo_n_steps = ppo_n_steps
        # "early"  → episode ends as soon as any agent's battery hits 0 (old behaviour)
        # "penalty" → episode always runs max_steps; dead agents get per-step penalty
        self.termination_mode = termination_mode
        self.eval_termination_mode = eval_termination_mode
        
        # Hyperparameters
        self.eps_init = eps_init
        self.eps_fin = eps_fin
        self.eps_dec = eps_dec
        self.eps = eps_init
        
        # Best reward tracking via deterministic evaluation
        self.best_eval_avg_reward = -np.inf
        self.best_eval_agent_reward = {i: -np.inf for i in range(num_agents)}
        self.evaluation_enabled = evaluation_enabled

        if self.train_all_mode not in (1, 2):
            raise ValueError("train_all_mode must be 1 or 2")
        if self.train_all_mode == 2 and self.rotation_episodes <= 0:
            raise ValueError("rotation_episodes must be greater than 0 when train_all_mode is 2")
        if self.total_steps is not None and self.total_steps <= 0:
            raise ValueError("total_steps must be greater than 0 when provided")

        np.random.seed(self.rng_seed)
        th.manual_seed(self.rng_seed)
        if th.cuda.is_available():
            th.cuda.manual_seed_all(self.rng_seed)
        
        os.makedirs(self.save_path, exist_ok=True)
        
        device = "cuda" if th.cuda.is_available() else "cpu"
        
        policy_kwargs = {}
        if self.use_deepsets or self.use_deepsets_spatial:
            policy_kwargs = dict(
                features_extractor_class=DeepSetsExtractor,
                features_extractor_kwargs=dict(
                    max_agents=self.max_agents,
                    features_dim=128,
                    use_spatial_index=self.use_deepsets_spatial
                )
            )
        elif self.use_lstm_prediction:
            policy_kwargs = dict(
                features_extractor_class=LSTMAttentionExtractor,
                features_extractor_kwargs=dict(
                    features_dim=96,       # 64 (base MLP) + 32 (attention)
                    attn_output_dim=32,
                    attn_hidden_dim=32,
                )
            )
        
        # Inject net_arch into policy_kwargs (controls hidden layers of pi/vf networks)
        if net_arch is not None:
            policy_kwargs.setdefault('net_arch', net_arch)
        
        if self.algo == "DQN":
            self.models = {i: DQN("MlpPolicy", EnvWrapper(self.eval_env, i), learning_rate=1e-4, buffer_size=100000, learning_starts=1000, batch_size=batch_size, train_freq=train_freq, target_update_interval=1000, exploration_fraction=0.5, verbose=0, tensorboard_log=None, device=device, policy_kwargs=policy_kwargs if policy_kwargs else None) for i in range(num_agents)}
        else:
            from stable_baselines3.common.vec_env import DummyVecEnv
            lr_schedule = self._linear_lr_schedule(self.ppo_initial_lr, self.ppo_final_lr)

            self.models = {}
            for i in range(num_agents):
                if self.num_envs > 1:
                    sub_envs = [copy.deepcopy(self.eval_env) for _ in range(self.num_envs)]
                    venv = DummyVecEnv([(lambda e=sub_envs[j], ag=i: EnvWrapper(e, ag))
                                        for j in range(self.num_envs)])
                else:
                    venv = EnvWrapper(self.eval_env, i)

                self.models[i] = PPO(
                    "MlpPolicy",
                    venv,
                    learning_rate=lr_schedule,
                    n_steps=self.ppo_n_steps,
                    batch_size=batch_size,
                    n_epochs=10,
                    verbose=0,
                    device=device,
                    policy_kwargs=policy_kwargs if policy_kwargs else None,
                )

        for i in range(num_agents):
            self.models[i].set_logger(configure(None, ["stdout", "csv"]))

        # TensorBoard writer for training rewards
        self.log_dir = os.path.join("tb_logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.tb_writer = \
             th.utils.tensorboard.SummaryWriter(log_dir=self.log_dir)

    def decode(self, action):
        """
        Decode an action from the policy.
        For PPO with MultiDiscrete([proc_rate+1, 3, num_agents, proc_rate+1]):
          action is an array [fti, oti, target, off_rate]
        For DQN with a legacy flat-integer encoding, falls back to modular arithmetic.
        """
        action = np.array(action).flatten()

        if len(action) == 4:
            # MultiDiscrete: [fti, oti, target, off_rate] — use directly
            return [float(action[0]), int(action[1]), int(action[2]), float(action[3])]

        # Legacy flat-integer encoding (DQN)
        a = int(action[0])
        fti = a % 21
        rem = a // 21
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

    @staticmethod
    def _seed_to_int(seed):
        if isinstance(seed, (int, np.integer)):
            return int(seed)
        if seed is None:
            return 0
        seed_text = str(seed).encode("utf-8")
        return int.from_bytes(hashlib.sha256(seed_text).digest()[:4], byteorder="little", signed=False)

    def _sample_or_policy_action(self, agent_id, obs, use_policy, deterministic=False):
        if not use_policy:
            action = self.models[agent_id].action_space.sample()
            return action, None, None

        if self.algo == "DQN":
            action, _ = self.models[agent_id].predict(obs, deterministic=deterministic)
            return action, None, None

        obs_tensor = th.as_tensor(obs).to(self.models[agent_id].device).unsqueeze(0)
        with th.no_grad():
            action, value, log_prob = self.models[agent_id].policy(obs_tensor)
        return action.cpu().numpy()[0], value, log_prob

    def _increment_total_steps(self, step_increment=1):
        if self.total_steps is None:
            return False

        self.total_steps_done += step_increment
        return self.total_steps_done >= self.total_steps

    def _episode_done(self, terminations, truncations):
        if self.termination_mode == "penalty":
            return all(truncations.values())
        return any(terminations.values()) or all(truncations.values())

    def _transition_done(self, termination, truncation):
        if self.termination_mode == "penalty":
            return truncation
        return termination or truncation

    def _run_evaluation(self):
        obs, _ = self.eval_env.reset(seed=self.seed, options={'evaluate': True})
        done = False
        eval_rewards = {i: 0.0 for i in range(self.num_agents)}
        while not done:
            actions = {}
            for agent_id in range(self.num_agents):
                # Deterministic prediction for evaluation
                action, _ = self.models[agent_id].predict(obs[agent_id], deterministic=True)
                actions[agent_id] = self.decode(action)

            next_obs, rewards, terminations, truncations, _ = self.eval_env.step(actions)
            
            for agent_id in range(self.num_agents):
                eval_rewards[agent_id] += rewards[agent_id]

            t_mode = self.eval_termination_mode if self.eval_termination_mode is not None else self.termination_mode
            if t_mode == "penalty":
                done = all(truncations.values())
            else:
                done = any(terminations.values()) or all(truncations.values())
            obs = next_obs

        return eval_rewards

    def _train_rotating(self):
        use_parallel = self.algo == "PPO" and self.num_envs > 1
        n = self.num_envs if use_parallel else 1
        env_handle = self.env if use_parallel else self.rotating_env

        print(
            f"Starting {self.algo} Training in rotating mode for {self.num_episodes} episodes "
            f"({'parallel ' + str(n) + ' envs' if use_parallel else 'single env'})..."
        )

        rewards_history = {i: [] for i in range(self.num_agents)}
        steps_history = []
        trained_agents = set()
        active_agent = 0
        episodes_in_rotation = 0
        stop_training = False

        for ep in range(self.num_episodes):
            if stop_training:
                break

            obs_list, _ = env_handle.reset(seed=self.seed)
            # Normalise: single-env reset returns a dict; multi-env returns a list of dicts.
            # We work with obs_list always being a list of length n.
            if not use_parallel:
                obs_list = [obs_list]   # wrap single dict into a list

            done = False
            ep_reward = {i: 0.0 for i in range(self.num_agents)}
            step_count = 0

            while not done:
                # -------------------------------------------------------------------
                # Collect actions for every env in the batch
                # -------------------------------------------------------------------
                # For PPO active agent: batch inference over all n envs at once.
                # For DQN / single-env: keep the original per-step logic.
                actions_list = [{} for _ in range(n)]   # decoded actions per env

                if use_parallel:
                    # ---- active agent: batch forward pass ----
                    obs_active = np.stack([obs_list[k][active_agent] for k in range(n)])
                    obs_tensor = th.as_tensor(obs_active).to(self.models[active_agent].device)
                    with th.no_grad():
                        act_batch, value_batch, lp_batch = self.models[active_agent].policy(obs_tensor)
                    act_batch_np = act_batch.cpu().numpy()   # (n, action_dim)

                    self.models[active_agent]._last_obs    = obs_active
                    self.models[active_agent]._last_value  = value_batch
                    self.models[active_agent]._last_log_prob = lp_batch

                    for k in range(n):
                        actions_list[k][active_agent] = self.decode(act_batch_np[k])

                    # ---- other agents: random or deterministic policy, per env ----
                    for agent_id in range(self.num_agents):
                        if agent_id == active_agent:
                            continue
                        use_pol = agent_id in trained_agents
                        obs_other = np.stack([obs_list[k][agent_id] for k in range(n)])
                        if use_pol:
                            obs_t = th.as_tensor(obs_other).to(self.models[agent_id].device)
                            with th.no_grad():
                                act_o, _, _ = self.models[agent_id].policy(obs_t)
                            act_o_np = act_o.cpu().numpy()
                            for k in range(n):
                                actions_list[k][agent_id] = self.decode(act_o_np[k])
                        else:
                            for k in range(n):
                                rnd = self.models[agent_id].action_space.sample()
                                actions_list[k][agent_id] = self.decode(rnd)

                    # ---- step all envs ----
                    next_obs_list, rew_list, term_list, trunc_list, _ = env_handle.step(actions_list)

                    # ---- accumulate rewards (track env-0 for logging) ----
                    for agent_id in range(self.num_agents):
                        ep_reward[agent_id] += rew_list[0][agent_id]

                    # ---- PPO rollout buffer update (active agent, n transitions) ----
                    rewards_arr  = np.array([rew_list[k][active_agent]  for k in range(n)])
                    is_done_arr  = np.array([
                        term_list[k][active_agent] or trunc_list[k][active_agent]
                        for k in range(n)
                    ])
                    episode_starts = np.array([step_count == 0] * n)

                    # act_batch shape (n, action_dim) → rollout buffer expects (n, action_dim)
                    action_buf = act_batch_np  # already (n, action_dim)

                    self.models[active_agent].rollout_buffer.add(
                        obs_active,
                        action_buf,
                        rewards_arr,
                        episode_starts,
                        value_batch,
                        lp_batch,
                    )
                    self.models[active_agent].num_timesteps += n
                    stop_training = self._increment_total_steps(n)

                    if self.models[active_agent].rollout_buffer.full:
                        next_obs_active = np.stack([next_obs_list[k][active_agent] for k in range(n)])
                        last_obs_t = th.as_tensor(next_obs_active).to(self.models[active_agent].device)
                        with th.no_grad():
                            last_val = self.models[active_agent].policy.predict_values(last_obs_t).flatten()
                        self.models[active_agent].rollout_buffer.compute_returns_and_advantage(
                            last_val, is_done_arr
                        )
                        progress = 1.0 - (ep / self.num_episodes)
                        self.models[active_agent]._current_progress_remaining = progress
                        self.models[active_agent].train()
                        self.models[active_agent].rollout_buffer.reset()

                    obs_list = next_obs_list
                    done = (
                        any(term_list[0].values()) or all(trunc_list[0].values())
                        or stop_training
                    )

                else:
                    # -----------------------------------------------------------
                    # Single-env path (original logic, kept intact)
                    # -----------------------------------------------------------
                    obs = obs_list[0]   # plain dict
                    actions_encoded = {}
                    actions = {}

                    for agent_id in range(self.num_agents):
                        if agent_id == active_agent:
                            if self.algo == "DQN":
                                if np.random.rand() < self.eps:
                                    act = self.models[agent_id].action_space.sample()
                                else:
                                    act, _ = self.models[agent_id].predict(obs[agent_id], deterministic=False)
                                value = None
                                log_prob = None
                            else:
                                obs_tensor = th.as_tensor(obs[agent_id]).to(self.models[agent_id].device).unsqueeze(0)
                                with th.no_grad():
                                    act, value, log_prob = self.models[agent_id].policy(obs_tensor)
                                act = act.cpu().numpy()[0]

                            actions_encoded[agent_id] = act
                            actions[agent_id] = self.decode(act)
                            self.models[agent_id]._last_value   = value
                            self.models[agent_id]._last_log_prob = log_prob
                            self.models[agent_id]._last_obs     = obs[agent_id]
                        elif agent_id in trained_agents:
                            act, _, _ = self._sample_or_policy_action(agent_id, obs[agent_id], use_policy=True, deterministic=True)
                            actions_encoded[agent_id] = act
                            actions[agent_id] = self.decode(act)
                        else:
                            act, _, _ = self._sample_or_policy_action(agent_id, obs[agent_id], use_policy=False)
                            actions_encoded[agent_id] = act
                            actions[agent_id] = self.decode(act)

                    next_obs, rewards, terminations, truncations, _ = self.rotating_env.step(actions)

                    for agent_id in range(self.num_agents):
                        ep_reward[agent_id] += rewards[agent_id]

                    if self.algo == "DQN":
                        is_done = terminations[active_agent] or truncations[active_agent]
                        self.models[active_agent].replay_buffer.add(
                            obs[active_agent],
                            next_obs[active_agent],
                            np.array([actions_encoded[active_agent]]),
                            rewards[active_agent],
                            is_done,
                            [{}],
                        )
                        if self.models[active_agent].num_timesteps > 1000 and self.models[active_agent].num_timesteps % self.train_freq == 0:
                            self.models[active_agent].train(gradient_steps=1, batch_size=self.batch_size)
                    else:
                        self.models[active_agent].rollout_buffer.add(
                            self.models[active_agent]._last_obs,
                            np.array([actions_encoded[active_agent]]),
                            rewards[active_agent],
                            np.array([step_count == 0]),
                            self.models[active_agent]._last_value,
                            self.models[active_agent]._last_log_prob,
                        )
                        if self.models[active_agent].rollout_buffer.full:
                            last_obs_t = th.as_tensor(next_obs[active_agent]).to(self.models[active_agent].device).unsqueeze(0)
                            with th.no_grad():
                                last_val = self.models[active_agent].policy.predict_values(last_obs_t)
                            self.models[active_agent].rollout_buffer.compute_returns_and_advantage(
                                last_val,
                                np.array([self._transition_done(terminations[active_agent], truncations[active_agent])]),
                            )
                            progress = 1.0 - (ep / self.num_episodes)
                            self.models[active_agent]._current_progress_remaining = progress
                            self.models[active_agent].train()
                            self.models[active_agent].rollout_buffer.reset()

                    self.models[active_agent].num_timesteps += 1
                    stop_training = self._increment_total_steps()

                    obs_list = [next_obs]
                    done = self._episode_done(terminations, truncations) or stop_training

                step_count += 1

            # ---- end-of-episode bookkeeping ----
            _episode_active_agent = active_agent   # capture before possible rotation
            episodes_in_rotation += 1
            if episodes_in_rotation >= self.rotation_episodes:
                trained_agents.add(active_agent)
                active_agent = (active_agent + 1) % self.num_agents
                episodes_in_rotation = 0

            self.eps = max(self.eps_fin, self.eps_init - ep * (self.eps_init - self.eps_fin) / (self.num_episodes * self.eps_dec))

            steps_history.append(step_count)
            for agent_id in range(self.num_agents):
                rewards_history[agent_id].append(ep_reward[agent_id])

            self.tb_writer.add_scalar("AverageReward/episode", np.mean(list(ep_reward.values())), ep)

            # ---- Periodic Deterministic Evaluation (mode=2) ----
            if ep % 20 == 0 or ep == self.num_episodes - 1:
                eval_rewards = self._run_evaluation()
                eval_avg_reward = np.mean(list(eval_rewards.values()))
                self.tb_writer.add_scalar("AverageReward/evaluation", eval_avg_reward, ep)
                
                # Update global average
                if eval_avg_reward > self.best_eval_avg_reward:
                    self.best_eval_avg_reward = eval_avg_reward
                
                # Check for personal bests
                for agent_idx in range(self.num_agents):
                    if eval_rewards[agent_idx] > self.best_eval_agent_reward[agent_idx]:
                        self.best_eval_agent_reward[agent_idx] = eval_rewards[agent_idx]
                        self.save_models(ep, eval_rewards[agent_idx], agent_id=agent_idx)
                
                print(f"--- Eval Ep {ep}: Det. Rewards: {eval_rewards} | Avg: {eval_avg_reward:.2f} ---")

            if ep % 20 == 0:
                last_obs = obs_list[0]
                print(
                    f"Ep {ep}/{self.num_episodes} | active={_episode_active_agent} | "
                    f"trained={sorted(trained_agents)} | Steps: {step_count} | "
                    f"Rewards: {ep_reward} | "
                    f"Obs: {[last_obs[i].tolist() for i in range(self.num_agents)]}"
                )

        # Final snapshot: ensure every agent has at least one saved checkpoint.
        print("Saving final model snapshot for all agents...")
        self.save_models(episode=self.num_episodes - 1)

        self.plot_results(rewards_history, steps_history)

        self.tb_writer.close()

        if use_parallel:
            self.env.close()

    def train(self):
        if self.train_all_mode == 2:
            self._train_rotating()
            return

        print(f"Starting {self.algo} Training for {self.num_episodes} episodes...")
        
        rewards_history = {i: [] for i in range(self.num_agents)}
        steps_history = []
        
        for ep in range(self.num_episodes):
            if self.total_steps is not None and self.total_steps_done >= self.total_steps:
                break

            obs, info = self.env.reset(seed=self.seed)
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

                if self.num_envs == 1:
                    stop_training = self._increment_total_steps()
                else:
                    stop_training = self._increment_total_steps(self.num_envs)
                
                for i in range(self.num_agents):
                    if self.num_envs == 1:
                        is_done = self._transition_done(terminations[i], truncations[i])
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
                        is_done_arr = np.array([
                            self._transition_done(terminations[env_idx][i], truncations[env_idx][i])
                            for env_idx in range(self.num_envs)
                        ])
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

                if self.total_steps is not None and stop_training:
                    done = True
                
                if self.num_envs == 1:
                    done = self._episode_done(terminations, truncations)
                else:
                    done = self._episode_done(terminations[0], truncations[0])

                if self.total_steps is not None and stop_training:
                    done = True

            # Update epsilon for DQN
            self.eps = max(self.eps_fin, self.eps_init - ep * (self.eps_init - self.eps_fin) / (self.num_episodes * self.eps_dec))
            
            steps_history.append(step_count)
            for i in range(self.num_agents):
                rewards_history[i].append(ep_reward[i])
            
            self.tb_writer.add_scalar("AverageReward/episode", np.mean(list(ep_reward.values())), ep)

            # ---- Periodic Deterministic Evaluation (mode=1) ----
            if self.evaluation_enabled and (ep % 10 == 0 or ep == self.num_episodes - 1):
                eval_rewards = self._run_evaluation()
                eval_avg_reward = np.mean(list(eval_rewards.values()))
                self.tb_writer.add_scalar("AverageReward/evaluation", eval_avg_reward, ep)
                
                # Save models if this is the best deterministic evaluation reward so far.
                if eval_avg_reward > self.best_eval_avg_reward:
                    self.best_eval_avg_reward = eval_avg_reward
                    self.save_models(ep, eval_avg_reward)
                    
                print(f"--- Eval Ep {ep}: Det. Rewards: {eval_rewards} | Avg: {eval_avg_reward:.2f} ---")

            if ep % 20 == 0:
                if self.num_envs == 1:
                    print(f"Ep {ep}/{self.num_episodes} - Steps: {step_count} - Obs: {[obs[i].tolist() for i in range(self.num_agents)]}")
                else:
                    print(f"Ep {ep}/{self.num_episodes} - Steps: {step_count} - Obs: {[obs[0][i].tolist() for i in range(self.num_agents)]}")
                
        self.tb_writer.close()

        if self.num_envs > 1:
            self.env.close()

    def save_models(self, episode=None, avg_reward=None, agent_id=None):
        """
        Save model checkpoints.
        - agent_id=int  → per-agent personal best (mode=2): writes agent_{i}_best.zip (overwrite)
        - agent_id=None → all agents (mode=1 / end-of-training): writes agent_{i}_final.zip (overwrite)
          and, when avg_reward is provided, also updates agent_{i}_best.zip.
        Only agent_{i}_best.zip is used by evaluate().
        """
        agents_to_save = [agent_id] if agent_id is not None else range(self.num_agents)
        for i in agents_to_save:
            fname = os.path.join(self.save_path, f"agent_{i}_best")
            self.models[i].save(fname)

        if avg_reward is not None:
            print(f"Saved Agent: {list(agents_to_save)} | reward: {avg_reward:.4f} | in: {self.save_path}")
        else:
            print(f"Saved Agent: {list(agents_to_save)} | in: {self.save_path}")

    def _load_latest_model_for_agent(self, agent_id):
        model_dir = Path(self.save_path)

        # Prefer agent_{i}_best.zip. Evaluation must use the best checkpoint
        # found during training, not the last saved snapshot.
        best_file = model_dir / f"agent_{agent_id}_best.zip"
        if best_file.exists():
            chosen = best_file
        else:
            return False

        print(f"Loading agent {agent_id}: {chosen}")

        # Do NOT pass env= to load(): SB3 would validate the action space against
        # the current environment, which fails if num_agents changed between runs.
        # For evaluation we only call model.predict(), so env binding is unnecessary.
        device = "cuda" if th.cuda.is_available() else "cpu"
        if self.algo == "DQN":
            model = DQN.load(str(chosen), device=device)
        else:
            model = PPO.load(str(chosen), device=device)

        self.models[agent_id] = model
        return True
    
    def save_args(self, args:dict):
        args_path = os.path.join(self.log_dir, "training_args.txt")
        with open(args_path, "w") as f:
            for k, v in args.items():
                f.write(f"{k}: {v}\n")
        print(f"Saved training arguments to {args_path}")

    def evaluate(self, model_paths: str = None, eval_days: int = 1):
        if model_paths:
            self.save_path = model_paths

        for i in range(self.num_agents):
            loaded = self._load_latest_model_for_agent(i)
            if not loaded:
                batt = int(self.battery_capacities[i])
                print(f"No saved model found for agent {i} ({batt}Wh), keeping fresh model")
        
        # Temporarily override max_steps for multi-day evaluation
        original_max_steps = self.eval_env.max_steps
        if eval_days > 1:
            self.eval_env.max_steps = original_max_steps * eval_days
            print(f"Multi-day evaluation: {eval_days} days ({self.eval_env.max_steps} steps)")

        # Evaluation should reflect the final safe behavior: stop on battery depletion.
        obs, _ = self.eval_env.reset(seed=self.seed, options={'evaluate': True})
        print(f"Evaluating on episode {self.eval_env.episode} (seed='{self.seed}', days={eval_days})")
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
                agents_logs[agent_id]["state"].append(actions[agent_id][1])

            t_mode = self.eval_termination_mode if self.eval_termination_mode is not None else self.termination_mode
            if t_mode == "penalty":
                terminate = all(truncations.values())
            else:
                terminate = any(terminations.values()) or all(truncations.values())

            obs = next_obs
        
        window_size = 50  # Più è alto, più appiattisce
        
        # Build time axis in hours
        steps_per_day = original_max_steps  # 288 for 5-min intervals
        num_steps_recorded = len(agents_logs[0]['battery'])
        hours = np.arange(num_steps_recorded) * (self.proc_interval / 3600.0)

        # Plot battery levels and processing decisions for all agents
        plt.figure(figsize=(max(12, 4 * eval_days), 4 * self.num_agents))
        for agent_id in range(self.num_agents):
            plt.subplot(self.num_agents, 1, agent_id + 1)
            plt.plot(hours, agents_logs[agent_id]['battery'], label='Battery Level')
            plt.plot(hours, agents_logs[agent_id]['processing'], label='Processing Decision')
            plt.plot(hours, agents_logs[agent_id]['panel_energy'], label='Panel Energy')
            plt.plot(hours, agents_logs[agent_id]['backlog'], label='Backlog')
            plt.plot(hours, agents_logs[agent_id]['state'], label='State')

            # Color area when battery is 0
            threshold = 0
            plt.fill_between(hours,
                            0,
                            1,
                            where=(np.array(agents_logs[agent_id]['battery']) <= threshold), 
                            color='red', alpha=0.2, label='Battery Depleted')

            # Draw vertical day-boundary lines for multi-day evaluation
            if eval_days > 1:
                for d in range(1, eval_days):
                    day_hour = d * 24
                    plt.axvline(x=day_hour, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
            
            plt.title(f'Agent {agent_id} Evaluation ({eval_days} day{"s" if eval_days > 1 else ""})')
            plt.xlabel('Time (hours)')
            plt.ylabel('Value')
            plt.legend(loc='upper right', fontsize='small')
            plt.grid(True, alpha=0.3)
        plt.tight_layout()
        file_name = f"evaluation_{self.num_episodes-1}_{self.eval_env.episode}_{self.proc_interval}_{self.w}_{self.num_agents}agents_{self.train_all_mode}_{eval_days}days.png"
        file_name = os.path.join(self.save_path, file_name)
        print("Saved in ", file_name)
        plt.savefig(file_name, dpi=150)
        plt.close() 

        print("Total processed frames during evaluation:", self.eval_env.total_frames_processed)
        print("Total rewards during evaluation:", total_rewards)
        print("Total transferred frames during evaluation:", self.eval_env.total_transferred_frames)
        
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

        # Restore original max_steps
        self.eval_env.max_steps = original_max_steps


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