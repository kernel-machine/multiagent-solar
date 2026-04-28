import gymnasium
from gymnasium.spaces import Box, Discrete, Dict, MultiDiscrete
import numpy as np
from stable_baselines3 import PPO

from custom_environment import CustomEnvironment

class EnvWrapper(gymnasium.Env):
    def __init__(self, env, agent_id):
        self.env = env
        self.agent_id = agent_id
        
        self.observation_space = env._observation_spaces[agent_id]
        
        # Use native MultiDiscrete representation instead of flattened
        self.action_space = env._action_spaces[agent_id]
        
    def reset(self, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)
        return obs[self.agent_id], infos[self.agent_id]
    
    def step(self, action):
        actions = {self.agent_id: action}
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        
        return (obs[self.agent_id], 
                rewards[self.agent_id], 
                terminations[self.agent_id], 
                truncations[self.agent_id], 
                infos[self.agent_id])
    
    def render(self):
        pass
    
    def close(self):
        pass


class ParallelSingleAgentEnv(gymnasium.Env):
    def __init__(
        self,
        env_kwargs,
        agent_id,
        rng_seed=None,
        teammate_model_paths=None,
        teammate_device="cpu",
    ):
        self.env = CustomEnvironment(**env_kwargs)
        self.agent_id = agent_id
        self.rng_seed = rng_seed
        self._last_obs = None

        self.teammate_policies = {}
        if teammate_model_paths is not None:
            for teammate_id, model_path in teammate_model_paths.items():
                if teammate_id == self.agent_id or model_path is None:
                    continue
                try:
                    self.teammate_policies[teammate_id] = PPO.load(model_path, device=teammate_device)
                except Exception:
                    # Fallback to random teammate actions when loading fails.
                    pass

        self.observation_space = self.env._observation_spaces[agent_id]
        self.action_space = self.env._action_spaces[agent_id]

    def reset(self, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)
        self._last_obs = obs
        return obs[self.agent_id], infos[self.agent_id]

    def step(self, action):
        actions = {}
        for other_agent_id in range(self.env._num_agents):
            if other_agent_id == self.agent_id:
                actions[other_agent_id] = action
            else:
                teammate_model = self.teammate_policies.get(other_agent_id)
                if teammate_model is not None and self._last_obs is not None:
                    teammate_action, _ = teammate_model.predict(self._last_obs[other_agent_id], deterministic=True)
                    actions[other_agent_id] = teammate_action
                else:
                    actions[other_agent_id] = self.env._action_spaces[other_agent_id].sample()

        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        self._last_obs = obs

        # If the underlying multi-agent env ended (e.g. another random agent died),
        # force termination so VecEnv resets this copy before the next step.
        terminated = terminations[self.agent_id] or (len(self.env.agents) == 0)
        truncated = truncations[self.agent_id]

        return (
            obs[self.agent_id],
            rewards[self.agent_id],
            terminated,
            truncated,
            infos[self.agent_id],
        )

    def render(self):
        pass

    def close(self):
        pass


def make_parallel_single_agent_env(
    env_kwargs,
    agent_id,
    rng_seed=None,
    teammate_model_paths=None,
    teammate_device="cpu",
):
    return ParallelSingleAgentEnv(
        env_kwargs=env_kwargs,
        agent_id=agent_id,
        rng_seed=rng_seed,
        teammate_model_paths=teammate_model_paths,
        teammate_device=teammate_device,
    )

class EnvWrapperDQN(gymnasium.Env):
    def __init__(self, env, agent_id):
        self.env = env
        self.agent_id = agent_id
        
        self.observation_space = env._observation_spaces[agent_id]
        
        self.orig_action_space = env._action_spaces[agent_id]
        self.nvec = self.orig_action_space.nvec
        total_actions = int(np.prod(self.nvec))
        
        # Flattened Discrete action space for DQN
        self.action_space = Discrete(total_actions)
        
    def _decode_action(self, action):
        decoded = []
        act = int(action)
        for n in reversed(self.nvec):
            decoded.append(act % n)
            act //= n
        return np.array(list(reversed(decoded)))

    def reset(self, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)
        return obs[self.agent_id], infos[self.agent_id]
    
    def step(self, action):
        # Decode action back to MultiDiscrete form before passing to env
        decoded_action = self._decode_action(action)
        actions = {self.agent_id: decoded_action}
        
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        
        return (obs[self.agent_id], 
                rewards[self.agent_id], 
                terminations[self.agent_id], 
                truncations[self.agent_id], 
                infos[self.agent_id])
    
    def render(self):
        pass
    
    def close(self):
        pass