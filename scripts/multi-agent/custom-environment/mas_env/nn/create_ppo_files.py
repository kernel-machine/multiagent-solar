import re

with open('once_a_time/sb3_mas_train.py', 'r') as f:
    train_code = f.read()

# 1. Imports
train_code = train_code.replace('from stable_baselines3 import DQN', 'from stable_baselines3 import PPO')

# 2. Class name
train_code = train_code.replace('class SB3_MAS_Train:', 'class SB3_MAS_Train_PPO:')

# 3. Model init
old_model_init = """        self.models = {i : DQN(
                policy="MlpPolicy",
                env=EnvWrapper(self.env, i),
                learning_rate=0.0001,
                buffer_size=10000,
                learning_starts=500,
                batch_size=self.batch_size,
                tau=1.0,
                gamma=0.99,
                train_freq=self.train_freq,
                gradient_steps=self.grad_steps,
                replay_buffer_class=None,
                replay_buffer_kwargs=None,
                optimize_memory_usage=False,
                n_steps=1,
                target_update_interval=((3600 / (proc_interval * 60))),
                exploration_fraction=1.0,
                exploration_initial_eps=1.0,
                exploration_final_eps=0.05,
                max_grad_norm=1,
                stats_window_size=100,
                tensorboard_log=None,
                policy_kwargs=None,
                verbose=0,
                seed=None,
                device=mode,
                _init_setup_model=True
            )
          for i in range(0, num_agents)}"""

new_model_init = """        self.models = {i : PPO(
                policy="MlpPolicy",
                env=EnvWrapper(self.env, i),
                learning_rate=0.0003,
                n_steps=2048,
                batch_size=self.batch_size,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.0,
                vf_coef=0.5,
                max_grad_norm=0.5,
                tensorboard_log=None,
                verbose=0,
                device=mode,
                _init_setup_model=True
            )
          for i in range(0, num_agents)}"""

train_code = train_code.replace(old_model_init, new_model_init)

# 4. Folder & filenames
train_code = train_code.replace('Path("./models/once_a_time")', 'Path("./models/once_a_time_ppo")')
train_code = train_code.replace('DQN_agent', 'PPO_agent')
train_code = train_code.replace('DQN.load', 'PPO.load')

# 5. get_action signature and logic
old_get_action = """    def get_action(self, agent_id, obs, actions_encoded, actions, active_training_node, trained_nodes):
        action = 0

        if agent_id == active_training_node:
            
            if(np.random.random() < self.eps):
                action = self.models[agent_id].action_space.sample()
            else:
                action, _ = self.models[agent_id].predict(obs, deterministic=False)
        elif self.train_all_mode == 2 and agent_id in trained_nodes:
            action, _ = self.models[agent_id].predict(obs, deterministic=True)
        else:
            action = self.models[agent_id].action_space.sample()
        
        actions_encoded[agent_id] = action
        actions[agent_id] = self.decode(action)"""

new_get_action = """    def get_action(self, agent_id, obs, actions_encoded, actions, active_training_node, trained_nodes, ppo_data):
        action = 0

        if agent_id == active_training_node:
            obs_tensor = torch.as_tensor(obs).float().to(self.models[agent_id].device).unsqueeze(0)
            with torch.no_grad():
                actions_tensor, values, log_probs = self.models[agent_id].policy(obs_tensor)
            action = int(actions_tensor.cpu().numpy()[0])
            value = values
            log_prob = log_probs
            
            ppo_data[agent_id] = {'value': value, 'log_prob': log_prob}
            
        elif self.train_all_mode == 2 and agent_id in trained_nodes:
            action, _ = self.models[agent_id].predict(obs, deterministic=True)
            action = int(action)
        else:
            action = int(self.models[agent_id].action_space.sample())
        
        actions_encoded[agent_id] = action
        actions[agent_id] = self.decode(action)"""

train_code = train_code.replace(old_get_action, new_get_action)

# 6. Train loop setup
# Find where the episode loop starts
ep_starts_insert_pos = train_code.find('for i in range(0, self.num_episodes):')
insert_code = '            episode_starts = {agent: True for agent in range(self.num_agents)}\n            '
train_code = train_code[:ep_starts_insert_pos] + insert_code + train_code[ep_starts_insert_pos:]

# Replace get_action call
train_code = train_code.replace(
"""                    for agent_id in range(0, self.num_agents):
                        self.get_action(
                            agent_id,
                            obs[agent_id],
                            actions_encoded,
                            actions,
                            active_training_node,
                            trained_nodes,
                        )""",
"""                    ppo_data = {}
                    for agent_id in range(0, self.num_agents):
                        self.get_action(
                            agent_id,
                            obs[agent_id],
                            actions_encoded,
                            actions,
                            active_training_node,
                            trained_nodes,
                            ppo_data
                        )"""
)

# Replace buffer logic
old_buffer_logic = """                        if agent_id == active_training_node:
                            action_encoded = actions_encoded[agent_id]

                            self.models[agent_id].replay_buffer.add(
                                obs=obs[agent_id],
                                next_obs=next_obs[agent_id],
                                action=np.array([action_encoded]),
                                reward=np.array(rewards[agent_id]),
                                done=np.array([done]),
                                infos=[{}],
                            )

                            self.models[agent_id].num_timesteps += 1

                        batteries_local[agent_id] += self.env.battery_energies[agent_id]
                        backlogs_local[agent_id] += self.env.backlogs[agent_id]

                        if agent_id == active_training_node:
                            if (
                                self.models[agent_id].num_timesteps > self.models[agent_id].learning_starts
                                and self.models[agent_id].num_timesteps % self.train_freq == 0
                            ):
                                self.models[agent_id].train(
                                    gradient_steps=self.grad_steps,
                                    batch_size=self.batch_size,
                                )"""

new_buffer_logic = """                        batteries_local[agent_id] += self.env.battery_energies[agent_id]
                        backlogs_local[agent_id] += self.env.backlogs[agent_id]
                        
                        if agent_id == active_training_node:
                            action_encoded = actions_encoded[agent_id]
                            value = ppo_data[agent_id]['value']
                            log_prob = ppo_data[agent_id]['log_prob']
                            
                            buffer = self.models[agent_id].rollout_buffer
                            buffer.add(
                                obs[agent_id],
                                np.array([action_encoded]),
                                np.array([rewards[agent_id]]),
                                np.array([episode_starts[agent_id]]),
                                value,
                                log_prob
                            )
                            episode_starts[agent_id] = done
                            self.models[agent_id].num_timesteps += 1
                            
                            if buffer.full:
                                with torch.no_grad():
                                    next_obs_tensor = torch.as_tensor(next_obs[agent_id]).float().to(self.models[agent_id].device).unsqueeze(0)
                                    next_value = self.models[agent_id].policy.predict_values(next_obs_tensor)
                                
                                buffer.compute_returns_and_advantage(last_values=next_value, dones=np.array([done]))
                                self.models[agent_id].train()
                                buffer.reset()"""

train_code = train_code.replace(old_buffer_logic, new_buffer_logic)


with open('once_a_time/sb3_mas_train_ppo.py', 'w') as f:
    f.write(train_code)

# ----------------- MAIN FILE -----------------
with open('once_a_time/sb3_main.py', 'r') as f:
    main_code = f.read()

main_code = main_code.replace('from sb3_mas_train import SB3_MAS_Train', 'from sb3_mas_train_ppo import SB3_MAS_Train_PPO')
main_code = main_code.replace('SB3_MAS_Train(', 'SB3_MAS_Train_PPO(')

with open('once_a_time/sb3_main_ppo.py', 'w') as f:
    f.write(main_code)

print("Files generated successfully!")
