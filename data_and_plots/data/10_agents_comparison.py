import numpy as np
import matplotlib.pyplot as plt
import csv

battery_capacities = [
    25, 
    100,  
    50,   
    37,   
    65,   
    80,   
    40,   
    75,   
    55,   
    90    
]


def compute_avg_rewards(episodes, day, interval, num_agents):
    rewards1 = [0 for episode in range(episodes+1)]
    rewards2 = [0 for episode in range(episodes+1)]
    
    
    for agent in range(num_agents):
        with open(f'./10agents/csvs_plots_hsp/final/rewards_agent_{battery_capacities[agent]}_{episodes}_{day}_{interval}_{num_agents}agents_cuda.csv') as file:
            csvFile = csv.reader(file)
            cnt = 0
            
            for line in csvFile:
                rewards1[cnt] += float(line[0])
                cnt += 1
            
    for id in range(episodes+1):
        rewards1[id] /= num_agents

    for agent in range(num_agents):
        with open(f'./10agents/iql/csvs_batch_256/rewards_agent_{battery_capacities[agent]}_{episodes}_{day}_{interval}_{num_agents}agents_cuda.csv') as file:
            csvFile = csv.reader(file)
            cnt = 0
            
            for line in csvFile:
                rewards2[cnt] += float(line[0])
                cnt += 1
            
    for id in range(episodes+1):
        rewards2[id] /= num_agents
             
    
    print(f"fine tuning: {np.mean(rewards1)}")
    print(f"IQL: {np.mean(rewards2)}")
    
             
    window = 10
    plt.suptitle("Average rewards")
    plt.title(f"Episodes: {episodes}, Day: {day}, Interval: {interval}, num_agents: {num_agents}, Mode: cuda")
    
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    
    plt.plot(range(window - 1, len(rewards1)), np.convolve(rewards1, np.ones(window)/window, mode='valid'), label = f"fine tuning", linewidth = 2.0,  alpha = 1.0)
    plt.plot(range(window - 1, len(rewards2)), np.convolve(rewards2, np.ones(window)/window, mode='valid'), label = f"heavy tuning", linewidth = 2.0, alpha = 1.0)

    plt.grid()
    plt.legend()
    # plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
    # plt.tight_layout()

    plt.savefig(f"./comparisons/rewards/average_episodes_{episodes}_{day}_{interval}_{num_agents}agents_once-a-time_reduced_states_cuda.pdf")
    plt.close()
    
def compute_avg_matchings(episodes, day, interval, num_agents):
    rewards1 = [0 for episode in range(episodes+1)]
    rewards2 = [0 for episode in range(episodes+1)]
    
    
    for agent in range(num_agents):
        with open(f'./10agents/csvs_plots_hsp/matchings/matchings_{battery_capacities[agent]}_{episodes}_{day}_{interval}_{num_agents}agents_cuda.csv') as file:
            csvFile = csv.reader(file)
            cnt = 0
            
            for line in csvFile:
                rewards1[cnt] += float(line[0])
                cnt += 1
            
    for id in range(episodes+1):
        rewards1[id] /= num_agents

    for agent in range(num_agents):
        with open(f'./10agents/iql/csvs_batch_256/matchings_{battery_capacities[agent]}_{episodes}_{day}_{interval}_{num_agents}agents_cuda.csv') as file:
            csvFile = csv.reader(file)
            cnt = 0
            
            for line in csvFile:
                rewards2[cnt] += float(line[0])
                cnt += 1
            
    for id in range(episodes+1):
        rewards2[id] /= num_agents
             
    window = 10
    plt.suptitle("Average offloading matchings")
    plt.title(f"Episodes: {episodes}, Day: {day}, Interval: {interval}, num_agents: {num_agents}, Mode: cuda")
    
    plt.xlabel("Episodes")
    plt.ylabel("Matchings")
    
    plt.plot(range(window - 1, len(rewards1)), np.convolve(rewards1, np.ones(window)/window, mode='valid'), label = f"fine tuning", linewidth = 2.0,  alpha = 1.0)
    plt.plot(range(window - 1, len(rewards2)), np.convolve(rewards2, np.ones(window)/window, mode='valid'), label = f"heavy tuning", linewidth = 2.0, alpha = 1.0)

    plt.grid()
    plt.legend()
    # plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
    # plt.tight_layout()

    plt.savefig(f"./comparisons/average_matchings_episodes_{episodes}_{day}_{interval}_{num_agents}agents_once-a-time_reduced_states_cuda.pdf")
    plt.close()
    
def compute_avg_time(episodes, day, interval, num_agents):
    rewards1 = [0 for episode in range(episodes+1)]
    rewards2 = [0 for episode in range(episodes+1)]
    
    
    with open(f'./10agents/csvs_plots_hsp/time/time_{episodes}_{day}_{interval}_{num_agents}agents_cuda.csv') as file:
        csvFile = csv.reader(file)
        cnt = 0
            
        for line in csvFile:
            rewards1[cnt] += float(line[0])
            cnt += 1
            
    # for id in range(episodes+1):
    #     rewards1[id] /= num_agents

    with open(f'./10agents/iql/csvs_batch_256/time_{episodes}_{day}_{interval}_{num_agents}agents_cuda.csv') as file:
        csvFile = csv.reader(file)
        cnt = 0
        
        for line in csvFile:
            rewards2[cnt] += float(line[0])
            cnt += 1
            
    # for id in range(episodes+1):
    #     rewards2[id] /= num_agents
    
    print(f"fine tuning: {np.mean(rewards1)}")
    print(f"IQL: {np.mean(rewards2)}")
    
             
    window = 10
    plt.suptitle("Average episodical")
    plt.title(f"Episodes: {episodes}, Day: {day}, Interval: {interval}, num_agents: {num_agents}, Mode: cuda")
    
    plt.xlabel("Episodes")
    plt.ylabel("Time")
    
    plt.plot(range(window - 1, len(rewards1)), np.convolve(rewards1, np.ones(window)/window, mode='valid'), label = f"fine tuning", linewidth = 2.0,  alpha = 1.0)
    plt.plot(range(window - 1, len(rewards2)), np.convolve(rewards2, np.ones(window)/window, mode='valid'), label = f"IQL", linewidth = 2.0, alpha = 1.0)

    plt.grid()
    plt.legend()
    # plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
    # plt.tight_layout()

    plt.savefig(f"./comparisons/average_time_episodes_{episodes}_{day}_{interval}_{num_agents}agents_once-a-time_reduced_states_cuda.pdf")
    plt.close()



def compute_avg_framerate(episodes, day, interval, num_agents):
    rewards1 = [0 for episode in range(episodes+1)]
    rewards2 = [0 for episode in range(episodes+1)]
    
    
    for agent in range(num_agents):
        with open(f'./10agents/csvs_plots_hsp/final/total_framerate_{battery_capacities[agent]}_{episodes}_{day}_{interval}_{num_agents}agents_cuda_FIXED.csv') as file:
            csvFile = csv.reader(file)
            cnt = 0
            
            for line in csvFile:
                rewards1[cnt] += float(line[0])
                cnt += 1
            
    for id in range(episodes+1):
        rewards1[id] /= num_agents

    for agent in range(num_agents):
        with open(f'./10agents/iql/csvs_batch_256/total_framerate_{battery_capacities[agent]}_{episodes}_{day}_{interval}_{num_agents}agents_cuda_FIXED.csv') as file:
            csvFile = csv.reader(file)
            cnt = 0
            
            for line in csvFile:
                rewards2[cnt] += float(line[0])
                cnt += 1
            
    for id in range(episodes+1):
        rewards2[id] /= num_agents
             
    window = 10
    plt.suptitle("Average framerate")
    plt.title(f"Episodes: {episodes}, Day: {day}, Interval: {interval}, num_agents: {num_agents}, Mode: cuda")
    
    plt.xlabel("Episodes")
    plt.ylabel("Framerate")
    
    plt.plot(range(window - 1, len(rewards1)), np.convolve(rewards1, np.ones(window)/window, mode='valid'), label = f"fine tuning", linewidth = 2.0, alpha = 1.0)
    plt.plot(range(window - 1, len(rewards2)), np.convolve(rewards2, np.ones(window)/window, mode='valid'), label = f"heavy tuning", linewidth = 2.0, alpha = 1.0)

    plt.grid()
    plt.legend()
    # plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
    # plt.tight_layout()
    plt.savefig(f"./comparisons/framerates/average_framerates_{episodes}_{day}_{interval}_{num_agents}agents_once-a-time_reduced_states_cuda.pdf")
    plt.close()
    

def compute_avg_backlog(episodes, day, interval, num_agents):
    backlog1_all_agents = []
    backlog1_samples = []
    
    backlog2_all_agents = []
    backlog2_samples = []
    
    for agent in range(num_agents):
        with open(f'./10agents/csvs_plots_hsp/final/backlog_{battery_capacities[agent]}_{episodes}_{day}_{interval}_{num_agents}agents_cuda.csv') as file:
            csvFile = csv.reader(file)
            next(csvFile)
            
            agent_episodes = []
            for line in csvFile:
                timesteps = [float(val) for val in line]
                episode_mean = np.mean(timesteps)
                agent_episodes.append(episode_mean)
            
            backlog1_all_agents.append(agent_episodes)
            
    backlog1_all_agents = np.array(backlog1_all_agents)
    backlog1_samples = np.mean(backlog1_all_agents, axis=0)

    for agent in range(num_agents):
        with open(f'./10agents/iql/csvs_batch_256/backlog_{battery_capacities[agent]}_{episodes}_{day}_{interval}_{num_agents}agents_cuda.csv') as file:
            csvFile = csv.reader(file)
            next(csvFile)
            
            agent_episodes = []
            for line in csvFile:
                timesteps = [float(val) for val in line]
                episode_mean = np.mean(timesteps)
                agent_episodes.append(episode_mean)
            
            backlog2_all_agents.append(agent_episodes)
            
    backlog2_all_agents = np.array(backlog2_all_agents)
    backlog2_samples = np.mean(backlog2_all_agents, axis=0)
    

    plt.suptitle("Average backlog (sampled episodes)")
    plt.title(f"Episodes: {episodes}, Day: {day}, Interval: {interval}, num_agents: {num_agents}, Mode: cuda")
    
    sample_episodes = [i * int(episodes / 10) for i in range(len(backlog2_samples)-1)]
        
    plt.xlabel("Episode * 400")
    plt.ylabel("Average Backlog")
    
    plt.plot(sample_episodes, backlog1_samples, 'o-', label=f"fine tuning", alpha=1.0, markersize=8, linewidth=2)
    plt.plot(sample_episodes, backlog2_samples[:-1], 's-', label=f"heavy tuning", alpha=1.0, markersize=8, linewidth=2)

    plt.grid(alpha=0.3)
    plt.legend()
    # plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
    # plt.tight_layout()

    plt.savefig(f"./comparisons/backlog/average_backlog_sampled_{episodes}_{day}_{interval}_{num_agents}agents_once-a-time_reduced_states_cuda.pdf")
    plt.close()
 
 
def compute_avg_battery(episodes, day, interval, num_agents):
    backlog1_all_agents = []
    backlog1_samples = []
    
    backlog2_all_agents = []
    backlog2_samples = []
    
    for agent in range(num_agents):
        with open(f'./10agents/csvs_plots_hsp/final/battery_{battery_capacities[agent]}_{episodes}_{day}_{interval}_{num_agents}agents_cuda.csv') as file:
            csvFile = csv.reader(file)
            next(csvFile)
            
            agent_episodes = []
            for line in csvFile:
                timesteps = [float(val) for val in line]
                episode_mean = np.mean(timesteps)
                agent_episodes.append(episode_mean)
            
            backlog1_all_agents.append(agent_episodes)
            
    backlog1_all_agents = np.array(backlog1_all_agents)
    backlog1_samples = np.mean(backlog1_all_agents, axis=0)

    for agent in range(num_agents):
        with open(f'./10agents/iql/csvs_batch_256/battery_{battery_capacities[agent]}_{episodes}_{day}_{interval}_{num_agents}agents_cuda.csv') as file:
            csvFile = csv.reader(file)
            next(csvFile)
            
            agent_episodes = []
            for line in csvFile:
                timesteps = [float(val) for val in line]
                episode_mean = np.mean(timesteps)
                agent_episodes.append(episode_mean)
            
            backlog2_all_agents.append(agent_episodes)
            
    backlog2_all_agents = np.array(backlog2_all_agents)
    backlog2_samples = np.mean(backlog2_all_agents, axis=0)
    

    plt.suptitle("Average battery (sampled episodes)")
    plt.title(f"Episodes: {episodes}, Day: {day}, Interval: {interval}, num_agents: {num_agents}, Mode: cuda")
    
    sample_episodes = [i * int(episodes / 10) for i in range(len(backlog2_samples)-1)]
        
    plt.xlabel("Episode * 400")
    plt.ylabel("Average Battery")
    
    plt.plot(sample_episodes, backlog1_samples, 'o-', label=f"fine tuning", alpha=1.0, markersize=8, linewidth=2)
    plt.plot(sample_episodes, backlog2_samples[:-1], 's-', label=f"heavy tuning", alpha=1.0, markersize=8, linewidth=2)

    plt.grid(alpha=0.3)
    plt.legend()
    # plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
    # plt.tight_layout()

    plt.savefig(f"./comparisons/backlog/average_battery_sampled_{episodes}_{day}_{interval}_{num_agents}agents_once-a-time_reduced_states_cuda.pdf")
    plt.close()

compute_avg_rewards(1000, 355, 60, 10)
compute_avg_framerate(1000, 355, 60, 10)
compute_avg_backlog(1000, 355, 60, 10)
compute_avg_battery(1000, 355, 60, 10)
compute_avg_time(1000, 355, 60, 10)

compute_avg_matchings(1000, 355, 60, 10)