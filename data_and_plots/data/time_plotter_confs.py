import numpy as np
import matplotlib.pyplot as plt
import csv

def conf_comparison(episodes, day, interval, num_agents, mode, mode1, mode2):
    mode1_local = 0
    mode1_vector = []
    mode1_vector_single = []
    
    mode2_local = 0
    mode2_vector = []
    mode2_vector_single = []
    
    with open(f'./{mode1}/csvs/csvs_batch_256/time_{episodes}_{day}_{interval}_{num_agents}agents_cuda.csv') as file:
            csvFile = csv.reader(file)
            for line in csvFile:
                mode1_vector_single.append(float(line[0]))
                mode1_local += float(line[0])
                mode1_vector.append(mode1_local)

    with open(f'./{mode2}/csvs/csvs_batch_256/time_{episodes}_{day}_{interval}_{num_agents}agents_cuda.csv') as file:
            csvFile = csv.reader(file)
            for line in csvFile:
                mode2_vector_single.append(float(line[0]))
                mode2_local += float(line[0])
                mode2_vector.append(mode2_local)
    
    
    window = 10
    plt.suptitle("Total time taken by execution comparison")
    plt.title(f"Episodes: {episodes}, Day: {day}, Interval: {interval}, num_agents: {num_agents}, Mode: {mode}")
    
    plt.xlabel("Episodes")
    plt.ylabel("Time [s]")
        
    print("mode: ", mode == "")
    
    plt.plot(range(window - 1, len(mode1_vector)), np.convolve(mode1_vector, np.ones(window)/window, mode='valid'), label = f"{mode1}", alpha = 1.0)
    plt.plot(range(window - 1, len(mode2_vector)), np.convolve(mode2_vector, np.ones(window)/window, mode='valid'), label = f"{mode2}", alpha = 1.0)
    
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./comparisons/time/long_time_comparison_{episodes}_{day}_{interval}_{num_agents}agents_{mode}_{mode1}_{mode2}.pdf")
    plt.close()

    plt.suptitle("Time taken per episode comparison")
    plt.title(f"Episodes: {episodes}, Day: {day}, Interval: {interval}, num_agents: {num_agents}, Mode: {mode}")
    
    plt.xlabel("Episodes")
    plt.ylabel("Time [s]")
    
    plt.plot(range(window - 1, len(mode1_vector_single)), np.convolve(mode1_vector_single, np.ones(window)/window, mode='valid'), label = f"{mode1}", alpha = 1.0)
    plt.plot(range(window - 1, len(mode2_vector_single)), np.convolve(mode2_vector_single, np.ones(window)/window, mode='valid'), label = f"{mode2}", alpha = 1.0)
    
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./comparisons/time/episodical_time_comparison_episode_{episodes}_{day}_{interval}_{num_agents}agents_{mode}_{mode1}_{mode2}.pdf")
    plt.close()
    
    
def triple_conf_comparison(episodes, day, interval, num_agents, mode, mode1, mode2, mode3):
    mode1_local = 0
    mode1_vector = []
    mode1_vector_single = []
    
    mode2_local = 0
    mode2_vector = []
    mode2_vector_single = []
    
    mode3_local = 0
    mode3_vector = []
    mode3_vector_single = []
    
    with open(f'./{mode1}/csvs/csvs_batch_256/time_{episodes}_{day}_{interval}_{num_agents}agents_cuda.csv') as file:
            csvFile = csv.reader(file)
            for line in csvFile:
                mode1_vector_single.append(float(line[0]))
                mode1_local += float(line[0])
                mode1_vector.append(mode1_local)

    with open(f'./{mode2}/csvs/csvs_batch_256/time_{episodes}_{day}_{interval}_{num_agents}agents_cuda.csv') as file:
            csvFile = csv.reader(file)
            for line in csvFile:
                mode2_vector_single.append(float(line[0]))
                mode2_local += float(line[0])
                mode2_vector.append(mode2_local)
    
    with open(f'./{mode3}/csvs/csvs_batch_256/time_{episodes}_{day}_{interval}_{num_agents}agents_cuda.csv') as file:
            csvFile = csv.reader(file)
            for line in csvFile:
                mode3_vector_single.append(float(line[0]))
                mode3_local += float(line[0])
                mode3_vector.append(mode3_local)
    
    
    window = 10
    plt.suptitle("Total time taken by execution comparison")
    plt.title(f"Episodes: {episodes}, Day: {day}, Interval: {interval}, num_agents: {num_agents}, Mode: {mode}")
    
    plt.xlabel("Episodes")
    plt.ylabel("Time [s]")
    
    print("mode: ", mode == "")
    
    plt.plot(range(window - 1, len(mode1_vector)), np.convolve(mode1_vector, np.ones(window)/window, mode='valid'), label = f"{mode1}", alpha = 1.0)
    plt.plot(range(window - 1, len(mode2_vector)), np.convolve(mode2_vector, np.ones(window)/window, mode='valid'), label = f"{mode2}", alpha = 1.0)
    plt.plot(range(window - 1, len(mode2_vector)), np.convolve(mode3_vector, np.ones(window)/window, mode='valid'), label = f"{mode3}", alpha = 1.0)
    
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./comparisons/time/long_time_comparison_{episodes}_{day}_{interval}_{num_agents}agents_{mode}_{mode1}_{mode2}_{mode3}.pdf")
    plt.close()

    plt.suptitle("Time taken per episode comparison")
    plt.title(f"Episodes: {episodes}, Day: {day}, Interval: {interval}, num_agents: {num_agents}, Mode: {mode}")
    
    plt.xlabel("Episodes")
    plt.ylabel("Time [s]")
    # plt.ylim(3.70)
    
    plt.plot(range(window - 1, len(mode1_vector_single)), np.convolve(mode1_vector_single, np.ones(window)/window, mode='valid'), label = f"{mode1}", alpha = 1.0)
    plt.plot(range(window - 1, len(mode2_vector_single)), np.convolve(mode2_vector_single, np.ones(window)/window, mode='valid'), label = f"{mode2}", alpha = 1.0)
    plt.plot(range(window - 1, len(mode3_vector_single)), np.convolve(mode3_vector_single, np.ones(window)/window, mode='valid'), label = f"{mode3}", alpha = 1.0)
    
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./comparisons/time/episodical_time_comparison_episode_{episodes}_{day}_{interval}_{num_agents}agents_{mode}_{mode1}_{mode2}_{mode3}.pdf")
    plt.close()
    
    
conf_comparison(4000, 355, 60, 5, "cuda", "reduced_states", "tanh_z_score")
# triple_conf_comparison(4000, 355, 60, 5, "cuda", "aggregated_states", "reduced_states", "local_only")