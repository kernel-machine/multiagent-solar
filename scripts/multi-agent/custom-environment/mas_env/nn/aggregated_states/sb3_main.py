
import os
import sys

from sb3_mas_train_threads import SB3_MAS_Train_Parallelized_Threads
from sb3_mas_train_processes import SB3_MAS_Train_Parallelized_Processes
from custom_environment import CustomEnvironment

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from sb3_mas_train import SB3_MAS_Train

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
# eps_dec = 0.999
eps_dec = 0.9985

# num_agents = 2
# battery_capacities = [25, 100]
# panel_surfaces = [1.0, 0.5]


# num_agents = 3
# battery_capacities = [25, 100, 50]
# panel_surfaces = [1.0, 0.5, 0.75]

# num_agents = 4
# battery_capacities = [25, 100, 50, 37]
# panel_surfaces = [1.0, 0.5, 0.75, 0.85]

num_agents = 5
battery_capacities = [25, 100, 50, 37, 65]
panel_surfaces = [1.0, 0.5, 0.75, 0.85, 0.65]

# num_agents = 10
# battery_capacities = [
#     25, 
#     100,  
#     50,   
#     37,   
#     65,   
#     80,   
#     40,   
#     75,   
#     55,   
#     90    
# ]

# panel_surfaces = [
#     1.00,  
#     0.50,  
#     0.75,  
#     0.85,  
#     0.65,  
#     0.55,  
#     0.90,  
#     0.60,  
#     0.80,  
#     0.52   
# ]


power_idle = 2.6
power_max = 6.0

w = 1.0

train_freq = 16
batch_size = 256

mode = 'cuda'

seed = "linbear" # "fixed_winter", "fixed_summer", "linear"

env = CustomEnvironment(
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


if __name__ == "__main__":
    trainer1 = SB3_MAS_Train(
        num_agents,
        num_episodes,
        irradiance_datapaths,
        delta_time,
        proc_interval,
        proc_rate,
        arrival_rate,
        eps_init,
        eps_fin,
        eps_dec,
        battery_capacities,
        panel_surfaces,
        power_idle,
        power_max,
        train_freq,
        w,
        mode,
        batch_size,
        seed,
        env = env
        )
    
    # trainer2 = SB3_MAS_Train_Parallelized_Threads(
    #     num_agents,
    #     num_episodes,
    #     irradiance_datapaths,
    #     delta_time,
    #     proc_interval,
    #     proc_rate,
    #     arrival_rate,
    #     eps_init,
    #     eps_fin,
    #     eps_dec,
    #     battery_capacities,
    #     panel_surfaces,
    #     power_idle,
    #     power_max,
    #     train_freq,
    #     w,
    #     mode,
    #     batch_size,
    #     seed
    #     )
    
    # trainer3 = SB3_MAS_Train_Parallelized_Processes(
    #     num_agents,
    #     num_episodes,
    #     irradiance_datapaths,
    #     delta_time,
    #     proc_interval,
    #     proc_rate,
    #     arrival_rate,
    #     eps_init,
    #     eps_fin,
    #     eps_dec,
    #     battery_capacities,
    #     panel_surfaces,
    #     power_idle,
    #     power_max,
    #     train_freq,
    #     w,
    #     mode,
    #     batch_size,
    #     seed
    #     )
    
    # trainer3.train()
    # trainer2.train()
    #trainer1.train()
    trainer1.evaluate()
