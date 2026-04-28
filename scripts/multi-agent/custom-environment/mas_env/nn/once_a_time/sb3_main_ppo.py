import argparse
from sb3_mas_train_ppo import SB3_MAS_Train_PPO


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
num_rounds = 5

# num_agents = 2
# battery_capacities = [25, 100]
# panel_surfaces = [1.0, 0.5]


# num_agents = 3
# battery_capacities = [25, 100, 50]
# panel_surfaces = [1.0, 0.5, 0.75]

# num_agents = 4
# battery_capacities = [25, 100, 50, 37]
# panel_surfaces = [1.0, 0.5, 0.75, 0.85]

# num_agents = 5
# battery_capacities = [25, 100, 50, 37, 65]
# panel_surfaces = [1.0, 0.5, 0.75, 0.85, 0.65]

num_agents = 4
battery_capacities = [
    50,     #
    100,    #
    50,     #
    37,     #
    65,     #
    80,     #
    40,     #
    75,     # X
    55,     #
    90      # x
]

panel_surfaces = [
    1.00,  
    0.50,  
    0.75,  
    0.85,  
    0.65,  
    0.55,  
    0.90,  
    0.60,  
    0.80,  
    0.52   
]


power_idle = 2.6
power_max = 6.0

smart_node = 0

w = 1.0

train_freq = 16
batch_size = 256
parallel_envs = 4

mode = 'cuda'
# seed = "fixed_summer"
seed = "fixed_winter"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-all-mode",
        type=int,
        choices=[1, 2],
        default=1,
        help=(
            "1: train each node one-by-one while others are random; "
            "2: round-robin training where teammates use latest learned checkpoints"
        ),
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=num_rounds,
        help="Number of round-robin turns when --train-all-mode 2 is enabled",
    )
    args = parser.parse_args()

    trainer1 = SB3_MAS_Train_PPO(
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
        parallel_envs,
        args.train_all_mode,
        args.num_rounds
        )
    
    #trainer1.train()
    trainer1.evaluate()

    # seed = "fixed_summer"
    
    # trainer1 = SB3_MAS_Train_PPO(
    #     num_agents,
    #     num_episodes,
    #     irradiance_datapaths,
    #     delta_time,
    #     proc_interval,
    #     proc_rate,
    #     arrival_rate,
    #     battery_capacities,
    #     panel_surfaces,
    #     power_idle,
    #     power_max,
    #     train_freq,
    #     w,
    #     mode,
    #     batch_size,
    #     smart_node,
    #     seed
    #     )
    
    #trainer1.train()
    #trainer1.evaluate()
