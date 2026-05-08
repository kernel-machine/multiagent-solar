
import argparse
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


battery_capacities = [50, 100, 50, 60, 65, 80, 50, 55, 90, 70]
panel_surfaces = [1.0, 0.5, 0.75, 0.85, 0.65, 0.55, 0.90, 0.60, 0.80, 0.52]

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
batch_size = 64

mode = 'cuda'

seed = "fixed_winter" # "fixed_winter", "fixed_summer", "linear"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-episodes", type=int, default=1001, help="Number of episodes to train.")
    parser.add_argument("--total-steps", type=int, default=None, help="Hard cap on total training steps.")
    parser.add_argument("--train-all-mode", type=int, choices=[1, 2], default=2, help="1: old joint training; 2: rotating training.")
    parser.add_argument("--rotation-episodes", type=int, default=1, help="Episodes assigned to each agent before rotating to the next one.")
    parser.add_argument("--rotation-cycles", type=int, default=None, help="Number of full agent rotations to run. Derives num_episodes as num_agents * rotation_episodes * rotation_cycles.")
    parser.add_argument("--num-envs", type=int, default=4, help="Number of parallel PPO envs (used in both mode 1 and mode 2; ignored for DQN).")
    parser.add_argument("--algo", type=str, choices=["DQN", "PPO"], default="PPO", help="RL algorithm to use.")
    parser.add_argument("--num-agents", type=int, default=5, help="Number of agents.")
    parser.add_argument(
        "--cross-attention",
        "--use-cross-attention",
        dest="cross_attention",
        action="store_true",
        default=False,
        help="Use cross-attention feature extractor instead of min/avg/max aggregation.",
    )
    parser.add_argument("--max-agents", type=int, default=None, help="Padded observation size for cross-attention (defaults to --num-agents). Set higher to train a policy that generalises to more agents.")
    parser.add_argument("--attn-d-model", type=int, default=16, help="Attention embedding dimension (default: 16).")
    args = parser.parse_args()

    effective_max_agents = args.max_agents if args.max_agents is not None else args.num_agents

    env = CustomEnvironment(
        args.num_agents,
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
        use_cross_attention=args.cross_attention,
        max_agents=effective_max_agents,
    )

    effective_num_episodes = args.num_episodes
    if args.rotation_cycles is not None:
        effective_num_episodes = args.num_agents * args.rotation_episodes * args.rotation_cycles
        print(
            f"Using rotation_cycles={args.rotation_cycles}; derived num_episodes={effective_num_episodes} "
            f"({args.num_agents} agents x {args.rotation_episodes} rotation_episodes)."
        )

    effective_total_steps = args.total_steps
    if effective_total_steps is not None:
        print(f"Hard training budget set to total_steps={effective_total_steps}.")

    save_path = (
        "models/aggregated_states/mode"
        + str(args.train_all_mode)
        + ("_cross_attention" if args.cross_attention else "")
        + f"/{args.num_agents}agents"
    )

    trainer1 = SB3_MAS_Train(
        args.num_agents,
        effective_num_episodes,
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
        env=env,
        save_path=save_path,
        num_envs=args.num_envs,
        algo=args.algo,
        train_all_mode=args.train_all_mode,
        rotation_episodes=args.rotation_episodes,
        total_steps=effective_total_steps,
        use_cross_attention=args.cross_attention,
        max_agents=effective_max_agents,
        attn_d_model=args.attn_d_model,
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
    trainer1.train()
    trainer1.evaluate(model_paths=save_path)