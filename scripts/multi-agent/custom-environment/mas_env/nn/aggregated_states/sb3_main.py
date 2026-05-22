import argparse
import os
import sys

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



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-episodes", type=int, default=1001, help="Number of episodes to train.")
    parser.add_argument("--total-steps", type=int, default=None, help="Hard cap on total training steps.")
    parser.add_argument("--train-all-mode", type=int, choices=[1, 2], default=1, help="1: old joint training; 2: rotating training.")
    parser.add_argument("--rotation-episodes", type=int, default=1, help="Episodes assigned to each agent before rotating to the next one.")
    parser.add_argument("--rotation-cycles", type=int, default=None, help="Number of full agent rotations to run. Derives num_episodes as num_agents * rotation_episodes * rotation_cycles.")
    parser.add_argument("--num-envs", type=int, default=4, help="Number of parallel PPO envs (used in both mode 1 and mode 2; ignored for DQN).")
    parser.add_argument("--algo", type=str, choices=["DQN", "PPO"], default="PPO", help="RL algorithm to use.")
    parser.add_argument("--termination-mode", type=str, choices=["early", "penalty"], default="early", help="early: end episode when an agent dies. penalty: continue to max_steps and apply penalty.")
    parser.add_argument("--eval-termination-mode", type=str, choices=["early", "penalty"], default=None, help="Termination mode during evaluation. Default is None (same as training).")
    parser.add_argument("--battery-hard-threshold", type=float, default=0.0, help="Battery ratio in [0, 1]. When an agent reaches this level, the environment forces a safe no-op mode; 0 disables the constraint.")
    parser.add_argument("--num-agents", type=int, default=5, help="Number of agents.")
    parser.add_argument("--max-agents", type=int, default=None, help="Padded observation size for cross-attention (defaults to --num-agents). Set higher to train a policy that generalises to more agents.")
    parser.add_argument("--cross-attention", action="store_true", help="Use cross-attention mechanism.")
    parser.add_argument("--deepsets", action="store_true", help="Use deep sets mechanism without spatial information.")
    parser.add_argument("--deepsets-spatial", action="store_true", help="Use deep sets mechanism with spatial information (index/position).")
    parser.add_argument("--attn-d-model", type=int, default=16, help="Attention embedding dimension (default: 16).")
    parser.add_argument("--random-nodes", type=int, default=0, help="Insert the battery and backlog values of X randomly selected nodes into the state.")
    parser.add_argument("--gossip", action="store_true", help="Enable gossip mechanism for information sharing.")
    parser.add_argument("--gossip-interval", type=int, default=5, help="Number of steps between gossip communications.")
    parser.add_argument("--gossip-targets", type=int, default=2, help="Number of random nodes to send info to during gossip.")
    parser.add_argument("--gossip-state-nodes", type=int, default=3, help="Number of nodes to include in the state from gossip memory.")
    parser.add_argument("--disable-evaluation", action="store_true", help="Disable evaluation after training.")
    parser.add_argument("--disable-offloading", action="store_true", help="Temporarily disable offloading mechanism in the environment.")
    parser.add_argument("--lstm-prediction", action="store_true", help="Enable LSTM-based GHI prediction with attention pooling (adds 32-dim forecast feature to agent state).")
    parser.add_argument("--lstm-prediction-demo", action="store_true", help="Like --lstm-prediction but uses real future GHI data instead of LSTM predictions (oracle baseline).")
    parser.add_argument("--net-width", type=int, default=64, help="Width (neurons) of each hidden layer in the policy/value networks (default: 64).")
    parser.add_argument("--net-layers", type=int, default=2, help="Number of hidden layers in the policy/value networks (default: 2).")
    parser.add_argument("--eval-days", type=int, default=1, help="Number of days to simulate during final evaluation (default: 1).")
    parser.add_argument("--train-days", type=int, default=1, help="Number of days per training episode (default: 1). Higher values teach agents to conserve battery across day boundaries.")
    parser.add_argument("--seed", type=str, default="random", help="Seed for the random number generator (options: fixed_winter, fixed_summer, linear, random).", choices=["fixed_winter", "fixed_summer", "linear", "random"])
    args = parser.parse_args()

    if args.lstm_prediction and args.lstm_prediction_demo:
        parser.error("--lstm-prediction and --lstm-prediction-demo are mutually exclusive.")

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
        args.seed,
        use_cross_attention=args.cross_attention,
        use_deepsets=args.deepsets,
        use_deepsets_spatial=args.deepsets_spatial,
        max_agents=effective_max_agents,
        random_nodes=args.random_nodes,
        use_gossip=args.gossip,
        gossip_interval=args.gossip_interval,
        gossip_targets=args.gossip_targets,
        gossip_state_nodes=args.gossip_state_nodes,
        battery_hard_threshold=args.battery_hard_threshold,
        use_random_battery=True,
        use_lstm_prediction=args.lstm_prediction,
        use_lstm_prediction_demo=args.lstm_prediction_demo,
        disable_offloading=args.disable_offloading,
    )

    # Extend training episodes to span multiple days
    if args.train_days > 1:
        env.max_steps = env.max_steps * args.train_days
        print(f"Multi-day training: {args.train_days} days/episode ({env.max_steps} steps)")

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

    suffix = ""
    if args.cross_attention:
        suffix = "_cross_attention"
    elif args.deepsets_spatial:
        suffix = "_deepsets_spatial"
    elif args.deepsets:
        suffix = "_deepsets"
    elif args.gossip:
        suffix = f"_gossip_{args.gossip_interval}_{args.gossip_targets}_{args.gossip_state_nodes}"
    elif args.random_nodes > 0:
        suffix = f"_random_nodes_{args.random_nodes}"
    
    if args.lstm_prediction:
        suffix += "_lstm_prediction"
    elif args.lstm_prediction_demo:
        suffix += "_lstm_prediction_demo"

    if args.train_days > 1:
        suffix += f"_{args.train_days}days"

    save_path = (
        "models/aggregated_states/mode"
        + str(args.train_all_mode)
        + suffix
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
        seed=1234,
        env=env,
        save_path=save_path,
        num_envs=args.num_envs,
        algo=args.algo,
        train_all_mode=args.train_all_mode,
        termination_mode=args.termination_mode,
        eval_termination_mode=args.eval_termination_mode,
        rotation_episodes=args.rotation_episodes,
        total_steps=effective_total_steps,
        max_agents=effective_max_agents,
        attn_d_model=args.attn_d_model,
        use_deepsets=args.deepsets,
        use_deepsets_spatial=args.deepsets_spatial,
        use_cross_attention=args.cross_attention,
        evaluation_enabled=not args.disable_evaluation,
        use_lstm_prediction=args.lstm_prediction or args.lstm_prediction_demo,
        net_arch=[args.net_width] * args.net_layers,
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
    trainer1.save_args(vars(args))
    trainer1.train()
    trainer1.evaluate(model_paths=save_path, eval_days=args.eval_days)