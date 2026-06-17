import argparse
from ilp_solver import SB3_MAS_Train

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
proc_rate = 30
arrival_rate = 20

eps_init = 1.0
eps_fin = 0.05
# eps_dec = 0.999
eps_dec = 0.9985

num_agents = 5
batt_moliplicator_factor = 0.5
battery_capacities = [50, 100, 50, 60, 65, 80, 50, 55, 90, 70]
battery_capacities = [b * batt_moliplicator_factor for b in battery_capacities]

panel_moltiplicator_factor = 0.5
panel_surfaces = [0.45, 0.4, 0.50, 0.35, 0.4, 0.275, 0.35, 0.3, 0.5, 0.275]
panel_surfaces = [p * panel_moltiplicator_factor for p in panel_surfaces]


power_idle = 2.8
power_max = 8

w = 1.0

irradiance_datapaths = irradiance_datapaths[:num_agents]
battery_capacities = battery_capacities[:num_agents]
panel_surfaces = panel_surfaces[:num_agents]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ILP solver for MAS scheduling')
    parser.add_argument('--variable-arrival-rate', action='store_true',
                        help='Use sinusoidal variable arrival rate (2h period, per-agent phase shift)')
    parser.add_argument("--days", type=int, default=1)
    args = parser.parse_args()

    s = SB3_MAS_Train(
        num_agents,
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
          w,
          initial_backlog = 100,
          initial_energy = 0.5,
          processing_days = 1,
          days_to_process = args.days,
          variable_arrival_rate = args.variable_arrival_rate
          )
    s.solve()
    s.print_solution()