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
proc_rate = 20
arrival_rate = 15

eps_init = 1.0
eps_fin = 0.05
# eps_dec = 0.999
eps_dec = 0.9985

num_agents = 5
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

w = 1.0

irradiance_datapaths = irradiance_datapaths[:num_agents]
battery_capacities = battery_capacities[:num_agents]
panel_surfaces = panel_surfaces[:num_agents]

if __name__ == '__main__':
    s = SB3_MAS_Train(num_agents,
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
          w)
    s.solve()
    s.print_solution()