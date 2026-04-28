import matplotlib.pyplot as plt
import numpy as np
from ilp_solver import SB3_MAS_Train

irradiance_datapaths = [
    '../../../../../dataset/csv_41.89109712745386_12.503566993103867_fixed_23_180_PT15M_2024.csv',
    '../../../../../dataset/csv_41.89109712745386_12.503566993103867_fixed_23_180_PT15M_2024.csv',
    '../../../../../dataset/csv_41.89109712745386_12.503566993103867_fixed_23_180_PT15M_2024.csv',
    '../../../../../dataset/csv_41.89109712745386_12.503566993103867_fixed_23_180_PT15M_2024.csv',
    '../../../../../dataset/csv_41.89109712745386_12.503566993103867_fixed_23_180_PT15M_2024.csv'
]

s = SB3_MAS_Train(5, irradiance_datapaths, 900, 300, 20, 15, 1.0, 0.05, 0.9985, [25, 100, 50, 37, 65], [1.0, 0.5, 0.75, 0.85, 0.65], 2.6, 6.0, 1.0)
s.solve()

agent_id = 0
tx_energy = [sum(s.x[agent_id, j, t].value * s.e_tx_rx for j in range(s.num_agents) if j != agent_id) for t in range(len(s.irradiance_arrays[0]))]
print("Max tx_energy for Agent 0:", max(tx_energy))
print("Min tx_energy for Agent 0:", min(tx_energy))
print("First 10 tx_energy values:", tx_energy[:10])

