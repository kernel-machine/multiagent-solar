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
proc_vals = s.x[agent_id, agent_id, :].value / (s.proc_rate * s.proc_interval_s)
buffer_vals = s.buffer[agent_id, :].value / (s.arrival_rate * s.proc_interval_s * 150)
tx_frames = [sum(s.x[agent_id, j, t].value for j in range(s.num_agents) if j != agent_id) / (s.proc_rate * s.proc_interval_s) for t in range(len(s.irradiance_arrays[0]))]
batt_vals = s.battery[agent_id, :].value / (s.battery_capacities_wh[agent_id] * 3600)

print("\n--- Agent 0 Values (t=175 to 195) ---")
for t in range(175, 196):
    print(f"t={t:3d} | Proc: {proc_vals[t]:.3f} | Tx: {tx_frames[t]:.3f} | Buff: {buffer_vals[t]:.3f} | Batt: {batt_vals[t]:.3f}")

print("\n--- Agent 0 Values (t=245 to 255) ---")
for t in range(245, 256):
    print(f"t={t:3d} | Proc: {proc_vals[t]:.3f} | Tx: {tx_frames[t]:.3f} | Buff: {buffer_vals[t]:.3f} | Batt: {batt_vals[t]:.3f}")

print("\n--- Agent 0 Buffer End ---")
print(f"t=288 | Buff: {buffer_vals[288]:.3f}")
