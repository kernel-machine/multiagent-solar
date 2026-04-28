from ilp_solver import SB3_MAS_Train

irradiance_datapaths = [
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
eps_dec = 0.9985

num_agents = 5
battery_capacities = [25, 100, 50, 37, 65]
panel_surfaces = [1.00, 0.50, 0.75, 0.85, 0.65]

power_idle = 2.6
power_max = 6.0
w = 1.0

s = SB3_MAS_Train(num_agents, irradiance_datapaths, delta_time, proc_interval, proc_rate, arrival_rate, eps_init, eps_fin, eps_dec, battery_capacities, panel_surfaces, power_idle, power_max, w)
s.solve()

print("\n--- Valori numerici per Agent 0 tra t=180 e t=200 ---")
for t in range(180, 200):
    processed = s.x[0, 0, t].value
    normalized = processed / (s.proc_rate * s.proc_interval_s)
    buffer_val = s.buffer[0, t].value
    tx = sum(s.x[0, j, t].value for j in range(5) if j != 0)
    battery = s.battery[0, t].value / (s.battery_capacities_wh[0] * 3600)
    print(f"t={t} | Processed: {normalized:.3f} ({processed:.0f} frames) | Buffer: {buffer_val:.0f} | Tx: {tx:.0f} | Batt: {battery:.3f}")

