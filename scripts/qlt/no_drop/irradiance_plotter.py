import matplotlib.pyplot as plt
import numpy as np
from interpol import interpolate
import os

def plot_daily_irradiance_timestep(
    datapath,
    day_of_year,
    delta_time,
    proc_interval,
    max_irradiance=1000
):
    data = interpolate(datapath, delta_time, proc_interval)

    steps_per_day = int((24 * 60 * 60) / proc_interval)
    day_idx = day_of_year - 1
    start_idx = day_idx * steps_per_day
    end_idx = start_idx + steps_per_day
    
    enlightened_steps = []
    total_energy = 0.0

    ghi = data.iloc[start_idx:end_idx]['ghi'].values
    timesteps = np.arange(len(ghi))

    step = 0
    start_step = 0
    last_step = 0
    
    for elem in ghi:
        if(elem > 0.00):
            if(len(enlightened_steps) == 0):
                start_step = step
                print(f"first light step: {step}")
                
            enlightened_steps.append(elem)
            total_energy += (elem)
            last_step = step
        
        step += 1

    print(f"Last step with light: {last_step}")

    total_time = len(enlightened_steps) * proc_interval
    print(f"total time: {total_time}s - total_energy: {total_energy * proc_interval}J/m² - avg_irradiance: {total_energy / len(enlightened_steps)} - total_avg_irradiance: {total_energy / len(ghi)}")
                
    num_ticks = 20
    tick_indices = np.linspace(0, len(timesteps) - 1, num_ticks, dtype=int)
    tick_labels = timesteps[tick_indices]

    ### irradiance along the day
    
    plt.plot(timesteps, ghi, linewidth=2)

    plt.scatter([start_step], [enlightened_steps[0]], color='g', zorder=5, label=f"Start step: {start_step}")
    plt.scatter([last_step], [enlightened_steps[-1]], color='r', zorder=5, label=f"Last step: {last_step}")

    plt.xticks(tick_indices, tick_labels, rotation=45)
    plt.xlabel("Timestep")
    plt.ylabel("Irradiance")
    plt.suptitle(f"Daily irradiance - day {day_of_year}")
    plt.title(f"start: {start_step} min , end: {last_step} min , tot avg irradiance: {round(total_energy / 86400, 2)} W/m²")
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(os.getcwd(), f"irradiance_day_{day_of_year}_timestep.pdf")
    print(f"Saving plot to: {save_path}")
    plt.savefig(save_path)
    plt.close()


    ### irradiance along the steps with "light"
    plt.plot(enlightened_steps)
    plt.title(f"Steps with light - day {day_of_year}")

    plt.ylabel("Irradiance")
    plt.xlabel("Timestep")
    plt.grid(True)
    plt.savefig(f"./steps_with_light-day_{day_of_year}.pdf")
    plt.close()

    # plt.show()

if __name__ == "__main__":
    datapath = '../../../dataset/csv_41.89109712745386_12.503566993103867_fixed_23_180_PT15M_2024.csv'
    day_of_year = 172
    delta_time = 15 * 60
    proc_interval = 1 * 60

    plot_daily_irradiance_timestep(datapath, day_of_year, delta_time, proc_interval)
