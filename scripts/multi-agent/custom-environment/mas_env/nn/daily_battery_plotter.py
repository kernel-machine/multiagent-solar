import numpy as np
import matplotlib.pyplot as plt
import csv
import os

folder_path   = "./csv_avg_plots"
episodes      = 4001
timesteps     = 1440
samplings     = 11
agents_tag    = "5agents"
WINDOW        = 40
PLOT_DIR      = "./daily_battery_plots"
os.makedirs(PLOT_DIR, exist_ok=True)

sample_labels = [i * int(episodes / 10) for i in range(samplings)]

def smooth(data, window):
    if len(data) < window:
        return np.array(data)
    return np.convolve(data, np.ones(window) / window, mode="valid")

def load_csv(filepath):
    rows = []
    with open(filepath) as f:
        reader = csv.reader(f)
        next(reader)
        for line in reader:
            if line:
                rows.append([float(v) for v in line])
    return np.array(rows)  # (samplings, timesteps)

design_daily = {}

for design in sorted(os.listdir(folder_path)):
    inner_dir = os.path.join(folder_path, design, "battery")
    if not os.path.isdir(inner_dir):
        continue

    accumulated = np.zeros((samplings, timesteps))
    count = 0

    for fname in os.listdir(inner_dir):
        if agents_tag not in fname or not fname.endswith(".csv"):
            continue
        try:
            data = load_csv(os.path.join(inner_dir, fname))
            if data.shape == (samplings, timesteps):
                accumulated += data
                count += 1
            else:
                print(f"  [WARN] incorrect shape {data.shape} in {fname}")
        except Exception as e:
            print(f"  [ERR] {fname}: {e}")

    if count > 0:
        design_daily[design] = accumulated / count
        print(f"  {design} — {agents_tag}: {count} files")
    else:
        print(f"  [WARN] no data for {design} — {agents_tag}")

if not design_daily:
    print("No data")
    exit()

x_smooth = np.arange(WINDOW - 1, timesteps)

for i, ep in enumerate(sample_labels):
    plt.figure(figsize=(10, 5))
    plt.suptitle("Multi-agent : daily battery")
    plt.title(f"Episode: {ep} — {agents_tag} — observation designs comparison")
    plt.xlabel("Timestep")
    plt.ylabel("Battery")

    for design, data in design_daily.items():
        s = smooth(data[i], WINDOW)
        plt.plot(x_smooth, s, label=design, linewidth=2.0, alpha=1.0)

    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"daily_battery_ep{ep}_{agents_tag}.pdf"))
    plt.close()
    print(f"saved: daily_battery_ep{ep}_{agents_tag}.pdf")

print("\n=== Done ===")