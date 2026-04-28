import numpy as np
import matplotlib.pyplot as plt
import csv
import os

folder_path = "./csv_avg_plots"
episodes    = 4001
WINDOW      = 100
PLOT_DIR    = "./time_plots"
os.makedirs(PLOT_DIR, exist_ok=True)

agents_list = ["2agents", "3agents", "5agents"]

def smooth(data, window):
    if len(data) < window:
        return np.array(data)
    return np.convolve(data, np.ones(window) / window, mode="valid")

def load_time_csv(filepath):
    values = []
    with open(filepath) as f:
        for line in csv.reader(f):
            if line:
                values.append(float(line[0]))
    return np.array(values)

all_data = {}

for design in sorted(os.listdir(folder_path)):
    inner_dir = os.path.join(folder_path, design, "time")
    if not os.path.isdir(inner_dir):
        continue

    all_data[design] = {}

    for fname in os.listdir(inner_dir):
        if not fname.endswith(".csv"):
            continue

        matched_tag = None
        for tag in agents_list:
            if tag in fname:
                matched_tag = tag
                break

        if matched_tag is None:
            print(f"no agent {fname}, skip")
            continue

        try:
            data = load_time_csv(os.path.join(inner_dir, fname))
            all_data[design][matched_tag] = data
            print(f"  {design} | {matched_tag}: {len(data)} episodi — "
                  f"avg  {data.mean():.3f}s, max {data.max():.3f}s")
        except Exception as e:
            print(f"error {fname}: {e}")

x_smooth_base = np.arange(WINDOW - 1, episodes)

for tag in agents_list:
    plt.figure(figsize=(10, 5))
    plt.suptitle("Multi-agent : average episode time")
    plt.title(f"{tag} — observation designs comparison")
    plt.xlabel("Episodes")
    plt.ylabel("Time (s)")

    has_data = False
    for design, tags in all_data.items():
        if tag not in tags:
            continue
        data = tags[tag]
        x_s = np.arange(WINDOW - 1, len(data))
        plt.plot(data, alpha=0.2)
        plt.plot(x_s, smooth(data, WINDOW),
                 label=f"{design} smooth", linewidth=2.0, alpha=1.0)
        has_data = True

    if not has_data:
        plt.close()
        print(f"missing data  {tag}")
        continue

    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"time_by_design_{tag}.pdf"))
    plt.close()
    print(f"saved: time_by_design_{tag}.pdf")


for design, tags in all_data.items():
    if not tags:
        continue

    plt.figure(figsize=(10, 5))
    plt.suptitle("Multi-agent : average episode time")
    plt.title(f"{design} — configurations comparison")
    plt.xlabel("Episodes")
    plt.ylabel("Time (s)")

    for tag, data in tags.items():
        x_s = np.arange(WINDOW - 1, len(data))
        plt.plot(data, alpha=0.2)
        plt.plot(x_s, smooth(data, WINDOW),
                 label=f"{tag} smooth", linewidth=2.0, alpha=1.0)

    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"time_by_nagents_{design}.pdf"))
    plt.close()
    print(f"saved: time_by_nagents_{design}.pdf")

labels  = []
boxes   = []

for design, tags in all_data.items():
    for tag, data in sorted(tags.items()):
        labels.append(f"{design}\n{tag}")
        boxes.append(data)

if boxes:
    fig, ax = plt.subplots(figsize=(max(8, len(boxes) * 1.5), 6))
    ax.boxplot(boxes, labels=labels, showfliers=False)
    ax.set_title("Episode time distribution — all design configurations")
    ax.set_ylabel("Time (s)")
    ax.set_xlabel("Design / Agents")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "time_boxplot_all.pdf"))
    plt.close()
    print("saved: time_boxplot_all.pdf")

print("\n=== Avg episode time summary ===")
print(f"{'Design':<35} {'Config':<12} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
print("-" * 85)
for design, tags in all_data.items():
    for tag, data in sorted(tags.items()):
        print(f"{design:<35} {tag:<12} {data.mean():>8.3f} "
              f"{data.std():>8.3f} {data.min():>8.3f} {data.max():>8.3f}")

print("\n=== Done ===")