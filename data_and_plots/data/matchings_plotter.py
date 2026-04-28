import numpy as np
import matplotlib.pyplot as plt
import csv
import os

folder_path = "./csv_avg_plots"
episodes    = 4001
timesteps   = 1440
agents_list = ["2agents", "3agents", "5agents"]
window_ep   = 100   
window_s    = 3 

def smooth(data, window):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode="valid")


def load_per_episode(folder_path, subfolder, prefix, agents_tag, suffix=""):
    inner_dir = os.path.join(folder_path, subfolder)
    if not os.path.isdir(inner_dir):
        return None, 0

    accumulated = [0.0 for _ in range(episodes)]
    count = 0

    for fname in os.listdir(inner_dir):
        if not fname.startswith(prefix):
            continue
        if agents_tag not in fname:
            continue
        if suffix and not fname.endswith(suffix + ".csv"):
            continue
        if not fname.endswith(".csv"):
            continue

        with open(os.path.join(inner_dir, fname)) as f:
            idx = 0
            for line in csv.reader(f):
                if idx < episodes:
                    accumulated[idx] += float(line[0])
                    idx += 1
        count += 1

    return accumulated, count


def load_sampled(folder_path, subfolder, prefix, agents_tag, samplings=11):
    inner_dir = os.path.join(folder_path, subfolder)
    if not os.path.isdir(inner_dir):
        return None, 0

    accumulated = [0.0 for _ in range(samplings)]
    count = 0

    for fname in os.listdir(inner_dir):
        if agents_tag not in fname:
            continue
        if not fname.endswith(".csv"):
            continue
        if not any(fname.startswith(p) for p in [prefix] if p):
            continue

        with open(os.path.join(inner_dir, fname)) as f:
            idx = 0
            for line in csv.reader(f):
                if idx > 0 and (idx - 1) < samplings:
                    for val in line:
                        accumulated[idx - 1] += float(val)
                idx += 1
        count += 1

    return accumulated, count

def compare_by_nagents_episode(metric_label, subfolder, prefix, suffix="",
                                ylabel=None, savename=None):
    global_means = {}

    for agents_tag in agents_list:
        accumulated = [0.0 for _ in range(episodes)]
        total = 0

        for design in os.listdir(folder_path):
            acc, cnt = load_per_episode(
                os.path.join(folder_path, design), subfolder, prefix, agents_tag, suffix
            )
            if acc is None:
                continue
            for i in range(episodes):
                accumulated[i] += acc[i]
            total += cnt

        if total == 0:
            print(f"  [WARN] no data {metric_label} for {agents_tag}")
            continue

        n = int(agents_tag.replace("agents", ""))
        global_means[agents_tag] = [v / total for v in accumulated]

    if not global_means:
        return

    plt.figure()
    plt.suptitle(f"Multi-agent : average {metric_label}")
    plt.title("Observation designs averages")
    plt.xlabel("Episodes")
    plt.ylabel(ylabel or metric_label)

    for agents_tag, data in global_means.items():
        # plt.plot(data, alpha=0.2)
        plt.plot(
            range(window_ep - 1, len(data)),
            smooth(data, window_ep),
            label=f"{agents_tag} smooth",
            linewidth=2.0,
            alpha=1.0
        )

    plt.grid()
    plt.legend()
    plt.tight_layout()
    fname = savename or f"{metric_label.lower().replace(' ', '_')}_comparison_by_nagents.pdf"
    plt.savefig(fname)
    plt.close()
    print(f"saved: {fname}")


def compare_by_nagents_sampled(metric_label, subfolder, prefix,
                                samplings=11, ylabel=None, savename=None):
    global_means = {}
    x_ticks = [i * int(episodes / 10) for i in range(samplings)]

    for agents_tag in agents_list:
        accumulated = [0.0 for _ in range(samplings)]
        total = 0

        for design in os.listdir(folder_path):
            inner_dir = os.path.join(folder_path, design, subfolder)
            if not os.path.isdir(inner_dir):
                continue

            for fname in os.listdir(inner_dir):
                if agents_tag not in fname or not fname.endswith(".csv"):
                    continue

                with open(os.path.join(inner_dir, fname)) as f:
                    idx = 0
                    for line in csv.reader(f):
                        if idx > 0 and (idx - 1) < samplings:
                            for val in line:
                                accumulated[idx - 1] += float(val)
                        idx += 1
                total += 1

        if total == 0:
            print(f"  [WARN] no data {metric_label} for {agents_tag}")
            continue

        global_means[agents_tag] = [v / (total * timesteps) for v in accumulated]

    if not global_means:
        return

    plt.figure()
    plt.suptitle(f"Multi-agent : average {metric_label}")
    plt.title("Observation designs averages")
    plt.xlabel("Episodes")
    plt.ylabel(ylabel or metric_label)

    for agents_tag, data in global_means.items():
        plt.plot(x_ticks, data, "o-", label=agents_tag, linewidth=2.0, alpha=1.0)

    plt.grid()
    plt.legend()
    plt.tight_layout()
    fname = savename or f"{metric_label.lower().replace(' ', '_')}_comparison_by_nagents.pdf"
    plt.savefig(fname)
    plt.close()
    print(f"saved: {fname}")


def compare_by_design_episode(metric_label, subfolder, prefix, agents_tag,
                               suffix="", ylabel=None, savename=None):
    design_means = {}

    for design in sorted(os.listdir(folder_path)):
        acc, cnt = load_per_episode(
            os.path.join(folder_path, design), subfolder, prefix, agents_tag, suffix
        )
        if acc is None or cnt == 0:
            continue

        n = int(agents_tag.replace("agents", ""))
        design_means[design] = [v / cnt for v in acc]

    if not design_means:
        return

    plt.figure()
    plt.suptitle(f"Multi-agent : average {metric_label} — {agents_tag}")
    plt.title("Observation designs comparison")
    plt.xlabel("Episodes")
    plt.ylabel(ylabel or metric_label)

    for design, data in design_means.items():
        # plt.plot(data, alpha=0.2)
        plt.plot(
            range(window_ep - 1, len(data)),
            smooth(data, window_ep),
            label=f"{design}",
            linewidth=2.0,
            alpha=1.0
        )

    plt.grid()
    plt.legend()
    plt.tight_layout()
    fname = savename or f"{metric_label.lower().replace(' ', '_')}_by_design_{agents_tag}.pdf"
    plt.savefig(fname)
    plt.close()
    print(f"saved: {fname}")


def compare_by_design_sampled(metric_label, subfolder, agents_tag,
                               samplings=11, ylabel=None, savename=None):
    design_means = {}
    x_ticks = [i * int(episodes / 10) for i in range(samplings)]

    for design in sorted(os.listdir(folder_path)):
        inner_dir = os.path.join(folder_path, design, subfolder)
        if not os.path.isdir(inner_dir):
            continue

        accumulated = [0.0 for _ in range(samplings)]
        count = 0

        for fname in os.listdir(inner_dir):
            if agents_tag not in fname or not fname.endswith(".csv"):
                continue

            with open(os.path.join(inner_dir, fname)) as f:
                idx = 0
                for line in csv.reader(f):
                    if idx > 0 and (idx - 1) < samplings:
                        for val in line:
                            accumulated[idx - 1] += float(val)
                    idx += 1
            count += 1

        if count == 0:
            continue

        design_means[design] = [v / (count * timesteps) for v in accumulated]

    if not design_means:
        return

    plt.figure()
    plt.suptitle(f"Multi-agent : average {metric_label} — {agents_tag}")
    plt.title("Observation designs comparison")
    plt.xlabel("Episodes")
    plt.ylabel(ylabel or metric_label)

    for design, data in design_means.items():
        plt.plot(x_ticks, data, "o-", label=design, linewidth=2.0, alpha=1.0)

    plt.grid()
    plt.legend()
    plt.tight_layout()
    fname = savename or f"{metric_label.lower().replace(' ', '_')}_by_design_{agents_tag}.pdf"
    plt.savefig(fname)
    plt.close()
    print(f"saved: {fname}")

if __name__ == "__main__":

    os.makedirs("./comparison_plots", exist_ok=True)

    print("\n=== Rewards by n_agents ===")
    compare_by_nagents_episode(
        "rewards", "rewards", "rewards_agent",
        ylabel="Rewards",
        savename="./comparison_plots/rewards_by_nagents.pdf"
    )

    print("\n=== Framerate by n_agents ===")
    compare_by_nagents_episode(
        "framerate", "framerate", "total_framerate", suffix="_FIXED",
        ylabel="Framerate",
        savename="./comparison_plots/framerate_by_nagents.pdf"
    )

    print("\n=== Matchings by n_agents ===")
    compare_by_nagents_episode(
        "matchings", "matchings", "matchings",
        ylabel="Matchings",
        savename="./comparison_plots/matchings_by_nagents.pdf"
    )

    print("\n=== Backlog by n_agents ===")
    compare_by_nagents_sampled(
        "backlog", "backlog", "backlog",
        ylabel="Backlog",
        savename="./comparison_plots/backlog_by_nagents.pdf"
    )

    print("\n=== Battery by n_agents ===")
    compare_by_nagents_sampled(
        "battery", "battery", "battery",
        ylabel="Battery",
        savename="./comparison_plots/battery_by_nagents.pdf"
    )

    for tag in agents_list:
        print(f"\n=== Rewards by design — {tag} ===")
        compare_by_design_episode(
            "rewards", "rewards", "rewards_agent", tag,
            ylabel="Rewards",
            savename=f"./comparison_plots/rewards_by_design_{tag}.pdf"
        )

        print(f"\n=== Framerate by design — {tag} ===")
        compare_by_design_episode(
            "framerate", "framerate", "total_framerate", tag, suffix="_FIXED",
            ylabel="Framerate",
            savename=f"./comparison_plots/framerate_by_design_{tag}.pdf"
        )

        print(f"\n=== Matchings by design — {tag} ===")
        compare_by_design_episode(
            "matchings", "matchings", "matchings", tag,
            ylabel="Matchings",
            savename=f"./comparison_plots/matchings_by_design_{tag}.pdf"
        )

        print(f"\n=== Backlog by design — {tag} ===")
        compare_by_design_sampled(
            "backlog", "backlog", tag,
            ylabel="Backlog",
            savename=f"./comparison_plots/backlog_by_design_{tag}.pdf"
        )

        print(f"\n=== Battery by design — {tag} ===")
        compare_by_design_sampled(
            "battery", "battery", tag,
            ylabel="Battery",
            savename=f"./comparison_plots/battery_by_design_{tag}.pdf"
        )

    print("\n=== Done ===")