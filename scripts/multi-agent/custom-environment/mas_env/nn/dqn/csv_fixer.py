import numpy as np
import matplotlib.pyplot as plt
import os
import glob


'''
    THIS FILE HAS BEEN USED TO CORRECT FRAMERATES CONTAINING AN
    ERROR RELATED TO HIGHER OFFLOADING RATES WITH RESPECT TO ITS
    REAL VALUE.
    
    THE FIXED VALUES CONSIDER THE RATE OVER THE NUMBER OF TIMESTEPS
    AND NOT OVER THE EFFECTIVE MATCHINGS.    
'''

proc_interval   = 60
max_steps   = int(24 * 60 * 60 / proc_interval)

num_agents  = 5
# battery_capacities_wh = [25, 100]
battery_capacities_wh = [25, 100, 50, 37, 65]   
battery_capacities  = [b for b in battery_capacities_wh]  

power_idle  = 2.6
power_max   = 6.0
proc_rate   = 20

num_episodes  = 4001
episode   = 355
mode  = "cuda"
w   = 1.0

CSV_DIR  = "./20260409_025840/csvs/csvs_batch_256"
PLOT_DIR   = "."
WINDOW = 10

def build_filename(prefix, battery_j):
  return os.path.join(
  CSV_DIR,
  f"{prefix}_{battery_j}_{num_episodes-1}_{episode}"
  f"_{proc_interval}_{num_agents}agents_{mode}.csv"
  )

def load_csv(path):
  if not os.path.exists(path):
    raise FileNotFoundError(f"File non trovato: {path}")
  return np.loadtxt(path)

def fixed_path(path):
  root, ext = os.path.splitext(path)
  return f"{root}_FIXED{ext}"

def save_csv(path, data):
  out = fixed_path(path)
  os.makedirs(os.path.dirname(out) if os.path.dirname(out) else ".", exist_ok=True)
  np.savetxt(out, data, fmt="%.8f")
  print(f"  saved : {out}")

def smooth(data, window):
  return np.convolve(data, np.ones(window) / window, mode="valid")

fs_all  = []
hs_fix_all  = []
total_fix_all = []
labels  = []

for agent_id, bat_j in enumerate(battery_capacities):
  bat_wh = battery_capacities_wh[agent_id]
  label  = f"{bat_wh} Wh"
  labels.append(label)

  fs_path   = build_filename("local_framerate",  bat_j)
  hs_raw_path   = build_filename("offloading_framerate", bat_j)
  match_path  = build_filename("matchings",  bat_j)
  total_path  = build_filename("total_framerate",  bat_j)

  fs   = load_csv(fs_path)
  hs_raw   = load_csv(hs_raw_path)
  matches  = load_csv(match_path)

  env_hs   = hs_raw * matches
  hs_fix   = env_hs / max_steps

  total_fix = fs + hs_fix

  print(f"\nAgente {agent_id} ({label})")
  print(f"  hs original : {hs_raw.mean():.4f}  (max {hs_raw.max():.4f})")
  print(f"  hs fixed  : {hs_fix.mean():.4f}  (max {hs_fix.max():.4f})")
  print(f"  total avg origin   : (fs+hs_raw) = {(fs + hs_raw).mean():.4f}")
  print(f"  total avg fixed   : {total_fix.mean():.4f}")

  fs_all.append(fs)
  hs_fix_all.append(hs_fix)
  total_fix_all.append(total_fix)

  save_csv(hs_raw_path, hs_fix)
  save_csv(total_path,  total_fix)


def make_title():
  return (f"P_i = {power_idle}, P_f = {power_max}, "
  f"fps = {proc_rate}, interval: {proc_interval}s")

def pdf_name(prefix):
  return os.path.join(
  PLOT_DIR,
  f"{prefix}_{num_episodes-1}_{episode}"
  f"_{proc_interval}_{w}_{num_agents}agents_{mode}_FIXED.pdf"
  )


plt.figure()
plt.suptitle("Multi-agent : local average framerate")
plt.title(make_title())
plt.xlabel("Episodes")
plt.ylabel("Framerate")
for i in range(num_agents):
  d = fs_all[i]
  plt.plot(range(WINDOW - 1, len(d)), smooth(d, WINDOW),
   label=f"smooth {labels[i]}", alpha=1.0)
  plt.plot(d, label=f"raw {labels[i]}", alpha=0.3)
plt.grid()
plt.legend(bbox_to_anchor=(0.5, -0.2), loc="upper center", ncol=3)
plt.tight_layout()
plt.savefig(pdf_name("local_framerate_plot"))
plt.close()
print(f"\nPlot saved: {pdf_name('local_framerate_plot')}")


plt.figure()
plt.suptitle("Multi-agent : offloading average framerate")
plt.title(make_title())
plt.xlabel("Episodes")
plt.ylabel("Framerate")
for i in range(num_agents):
  d = hs_fix_all[i]
  plt.plot(range(WINDOW - 1, len(d)), smooth(d, WINDOW),
   label=f"smooth {labels[i]}", alpha=1.0)
  plt.plot(d, label=f"raw {labels[i]}", alpha=0.3)
plt.grid()
plt.legend(bbox_to_anchor=(0.5, -0.2), loc="upper center", ncol=3)
plt.tight_layout()
plt.savefig(pdf_name("offloading_framerate_plot"))
plt.close()
print(f"Plot saved: {pdf_name('offloading_framerate_plot')}")


plt.figure()
plt.suptitle("Multi-agent : average framerate")
plt.title(make_title())
plt.xlabel("Episodes")
plt.ylabel("Framerate")
for i in range(num_agents):
  d = total_fix_all[i]
  plt.plot(range(WINDOW - 1, len(d)), smooth(d, WINDOW),
   label=f"smooth {labels[i]}", alpha=1.0)
  plt.plot(d, label=f"raw {labels[i]}", alpha=0.3)
plt.grid()
plt.legend(bbox_to_anchor=(0.5, -0.2), loc="upper center", ncol=3)
plt.tight_layout()
plt.savefig(pdf_name("framerate_plot"))
plt.close()
print(f"Plot saved: {pdf_name('framerate_plot')}")


for i in range(num_agents):
  for data, suptitle, prefix in [
  (fs_all[i],
   "Multi-agent : local average framerate",
   "local_framerate_plot"),
  (hs_fix_all[i],
   "Multi-agent : offloading average framerate",
   "offloading_framerate_plot"),
  (total_fix_all[i],
   "Multi-agent : average framerate",
   "framerate_plot"),
  ]:
      
    plt.figure()
    plt.suptitle(suptitle)
    plt.title(f"B: {battery_capacities_wh[i]} Wh - " + make_title())
    plt.xlabel("Episodes")
    plt.ylabel("Framerate")
    plt.plot(range(WINDOW - 1, len(data)), smooth(data, WINDOW),
    label=f"smooth {labels[i]}", alpha=1.0)
    plt.plot(data, label=f"raw {labels[i]}", alpha=0.3)
    plt.grid()
    plt.legend(bbox_to_anchor=(0.5, -0.2), loc="upper center", ncol=3)
    plt.tight_layout()
    fname = os.path.join(
    PLOT_DIR,
    f"{prefix}_{battery_capacities_wh[i]}Wh_{num_episodes-1}_{episode}"
    f"_{proc_interval}_{w}_{num_agents}agents_{mode}_FIXED.pdf"
    )
    plt.savefig(fname)
    plt.close()
    print(f"Plot saved: {fname}")
