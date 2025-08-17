import json
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 13,
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.3,
    "figure.dpi": 150,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
})

def smooth_curve(y, window=3):
    if len(y) < window:
        return y
    return np.convolve(y, np.ones(window)/window, mode='valid')

files_FedAvg = {
    "Baseline (All Visible)": "results/results_FedAvg/FedAvg_baseline.json",
    "Walker Star": "results/results_FedAvg/FedAvg_walker_star.json",
    "Polar SSO": "results/results_FedAvg/FedAvg_polar_sso.json",
    "Inclined Sparse": "results/results_FedAvg/FedAvg_inclined_sparse.json",
    "Retrograde Polar": "results/results_FedAvg/FedAvg_retrograde_polar.json",
    "Equatorial": "results/results_FedAvg/FedAvg_equatorial.json",
}

files_FedAsync = {
    "Baseline (All Visible)": "results/results_FedAsync/FedAsync_baseline.json",
    "Walker Star": "results/results_FedAsync/FedAsync_walker_star.json",
    "Polar SSO": "results/results_FedAsync/FedAsync_polar_sso.json",
    "Inclined Sparse": "results/results_FedAsync/FedAsync_inclined_sparse.json",
    "Retrograde Polar": "results/results_FedAsync/FedAsync_retrograde_polar.json",
    "Equatorial": "results/results_FedAsync/FedAsync_equatorial.json",
}

fig, axs = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

# Store handles and labels for the shared legend
handles_labels = []

# --- FedAvg ---
ax = axs[0]
for label, path in files_FedAvg.items():
    with open(path, "r") as f:
        data = json.load(f)
    times = np.array(data["times"])
    accs = np.array(data["accuracies"])
    accs_smooth = smooth_curve(accs, window=3)
    times_smooth = times[1:-1]
    wall_clock_durations = data.get("wall_clock_durations", [])
    avg_time = np.mean(wall_clock_durations) if wall_clock_durations else None
    legend_label = f"{label} (Avg. round time: {avg_time:.1f}s)" if avg_time else label
    line, = ax.plot(times_smooth, accs_smooth, label=legend_label, linewidth=2)
    handles_labels.append((line, legend_label))
ax.set_title("FedAvg")
ax.set_xlabel("Simulated Time (hours)")
ax.set_ylabel("Top-1 Accuracy")
ax.grid(True)
ax.set_xlim(times_smooth[0], times_smooth[-1])
ax.margins(x=0)

# --- FedAsync ---
ax = axs[1]
for idx, (label, path) in enumerate(files_FedAsync.items()):
    with open(path, "r") as f:
        data = json.load(f)
    times = np.array(data["times"])
    accs = np.array(data["accuracies"])
    accs_smooth = smooth_curve(accs, window=3)
    times_smooth = times[1:-1]
    wall_clock_durations = data.get("wall_clock_durations", [])
    avg_time = np.mean(wall_clock_durations) if wall_clock_durations else None
    legend_label = f"{label} (Avg. round time: {avg_time:.1f}s)" if avg_time else label
    # Use the same color as FedAvg for the same constellation
    color = handles_labels[idx][0].get_color()
    ax.plot(times_smooth, accs_smooth, label=legend_label, linewidth=2, color=color)
ax.set_title("FedAsync")
ax.set_xlabel("Simulated Time (hours)")
ax.set_ylabel("Top-1 Accuracy") 
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()
ax.grid(True)
ax.set_xlim(times_smooth[0], times_smooth[-1])
ax.margins(x=0)

plt.tight_layout()
plt.subplots_adjust(left=0.10, right=0.90, bottom=0.20)

# Only one legend, centered below both plots...
handles, labels = zip(*handles_labels)
fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=14.3, frameon=False)

plt.savefig("FedAvg_vs_FedAsync_all_constellations.png", dpi=300, bbox_inches="tight")

