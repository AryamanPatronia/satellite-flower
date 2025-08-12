import json
import matplotlib.pyplot as plt
import numpy as np

# Use consistent, paper-style appearance
plt.rcParams.update({
    "font.size": 13,
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})

def smooth_curve(y, window=3):
    if len(y) < window:
        return y
    return np.convolve(y, np.ones(window)/window, mode='valid')

# Map of constellation name to result file
files = {
    "Baseline (All Visible)": "results/results_FedAvg/FedAvg_baseline.json",
    "Walker Star": "results/results_FedAvg/FedAvg_walker_star.json",
    "Polar SSO": "results/results_FedAvg/FedAvg_polar_sso.json",
    "Inclined Sparse": "results/results_FedAvg/FedAvg_inclined_sparse.json",
    "Retrograde Polar": "results/results_FedAvg/FedAvg_retrograde_polar.json",
    "Equatorial": "results/results_FedAvg/FedAvg_equatorial.json",
}

# Square-shaped figure (equal width and height)
plt.figure(figsize=(8, 6))  # Increased width from 6 to 8

legend_labels = []

for label, path in files.items():
    with open(path, "r") as f:
        data = json.load(f)
    times = np.array(data["times"])
    accs = np.array(data["accuracies"])
    accs_smooth = smooth_curve(accs, window=3)
    times_smooth = times[1:-1]  # Align with smoothing

    # Calculate average round time for this constellation
    wall_clock_durations = data.get("wall_clock_durations", [])
    avg_time = np.mean(wall_clock_durations) if wall_clock_durations else None
    if avg_time:
        legend_label = f"{label} (Avg. round time: {avg_time:.1f}s)"
    else:
        legend_label = label

    plt.plot(times_smooth, accs_smooth, label=legend_label, linewidth=2)

plt.title("FedAvg: Central Test Accuracy Across Constellations")
plt.xlabel("Simulated Time (hours)")
plt.ylabel("Central Test Accuracy")
plt.legend(fontsize=10, loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.xlim(times[1], times[-2])  # Set x-axis to data range
plt.margins(x=0) 
plt.savefig("FedAvg_all_constellations.png", dpi=300)
