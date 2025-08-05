import json
import matplotlib.pyplot as plt
import numpy as np

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

plt.figure(figsize=(10, 6))

for label, path in files.items():
    with open(path, "r") as f:
        data = json.load(f)
    times = np.array(data["times"])
    accs = np.array(data["accuracies"])
    # 3-point moving average, dissertation-ready style
    accs_smooth = np.convolve(accs, np.ones(3)/3, mode='valid')
    times_smooth = times[1:-1]  # Align with 'valid' mode
    plt.plot(times_smooth, accs_smooth, label=label, linewidth=2)

plt.title("FedAvg: Central Test Accuracy Across Constellations")
plt.xlabel("Time [arbitrary units or minutes]")
plt.ylabel("Central Test Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()