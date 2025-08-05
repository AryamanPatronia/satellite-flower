import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

CONSTELLATION_FILES = {
    "walker_star": "walker_star.json",
    "polar_sso": "polar_sso.json",
    "equatorial": "equatorial.json",
    "inclined_sparse": "inclined_sparse.json",
    "retrograde_polar": "retrograde_polar.json",
    "baseline": "baseline.json"
}

SATELLITE_IDS = [f"S{i}" for i in range(1, 6)]
CLUSTER_LABELS = SATELLITE_IDS  
CLUSTERS = [[f"satellite_{i}"] for i in range(1, 6)]

# Every constellation has 5 satellites, each with a unique ID...
# Every round is 4 hours long, and there are 30 rounds in total...
ROUNDS = 30
HOURS_PER_ROUND = 4
TOTAL_HOURS = ROUNDS * HOURS_PER_ROUND


def plot_cluster_and_satellite_visibility(constellation_name, filepath):
    with open(filepath, "r") as f:
        schedule = json.load(f)

    fig, ax = plt.subplots(figsize=(5.5, 2.7)) 

    for idx, cluster in enumerate(CLUSTERS):
        y_base = idx * 1.5

        # Not visible: gray background for the entire cluster...
        ax.broken_barh([(0, TOTAL_HOURS)], (y_base - 0.5, 1.0), facecolors="#eeeeee", zorder=0)

        # Gray bars: shading for each satellite in the cluster...
        for j, sat in enumerate(cluster):
            for r in range(1, ROUNDS + 1):
                t_start = (r - 1) * HOURS_PER_ROUND
                visible = schedule.get(f"round_{r}", [])
                if sat in visible:
                    ax.broken_barh([(t_start, HOURS_PER_ROUND)], (y_base - 0.3 + j * 0.2, 0.15),
                                   facecolors="#999999", zorder=1)

        # Red bar: visible to server...
        for r in range(1, ROUNDS + 1):
            t_start = (r - 1) * HOURS_PER_ROUND
            visible = schedule.get(f"round_{r}", [])
            if any(sat in visible for sat in cluster):
                ax.broken_barh([(t_start, HOURS_PER_ROUND)], (y_base - 0.2, 0.6), facecolors="#0072B2", zorder=2)

    # Styling...
    ax.set_yticks([i * 1.5 for i in range(len(CLUSTER_LABELS))])
    ax.set_yticklabels(SATELLITE_IDS)
    ax.set_xlim(0, TOTAL_HOURS)
    ax.set_ylim(-1, len(CLUSTER_LABELS) * 1.5)
    ax.set_xticks(range(0, TOTAL_HOURS + 1, 12))
    ax.set_xlabel("Time(H)", fontsize=10, labelpad=6)
    ax.set_title(f"{constellation_name.replace('_', ' ').upper()} CONSTELLATION", fontsize=10, fontweight="bold")
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)
    ax.tick_params(axis='both', labelsize=8)

    # Legend...
    fig.legend(
        [patches.Patch(color='#0072B2'), patches.Patch(color='#eeeeee')],
        ["Satellite visible to server", "Satellite not visible"],
        loc="upper center", ncol=2, fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.0)
    )

    os.makedirs("CMaps", exist_ok=True)
    save_path = f"CMaps/map_{constellation_name}.png"
    plt.subplots_adjust(bottom=0.28)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved enhanced map: {save_path}")


if __name__ == "__main__":
    for name, file in CONSTELLATION_FILES.items():
        full_path = os.path.join("constellations", file)
        if os.path.exists(full_path):
            plot_cluster_and_satellite_visibility(name, full_path)
        else:
            print(f"[!] File missing: {file}")
