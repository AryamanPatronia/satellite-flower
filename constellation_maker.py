import json
import os

# Always save in the 'constellations' folder relative to this script
CONSTELLATION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "constellations")

# Satellite-to-class mapping...
client_classes = {
    "satellite_1": [0, 1, 2, 3],
    "satellite_2": [2, 3, 4, 5],
    "satellite_3": [4, 5, 6, 7],
    "satellite_4": [6, 7, 8, 9],
    "satellite_5": [0, 1, 8, 9],
}

def save_schedule(name, schedule):
    path = os.path.join(CONSTELLATION_DIR, f"{name}.json")
    with open(path, "w") as f:
        f.write("{\n")
        rounds = list(schedule.items())
        for idx, (rnd, sats) in enumerate(rounds):
            line = f'  "{rnd}": {json.dumps(sats)}'
            if idx < len(rounds) - 1:
                line += ","
            f.write(line + "\n")
        f.write("}\n")
    print(f"Saved {name}.json in {CONSTELLATION_DIR}")

def make_baseline():
    # All satellites visible every round (best availability)
    schedule = {f"round_{i+1}": [f"satellite_{j+1}" for j in range(5)] for i in range(30)}
    save_schedule("baseline", schedule)

def make_walker_star():
    # 4 satellites per round, rotating window (high availability)
    satellites = [f"satellite_{i+1}" for i in range(5)]
    schedule = {}
    for i in range(30):
        group = [satellites[(i+j)%5] for j in range(4)]
        schedule[f"round_{i+1}"] = group
    save_schedule("walker_star", schedule)

def make_polar_sso():
    # 3 satellites per round, rotating triplets (moderate availability)
    satellites = [f"satellite_{i+1}" for i in range(5)]
    triplets = []
    for i in range(5):
        for j in range(i+1, 5):
            for k in range(j+1, 5):
                triplets.append([satellites[i], satellites[j], satellites[k]])
    schedule = {}
    for i in range(30):
        schedule[f"round_{i+1}"] = triplets[i % len(triplets)]
    save_schedule("polar_sso", schedule)

def make_equatorial():
    # 2 satellites per round, rotating pairs, every 10th round all 5 (low-moderate availability)
    satellites = [f"satellite_{i+1}" for i in range(5)]
    pairs = [
        ["satellite_1", "satellite_2"],
        ["satellite_2", "satellite_3"],
        ["satellite_3", "satellite_4"],
        ["satellite_4", "satellite_5"],
        ["satellite_5", "satellite_1"],
    ]
    schedule = {}
    for i in range(30):
        if i % 10 == 9:
            schedule[f"round_{i+1}"] = satellites  # all 5
        else:
            schedule[f"round_{i+1}"] = pairs[i % len(pairs)]
    save_schedule("equatorial", schedule)

def make_inclined_sparse():
    # 2 satellites per round, cycling through all pairs (low availability)
    satellites = [f"satellite_{i+1}" for i in range(5)]
    pairs = []
    for i in range(5):
        for j in range(i+1, 5):
            pairs.append([satellites[i], satellites[j]])
    schedule = {}
    for i in range(30):
        schedule[f"round_{i+1}"] = pairs[i % len(pairs)]
    save_schedule("inclined_sparse", schedule)

def make_retrograde_polar():
    # Mostly single satellite per round, sometimes a pair (extremely low availability)
    satellites = [f"satellite_{i+1}" for i in range(5)]
    schedule = {}
    for i in range(30):
        if i % 3 == 0:
            schedule[f"round_{i+1}"] = ["satellite_5"]
        elif i % 3 == 1:
            schedule[f"round_{i+1}"] = ["satellite_4", "satellite_2"]
        else:
            schedule[f"round_{i+1}"] = ["satellite_5", satellites[(i//3)%5]]
    save_schedule("retrograde_polar", schedule)

if __name__ == "__main__":
    make_baseline()
    make_walker_star()
    make_polar_sso()
    make_equatorial()
    make_inclined_sparse()
    make_retrograde_polar()