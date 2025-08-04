import os
import shutil
import subprocess
import time
import json

CONSTELLATIONS = {
    "walker_star": "walker_star.json",
    "polar_sso": "polar_sso.json",
    "equatorial": "equatorial.json",
    "inclined_sparse": "inclined_sparse.json",
    "retrograde_polar": "retrograde_polar.json"
}

# Get strategy from environment variable (FedAsync or FedAvg)
SERVER_TYPE = os.environ.get("SERVER_TYPE", "FedAsync")
RESULTS_DIR = f"results/results_{SERVER_TYPE}"
SERVER_LOG_PATH = "shared_logs/server.log"

def wait_for_training_finish(expected_rounds=30, timeout=1800):
    print(f"[+] Waiting for training to complete (watching {RESULTS_DIR}/server_history.json)...")
    history_path = f"{RESULTS_DIR}/server_history.json"
    waited = 0
    while waited < timeout:
        if os.path.exists(history_path):
            try:
                with open(history_path, "r") as f:
                    data = json.load(f)
                if "accuracies" in data:
                    print(f"[DEBUG] Found {len(data['accuracies'])} rounds so far")
                # Check if the expected number of rounds is reached
                if "accuracies" in data and len(data["accuracies"]) >= expected_rounds:
                    print("[✓] Training complete!")
                    return True
            except Exception as e:
                print(f"[!] Error reading server_history.json: {e}")
        time.sleep(5)
        waited += 5
    print("[-] Timeout waiting for training to finish.")
    return False

def run_simulation(name, file):
    print(f"\n[+] Running {SERVER_TYPE} for constellation: {name}")

    # 1. Copy visibility schedule
    shutil.copyfile(
        os.path.join("constellations", file),
        "visibility/visibility_schedule.json"
    )

    # 2. Clean
    subprocess.run(["docker-compose", "-f", "docker/docker-compose.yml", "down", "-v", "--remove-orphans"])
    subprocess.run(["docker", "image", "prune", "-a", "-f"])

    # 3. Clean previous log
    if os.path.exists(SERVER_LOG_PATH):
        os.remove(SERVER_LOG_PATH)

    # 4. Launch containers
    os.environ["RESULTS_DIR"] = RESULTS_DIR
    os.environ["SERVER_TYPE"] = SERVER_TYPE

    subprocess.run(
        ["docker-compose", "-f", "docker/docker-compose.yml", "up", "--build", "-d"],
        env={**os.environ, "RESULTS_DIR": RESULTS_DIR, "SERVER_TYPE": SERVER_TYPE}
    )

    # 5. Wait for training to finish
    if not wait_for_training_finish():
        print("[-] Timeout or error: Training may have failed.")
        return

    # 6. Move accuracy plot
    src_chart = f"{RESULTS_DIR}/{SERVER_TYPE}_accuracy_vs_time.png"
    dst_chart = f"{RESULTS_DIR}/{SERVER_TYPE}_{name}.png"
    if os.path.exists(src_chart):
        shutil.move(src_chart, dst_chart)
        print(f"[✓] Saved result chart as {dst_chart}")
    else:
        print("[-] Warning: Accuracy chart not found!")

    # 7. Move server history
    src_history = f"{RESULTS_DIR}/server_history.json"
    dst_history = f"{RESULTS_DIR}/{SERVER_TYPE}_{name}.json"
    if os.path.exists(src_history):
        shutil.move(src_history, dst_history)
        print(f"[✓] Saved history JSON as {dst_history}")
    else:
        print("[-] Warning: Server history not found!")

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for name, file in CONSTELLATIONS.items():
        run_simulation(name, file)

    print(f"\n[✓] All {SERVER_TYPE} simulations complete.")
