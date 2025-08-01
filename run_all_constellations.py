import os
import shutil
import subprocess
import time

CONSTELLATIONS = {
    "walker_star": "walker_star.json",
    "polar_sso": "polar_sso.json",
    "equatorial": "equatorial.json",
    "inclined_sparse": "inclined_sparse.json",
    "retrograde_polar": "retrograde_polar.json"
}

def wait_for_training_finish(timeout=600):
    print("[+] Waiting for training to complete (watching results/server_history.json)...")
    history_path = "results/server_history.json"
    waited = 0
    while waited < timeout:
        if os.path.exists(history_path):
            return True
        time.sleep(5)
        waited += 5
    return False

def run_simulation(name, file):
    print(f"\n[+] Running FedAsync for constellation: {name}")

    # 1. Copy visibility schedule
    shutil.copyfile(
        os.path.join("constellations", file),
        "visibility/visibility_schedule.json"
    )

    # 2. Clean
    subprocess.run(["docker-compose", "-f", "docker/docker-compose.yml", "down", "-v", "--remove-orphans"])
    subprocess.run(["docker", "image", "prune", "-a", "-f"])

    # 3. Clean server log
    log_path = "shared_logs/server.log"
    if os.path.exists(log_path):
        os.remove(log_path)

    # 4. Launch containers
    subprocess.run(["docker-compose", "-f", "docker/docker-compose.yml", "up", "--build", "-d"])

    # 5. Wait for training to finish
    if not wait_for_training_finish():
        print("[-] Timeout or error: Training may have failed.")
        return

    # 6. Rename chart
    original_chart = "results/FedAsync_accuracy_vs_time.png"
    renamed_chart = f"results/FedAsync_{name}.png"
    if os.path.exists(original_chart):
        shutil.move(original_chart, renamed_chart)
        print(f"[✓] Saved result as {renamed_chart}")
    else:
        print("[-] Warning: Accuracy chart not found!")

    # 7. Save server history JSON uniquely
    history_original = "results/server_history.json"
    history_renamed = f"results/FedAsync_{name}.json"
    if os.path.exists(history_original):
        shutil.move(history_original, history_renamed)
        print(f"[✓] Saved server history as {history_renamed}")
    else:
        print("[-] Warning: Server history not found!")

if __name__ == "__main__":
    for name, file in CONSTELLATIONS.items():
        run_simulation(name, file)

    print("\n[✓] All FedAsync simulations complete.")
