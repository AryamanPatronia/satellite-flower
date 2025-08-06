# server_FedAsync.py
import os
import time
import json
import torch
import logging
import numpy as np
from typing import Dict, List
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import flwr as fl
from flwr.common import FitIns, GetParametersIns
from flwr.server.strategy import Strategy
import sys

sys.path.append("/app/client")
from train import Net

class FedAsyncStrategy(Strategy):
    def __init__(self, visibility_path: str):
        self.logger = logging.getLogger("Server")
        self.visibility_path = visibility_path
        self.visibility_schedule = self._load_schedule()
        self.current_weights = None
        self.client_staleness: Dict[str, int] = {}
        self.round_accuracies = []
        self.round_times = []
        self.client_id_map = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_testset = self._load_global_testset()
        self.wall_clock_durations = []  # <-- Add this line
        self.last_round_end_time = None  # <-- Add this line

    def _load_schedule(self):
        self.logger.info(f"Visibility schedule path: {self.visibility_path}")
        if os.path.exists(self.visibility_path):
            with open(self.visibility_path, "r") as f:
                data = json.load(f)
                self.logger.info(f"Loaded schedule with {len(data)} rounds")
                return data
        return {}

    def _load_global_testset(self):
        path = "/app/data/central_testset"
        x_test = torch.load(os.path.join(path, "x.pt"))
        y_test = torch.load(os.path.join(path, "y.pt"))
        return DataLoader(TensorDataset(x_test, y_test), batch_size=32)

    def _get_visible_ids(self, rnd: int) -> List[str]:
        return self.visibility_schedule.get(f"round_{rnd}", [])

    EXPECTED_CLIENTS = 5

    def initialize_parameters(self, client_manager):
        self.logger.info("Waiting for all clients to connect...")
        waited = 0
        while len(client_manager.all()) < self.EXPECTED_CLIENTS:
            time.sleep(1)
            waited += 1
            if waited > 300:
                raise RuntimeError(f"Only {len(client_manager.all())} clients connected after {waited}s")

        # Try to use explicit client_id property if available, else fallback to satellite_X
        for idx, (uuid, client_proxy) in enumerate(sorted(client_manager.all().items()), 1):
            client_id = None
            # Try to get explicit client_id property
            if hasattr(client_proxy, "properties"):
                client_id = client_proxy.properties.get("client_id", None)
            if client_id is None:
                client_id = f"satellite_{idx}"
            self.client_id_map[uuid] = client_id

        client = list(client_manager.all().values())[0]
        res = client.get_parameters(GetParametersIns(config={}), timeout=60, group_id="default")
        return res.parameters

    def configure_fit(self, server_round, parameters, client_manager):
        all_clients = list(client_manager.all().values())
        allowed_ids = self._get_visible_ids(server_round)
        visible_clients = [c for c in all_clients if self.client_id_map.get(c.cid, "") in allowed_ids]

        if not visible_clients:
            self.logger.warning(f"[ROUND {server_round}] No visible clients, skipping.")
            self.round_accuracies.append(0.0)
            self.round_times.append(server_round * 4)
            return []

        for client in visible_clients:
            self.client_staleness.setdefault(client.cid, 0)

        self.logger.info(f"[ROUND {server_round}] Training on: {[self.client_id_map[c.cid] for c in visible_clients]}")
        return [(client, FitIns(parameters, {"server_round": server_round})) for client in visible_clients]

    def aggregate_fit(self, server_round, results, failures):
        # --- Wall-clock timing start ---
        now = time.time()
        if self.last_round_end_time is not None:
            round_duration = now - self.last_round_end_time
            self.logger.info(f"[ROUND {server_round}] Wall-clock duration: {round_duration:.2f} seconds")
            self.wall_clock_durations.append(round_duration)
        self.last_round_end_time = now
        # --- Wall-clock timing end ---

        if not results:
            return self.current_weights, {}

        for client, fit_res in results:
            staleness = self.client_staleness.get(client.cid, 0)
            alpha = np.exp(-staleness)
            client_weights = fl.common.parameters_to_ndarrays(fit_res.parameters)

            if self.current_weights is None:
                self.current_weights = client_weights
            else:
                momentum = 0.9
                self.current_weights = [
                    momentum * cw + (1 - momentum) * lw for cw, lw in zip(self.current_weights, client_weights)
                ]
            self.client_staleness[client.cid] = 0

        for cid in self.client_staleness:
            if cid not in [c.cid for c, _ in results]:
                self.client_staleness[cid] += 1

        acc, _ = self.evaluate(server_round, fl.common.ndarrays_to_parameters(self.current_weights))
        self.round_accuracies.append(acc)
        self.round_times.append(server_round * 4)
        self.logger.info(f"[ROUND {server_round}] Central Eval Accuracy: {acc:.4f}")
        return fl.common.ndarrays_to_parameters(self.current_weights), {}

    def configure_evaluate(self, server_round, parameters, client_manager):
        return []

    def evaluate(self, server_round: int, parameters):
        if self.current_weights is None:
            return 0.0, {}

        model = Net().to(self.device)
        ndarrays = fl.common.parameters_to_ndarrays(parameters)
        state_dict = dict(zip(model.state_dict().keys(), [torch.tensor(w) for w in ndarrays]))
        model.load_state_dict(state_dict, strict=True)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in self.global_testset:
                x, y = x.to(self.device), y.to(self.device)
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        return correct / total, {}

    def aggregate_evaluate(self, server_round, results, failures):
        return 0.0, {"accuracy": 0.0}

    def plot_accuracy_chart(self):
        if not self.round_accuracies:
            return

        strategy_name = os.environ.get("SERVER_TYPE", "FedAsync")
        results_dir = os.path.join("results", f"results_{strategy_name}")
        os.makedirs(results_dir, exist_ok=True)

        x = np.array(self.round_times)
        y = np.array(self.round_accuracies)

        if len(x) > 3:
            x_smooth = np.linspace(x.min(), x.max(), 300)
            y_smooth = make_interp_spline(x, y, k=3)(x_smooth)
        else:
            x_smooth, y_smooth = x, y

        plt.figure(figsize=(10, 6), dpi=200)
        plt.plot(x_smooth, y_smooth, color='seagreen', linewidth=2.5, label=f"{strategy_name} (smoothed)")
        plt.scatter(x, y, color='tomato', s=40, zorder=5, label="Actual Rounds")
        plt.title(f"{strategy_name}: Global Accuracy vs. Simulated Time", fontsize=18, fontweight='bold')
        plt.xlabel("Simulated Time (hours)", fontsize=16)
        plt.ylabel("Accuracy", fontsize=16)
        plt.legend(fontsize=13)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()

        # Annotate average wall-clock round duration at the bottom right
        if self.wall_clock_durations:
            avg_wall = np.mean(self.wall_clock_durations)
            plt.text(
                0.98, 0.02,
                f"Avg. wall-clock round duration: {avg_wall:.2f} s",
                fontsize=12,
                color="royalblue",
                ha="right", va="bottom",
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
            )

        plot_path = os.path.join(results_dir, f"{strategy_name}_accuracy_vs_time.png")
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        self.logger.info(f"Saved chart to {plot_path}")

        history_path = os.path.join(results_dir, "server_history.json")
        with open(history_path, "w") as f:
            json.dump({
                "times": self.round_times,
                "accuracies": self.round_accuracies,
                "wall_clock_durations": self.wall_clock_durations,
            }, f, indent=2)
        self.logger.info(f"Saved server history to {history_path}")

def main():
    strategy = FedAsyncStrategy("/app/visibility/visibility_schedule.json")
    fl.server.start_server(
        server_address="[::]:8080",
        config=fl.server.ServerConfig(num_rounds=30),  # Adjusted to 30 rounds.. Because 0-indexed..
        strategy=strategy,
    )
    os.makedirs("results", exist_ok=True)
    strategy.plot_accuracy_chart()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[Server] %(message)s")
    main()
