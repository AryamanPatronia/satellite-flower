import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from typing import Dict, List, Optional, Tuple
import logging
import json
import os
import matplotlib.pyplot as plt
import warnings
import time
import numpy as np
from scipy.interpolate import make_interp_spline

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import sys

# Load the model from client code
sys.path.append("/app/client")
from train import Net

warnings.filterwarnings("ignore", category=DeprecationWarning)


class VisibleClientFedAvg(FedAvg):
    def __init__(self, visibility_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.visibility_path = visibility_path
        self.visibility_schedule = {}
        self.logger = logging.getLogger("Server")
        self.round_accuracies = []
        self.round_times = []
        self.current_weights = None
        self.client_id_map = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_testset = self.load_global_testset()
        self.last_round_end_time = None 

        self.logger.info(f"Visibility schedule path: {visibility_path}")
        if os.path.exists(visibility_path):
            with open(visibility_path, "r") as f:
                self.visibility_schedule = json.load(f)
                self.logger.info(f"Loaded schedule with {len(self.visibility_schedule)} rounds")
        else:
            self.logger.warning("No visibility schedule found. All clients will be used.")

    def load_global_testset(self):
        path = "/app/data/central_testset"
        x_test = torch.load(os.path.join(path, "x.pt"))
        y_test = torch.load(os.path.join(path, "y.pt"))
        return DataLoader(TensorDataset(x_test, y_test), batch_size=32)

    def _get_visible_ids(self, rnd: int) -> List[str]:
        return self.visibility_schedule.get(f"round_{rnd}", [])

    def evaluate(self, server_round: int, parameters):
        if parameters is None:
            return 0.0, {}

        model = Net().to(self.device)
        ndarrays = fl.common.parameters_to_ndarrays(parameters)
        params_dict = zip(model.state_dict().keys(), ndarrays)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in self.global_testset:
                x, y = x.to(self.device), y.to(self.device)
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        acc = correct / total
        self.round_accuracies.append(acc)
        self.round_times.append(server_round * 4)  # Each round = 4 simulated hours
        self.logger.info(f"[ROUND {server_round}] Central Test Accuracy: {acc:.4f}")
        return 0.0, {"accuracy": acc}

    def aggregate_evaluate(self, server_round, results, failures):
        # Skip client-side evaluation and use central evaluation instead
        return self.evaluate(server_round, self.current_weights)

    def aggregate_fit(self, server_round, results, failures):
        now = time.time()
        if self.last_round_end_time is not None:
            round_duration = now - self.last_round_end_time
            self.logger.info(f"[ROUND {server_round}] Wall-clock duration: {round_duration:.2f} seconds")
            if not hasattr(self, "wall_clock_durations"):
                self.wall_clock_durations = []
            self.wall_clock_durations.append(round_duration)
        self.last_round_end_time = now

        round_start = time.time()  # Start timing

        if not results or sum(fit_res.num_examples for _, fit_res in results) == 0:
            self.logger.warning(f"[ROUND {server_round}] No training examples, skipping aggregation.")
            return self.current_weights, {}

        weights_results = super().aggregate_fit(server_round, results, failures)
        self.current_weights = weights_results[0]

        round_end = time.time()  # End timing
        round_duration = round_end - round_start
        self.logger.info(f"[ROUND {server_round}] Duration: {round_duration:.2f} seconds")
        # Optionally, store durations for later analysis/plotting:
        if not hasattr(self, "round_durations"):
            self.round_durations = []
        self.round_durations.append(round_duration)

        return weights_results

    def plot_accuracy_chart(self):
        if not self.round_accuracies or not self.round_times:
            self.logger.warning("No accuracy/time data to plot.")
            return

        strategy_name = os.environ.get("SERVER_TYPE", "FedAvg")
        results_dir = os.path.join("results", f"results_{strategy_name}")
        os.makedirs(results_dir, exist_ok=True)

        x = np.array(self.round_times)
        y = np.array(self.round_accuracies)

        if len(x) > 3:
            x_smooth = np.linspace(x.min(), x.max(), 300)
            spline = make_interp_spline(x, y, k=3)
            y_smooth = spline(x_smooth)
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
        if hasattr(self, "wall_clock_durations") and self.wall_clock_durations:
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
                "wall_clock_durations": getattr(self, "wall_clock_durations", []),
            }, f, indent=2)
        self.logger.info(f"Saved server history to {history_path}")

    def configure_fit(self, server_round: int, parameters, client_manager):
        all_clients = list(client_manager.all().values())
        allowed_ids = self._get_visible_ids(server_round)

        if not self.client_id_map and all_clients:
            sorted_clients = sorted(all_clients, key=lambda c: c.cid)
            for idx, c in enumerate(sorted_clients, start=1):
                self.client_id_map[c.cid] = f"satellite_{idx}"
            self.logger.info(f"Client ID map: {self.client_id_map}")

        visible_clients = [
            c for c in all_clients
            if self.client_id_map.get(c.cid, "") in allowed_ids
        ]

        if not visible_clients:
            self.logger.warning(f"[ROUND {server_round}] No visible clients, skipping.")
            return []

        self.logger.info(f"[ROUND {server_round}] Training on: {[self.client_id_map[c.cid] for c in visible_clients]}")
        return [(client, fl.common.FitIns(parameters, {"server_round": server_round})) for client in visible_clients]


def main():
    strategy = VisibleClientFedAvg(
        visibility_path="/app/visibility/visibility_schedule.json",
        fraction_fit=1.0,
        fraction_evaluate=0.0,  # Not using client evaluation
        min_fit_clients=5,
        min_evaluate_clients=0,
        min_available_clients=5,
        on_fit_config_fn=lambda rnd: {"server_round": rnd},
    )

    fl.server.start_server(
        server_address="[::]:8080",
        config=fl.server.ServerConfig(num_rounds=31),  # Adjusted to 31 rounds.. Because 0-indexed..
        strategy=strategy,
    )

    os.makedirs("results", exist_ok=True)
    strategy.plot_accuracy_chart()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[Server] %(message)s")
    main()
