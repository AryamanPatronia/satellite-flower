import flwr as fl
from flwr.common import FitIns, EvaluateIns, GetParametersIns
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from typing import Dict, List, Optional, Tuple
import logging
import json
import os
import matplotlib.pyplot as plt
import warnings
import time
import numpy as np
from scipy.interpolate import make_interp_spline

warnings.filterwarnings("ignore", category=DeprecationWarning)

class FedAsyncStrategy(Strategy):
    def __init__(self, visibility_path: str):
        self.visibility_path = visibility_path
        self.visibility_schedule = {}
        self.logger = logging.getLogger("Server")
        self.current_weights = None
        self.client_staleness: Dict[str, int] = {}
        self.round_accuracies = []
        self.round_times = []
        self.client_id_map = {}  # Maps UUID -> satellite_X

        self.logger.info(f"Visibility schedule path: {visibility_path}")
        if os.path.exists(visibility_path):
            with open(visibility_path, "r") as f:
                self.visibility_schedule = json.load(f)
                self.logger.info("Visibility schedule exists")
                self.logger.info(f"Loaded schedule with {len(self.visibility_schedule)} rounds")
        else:
            self.logger.warning("No visibility schedule found. All clients will be used.")

    def _get_visible_ids(self, rnd: int) -> List[str]:
        key = f"round_{rnd}"
        return self.visibility_schedule.get(key, [])

    def initialize_parameters(self, client_manager):
        self.logger.info("Initializing parameters from one client")
        waited = 0
        while not client_manager.all():
            time.sleep(1)
            waited += 1
            if waited > 60:
                raise RuntimeError("No clients connected after 60 seconds")

        # Build UUID -> satellite_X map (based on connection order)
        sorted_clients = sorted(client_manager.all().items())
        for idx, (uuid, _) in enumerate(sorted_clients, start=1):
            self.client_id_map[uuid] = f"satellite_{idx}"
        self.logger.info(f"Client ID Map: {self.client_id_map}")

        # Pick any client for parameter init
        client = list(client_manager.all().values())[0]
        ins = GetParametersIns(config={})
        res = client.get_parameters(ins, timeout=60, group_id="default")
        return res.parameters

    def configure_fit(self, server_round, parameters, client_manager):
        all_clients = list(client_manager.all().values())

        allowed_ids = self._get_visible_ids(server_round)
        self.logger.info(f"Allowed IDs for round {server_round}: {allowed_ids}")
        self.logger.info(f"Connected UUIDs: {[c.cid for c in all_clients]}")

        visible_clients = [
            c for c in all_clients
            if self.client_id_map.get(c.cid, "") in allowed_ids
        ]

        if not visible_clients:
            self.logger.warning(f"No visible clients for round {server_round}, using all clients.")
            visible_clients = all_clients

        for client in visible_clients:
            self.client_staleness.setdefault(client.cid, 0)

        return [
            (client, FitIns(parameters, {"server_round": server_round}))
            for client in visible_clients
        ]

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return self.current_weights, {}

        for client, fit_res in results:
            staleness = self.client_staleness.get(client.cid, 0)
            alpha = 1.0 / (staleness + 1)

            client_weights = fl.common.parameters_to_ndarrays(fit_res.parameters)
            if self.current_weights is None:
                self.current_weights = client_weights
            else:
                self.current_weights = [
                    (1 - alpha) * cw + alpha * lw for cw, lw in zip(self.current_weights, client_weights)
                ]

            self.client_staleness[client.cid] = 0

        for cid in self.client_staleness:
            if cid not in [c.cid for c, _ in results]:
                self.client_staleness[cid] += 1

        return fl.common.ndarrays_to_parameters(self.current_weights), {}

    def configure_evaluate(self, server_round, parameters, client_manager):
        clients = list(client_manager.all().values())
        return [
            (client, EvaluateIns(parameters, {"server_round": server_round}))
            for client in clients
        ]

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            self.round_accuracies.append(0.0)
            self.round_times.append(server_round * 4)
            self.logger.warning(f"[ROUND {server_round}] No evaluation results, defaulting to 0.0 accuracy")
            return 0.0, {"accuracy": 0.0}

        losses = [r.loss for _, r in results]
        examples = [r.num_examples for _, r in results]
        accs = [r.metrics.get("accuracy", 0.0) for _, r in results]
        total_examples = sum(examples)

        weighted_loss = sum(loss * n for loss, n in zip(losses, examples)) / total_examples
        weighted_acc = sum(acc * n for acc, n in zip(accs, examples)) / total_examples

        self.round_accuracies.append(weighted_acc)
        self.round_times.append(server_round * 4)
        self.logger.info(f"[ROUND {server_round}] Async Aggregated Accuracy: {weighted_acc:.4f}")
        return weighted_loss, {"accuracy": weighted_acc}

    def evaluate(self, server_round: int, parameters):
        return None

    def plot_accuracy_chart(self):
        if not self.round_accuracies or not self.round_times:
            self.logger.warning("No accuracy/time data to plot.")
            return

        results_dir = "/app/results"
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
        plt.plot(x_smooth, y_smooth, color='green', linewidth=2.5, label="FedAsync (smoothed)")
        plt.scatter(x, y, color='orange', s=40, zorder=5, label="Actual Rounds")
        plt.title("FedAsync: Global Accuracy vs. Simulated Time", fontsize=18, fontweight='bold')
        plt.xlabel("Simulated Time (hours)", fontsize=16)
        plt.ylabel("Accuracy", fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend(fontsize=13)
        plt.tight_layout()
        plot_path = os.path.join(results_dir, "FedAsync_accuracy_vs_time.png")
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        self.logger.info(f"Saved FedAsync accuracy plot to {plot_path}")

        # Save accuracy/time to JSON for external plotting
        history_path = os.path.join(results_dir, "server_history.json")
        with open(history_path, "w") as f:
            json.dump({
                "times": self.round_times,
                "accuracies": self.round_accuracies,
            }, f, indent=2)
        self.logger.info(f"Saved training history to {history_path}")

def main():
    strategy = FedAsyncStrategy(
        visibility_path="/app/visibility/visibility_schedule.json"
    )

    history = fl.server.start_server(
        server_address="[::]:8080",
        config=fl.server.ServerConfig(num_rounds=30),
        strategy=strategy,
    )

    os.makedirs("results", exist_ok=True)
    strategy.plot_accuracy_chart()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[Server] %(message)s")
    main()
