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

warnings.filterwarnings("ignore", category=DeprecationWarning)


# Using FedAvg strategy with default visibility schedule...
class VisibleClientFedAvg(FedAvg):
    def __init__(self, visibility_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.visibility_path = visibility_path
        self.visibility_schedule = {}
        self.logger = logging.getLogger("Server")
        self.round_accuracies = []
        self.round_times = []
        self.start_time = None
        self.client_id_map = {}  # <-- Add this line

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

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, fl.common.EvaluateRes]],
        failures: List[BaseException],
    ) -> Optional[Tuple[float, Dict[str, float]]]:
        if not results:
            return None

        # Aggregate loss (weighted average)
        losses = [r.loss for _, r in results]
        examples = [r.num_examples for _, r in results]
        total_examples = sum(examples)
        if total_examples == 0:
            self.logger.warning(f"[ROUND {rnd}] No examples returned, defaulting to 0.0 accuracy/loss")
            self.round_accuracies.append(0.0)
            self.round_times.append(rnd * 4)
            return 0.0, {"accuracy": 0.0}

        weighted_loss = sum(loss * n for loss, n in zip(losses, examples)) / total_examples

        # Aggregate accuracy (weighted average)
        accs = [r.metrics.get("accuracy", 0.0) for _, r in results]
        weighted_acc = sum(acc * n for acc, n in zip(accs, examples)) / total_examples

        self.round_accuracies.append(weighted_acc)
        simulated_time = rnd * 4  # Each round = 4 simulated hours
        self.round_times.append(simulated_time)
        self.logger.info(f"[ROUND {rnd}] Aggregated Accuracy: {weighted_acc:.4f} (Simulated Time: {simulated_time}h)")

        return weighted_loss, {"accuracy": weighted_acc}

    def plot_accuracy_chart(self):
        if not self.round_accuracies or not self.round_times:
            self.logger.warning("No accuracy/time data to plot.")
            return

        # Use the strategy name as a subfolder
        strategy_name = os.environ.get("SERVER_TYPE", "FedAvg")
        results_dir = os.path.join("/app/results", f"results_{strategy_name}")
        os.makedirs(results_dir, exist_ok=True)

        x = np.array(self.round_times)
        y = np.array(self.round_accuracies)

        # Spline interpolation for smooth curve (only if enough points)
        if len(x) > 3:
            x_smooth = np.linspace(x.min(), x.max(), 300)
            spline = make_interp_spline(x, y, k=3)
            y_smooth = spline(x_smooth)
        else:
            x_smooth, y_smooth = x, y

        plt.figure(figsize=(10, 6), dpi=200)
        strategy_name = "FedAvg"
        plt.plot(x_smooth, y_smooth, color='royalblue', linewidth=2.5, label=f"{strategy_name} (smoothed)")
        plt.scatter(x, y, color='crimson', s=40, zorder=5, label="Actual Rounds")
        plt.title(f"{strategy_name}: Global Accuracy vs. Simulated Time", fontsize=18, fontweight='bold')
        plt.xlabel("Simulated Time (hours)", fontsize=16)
        plt.ylabel("Accuracy", fontsize=16)
        plt.legend(fontsize=13)
        plt.tight_layout()
        plot_path = os.path.join(results_dir, f"{strategy_name}_accuracy_vs_time.png")
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        self.logger.info(f"Saved accuracy vs. time plot to {plot_path}")

        # Save training history to JSON
        history_path = os.path.join(results_dir, "server_history.json")
        with open(history_path, "w") as f:
            json.dump({
                "times": self.round_times,
                "accuracies": self.round_accuracies,
            }, f, indent=2)
        self.logger.info(f"Saved training history to {history_path}")

    def configure_fit(
        self,
        server_round: int,
        parameters,
        client_manager,
    ):
        all_clients = list(client_manager.all().values())
        allowed_ids = self._get_visible_ids(server_round)

        # Build the mapping from UUID to satellite name on the first round
        if not self.client_id_map and all_clients:
            sorted_clients = sorted(all_clients, key=lambda c: c.cid)
            for idx, c in enumerate(sorted_clients, start=1):
                self.client_id_map[c.cid] = f"satellite_{idx}"
            self.logger.info(f"Client ID map: {self.client_id_map}")

        # Use the mapping for visibility
        visible_clients = [
            c for c in all_clients
            if self.client_id_map.get(c.cid, "") in allowed_ids
        ]

        self.logger.debug(f"[ROUND {server_round}] Allowed IDs: {set(allowed_ids)}")
        self.logger.debug(f"[ROUND {server_round}] Available client names: {[self.client_id_map.get(c.cid, '') for c in all_clients]}")

        if not visible_clients:
            self.logger.warning(f"No visible clients for round {server_round}, skipping round.")
            return []  # Skip the round

        # Only use visible clients
        return super().configure_fit(server_round, parameters, client_manager)


def main():
    strategy = VisibleClientFedAvg(
        visibility_path="/app/visibility/visibility_schedule.json",
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=5,
        min_evaluate_clients=5,
        min_available_clients=5,
        on_fit_config_fn=lambda rnd: {"server_round": rnd},
        on_evaluate_config_fn=lambda rnd: {"server_round": rnd},
    )

    # Start the server and store the returned history
    history = fl.server.start_server(
        server_address="[::]:8080",
        config=fl.server.ServerConfig(num_rounds=30),
        strategy=strategy,
    )


    strategy.plot_accuracy_chart()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[Server] %(message)s")
    main()
