import flwr as fl
import warnings
import logging
import os
import json

warnings.filterwarnings("ignore", category=DeprecationWarning)

logging.basicConfig(
    level=logging.INFO,
    format='[Server] %(message)s'
)
logger = logging.getLogger("server")

class VisibleClientFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, visibility_schedule_path, **kwargs):
        super().__init__(**kwargs)
        self.visibility_schedule_path = visibility_schedule_path

    def configure_fit(self, server_round, parameters, client_manager):
        # Load visibility schedule
        try:
            with open(self.visibility_schedule_path, "r") as f:
                schedule = json.load(f)
        except FileNotFoundError:
            logger.warning("Visibility schedule not found, using all clients.")
            return super().configure_fit(server_round, parameters, client_manager)

        round_key = f"round_{server_round}"
        allowed_ids = set(schedule.get(round_key, []))

        available_clients = list(client_manager.all().values())
        visible_clients = [c for c in available_clients if c.cid in allowed_ids]

        if not visible_clients:
            logger.warning(f"No visible clients for round {server_round}, using all clients.")
            visible_clients = available_clients

        config = {"server_round": server_round}
        # Return a list of (client, FitIns)
        return [
            (client, fl.common.FitIns(parameters, config))
            for client in visible_clients
        ]

    def configure_evaluate(self, server_round, parameters, client_manager):
        try:
            with open(self.visibility_schedule_path, "r") as f:
                schedule = json.load(f)
        except FileNotFoundError:
            logger.warning("Visibility schedule not found, using all clients.")
            return super().configure_evaluate(server_round, parameters, client_manager)

        round_key = f"round_{server_round}"
        allowed_ids = set(schedule.get(round_key, []))

        available_clients = list(client_manager.all().values())
        visible_clients = [c for c in available_clients if c.cid in allowed_ids]

        if not visible_clients:
            logger.warning(f"No visible clients for round {server_round}, using all clients.")
            visible_clients = available_clients

        config = {"server_round": server_round}
        return [
            (client, fl.common.EvaluateIns(parameters, config))
            for client in visible_clients
        ]

    def aggregate_fit(self, server_round, results, failures):
        filtered_results = [
            (client, fit_res)
            for client, fit_res in results
            if getattr(fit_res, "num_examples", 0) > 0
        ]
        if not filtered_results:
            logger.warning(f"No clients contributed training examples in round {server_round}. Skipping aggregation.")
            return None, {}
        return super().aggregate_fit(server_round, filtered_results, failures)

    def aggregate_evaluate(self, server_round, results, failures):
        filtered_results = [
            (client, eval_res)
            for client, eval_res in results
            if getattr(eval_res, "num_examples", 0) > 0
        ]
        if not filtered_results:
            logger.warning(f"No clients contributed evaluation examples in round {server_round}. Skipping evaluation aggregation.")
            return None
        return super().aggregate_evaluate(server_round, filtered_results, failures)

def main():
    logger.info("Starting Flower server...")

    num_rounds = int(os.environ.get("NUM_ROUNDS", "10"))

    strategy = VisibleClientFedAvg(
        visibility_schedule_path="/app/visibility/visibility_schedule.json",
        fraction_fit=1.0,
        min_fit_clients=1,
        min_available_clients=1,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    logger.info("Server finished training.")

if __name__ == "__main__":
    main()
