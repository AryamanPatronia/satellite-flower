import json
from flwr.server.strategy import FedAvg

class VisibleClientFedAvg(FedAvg):
    def __init__(self, visibility_schedule_path, **kwargs):
        super().__init__(**kwargs)
        self.visibility_schedule_path = visibility_schedule_path

    def configure_fit(self, server_round, parameters, client_manager):
        visible_clients = self._get_visible_clients(server_round, client_manager)
        if not visible_clients:
            # Fallback: use all clients if none are visible
            visible_clients = list(client_manager.all().values())
        # Sample the required number of clients
        sample_size = min(self.min_fit_clients, len(visible_clients))
        sampled_clients = visible_clients[:sample_size]
        return [
            (client, self.fit_ins(parameters, {}))
            for client in sampled_clients
        ]

    def configure_evaluate(self, server_round, parameters, client_manager):
        visible_clients = self._get_visible_clients(server_round, client_manager)
        if not visible_clients:
            visible_clients = list(client_manager.all().values())
        sample_size = min(self.min_available_clients, len(visible_clients))
        sampled_clients = visible_clients[:sample_size]
        return [
            (client, self.evaluate_ins(parameters, {}))
            for client in sampled_clients
        ]

    def _get_visible_clients(self, server_round, client_manager):
        try:
            with open(self.visibility_schedule_path, "r") as f:
                schedule = json.load(f)
        except FileNotFoundError:
            return list(client_manager.all().values())

        round_key = f"round_{server_round}"
        allowed_ids = schedule.get(round_key, [])
        available_clients = list(client_manager.all().values())
        return [c for c in available_clients if c.cid in allowed_ids]

def aggregate_fit(self, server_round, results, failures):
    filtered_results = [
        (client, fit_res)
        for client, fit_res in results
        if getattr(fit_res, "num_examples", 0) > 0
    ]
    if not filtered_results:
        print(f"[Server] WARNING: No clients contributed training examples in round {server_round}. Skipping aggregation.")
        # Return dummy parameters to avoid crashing
        return [], {}  # <- Return valid fallback
    return super().aggregate_fit(server_round, filtered_results, failures)

def aggregate_evaluate(self, server_round, results, failures):
    filtered_results = [
        (client, eval_res)
        for client, eval_res in results
        if getattr(eval_res, "num_examples", 0) > 0
    ]
    if not filtered_results:
        print(f"[Server] WARNING: No clients contributed evaluation examples in round {server_round}. Skipping evaluation aggregation.")
        return 0.0, {}  # <- Return fallback to avoid unpacking None
    return super().aggregate_evaluate(server_round, filtered_results, failures)