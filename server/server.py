import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from typing import List, Optional, Tuple
import logging
import json
import os

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class VisibleClientFedAvg(FedAvg):
    def __init__(self, visibility_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.visibility_path = visibility_path
        self.visibility_schedule = {}
        self.logger = logging.getLogger("Server")

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

def configure_fit(
    self,
    server_round: int,
    parameters,
    client_manager,
):
    all_clients = list(client_manager.all().values())
    allowed_ids = self._get_visible_ids(server_round)
    visible_clients = [c for c in all_clients if c.cid in allowed_ids]

    self.logger.debug(f"[ROUND {server_round}] Allowed IDs: {set(allowed_ids)}")
    self.logger.debug(f"[ROUND {server_round}] Available client IDs: {[c.cid for c in all_clients]}")

    if not visible_clients:
        self.logger.warning(f"No visible clients for round {server_round}, using all clients.")
        visible_clients = all_clients

    return super().configure_fit(server_round, parameters, client_manager)


    def configure_evaluate(
        self,
        rnd: int,
        parameters,
        client_manager,
    ) -> List[Tuple[ClientProxy, dict]]:
        return super().configure_evaluate(rnd, parameters, client_manager=client_manager)

def main():
    strategy = VisibleClientFedAvg(
        visibility_path="/app/visibility/visibility_schedule.json",
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        on_fit_config_fn=lambda rnd: {"server_round": rnd},
        on_evaluate_config_fn=lambda rnd: {"server_round": rnd},
    )

    fl.server.start_server(
        server_address="[::]:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[Server] %(message)s")
    main()
