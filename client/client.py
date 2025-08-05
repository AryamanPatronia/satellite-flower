import flwr as fl
import torch
from train import Net, load_data, train, test
from torch.utils.data import DataLoader
import os
import warnings
import logging
import json

import socket # For checking server availability

warnings.filterwarnings("ignore", category=DeprecationWarning)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_logger(cid):
    logger = logging.getLogger(f"Satellite_{cid}")
    handler = logging.StreamHandler()
    formatter = logging.Formatter(f"[Satellite {cid}] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

class SatelliteClient(fl.client.NumPyClient):
    def __init__(self, cid):
        self.cid = f"satellite_{cid}"
        self.logger = get_logger(self.cid)
        self.logger.info("Satellite client initialized")

        self.model = Net().to(DEVICE)
        self.trainset, self.testset = load_data(cid)
        self.trainloader = DataLoader(self.trainset, batch_size=32, shuffle=True)
        self.testloader = DataLoader(self.testset, batch_size=32)

        self.status_file = f"/app/shared_logs/satellite_{cid}.txt"
        self.update_status("Initialized")

    def update_status(self, msg):
        with open(self.status_file, "w") as f:
            f.write(msg)

    def is_visible(self, round_num):
        schedule_path = "/app/visibility/visibility_schedule.json"
        if not os.path.exists(schedule_path):
            return True

        with open(schedule_path, "r") as f:
            visibility = json.load(f)

        round_key = f"round_{int(round_num)}"
        allowed = visibility.get(round_key, [])
        return self.cid in allowed

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for k, v in zip(state_dict.keys(), parameters):
            state_dict[k] = torch.tensor(v)
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        current_round = int(config.get("server_round", 1))
        self.logger.info(f"Config received: {config}")

        if not self.is_visible(current_round):
            self.logger.info(f"Skipping round {current_round} (not visible)")
            self.update_status(f"Skipped round {current_round}")
            return self.get_parameters(config={}), 0, {}

        self.logger.info(f"Training for round {current_round}")
        self.update_status("Training")
        self.set_parameters(parameters)
        train(self.model, self.trainloader, epochs=3, device=DEVICE)
        self.logger.info("Training completed")
        self.update_status("Trained")
        return self.get_parameters(config={}), len(self.trainset), {}

    def evaluate(self, parameters, config):
        current_round = int(config.get("server_round", 1))
        self.logger.info(f"Config received: {config}")

        if not self.is_visible(current_round):
            self.logger.info(f"Skipping eval in round {current_round} (not visible)")
            self.update_status(f"Skipped eval round {current_round}")
            return 0.0, 0, {"accuracy": 0.0}

        self.set_parameters(parameters)
        self.update_status("Evaluating")
        acc = test(self.model, self.testloader, device=DEVICE)
        self.logger.info(f"Evaluation completed. Accuracy: {acc:.4f}")
        self.update_status("Done")
        return float(1.0 - acc), len(self.testset), {"accuracy": float(acc)}
    
    @staticmethod
    def wait_for_server(server_address):
        host, port = server_address.split(":")
        port = int(port)
        print(f"[Client] Waiting for server {host}:{port} to be available...")
        while True:
            try:
                with socket.create_connection((host, port), timeout=2):
                    print("[Client] Server is ready.")
                    break
            except OSError:
                print("[Client] Server not ready yet, retrying in 2s...")
                import time
                time.sleep(2)

if __name__ == "__main__":
    cid = os.environ.get("CLIENT_ID", "1")
    server_address = os.environ.get("SERVER_ADDRESS", "server:8080")
    
    # Wait for gRPC server to become available
    SatelliteClient.wait_for_server(server_address)

    client = SatelliteClient(cid)
    fl.client.start_client(
        server_address=server_address,
        client=client,
    )
