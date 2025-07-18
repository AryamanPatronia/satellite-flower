import flwr as fl
import torch
from train import Net, load_data, train, test
from torch.utils.data import DataLoader
import sys
import os
import warnings
import logging

# Suppress deprecation warnings from Flower
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Device config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup logger for each client
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
        self.cid = cid
        self.logger = get_logger(cid)
        self.logger.info("Satellite client initialized")

        self.model = Net().to(DEVICE)
        self.trainset, self.testset = load_data(cid)
        self.trainloader = DataLoader(self.trainset, batch_size=32, shuffle=True)
        self.testloader = DataLoader(self.testset, batch_size=32)

        #  Create a status file for logging
        self.status_file = f"/app/shared_logs/satellite_{cid}.txt"
        self.update_status("Initialized")

    def update_status(self, msg):
        with open(self.status_file, "w") as f:
            f.write(msg)



    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for k, v in zip(state_dict.keys(), parameters):
            state_dict[k] = torch.tensor(v)
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.logger.info("Training started for new round")

        # Update status to "Training"
        self.update_status("Training")

        self.set_parameters(parameters)
        train(self.model, self.trainloader, epochs=3, device=DEVICE)
        self.logger.info("Training completed")
        return self.get_parameters(config={}), len(self.trainset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        # Update status to "Evaluating"
        self.update_status("Evaluating")

        acc = test(self.model, self.testloader, device=DEVICE)
        self.logger.info(f"Evaluation completed. Accuracy: {acc:.4f}")
        
        # Update status to "Done"
        self.update_status("Done")

        return float(1.0 - acc), len(self.testset), {"accuracy": float(acc)}

if __name__ == "__main__":
    cid = os.environ.get("CLIENT_ID", "0")
    fl.client.start_numpy_client(server_address="server:8080", client=SatelliteClient(cid))
