import flwr as fl
import warnings
import logging

# Suppress deprecated warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Configure server-side logger
logging.basicConfig(
    level=logging.INFO,
    format='[Server] %(message)s'
)
logger = logging.getLogger("server")

def main():
    logger.info("Starting Flower server...")

    # Define FL strategy (FedAvg)
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        min_fit_clients=1,
        min_available_clients=1
    )

    # Start Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )

    logger.info("Server finished training.")

if __name__ == "__main__":
    main()
